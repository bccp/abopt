def default_addmul(a, b, s):
    if s is 0:
        return 1.0 * a # always a new instance is created
    if a is 0:
        return b * s
    return a + b * s

def default_dot(a, b):
    if hasattr(a, 'dot'):
        return a.dot(b)
    try:
        return sum(a * b)
    except TypeError:
        return float(a * b)

class parameter(object):
    """ The parameter decorator declares a parameter to
        the optimizer object.

        This class follows the descriptor pattern in Python.

        The return value of the accessor is used to set the attribute.
    """
    def __init__(self, convert):
        self.name = convert.__name__
        self.default = convert.__defaults__[0]
        self.convert = convert
        self.__doc__ = convert.__doc__

    def __get__(self, instance, owner):
        if isinstance(instance, Optimizer):
            return instance.config.get(self.name, self.default)
        else:
            return self

    def __set__(self, instance, value):
        instance.config[self.name] = self.convert(value)

class Optimizer(object):
    @parameter
    def tol(value=1e-6):
        """Tolerance of objective"""
        assert value > 0
        return value
    @parameter
    def gtol(value=1e-6):
        """Tolerance of gradient"""
        assert value >= 0
        return value
    @parameter
    def maxsteps(value=1000):
        """Maximium  number of iterations"""
        return int(value)

    def copy(self, a):
        return self.addmul(a, a, 0)

    def mul(self, a, s):
        return self.addmul(0, a, s)

    def __setattr__(self, key, value):
        # only allow setting parameters
        if hasattr(type(self), key) and isinstance(getattr(type(self), key), parameter):
            return object.__setattr__(self, key, value)
        else:
            raise AttributeError("Setting attribute %s on an Optimizer of type %s is not supported" % (key, type(self)))

    def __init__(self,
                 addmul=default_addmul,
                 dot=default_dot,
                 ):
        """
            Parameters
            ----------
            addmul : callable
                addmul(a, b, s) returns a + b * s as a new vector from vector a, b and a Python scalar s.
                when s is 0 (not 0.0), it returns a copy of a, serving as a constructor of new vectors.
                when a is 0, it returns b * s. The default is simply `a + b * s` with optimizations
                for zeros.

            dot : callable
                dot(a, b) returns the inner product of vectors a and b as a Python scalar; the default
                works for most cases by first looking for a `dot` method, then fallback to `sum` of
                the `*` operator.
        """
        # FIXME: use check the function signature is correct.
        d = self.__dict__
        d['dot'] = dot
        d['addmul'] = addmul
        d['config'] = {}

    def minimize(self, objective, gradient, x0, monitor=None):
        raise NotImplementedError

class State(dict):
    def __init__(self, optimizer, **kwargs):
        dict.__init__(self)
        self.update(kwargs)
        if 'xnorm' not in kwargs:
            self['xnorm'] = optimizer.dot(self['x'], self['x']) ** 0.5
        self.optimizer = optimizer

    def __str__(self):
        d = {}
        d['it'] = self['it']
        d['y'] = self['y']
        d['xnorm'] = self['xnorm']
        d['gnorm'] = self['gnorm']
        d['fev'] = self['fev']
        d['gev'] = self['gev']
        d['dy'] = self['dy']
        if self['dy'] is None:
            d['dy'] = 'None'
        else:
            d['dy'] = '%g' % self['dy']
        return "Iteration %(it)d: y = %(y)g dy = %(dy)s fev = %(fev)d gev = %(gev)d gnorm = %(gnorm)g xnorm = %(xnorm)g" % d

class GradientDescent(Optimizer):
    @parameter
    def gamma(value=1e-3):
        """descent rate parameter"""
        assert value > 0
        return value

    def minimize(self, objective, gradient, x0, monitor=None):
        # FIXME: line search
        it = 0
        dy = None # initial it
        y0 = objective(x0)
        while it < self.maxsteps:
            dx0 = gradient(x0)
            gnorm = self.dot(dx0, dx0) ** 0.5
            state = State(self, x=x0, y=y0, dy=dy, fev=it+1, gev=it+1, g=dx0, gnorm=gnorm, it=it)
            if monitor:
                monitor(state)

            if gnorm < self.gtol: break
            if dy is not None and dy < self.tol: break

            # move to the next point
            x1 = self.addmul(x0, dx0, -self.gamma)
            y1 = objective(x1)
            dy = abs(y0 - y1)
            x0 = x1
            y0 = y1
            it = it + 1

        return state

class LBFGS(Optimizer):
    @parameter
    def m(value=10):
        """number of vectors for approximating Hessian"""
        return int(value)

    def minimize(self, objective, gradient, x0, monitor=None):

        it = 0
        x = self.copy(x0)
        val = objective(x)
        g = gradient(x)
        gnorm = self.dot(g, g) ** 0.5

        rho = []
        S = []
        Y = []

        xprev = self.copy(x)
        gprev = self.copy(g)
        H0k = 1.0
        it = 0
        rate = 1.0

        fev, gev = 0, 0

        converged_iters = 0

        dy = None
        use_steepest_descent = False
        converged_state = "NO"

        state = State(self, x=x, y=val, dy=dy, g=g, gnorm=gnorm,
                      it=it, fev=fev, gev=gev, steepest_descent=use_steepest_descent, converged=converged_state)
        if monitor:
            monitor(state)

        while True:
            q = self.copy(g)

            alpha = []
            for i in range(len(S)):
                dotproduct = self.dot(S[i], q)
                alpha.append(rho[i] * dotproduct)
                q = self.addmul(q, Y[i], -alpha[i])

            z = self.mul(q, H0k)

            for i in reversed(list(range(len(S)))):
                dotproduct = self.dot(Y[i], z)
                beta = rho[i] * dotproduct
                z = self.addmul(z, S[i], alpha[i] - beta)

            use_steepest_descent = False
            znorm = self.dot(z, z) ** 0.5
            zg = 0.0 if znorm == 0 else self.dot(z, g) / znorm
            zg_grannorm = 0.0 if gnorm == 0 else zg / gnorm

            if zg_grannorm < 0.01:
                z = self.copy(g)
                zg = 1.0
                use_steepest_descent = True

            oldval = val

            rate = 1.0 / gnorm if (it == 0 or use_steepest_descent) else 1.0

            # doing only backtracking line search
            # FIXME: implement more-thuente
            tau = 0.5
            c = 1e-5
            search_x = self.addmul(x, z, -rate)
            newval = objective(search_x)
            fev += 1
            while True:
                valmax = max(abs(val), abs(newval))
                if abs(val - newval) / max(valmax, 1.0) < self.tol and val >= newval:
                    break
                if val - newval >= rate * c * zg:
                    break

                rate *= tau
                search_x = self.addmul(x, z, -rate)
                newval = objective(search_x)
                fev += 1

            # now move
            x = self.copy(search_x)
            val = newval
            g = gradient(x)
            gev = gev + 1
            gnorm = self.dot(g, g) ** 0.5

            # end of backtracking line search

            it += 1
            dy = abs(val - oldval)
            valmax = max(abs(val), abs(oldval))
            ratio = dy / max(valmax, 1.0)
            min_iter = 10

            if ratio < self.tol and it >= min_iter:
                converged_iters += 1
                if converged_iters >= 3:
                    converged_state = "YES: Tolerance achieved"
                    break
            else:
                converged_iters = 0

            if gnorm < self.gtol:
                converged_state = "YES: Gradient tolerance achieved"
                break

            # move everything down
            S.insert(0, self.addmul(x, xprev, -1))
            Y.insert(0, self.addmul(g, gprev, -1))
            ys = self.dot(S[0], Y[0])
            yy = self.dot(Y[0], Y[0])

            if ys == 0.0:
                converged_state = "NO: LBFGS didn't move for some reason ys is 0, QUITTING"
                break
            if yy == 0.0:
                converged_state = "NO: LBFGS didn't move for some reason yy is 0, QUITTING"
                break

            rho.insert(0, 1.0 / ys)

            if len(S) > self.m:
                del S[-1]
                del Y[-1]
                del rho[-1]

            H0k = ys / yy

            xprev = self.copy(x)
            gprev = self.copy(g)

            if it > self.maxsteps:
                converged_state = "NO: but maximum number of iterations reached. QUITTING."
                break

            state = State(self, x=x, y=val, dy=dy, g=g, gnorm=gnorm, it=it, fev=fev, gev=gev, steepest_descent=use_steepest_descent, converged=converged_state)
            if monitor:
                monitor(state)

        # update the state one last time
        state=State(self, x=x, y=val, dy=dy, g=g, gnorm=gnorm, it=it, fev=fev, gev=gev, steepest_descent=use_steepest_descent, converged=converged_state)
        if monitor:
            monitor(state)

        return state

