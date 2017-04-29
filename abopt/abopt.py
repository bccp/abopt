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

class ConvergenceStatus(BaseException): pass 
class TooManySteps(ConvergenceStatus): pass
class BadDirection(ConvergenceStatus): pass
class Converged(ConvergenceStatus): pass

class Optimizer(object):
    @parameter
    def tol(value=1e-6):
        """Relative tolerance of change in objective. Terminate if dy < tol * y + atol"""
        assert value >= 0
        return value

    @parameter
    def atol(value=0):
        """Absolute tolerance of change objective. Terminate if dy < tol * y + atol """
        assert value >= 0
        return value

    @parameter
    def ymin(value=None):
        """ending objective. Terminate if y < ymin """
        return value

    @parameter
    def gtol(value=1e-6):
        """Absolute tolerance of gradient. Terminate if gnorm < gtol """
        assert value >= 0
        return value

    @parameter
    def maxsteps(value=1000):
        """Maximium  number of iterations"""
        return int(value)

    @parameter
    def csteps(value=3):
        """ number of iterations dy < threshold before confirming convergence """
        return int(value)
    @parameter
    def minsteps(value=10):
        """ minimum number of steps """
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

def simpleproperty(varname, mode):
    def fget(self): return getattr(self, varname)
    def fset(self, value): return setattr(self, varname, value)
    if 'w' in mode:
        r = property(fget, fset)
    else:
        r = property(fget)
    return r

class BaseState(object):
    __slots__ = []

    x = simpleproperty('_x', mode='r')
    xnorm = simpleproperty('_xnorm', mode='r')
    @x.setter
    def x(self, value):
        self._x = value
        self._xnorm = self.optimizer.dot(value, value) ** 0.5

    g = simpleproperty('_g', mode='r')
    gnorm = simpleproperty('_gnorm', mode='r')
    @g.setter
    def g(self, value):
        self._g = value
        self._gnorm = self.optimizer.dot(value, value) ** 0.5

    it = simpleproperty('_it', mode='rw')
    fev = simpleproperty('_fev', mode='rw')
    gev = simpleproperty('_gev', mode='rw')
    dy = simpleproperty('_dy', mode='rw')
    status = simpleproperty('_status', mode='rw')

    def __getitem__(self, name):
        return getattr(self, name)

    def __init__(self, optimizer):
        self.__dict__['optimizer'] = optimizer

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
    """ GradientDescent ignores minsteps, csteps, tol and atol. It always run for maxsteps
         -- Since there is no linear search we can never know about the convergence.
    """
    @parameter
    def gamma(value=1e-3):
        """descent rate parameter"""
        assert value > 0
        return value

    class State(BaseState):
        cit = simpleproperty('_cit', mode='rw')
        pass

    def minimize(self, objective, gradient, x0, monitor=None):
        if isinstance(x0, GradientDescent.State):
            state = x0
        else:
            state = GradientDescent.State(self)

            # FIXME: line search
            state.dy = None # initial it
            state.x = x0
            state.y = objective(x0)
            state.fev = 1
            state.gev = 0
            state.cit = 0 # number of contiguous small dy steps
            state.status = None

        state.it = 0

        while state.it < self.maxsteps:
            state.g = gradient(state.x)
            state.gev = state.gev + 1

            if monitor:
                monitor(state)

            if state.gnorm < self.gtol: break
            if self.ymin is not None and state.y < self.ymin : break
            # move to the next point
            x1 = self.addmul(state.x, state.g, -self.gamma)
            y1 = objective(x1)
            state.fev = state.fev + 1

            state.dy = abs(y1 - state.y)
            state.x = x1
            state.y = y1
            state.it = state.it + 1

        return state

class LBFGS(Optimizer):
    @parameter
    def m(value=10):
        """number of vectors for approximating Hessian"""
        return int(value)

    class State(BaseState):
        H0k = simpleproperty('_H0k', mode='rw')
        rho = simpleproperty('_rho', mode='rw')
        S = simpleproperty('_S', mode='rw')
        Y = simpleproperty('_Y', mode='rw')

    def linesearch(self, objective, state, z, zg, rate):
        # doing only backtracking line search
        # FIXME: implement more-thuente
        tau = 0.5
        c = 1e-5
        x1 = self.addmul(state.x, z, -rate)
        y1 = objective(x1)
        state.fev = state.fev + 1
        while True:
            valmax = max(abs(y1), abs(state.y))
            thresh = self.tol * max(valmax, 1.0) + self.atol

            #print(rate, state.y, y1, state.x, x1)
            if self.converged(state, y1): break

            # sufficient descent
            if state.y - y1 >= rate * c * zg:
                break

            rate *= tau
            x1 = self.addmul(state.x, z, -rate)
            y1 = objective(x1)
            #print('new point ', x1, y1, state.x, state.y)
            state.fev = state.fev + 1

        return x1, y1

    def converged(self, state, y1):
        valmax = max(abs(y1), abs(state.y))
        thresh = self.tol * max(valmax, 1.0) + self.atol
        return abs(y1 - state.y) < thresh and state.y >= y1

    def minimize(self, objective, gradient, x0, monitor=None):
        if isinstance(x0, LBFGS.State):
            state = x0
        else:
            state = LBFGS.State(self)

            state.x = self.copy(x0)
            state.rho = []
            state.S = []
            state.Y = []
            state.dy = None

            state.H0k = 1.0

        converged_iters = 0

        dy = None
        use_steepest_descent = False

        state.y = objective(state.x)
        state.g = gradient(state.x)
        state.fev, state.gev = 1, 1
        state.it = 0
        if monitor: monitor(state)


        while True:
            q = self.copy(state.g)
            alpha = []
            for i in range(len(state.S)):
                dotproduct = self.dot(state.S[i], q)
                alpha.append(state.rho[i] * dotproduct)
                q = self.addmul(q, state.Y[i], -alpha[i])

            z = self.mul(q, state.H0k)

            for i in reversed(list(range(len(state.S)))):
                dotproduct = self.dot(state.Y[i], z)
                beta = state.rho[i] * dotproduct
                z = self.addmul(z, state.S[i], alpha[i] - beta)

            use_steepest_descent = False
            znorm = self.dot(z, z) ** 0.5
            zg = 0.0 if znorm == 0 else self.dot(z, state.g) / znorm
            zg_grannorm = 0.0 if state.gnorm == 0 else zg / state.gnorm

            if zg_grannorm < 0.01:
                # L-BFGS gave a bad direction.
                z = self.copy(state.g)
                zg = 1.0
                use_steepest_descent = True

            rate = 1.0 / state.gnorm if (state.it == 0 or use_steepest_descent) else 1.0

            x1, y1 = self.linesearch(objective, state, z, zg, rate)

            if self.converged(state, y1) and state.it >= self.minsteps:
                converged_iters += 1
            else:
                converged_iters = 0

            g1 = gradient(x1)
            state.gev = state.gev + 1

            # hessian update
            # XXX: shall we do this when use_steepest_descent is true?
            state.S.insert(0, self.addmul(x1, state.x, -1))
            state.Y.insert(0, self.addmul(g1, state.g, -1))
            ys = self.dot(state.S[0], state.Y[0])
            yy = self.dot(state.Y[0], state.Y[0])

            if ys == 0.0:
                state.status = BadDirection("LBFGS didn't move for some reason ys is 0, QUITTING")
                break
            if yy == 0.0:
                state.status = BadDirection("LBFGS didn't move for some reason yy is 0, QUITTING")
                break

            state.rho.insert(0, 1.0 / ys)

            if len(state.S) > self.m:
                del state.S[-1]
                del state.Y[-1]
                del state.rho[-1]

            state.H0k = ys / yy

            state.dy = abs(y1 - state.y)
            state.x = x1
            state.y = y1
            state.g = g1

            state.it = state.it + 1

            if monitor:
                monitor(state)

            if converged_iters >= self.csteps:
                state.status = Converged("YES: Tolerance achieved")
                break

            if state.gnorm < self.gtol:
                state.status = Converged("YES: Gradient tolerance achieved")
                break

            if self.ymin is not None and state.y < self.ymin : 
                state.status = Converged("YES: Objective below threshold.")
                break

            if state.it > self.maxsteps:
                state.status = TooManySteps("maximum number of iterations reached. QUITTING.")
                break

        return state

