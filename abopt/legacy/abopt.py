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

    @parameter
    def local_linesearch(value=True):
        """ whether the blind search tries to use local minima """
        return bool(value)

    @parameter
    def use_linesearch(value=True):
        """ whether use line search """
        return bool(value)

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

    def get_thresh(self, y1, y2):
        valmax = max(abs(y1), abs(y2))
        thresh = self.tol * max(valmax, 1.0) + self.atol
        return thresh

    def converged(self, state, y1):
        return abs(y1 - state.y) < self.get_thresh(state.y, y1) and state.y >= y1

    def backtrace_optimal(self, objective, state, z, zg, rate):
        """ This finds the minimal along the proposal direction. """
        from scipy.optimize import minimize_scalar
        def func(tau):
            if tau == 0: return state.y
            x1 = self.addmul(state.x, z, -tau * rate)
            state.fev = state.fev + 1
            return objective(x1)

        r = minimize_scalar(func, (0, 1), bounds=(0, 1), method='bounded', options={'maxiter':10}, )

        x1 = self.addmul(state.x, z, -r.x * rate)
        return x1, r.fun

    def backtrace(self, objective, state, z, zg, rate):

        # doing only backtracking line search
        # FIXME: implement more-thuente
        tau = 0.5
        c = 1e-5
        x1 = self.addmul(state.x, z, -rate)
        y1 = objective(x1)
        state.fev = state.fev + 1
        i = 0
        while i < 10:
            valmax = max(abs(y1), abs(state.y))
            thresh = self.tol * max(valmax, 1.0) + self.atol

            #print(rate, state.y, y1, state.x, x1, z)
            #if self.converged(state, y1): break

            # sufficient descent
            if y1 < state.y and abs(y1 - state.y) >= abs(rate * c * zg):
                break

            rate *= tau
            x1 = self.addmul(state.x, z, -rate)
            y1 = objective(x1)
            #print('new point ', x1, y1, state.x, state.y)
            state.fev = state.fev + 1
            i = i + 1
        return x1, y1

    def linesearch(self, objective, gradient, state, z, zg, rate):
        # This tries to go a bigger step until the direction of
        # gradient changes or the objective increases.
        # Useful when there is no hessian
        # e.g. with gradient descent but really just a hack -- it may hop around
        # to a different local stationary point.
        # need a better algorithm than this..
        if not self.use_linesearch:
            return self.backtrace(objective, state, z, zg, rate)
        tau = 2.0
        x1 = self.addmul(state.x, z, -rate)
        y1 = objective(x1)
        state.fev = state.fev + 1
        dy = state.y - y1
        while True:
            rate *= tau
            x1 = self.addmul(state.x, z, -rate)
            y1 = objective(x1)
            state.fev = state.fev + 1

            if self.local_linesearch:
                g1 = gradient(x1)
                state.gev = state.gev + 1

                # change direction
                if self.dot(g1, state.g) / state.gnorm < 0.01:
                    break

            # worse descent
            if state.y - y1 < dy:
                break
            dy = state.y - y1


        return self.backtrace(objective, state, z, zg, rate)

class BaseState(object):
    #__slots__ = ['it', 'fev', 'gev', 'dy', 'statue']

    @property
    def x(self): return self._x
    @property
    def xnorm(self): return self._xnorm

    @x.setter
    def x(self, value):
        self._x = value
        self._xnorm = self.optimizer.dot(value, value) ** 0.5

    @property
    def g(self): return self._g
    @property
    def gnorm(self): return self._gnorm

    @g.setter
    def g(self, value):
        self._g = value
        self._gnorm = self.optimizer.dot(value, value) ** 0.5

    def __getitem__(self, name):
        return getattr(self, name)

    def __init__(self, optimizer, objective, gradient, x0):
        self.__dict__['optimizer'] = optimizer

        if isinstance(x0, type(self)):
            self.__dict__.update(x0.__dict__)
        else:
            self.x = optimizer.copy(x0)
            self.dy = None

            self.fev, self.gev = 0, 0
            self.it = 0

        self.status = None
        self.y = objective(self.x)
        self.g = gradient(self.x)
        self.fev = self.fev + 1
        self.gev = self.gev + 1


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
        def __init__(self, optimizer, objective, gradient, x0):
            BaseState.__init__(self, optimizer, objective, gradient, x0)

    def minimize(self, objective, gradient, x0, monitor=None):
        state = self.State(self, objective, gradient, x0)

        while state.it < self.maxsteps:
            if monitor:
                monitor(state)

            if state.gnorm < self.gtol: break
            if self.ymin is not None and state.y < self.ymin : break
            # move to the next point
            x1 = self.addmul(state.x, state.g, -self.gamma)
            y1 = objective(x1)
            state.fev = state.fev + 1

            state.g = gradient(state.x)
            state.gev = state.gev + 1


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
        def __init__(self, optimizer, objective, gradient, x0):
            BaseState.__init__(self, optimizer, objective, gradient, x0)
            self.rho = []
            self.S = []
            self.Y = []
            self.H0k = 1.0
            self.converged_it = 0
            self.z = self.g
            self.use_steepest_descent = False

        def __str__(self):
            return BaseState.__str__(self) + ' GD: %s' % self.use_steepest_descent

    def one(self, objective, gradient, state):
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

        state.use_steepest_descent = False

        znorm = self.dot(z, z) ** 0.5
        zg = 0.0 if znorm == 0 else self.dot(z, state.g) / znorm
        zg_grannorm = 0.0 if state.gnorm == 0 else zg / state.gnorm

        if zg_grannorm < 0.01:
            # L-BFGS gave a bad direction.
            z = self.copy(state.g)
            zg = 1.0
            state.use_steepest_descent = True
        """
        print('--------')
        print(alpha)
        print([1 / tmp for tmp in state.rho])
        print('q', q)
        print('z', z)
        print(state.H0k)
        print(state.Y)
        print(state.S)
        """
        state.z = z
        if state.it == 0 or state.use_steepest_descent:
            rate = 1.0 / state.gnorm
            #x1, y1 = self.linesearch(objective, gradient, state, z, zg, rate)
            x1, y1 = self.backtrace(objective, state, z, zg, rate)
        else: 
            rate = 1.0
            x1, y1 = self.backtrace(objective, state, z, zg, rate)

        g1 = gradient(x1)
        state.gev = state.gev + 1

        if self.converged(state, y1):
            state.converged_it = state.converged_it + 1
        else:
            state.converged_it = 0
            # hessian update

            # XXX: shall we do this when use_steepest_descent is true?
            state.S.insert(0, self.addmul(x1, state.x, -1))
            state.Y.insert(0, self.addmul(g1, state.g, -1))
            ys = self.dot(state.S[0], state.Y[0])
            yy = self.dot(state.Y[0], state.Y[0])

            if ys == 0.0:
                state.status = BadDirection("LBFGS didn't move for some reason ys is 0, QUITTING")
            if yy == 0.0:
                state.status = BadDirection("LBFGS didn't move for some reason yy is 0, QUITTING")

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


    def minimize(self, objective, gradient, x0, monitor=None):
        state = self.State(self, objective, gradient, x0)

        if monitor: monitor(state)

        while True:
            self.one(objective, gradient, state)

            if monitor:
                monitor(state)

            if state.status is not None:
                break

            if state.converged_it >= self.csteps and state.it >= self.minsteps:
                state.status = Converged("YES: Tolerance achieved")
                break

            if state.gnorm < self.gtol:
                state.status = Converged("YES: Gradient tolerance achieved")

            if self.ymin is not None and state.y < self.ymin : 
                state.status = Converged("YES: Objective below threshold.")

            if state.it > self.maxsteps:
                state.status = TooManySteps("maximum number of iterations reached. QUITTING.")

        return state

class SLBFGS(LBFGS):
    """ Stochastic LBFGS """
    @parameter
    def oracle(value=None):
        """Function to generates a random displacement in the parameter space """
        return value

    @parameter
    def N0(value=10):
        return int(value)

    @parameter
    def gthresh(value=1.0):
        return float(value)

    @parameter
    def noiseamp(value=0.1):
        return float(value)

    class State(LBFGS.State):
        def __init__(self, optimizer, objective, gradient, x0):
            LBFGS.State.__init__(self, optimizer, objective, gradient, x0)
            self.it_noise = - optimizer.N0 - 1
            self.x_noise = None
            self.y_noise = None
            self.g_noise = None
            self.noise = None

    def one(self, objective, gradient, state):
        state.noise = None

        if state.it - state.it_noise > self.N0 and state.gnorm < self.gthresh:
            state.x_noise = state.x
            state.it_noise = state.it
            state.g_noise = state.g
            state.y_noise = state.y
            noise = self.oracle(state)
            state.noise = noise
            state.x = self.addmul(state.x_noise, state.noise, self.gthresh * self.noiseamp)
            print('hop', state.x, state.x_noise, state.noise, self.gthresh, self.noiseamp)
            state.y = objective(state.x)
            state.g = gradient(state.x)
            state.fev = state.fev + 1
            state.gev = state.gev + 1
            state.it = state.it + 1
        elif state.it - state.it_noise == self.N0 and state.y - state.y_noise > - self.get_thresh(state.y, state.y_noise):
            state.x = state.x_noise
            state.y = state.y_noise
            state.g = state.g_noise
            #state.status = Converged("Perturbed Gradient Descent converged")
            state.it = state.it + 1
        else:
            state.status = LBFGS.one(self, objective, gradient, state)

