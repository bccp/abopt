from .vectorspace import VectorSpace

def NullPrecondition(x): return x

def NotImplementedHvp(x):
    raise NotImplementedError("HessianVectorProduct is not implemented. Use a method that does not require it")

class State(object):
    def __init__(self):
        self.nit = 0
        self.fev = 0
        self.gev = 0
        self.dy = None
        self.converged = False
        self.y_ = []

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        d = dict([(k, self[k]) for k in ['nit', 'fev', 'gev', 'dy', 'converged', 'y', 'xnorm', 'gnorm']])
        return str(d)

class Problem(object):
    def __init__(self, objective, gradient, Hvp=NotImplementedHvp, **kwargs):
        self.objective = objective
        self.gradient = gradient
        self.Hvp = Hvp
        self.Pvp = NullPrecondition
        self.vPp = NullPrecondition
        self.maxiter = 1000
        self.atol = 1e-7
        self.rtol = 1e-7
        self.gtol = 0
        # this updates 
        self.__dict__.update(kwargs)

    def f(self, Px):
        x = self.vPp(Px)
        return self.objective(x)

    def g(self, Px):
        x = self.vPp(Px)
        return self.Pvp(self.gradient(x))

class Optimizer(object):
    problem_defaults = {} # placeholder for subclasses to replace

    from .vectorspace import real_vector_space
    from .vectorspace import complex_vector_space

    def __init__(self, vs=real_vector_space):
        self.vs = vs

    def terminated(self, problem, state):
        if state.dy is None: return False

        if state.nit > problem.maxiter: return True

        if state.converged : return True

        return False

    def move(self, problem, state, Px1, y1, Pg1, r1):
        dot = self.vs.dot

        if state.nit > 0:
            state.dy = y1 - state.y

        state.y_.append(y1)
        if len(state.y_) > 2:
            del state.y_[0]
        state.Px = Px1
        state.y = y1
        state.rate = r1
        state.Pxnorm = dot(Px1, Px1) ** 0.5
        state.Pg = Pg1
        state.Pgnorm = dot(Pg1, Pg1) ** 0.5

        # now move the un-preconditioned variables
        state.x = problem.vPp(Px1)
        state.g = problem.vPp(Pg1)

        if problem.Pvp is not NullPrecondition:
            state.xnorm = dot(state.x, state.x) ** 0.5
            state.gnorm = dot(state.g, state.g) ** 0.5
        else:
            state.xnorm = state.Pxnorm
            state.gnorm = state.Pgnorm

    def single_iteration(self, problem, state):
        raise NotImplementedError

    def post_single_iteration(self, problem, state, Px1, y1, Pg1, r1):

        state.converged = check_convergence(state, y1, atol=problem.atol, rtol=problem.rtol)
        state.nit = state.nit + 1
        self.move(problem, state, Px1, y1, Pg1, r1)

    def minimize(optimizer, problem, x0, monitor=None, **kwargs):
        state = State()

        Px0 = problem.Pvp(x0)

        y0 = problem.f(Px0)
        Pg0 = problem.g(Px0)
        state.fev = 1
        state.gev = 1

        optimizer.move(problem, state, Px0, y0, Pg0, 1.0)

        if monitor is not None:
            monitor(state)

        while not optimizer.terminated(problem, state):
            optimizer.single_iteration(problem, state)

            if monitor is not None:
                monitor(state)

        return state


class TrustRegionOptimizer(Optimizer):
    def __init__(self, vs=Optimizer.real_vector_space, trustregion=None):
        Optimizer.__init__(self, vs)
        self.trustregion = trustregion


class GradientDescent(Optimizer):
    from .linesearch import backtrace
    def __init__(self, vs=Optimizer.real_vector_space, linesearch=backtrace):
        Optimizer.__init__(self, vs)
        self.linesearch = linesearch

    def single_iteration(self, problem, state):
        mul = self.vs.mul

        z = mul(state.Pg, 1 / state.gnorm)

        Px1, y1, Pg1, r1 = self.linesearch(self.vs, problem, state, z, state.rate * 2)

        if Pg1 is None:
            Pg1 = problem.g(Px1)
            state.gev = state.gev + 1

        self.post_single_iteration(problem, state, Px1, y1, Pg1, r1)

        if state.gnorm <= problem.gtol: 
            state.converged = True


def check_convergence(state, y1, rtol, atol):
    valmax = max(abs(state.y), abs(y1))
    thresh = rtol * max(valmax, 1.0) + atol

    if y1 > state.y : return False
    if abs(state.y - y1) < thresh: return True

    return False

from .lbfgs import LBFGS

def minimize(optimizer, objective, gradient, x0, monitor=None, Pvp=NullPrecondition, vPp=NullPrecondition, **kwargs):
    problem_args = {}
    problem_args.update(optimizer.problem_defaults)
    problem_args.update(kwargs)

    problem = Problem(objective, gradient, **problem_args)
    return optimizer.minimize(problem, x0, monitor=None, **kwargs)
