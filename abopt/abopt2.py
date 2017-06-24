from .vectorspace import VectorSpace

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
    def __init__(self, objective, gradient, **kwargs):
        self.objective = objective
        self.gradient = gradient
        self.maxiter = 1000
        self.atol = 1e-7
        self.rtol = 1e-7
        self.gtol = 0
        self.__dict__.update(kwargs)

class PreconditionedProblem(Problem):
    def __init__(self, objective, gradient, P, PT, **kwargs):
        self.P = P
        self.PT = PT
        Problem.__init__(self)

    def objectivePT(self, Px):
        return self._objective(self.PT(Px))

    def PgradientPT(self, Px):
        return self.P(self._gradient(self.PT(Px)))

class Optimizer(object):
    problem_defaults = {} # placeholder for subclasses to replace

    from .vectorspace import real_vector_space
    from .vectorspace import complex_vector_space
    from .linesearch import backtrace

    def __init__(self, vs=real_vector_space, linesearch=backtrace):
        self.vs = vs
        self.linesearch = linesearch

    def terminated(self, problem, state):
        if state.dy is None: return False

        if state.nit > problem.maxiter: return True

        if state.converged : return True

        return False

    def move(self, problem, state, x1, y1, g1, r1):
        dot = self.vs.dot

        if state.nit > 0:
            state.dy = y1 - state.y

        state.y_.append(y1)
        if len(state.y_) > 2:
            del state.y_[0]
        state.x = x1
        state.y = y1
        state.rate = r1
        state.xnorm = dot(x1, x1) ** 0.5
        state.g = g1
        state.gnorm = dot(g1, g1) ** 0.5


    def single_iteration(self, problem, state):
        raise NotImplementedError

    def post_single_iteration(self, problem, state, x1, y1, g1, r1):

        state.converged = check_convergence(state, y1, atol=problem.atol, rtol=problem.rtol)
        state.nit = state.nit + 1
        self.move(problem, state, x1, y1, g1, r1)

    def minimize(optimizer, objective, gradient, x0, P=None, PT=None, monitor=None, **kwargs):
        if P is not None:
            return minimize_p(optimizer, objective, gradient, x0, P, PT, monitor, **kwargs)
        else:
            return minimize(optimizer, objective, gradient, x0, monitor, **kwargs)

class GradientDescent(Optimizer):
    def single_iteration(self, problem, state):
        mul = self.vs.mul

        z = mul(state.g, 1 / state.gnorm)

        x1, y1, g1, r1 = self.linesearch(self.vs, problem, state, z, state.rate * 2)

        if g1 is None:
            g1 = problem.gradient(x1)
            state.gev = state.gev + 1

        self.post_single_iteration(problem, state, x1, y1, g1, r1)

        if state.gnorm <= problem.gtol: 
            state.converged = True

def minimize_p(optimizer, objective, gradient, x0, P, PT, monitor=None, **kwargs):

    def objectivePT(Px):
        return objective(PT(Px))

    def PgradientPT(Px):
        return P(gradient(PT(Px)))

    Px0 = P(x0)

    def Pmonitor(state):
        if monitor is None: return
        dot = optimizer.vs.dot

        Px = state.x
        Pg = state.g
        state.x = PT(Px)
        state.g = PT(Pg)
        state.gnorm = dot(state.g, state.g) ** 0.5
        state.xnorm = dot(state.x, state.x) ** 0.5
        monitor(state)
        state.x = Px
        state.g = Pg

    state = minimize(optimizer, objectivePT, PgradientPT, Px0, monitor=Pmonitor, **kwargs)

    state.x = PT(state.x)
    state.g = PT(state.g)
    return state

def minimize(optimizer, objective, gradient, x0, monitor=None, **kwargs):
    problem_args = {}
    problem_args.update(optimizer.problem_defaults)
    problem_args.update(kwargs)

    problem = Problem(objective, gradient, **problem_args)
    state = State()

    y0 = problem.objective(x0)
    g0 = problem.gradient(x0)
    state.fev = 1
    state.gev = 1

    optimizer.move(problem, state, x0, y0, g0, 1.0)

    if monitor is not None:
        monitor(state)
    while not optimizer.terminated(problem, state):
        optimizer.single_iteration(problem, state)
        if monitor is not None:
            monitor(state)

    return state


def check_convergence(state, y1, rtol, atol):
    valmax = max(abs(state.y), abs(y1))
    thresh = rtol * max(valmax, 1.0) + atol

    if y1 > state.y : return False
    if abs(state.y - y1) < thresh: return True

    return False

from .lbfgs import LBFGS

