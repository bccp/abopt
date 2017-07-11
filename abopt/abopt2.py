"""

    Data model

    A ``Problem`` is defined on a ``VectorSpace``.
    A ``Problem`` consists of a differentiable function, up to second order. `Gradient` and `HessianVectorProduct`
    A ``Problem`` can be `minimize`d by an ``Optimizer``, yielding a sequence of `State`s.

    A ``Problem`` can be ``Preconditioned``, in which case Optimization yields a sequence of ``PreconditionedState``s

"""
from .vectorspace import VectorSpace
from .vectorspace import real_vector_space
from .vectorspace import complex_vector_space

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

    def __init__(self, objective, gradient, hessian_vector_product=NotImplementedHvp, vs=real_vector_space):
        self.objective = objective
        self.gradient = gradient
        self.hessian_vector_product = hessian_vector_product
        self.Pvp = NullPrecondition
        self.vPp = NullPrecondition
        self.vs = vs

    def f(self, Px):
        x = self.vPp(Px)
        return self.objective(x)

    def g(self, Px):
        x = self.vPp(Px)
        return self.Pvp(self.gradient(x))

    def Hvp(self, Px, Pg):
        x = self.vPp(Px)
        g = self.vPp(Pg)
        return self.Pvp(self.hessian_vector_product(x, g))

class Optimizer(object):
    problem_defaults = {}

    def __init__(self, **kwargs):
        self.maxiter = 1000
        self.atol = 1e-7
        self.rtol = 1e-7
        self.gtol = 0

        # this updates the attributes
        self.__dict__.update(type(self).problem_defaults)
        self.__dict__.update(kwargs)

    def terminated(self, problem, state):
        if state.dy is None: return False

        if state.nit > self.maxiter: return True

        if state.converged : return True

        return False

    def move(self, problem, state, Px1, y1, Pg1, r1):
        dot = problem.vs.dot

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

        state.converged = check_convergence(state, y1, atol=self.atol, rtol=self.rtol)
        state.nit = state.nit + 1
        self.move(problem, state, Px1, y1, Pg1, r1)

    def minimize(optimizer, problem, x0, monitor=None):
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
    pass

class GradientDescent(Optimizer):
    from .linesearch import backtrace

    problem_defaults = {
            'linesearch' : backtrace
    }

    def single_iteration(self, problem, state):
        mul = problem.vs.mul

        z = mul(state.Pg, 1 / state.gnorm)

        Px1, y1, Pg1, r1 = self.linesearch(problem, state, z, state.rate * 2)

        if Pg1 is None:
            Pg1 = problem.g(Px1)
            state.gev = state.gev + 1

        self.post_single_iteration(problem, state, Px1, y1, Pg1, r1)

        if state.gnorm <= self.gtol: 
            state.converged = True


def check_convergence(state, y1, rtol, atol):
    valmax = max(abs(state.y), abs(y1))
    thresh = rtol * max(valmax, 1.0) + atol

    if y1 > state.y : return False
    if abs(state.y - y1) < thresh: return True

    return False

from .lbfgs import LBFGS

def minimize(optimizer, objective, gradient, x0, hessian_vector_product=NotImplementedHvp,
    monitor=None, vs=real_vector_space, Pvp=NullPrecondition, vPp=NullPrecondition):

    problem = Problem(objective, gradient, hessian_vector_product=hessian_vector_product, vs=vs)

    d = vs.addmul(problem.vPp(problem.Pvp(x0)), x0, -1)

    # assert vPv and Pvp are inverses
    assert vs.dot(d, d) ** 0.5 < 1e-15

    return optimizer.minimize(problem, x0, monitor=None)


