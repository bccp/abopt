"""

    Data model

    A ``Problem`` is defined on a ``VectorSpace``; the problem is to minimize a function to a given tolerance.
    A ``Problem`` consists of a differentiable function, up to second order. `Gradient` and `HessianVectorProduct`
    A ``Problem`` can be `minimize`d by an ``Optimizer``, yielding a sequence of `State`s.
    A ``Problem`` is ``Preconditioned``, the ``Optimizer`` only operates on preconditioned variables.
    An ``Optimizer`` implements a minimization policy (algorithm)

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

class Preconditioner(object):
    """ A preconditioner has three functions:

        x~ i  = P_ij x_j -> Pvp(x)

        x_j = Q_ij x~_i -> vQp(x~)

        g~_i = g_j Q_ij -> Qvp(g)

        H~_ij v_j = Q_ia Q_jb v_j H_ab -> Qvp(Hvp(vQp(v)))

    """
    def __init__(self, Pvp, Qvp, vQp):
        self.Pvp = Pvp
        self.Qvp = Qvp
        self.vQp = vQp

NullPreconditioner = Preconditioner(lambda x:x, lambda x:x, lambda x:x)

class Problem(object):

    def __init__(self, objective, gradient,
        hessian_vector_product=NotImplementedHvp,
        vs=real_vector_space,
        atol=1e-7,
        rtol=1e-7,
        gtol=0,
        precond=NullPreconditioner,
        ):

        self.objective = objective
        self.gradient = gradient
        self.hessian_vector_product = hessian_vector_product
        self.atol = atol
        self.rtol = rtol
        self.gtol = gtol

        if not isinstance(vs, VectorSpace):
            raise TypeError("expecting a VectorSpace object for vs, got type(vs) = %s", repr(type(vs)))

        if not isinstance(precond, Preconditioner):
            raise TypeError("expecting a VPreconditioner object for precond, got type(vs) = %s", repr(type(precond)))

        self.precond = precond
        self.vs = vs

    def f(self, x):
        return self.objective(x)

    def g(self, x):
        g = self.gradient(x)
        return g, self.precond.Qvp(g)

    def Hvp(self, x, v):
        vQ = self.precond.vQp(v)
        return self.precond.Qvp(self.hessian_vector_product(x, vQ))

class Optimizer(object):
    problem_defaults = {}

    def __init__(self, **kwargs):
        self.maxiter = 1000

        # this updates the attributes
        self.__dict__.update(type(self).problem_defaults)
        self.__dict__.update(kwargs)

    def terminated(self, problem, state):
        if state.dy is None: return False

        if state.nit > self.maxiter: return True

        if state.converged : return True

        return False

    def move(self, problem, state, x1, Px1, y1, g1, Pg1, r1):
        dot = problem.vs.dot

        if state.nit > 0:
            state.dy = y1 - state.y

        state.y_.append(y1)
        if len(state.y_) > 2:
            del state.y_[0]
        state.y = y1
        state.Px = Px1
        state.rate = r1
        state.Pxnorm = dot(Px1, Px1) ** 0.5
        state.Pg = Pg1
        state.Pgnorm = dot(Pg1, Pg1) ** 0.5

        # now move the un-preconditioned variables
        state.x = x1
        state.g = g1

        if problem.precond is not NullPrecondition:
            state.xnorm = dot(state.x, state.x) ** 0.5
            state.gnorm = dot(state.g, state.g) ** 0.5
        else:
            state.xnorm = state.Pxnorm
            state.gnorm = state.Pgnorm

    def single_iteration(self, problem, state):
        raise NotImplementedError

    def post_single_iteration(self, problem, state, x1, Px1, y1, g1, Pg1, r1):

        state.converged = check_convergence(state, y1, atol=problem.atol, rtol=problem.rtol)
        state.nit = state.nit + 1
        self.move(problem, state, x1, Px1, y1, g1, Pg1, r1)

    def minimize(optimizer, problem, x0, monitor=None):
        state = State()

        Px0 = problem.precond.Pvp(x0)

        y0 = problem.f(x0)
        g0, Pg0 = problem.g(x0)
        state.fev = 1
        state.gev = 1

        optimizer.move(problem, state, x0, Px0, y0, g0, Pg0, 1.0)

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

        x1_and_Px1, y1, g1_and_Pg1, r1 = self.linesearch(problem, state, z, state.rate * 2)

        x1, Px1 = x1_and_Px1

        if g1_and_Pg1 is None:
            g1, Pg1 = problem.g(x1)
            state.gev = state.gev + 1
        else:
            g1, Pg1 = g1_and_Pg1

        self.post_single_iteration(problem, state, x1, Px1, y1, g1, Pg1, r1)

        if state.gnorm <= problem.gtol: 
            state.converged = True


def check_convergence(state, y1, rtol, atol):
    valmax = max(abs(state.y), abs(y1))
    thresh = rtol * max(valmax, 1.0) + atol

    if y1 > state.y : return False
    if abs(state.y - y1) < thresh: return True

    return False

from .lbfgs import LBFGS

def minimize(optimizer, objective, gradient, x0, hessian_vector_product=NotImplementedHvp,
    monitor=None, vs=real_vector_space, precond=NullPreconditioner):

    problem = Problem(objective, gradient, hessian_vector_product=hessian_vector_product, vs=vs, precond=precond)

    d = vs.addmul(problem.precond.vQp(problem.precond.Pvp(x0)), x0, -1)

    # assert vPv and Pvp are inverses
    assert vs.dot(d, d) ** 0.5 < 1e-15

    return optimizer.minimize(problem, x0, monitor=monitor)


