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

class Proposal(object):
    def __init__(self, problem, y=None, x=None, Px=None, g=None, Pg=None):
        """ A proposal is a collection of variable and gradients.

            We will generate the variables if they are not provided.

        """
        if x is None and Px is not None:
            x = problem.Px2x(Px)

        if Pg is None and g is not None:
            Pg = problem.g2Pg(g)

        self.y = y
        self.x = x
        self.Px = Px
        self.g = g
        self.Pg = Pg
        self.problem = problem

    def complete(self, state):
        self.complete_y(state)
        self.complete_g(state)
        return self

    def complete_y(self, state):
        problem = self.problem

        if self.y is None:
            self.y = problem.f(self.x)
            state.fev = state.fev + 1
        return self

    def complete_g(self, state):
        problem = self.problem

        # fill missing values in prop
        if self.g is None:
            self.g = problem.g(self.x)
            state.gev = state.gev + 1

        if self.Pg is None:
            self.Pg = problem.g2Pg(self.g)
        return self

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

    def Px2x(self, Px):
        return self.precond.vQp(Px)

    def g2Pg(self, g):
        return self.precond.Qvp(g)

    def f(self, x):
        return self.objective(x)

    def g(self, x):
        """ This returns the gradient for the original variable"""
        g = self.gradient(x)
        return g

    def Hvp(self, x, v):
        """ This returns the hessian product of the preconditioned variable against
            a vector of the preconditioned variable.
        """
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
        if state.converged : return True

        if state.dy is None: return False

        if state.nit > self.maxiter: return True

        return False

    def move(self, problem, state, prop):
        dot = problem.vs.dot

        prop.complete(state) # filling in all missing values

        if state.nit > 0:
            state.dy = prop.y - state.y
            state.converged = check_convergence(state, prop.y, atol=problem.atol, rtol=problem.rtol)
        else:
            state.converged = False

        state.y_.append(prop.y)
        if len(state.y_) > 2: # only store a short history
            del state.y_[0]

        state.y = prop.y
        state.Px = prop.Px
        state.Pg = prop.Pg

        # now move the un-preconditioned variables
        state.x = prop.x
        state.g = prop.g

        state.Pxnorm = dot(state.Px, state.Px) ** 0.5
        state.Pgnorm = dot(state.Pg, state.Pg) ** 0.5
        state.xnorm = dot(state.x, state.x) ** 0.5
        state.gnorm = dot(state.g, state.g) ** 0.5

        if state.gnorm <= problem.gtol:
            state.converged = True

    def single_iteration(self, problem, state):
        # it shall return a Proposal object
        raise NotImplementedError

    def minimize(optimizer, problem, x0, monitor=None):
        state = State()

        Px0 = problem.precond.Pvp(x0)
        y0 = problem.f(x0)
        g0 = problem.g(x0)
        state.fev = 1
        state.gev = 1

        prop = Proposal(problem, y=y0, x=x0, Px=Px0, g=g0)

        optimizer.move(problem, state, prop)

        if monitor is not None:
            monitor(state)

        while not optimizer.terminated(problem, state):

            prop = optimizer.single_iteration(problem, state)

            optimizer.move(problem, state, prop)

            state.nit = state.nit + 1

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

    def move(self, problem, state, prop):
        if hasattr(prop, "rate"):
            state.rate = prop.rate
        else:
            state.rate = 1.0

        Optimizer.move(self, problem, state, prop)

        if state.Pgnorm == 0:
            # cannot move if Pgnorm is 0
            self.converged = True

    def single_iteration(self, problem, state):
        mul = problem.vs.mul

        z = mul(state.Pg, 1 / state.Pgnorm)

        prop, r1 = self.linesearch(problem, state, z, state.rate * 2)

        prop.rate = r1
        return prop


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


