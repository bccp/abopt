"""

    Data model

    A ``Problem`` is defined on a ``VectorSpace``; the problem is to minimize a function to a given tolerance.
    A ``Problem`` consists of a differentiable function, up to second order. `Gradient` and `HessianVectorProduct`
    A ``Problem`` can be `minimize`d by an ``Optimizer``, yielding a sequence of `State`s.
    A ``Problem`` is ``Preconditioned``, the ``Optimizer`` only operates on preconditioned variables.
    An ``Optimizer`` implements a minimization policy (algorithm)

    Problem parameters and Optimizer parameters
    -------------------------------------------
    Problem parameters are related to the accuracy, atol, rtol, etc.
    Optimizer parameters only controls the behavior of the optimizer; maxiter, etc.

    An easy way to see this is that if we redefine the vector variable by a factor of 10,
    if a parameters shall be adjusted, then it belongs to the problem;
    if it shall not be adjusted, then it belongs to the optimizer.
"""
from .vectorspace import VectorSpace
from .vectorspace import real_vector_space
from .vectorspace import complex_vector_space

class ContinueIteration(str): pass
class ConvergedIteration(str): pass
class FailedIteration(str): pass

class State(object):
    def __init__(self):
        self.nit = 0
        self.fev = 0
        self.gev = 0
        self.hev = 0
        self.dy = None
        self.dxnorm = None
        self.assessment = None
        self.conviter = 0
        self.converged = False
        self.message = ""
        self.y_ = []
        self.z = None
        self.Pg = None

    def __getitem__(self, key):
        return getattr(self, key)
    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        d = [(k, self[k]) for k in ['nit', 'fev', 'gev', 'hev', 'y', 'dy', 'xnorm', 'dxnorm', 'gnorm', 'theta', 'converged', 'message', 'assessment', 'radius', 'B', 'rate', 'rho', 'conviter'] if k in self]
        return repr(d)

class Proposal(object):
    def __init__(self, problem, y=None, x=None, Px=None, g=None, Pg=None, z=None):
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
        self.z = z
        self.problem = problem
        self.message = "normal"

    def complete(self, state):
        dot = self.problem.vs.dot
        addmul = self.problem.vs.addmul
        self.xnorm = dot(self.x, self.x) ** 0.5
        self.Pxnorm = dot(self.Px, self.Px) ** 0.5
        self.complete_y(state)
        self.complete_g(state)

        self.dy = self.y - state.y
        dx = addmul(self.x, state.x, -1)
        self.dxnorm = dot(dx, dx) ** 0.5
        self.znorm = dot(self.z, self.z) ** 0.5
        if self.Pgnorm == 0:
            self.theta = 1
        else:
            self.theta = dot(self.z, self.Pg) / (self.znorm * self.Pgnorm)
        return self

    def complete_y(self, state):
        problem = self.problem

        if self.y is None:
            self.y = problem.f(self.x)
            state.fev = state.fev + 1
        return self

    def complete_g(self, state):
        dot = self.problem.vs.dot
        problem = self.problem

        # fill missing values in prop
        if self.g is None:
            self.g = problem.g(self.x)
            state.gev = state.gev + 1

        if self.Pg is None:
            self.Pg = problem.g2Pg(self.g)

        self.Pgnorm = dot(self.Pg, self.Pg) ** 0.5
        self.gnorm = dot(self.g, self.g) ** 0.5

        return self

class InitialProposal(Proposal):
    def complete(self, state):
        dot = self.problem.vs.dot
        addmul = self.problem.vs.addmul
        self.xnorm = dot(self.x, self.x) ** 0.5
        self.Pxnorm = dot(self.Px, self.Px) ** 0.5
        self.complete_y(state)
        self.complete_g(state)

        self.dy = None
        self.dxnorm = None
        self.znorm = None
        self.theta = None
        return self

class Preconditioner(object):
    """ A preconditioner has four functions, corresponding
        to left dot and right dot of P and Q on a vector.

        P = [Q ^{-1}]^T,

        P and Q are Given by the following coordinate transformations
        from x to x~.

            x~ i  = P_ij x_j -> Pvp(x)

            x_j = Q_ij x~_i -> vQp(x~)

        The gradient transformation is from chain rule,

            g~_i = g_j Q_ij -> Qvp(g)
            g_j = g~_i P_ij -> vPp(g~)

        Hessian vector product and inverse Hessian vector product,

            H~_ij v_j = Q_ia Q_jb v_j H_ab -> Qvp(Hvp(vQp(v)))
            h~_ij v_j = P_ai P_bj v_j h_ab -> vPp(hvp(Pvp(v)))

        The input functions are

            Pvp(v, direction),
            vPp(v, direction);
    """
    def __init__(self, Pvp, vPp):
        self.Pvp = Pvp
        self.vPp = vPp

class Problem(object):
    """ Defines a problem.

    """
    def __init__(self, objective, gradient,
        hessian_vector_product=None,
        inverse_hessian_vector_product=None,
        vs=real_vector_space,
        atol=0,
        rtol=1e-7,
        xtol=1e-7,
        gtol=1e-8,
        precond=None,
        ):
        if precond is None:
            precond = Preconditioner(lambda x, direction:x, lambda x, direction:x)

        if not isinstance(vs, VectorSpace):
            raise TypeError("expecting a VectorSpace object for vs, got type(vs) = %s", repr(type(vs)))

        if not isinstance(precond, Preconditioner):
            raise TypeError("expecting a VPreconditioner object for precond, got type(vs) = %s", repr(type(precond)))

        self._precond = precond
        self.vs = vs

        self._objective = objective
        self._gradient = gradient
        self._hessian_vector_product = hessian_vector_product
        self._inverse_hessian_vector_product = inverse_hessian_vector_product
        self.atol = atol
        self.rtol = rtol
        self.xtol = xtol
        self.gtol = gtol

    def Px2x(self, Px):
        return self._precond.vPp(Px, direction=-1)

    def x2Px(self, x):
        return self._precond.Pvp(x, direction=1)

    def g2Pg(self, g):
        return self._precond.Pvp(g, direction=-1)

    def Pg2g(self, Pg):
        return self._precond.vPp(Pg, direction=1)

    def check_preconditioner(self, x0):
        vs = self.vs
            
        d = vs.addmul(self.Px2x(self.x2Px(x0)), x0, -1)
        if vs.dot(d, d) > 1e-6 * vs.dot(x0, x0):
            raise ValueError("Preconditioner's vQp and Pvp are not inverses.")

        d = vs.addmul(self.Pg2g(self.g2Pg(x0)), x0, -1)
        if vs.dot(d, d) > 1e-6 * vs.dot(x0, x0):
            raise ValueError("Preconditioner's vPp and Qvp are not inverses.")


    def f(self, x):
        return self._objective(x)

    def g(self, x):
        """ This returns the gradient for the original variable"""
        g = self._gradient(x)
        return g

    def Hvp(self, x, v):
        """ This returns the raw hessian product H_x v
            uppercase H means Hessian, not Hessian inverse.

            v is not preconditioned.
            x is not preconditioned.

            result is not preconditioned, and act like x.
        """
        if self._hessian_vector_product is None:
            raise ValueError("hessian vector product is not defined")
        return self._hessian_vector_product(x, v)


    def PHvp(self, x, v):
        """ This returns the preconditioned hessian times v

            uppercase H means Hessian, not Hessian inverse.

            v is usually preconditioned (irrelevant)
            x is always not preconditioned.

            result is preconditioned, and act like Px.

            ~H_x v = Q [H_x (v Q^T)]
        """
        if self._hessian_vector_product is None:
            raise ValueError("hessian vector product is not defined")
        vQ = self._precond.vPp(v, direction=-1)
        return self._precond.Pvp(self._hessian_vector_product(x, vQ), direction=-1)

    def Phvp(self, x, v):
        """ This returns the preconditioned inverse hessian times v

            lowercase h means inverse of Hessian

            v is usually preconditioned (irrelevant)
            x is always not preconditioned.

            result is preconditioned, and act like Px.

            ~h_x v = [h_x (P v)] P^T
        """
        if self._inverse_hessian_vector_product is None:
            raise ValueError("inverse_hessian vector product is not defined")
        Pv = self._precond.Pvp(v, direction=1)
        return self._precond.vPp(self._inverse_hessian_vector_product(x, Pv), direction=1)

    def get_ytol(self, y):
        thresh = self.rtol * abs(y) + self.atol
        return thresh

    def check_convergence(self, y0, y1):
        if y1 > y0 :
            return False
            # probably shall do this
            # raise RuntimeError("new proposal is larger than previous value")

        valmax = max(abs(y0), abs(y1))

        thresh = self.get_ytol(valmax)

        if abs(y0 - y1) < thresh: return True

        return False


class Optimizer(object):
    optimizer_defaults = {}

    def __init__(self, **kwargs):
        # this updates the attributes
        self.__dict__.update(type(self).optimizer_defaults)
        self.__dict__.update(kwargs)

    def terminated(self, problem, state):
        if isinstance(state.assessment, ConvergedIteration):
            if state.Pgnorm == 0:
                return True
            if state.conviter >= self.conviter:
                return True
            else:
                return False
        if isinstance(state.assessment, FailedIteration):
            return True
        if state.nit > self.maxiter: return True
        if state.dy is None: return False

        return False

    def move(self, problem, state, prop):

        state.message = prop.message

        state.y_.append(prop.y)

        if len(state.y_) > 2: # only store a short history
            del state.y_[0]

        state.y = prop.y
        state.dy = prop.dy

        state.x = prop.x
        state.g = prop.g
        state.z = prop.z
        state.theta = prop.theta
        state.Px = prop.Px
        state.Pg = prop.Pg

        state.xnorm = prop.xnorm
        state.gnorm = prop.gnorm
        state.Pxnorm = prop.Pxnorm
        state.Pgnorm = prop.Pgnorm
        state.dxnorm = prop.dxnorm

    def assess(self, problem, state, prop):
        if prop is None:
            return FailedIteration("no proposal is made")

        prop = prop.complete(state)

        if prop.gnorm <= problem.gtol:
            return ConvergedIteration("Gradient is sufficiently small")

        if prop.dxnorm <= problem.xtol:
            return ConvergedIteration("Solution stopped moving")

        if problem.check_convergence(state.y, prop.y):
            return ConvergedIteration("Objective stopped improving")

        return ContinueIteration("continue iteration")

    def single_iteration(self, problem, state):
        # it shall return a Proposal object
        raise NotImplementedError
        # here is an example that doesn't yield a new solution
        return Proposal(Px=state.Px)

    def start(self, problem, state, x0):
        # make a full initial proposal with x and g
        Px0 = problem.x2Px(x0) # the only place we convert from x to Px
        prop = InitialProposal(problem, x=x0, Px=Px0).complete(state)
        return prop

    def restart(optimizer, problem, state, monitor=None):

        if monitor is not None:
            monitor(state)

        while not optimizer.terminated(problem, state):

            prop = optimizer.single_iteration(problem, state)

            # assessment must be before the move, for it needs to see dy
            assessment = optimizer.assess(problem, state, prop)

            state.assessment = assessment

            if isinstance(assessment, (ContinueIteration, ConvergedIteration)):
                optimizer.move(problem, state, prop)
                state.nit = state.nit + 1

            if isinstance(assessment, ConvergedIteration):
                state.conviter = state.conviter + 1
                state.converged = True

            if isinstance(assessment, ContinueIteration):
                state.conviter = 0
                state.converged = False

            if monitor is not None:
                monitor(state)

        return state

    def minimize(optimizer, problem, x0, monitor=None, **state_args):

        state = State()

        # initialize state with args
        for key, value in state_args.items():
            setattr(state, key, value)

        # check the preconditioner.
        problem.check_preconditioner(x0)

        prop = optimizer.start(problem, state, x0)

        optimizer.move(problem, state, prop)

        optimizer.restart(problem, state, monitor)

        return state


from .lbfgs import LBFGS
from .naivemethods import GradientDescent, DirectNewton
from .trustregion import TrustRegionCG

def minimize(optimizer, objective, gradient, x0, hessian_vector_product=None,
    monitor=None, vs=real_vector_space, precond=None):

    problem = Problem(objective, gradient, hessian_vector_product=hessian_vector_product, vs=vs, precond=precond)

    return optimizer.minimize(problem, x0, monitor=monitor)


