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

class ContinueIteration(str): pass
class ConvergedIteration(str): pass
class FailedIteration(str): pass

import time

class State(object):
    def __init__(self):
        self.nit = 0
        self.fev = 0
        self.gev = 0
        self.hev = 0
        self.y = None
        self.dy = None
        self.dxnorm = None
        self.xnorm = None
        self.gnorm = None
        self.Pxnorm = None
        self.Pgnorm = None
        self.dxnorm = None
        self.assessment = None
        self.conviter = 0
        self.converged = False
        self.message = ""
        self.y_ = []
        self.z = None
        self.Pg = None
        self.timestamp = time.time()
        self.wallclock = 0

        self.default_format = dict(
        [
            ('wallclock', '[ %08.4f ]',),
            ('nit', '%04d',),
            ('fev', '%04d',),
            ('gev', '%04d',),
            ('hev', '%04d',),
            ('y', '% 13.6e'),
            ('dy', '% 13.6e'),
            ('xnorm', '% 11.4e'),
            ('gnorm', '% 11.4e'),
            ('theta', '% 4.2f'),
            ('radius', '% 9.2e'),
            ('B', '%10s'),
            ('rate', '% 8.2f'),
            ('rho', '% 5.2f'),
            ('conviter', '%04d'),
            ('converged', '% 6s'),
            ('message', '% 20s'),
            ('assessment', '% 20s')
        ])


    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        return self.format()

    def format(self, columns=None, header=False, sp='|'):
        """ format a state object.

            Parameters
            ----------
            columns : list of string or tuples.
                for each item,
                if tuple, the format of the column. if string, use a default format
                for the column.
            header : bool
                format the column headers if True
        """

        dd = dict(self.default_format)

        if columns is None:
            columns = self.default_format

        c2 = []
        for item in columns:
            if not isinstance(item, tuple):
                item = (item, dd.get(item, '%10s'))
            c2.append((item[0], item[1]))

        keys = { key : fmt for key, fmt in c2}

        def get_width(key):
            fmt = keys[key]
            width = fmt[1:-1].split('.')[0]
            while len(width) and width[0] not in '0123456789':
                width = width[1:]
            width = int(width.strip())
            if width < len(key):
                width = len(key)
            return width

        def get_strfmt(key, strict):
            if strict:
                return ('%%%d.%ds' % (get_width(key), get_width(key)))
            else:
                return ('%%%ds' % (get_width(key)))

        if header:
            return (sp.join(get_strfmt(key, True) % key for key, fmt in c2))

        else:
            def fmt_field(key):
                if key not in self:
                    s = "N/A"
                else:
                    try:
                        s = keys[key] % self[key]
                    except TypeError:
                        s = str(self[key])

                return get_strfmt(key, False) % s

            return (sp.join(fmt_field(key)  for key, fmt in c2))

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
        self.znorm = dot(self.z, self.z) ** 0.5
        self.xnorm = dot(self.x, self.x) ** 0.5
        self.Pxnorm = dot(self.Px, self.Px) ** 0.5
        self.complete_y(state)
        self.complete_g(state)

        self.dy = self.y - state.y
        dx = addmul(self.x, state.x, -1)
        self.dxnorm = dot(dx, dx) ** 0.5
        if state.Pgnorm == 0:
            self.theta = 1
        else:
            self.theta = dot(self.z, state.Pg) / (self.znorm * state.Pgnorm)
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
        vs=None,
        atol=0,
        rtol=1e-7,
        xtol=1e-7,
        gtol=1e-8,
        precond=None,
        ):
        if precond is None:
            precond = Preconditioner(lambda x, direction:x, lambda x, direction:x)

        if vs is None:
            from .vectorspace import real_vector_space
            vs = real_vector_space

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
        # This condition may need to be removed.
        # some optimizers do not need the objectve to have been strictly decreasing
        # for convergence.
        if y1 > y0 :
            return False

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
        # check for maxiter first to overrride
        # continuing due to ConvergedItreration.

        if state.nit > self.maxiter: return True

        if isinstance(state.assessment, (ConvergedIteration, FailedIteration)):
            if state.Pgnorm == 0:
                return True
            if state.conviter >= self.conviter:
                return True
            else:
                return False

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
        state.wallclock = time.time() - state.timestamp
        state.timestamp = timestamp = time.time()

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

    def _minimize(optimizer, problem, state, monitor=None):

        first_iteration = True
        while not optimizer.terminated(problem, state):
            if isinstance(state.assessment, (ConvergedIteration, FailedIteration)) \
              or first_iteration:
                prop = optimizer.start(problem, state, state['x'])

                optimizer.move(problem, state, prop)

                if monitor is not None:
                    monitor(state)

            first_iteration = False

            prop = optimizer.single_iteration(problem, state)

            # assessment must be before the move, for it needs to see dy
            assessment = optimizer.assess(problem, state, prop)

            state.assessment = assessment

            if isinstance(assessment, (ContinueIteration, ConvergedIteration)):
                optimizer.move(problem, state, prop)
                state.nit = state.nit + 1

            if isinstance(assessment, (ConvergedIteration, FailedIteration)):
                state.conviter = state.conviter + 1
                # converged or failed -- restart cleanly
                state.converged = True

            if isinstance(assessment, ContinueIteration):
                state.conviter = 0
                state.converged = False

            if monitor is not None:
                monitor(state)

        return state

    def minimize(optimizer, problem, x0, monitor=None, **state_args):
        """ minimize a problem starting from state x0

            Parameters
            ----------
            problem : Problem
                the problem object, which defines the objective function
                and the vector space of the parameters.

            x0 : object or a State
                the initial state.
                if a object is given, a state is constructed by setting state_args
                positioning the parameters at x0.
                if a State object is given it will be used and updated.

            monitor: function(state)
                a function that gets called on each iteration.
                if state.dy is None then it is the first time monitor is called.

            Returns
            -------
            state : a State object of the final minimization result.
            state.converged : if the solution has converged
            state.x : the new value of the parameter.
            state.y : the new value of the objective

        """
        if not isinstance(x0, State):
            state = State()

            state.x = x0
            # initialize state with args
            for key, value in state_args.items():
                setattr(state, key, value)

            # check the preconditioner.
            problem.check_preconditioner(x0)
        else:
            state = x0

        optimizer._minimize(problem, state, monitor)

        return state

class VectorSpace(object):
    def __init__(self, addmul=None, dot=None):
        if addmul:
            self.addmul = addmul
        if dot:
            self.dot = dot

    def copy(self, a):
        r = self.addmul(0, a, 1)
        assert type(r) is type(a)
        return r

    def ones_like(self, b):
        r = self.addmul(1, b, 0)
        assert type(r) is type(b)
        return r

    def zeros_like(self, b):
        r = self.addmul(0, b, 0)
        assert type(r) is type(b)
        return r

    def mul(self, b, c, p=1):
        return self.addmul(0, b, c, p)

    def pow(self, c, p):
        i = self.ones_like(c)
        return self.addmul(0, i, c, p)

    def addmul(self, a, b, c, p=1):
        """ Defines the addmul operation.

            either subclass this method or supply a method in the constructor, __init__

            addmul(a, b, c, p) := a + b * c ** p

            The result shall be a vector like b.

            b is always a vector for this VectorSpace; though be aware
            that there can be multiple valid Python types defined on the same
            VectorSpace. For example, particle positions are straight numpy.ndarray,
            An overdensity field may be a ComplexField or a RealField object.
        """

        raise NotImplementedError

    def dot(self, a, b):
        """ defines the inner product operation. 

            dot(a, b) := a @ b

            The result shall be a scalar floating point number.

            a and b are always vector for this VectorSpace, and are guarenteed
            to be of the same Python type -- if not there is a bug from upstream
            callers.

        """
        raise NotImplementedError


