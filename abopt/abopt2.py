
class VectorSpace(object):
    def __init__(self, addmul=None, dot=None):
        if addmul:
            self.addmul = addmul
        if dot:
            self.dot = dot

    @staticmethod
    def addmul(a, b, s):
        if s is 0:
            return 1.0 * a # always a new instance is created
        if a is 0:
            return b * s
        return a + b * s

    @staticmethod
    def dot(a, b):
        if hasattr(a, 'dot'):
            return a.dot(b)
        try:
            return sum(a * b)
        except TypeError:
            return float(a * b)

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

    def __init__(self, vs=None, linesearch=None):
        if vs is None:
            vs = VectorSpace()

        self.vs = vs

        if linesearch is None:
            linesearch = globals()['backtrace_linesearch']
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
        addmul = self.vs.addmul

        z = addmul(0, state.g, 1 / state.gnorm)

        x1, y1, g1, r1 = self.linesearch(self.vs, problem, state, z, state.rate * 2)

        if g1 is None:
            g1 = problem.gradient(x1)
            state.gev = state.gev + 1

        self.post_single_iteration(problem, state, x1, y1, g1, r1)

        if state.gnorm <= problem.gtol: 
            state.converged = True

class LBFGS(Optimizer):
    problem_defaults = {'m' : 6}

    def move(self, problem, state, x1, y1, g1, r1):
        addmul = self.vs.addmul
        dot = self.vs.dot

        if state.nit == 0:
            state.Y = [] # Delta G
            state.S = [] # Delta S
            state.YS = []
            state.H0 = 1.0
            state.z = g1
        else:
            state.Y.append(addmul(g1, state.g, -1))
            state.S.append(addmul(x1, state.x, -1))
            state.YS.append(dot(state.Y[-1], state.S[-1]))

            if len(state.Y) > problem.m:
                del state.Y[0]
                del state.S[0]
                del state.YS[0]

        Optimizer.move(self, problem, state, x1, y1, g1, r1)

    def single_iteration(self, problem, state):
        """ Line by line translation of the LBFGS on wikipedia """
        addmul = self.vs.addmul
        dot = self.vs.dot

        if state.gnorm == 0:
            raise RuntimeError("gnorm is zero. This shall not happen because terminated() checks for this")

        try:
            q = state.g

            if len(state.Y) == 0: # first step
                raise StopIteration

            alpha = list(range(len(state.Y)))
            beta = list(range(len(state.Y)))

            YY = dot(state.Y[-1], state.Y[-1])
            if YY == 0 or state.YS[-1] == 0: # failed LBFGS
                raise StopIteration

            H0 = state.YS[-1] / YY

            state.H0 = H0

            for i in range(len(state.Y) - 1, -1, -1):
                alpha[i] = dot(state.S[i], q) / state.YS[i]
                q = addmul(q, state.Y[i], -alpha[i])

            z = addmul(0, q, H0)
            for i in range(len(state.Y)):
                beta[i] = 1.0 / state.YS[i] * dot(state.Y[i], z)
                z = addmul(z, state.S[i], (alpha[i] - beta[i]))

            """
            print('-----')
            print('alpha', alpha)
            print('YS', state.YS)
            print('q', q)
            print('z', z)
            print('x', state.x)
            print('H0', state.H0)
            print('Y', state.Y)
            print('S', state.S)
            """
            x1, y1, g1, r1 = self.linesearch(self.vs, problem, state, z, 1.0)

            if x1 is None: # failed line search
                raise StopIteration

        except StopIteration:
            # LBFGS failed. Abandon LBFGS and restart with GD
            state.Y = []
            state.S = []
            state.YS = []

            z = addmul(0, state.g, 1 / state.gnorm)

            x1, y1, g1, r1 = self.linesearch(self.vs, problem, state, z, 1.0)

            if x1 is None: raise ValueError("Line search failed.")

        if g1 is None:
            g1 = problem.gradient(x1)
            state.gev = state.gev + 1

        state.z = z
        self.post_single_iteration(problem, state, x1, y1, g1, r1)

        if len(state.Y) < problem.m:
            # must have 'good' approximation for the hessian
            state.converged = False

        if state.gnorm <= problem.gtol: 
            # but if gnorm is small, converged too
            state.converged = True

def minimize_p(optimizer, objective, gradient, x0, P, PT, monitor=None, **kwargs):

    def objectivePT(Px):
        return objective(PT(Px))

    def PgradientPT(Px):
        return P(gradient(PT(Px)))

    Px0 = P(x0)

    def Pmonitor(state):
        if monitor is None: return

        Px = state.x
        Pg = state.g
        state.x = PT(Px)
        state.g = PT(Pg)
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

def backtrace_linesearch(vs, problem, state, z, rate, c=1e-5, tau=0.5):
    addmul = vs.addmul
    dot = vs.dot

    objective = problem.objective

    zz = dot(z, z)
    zg = dot(z, state.g) / zz

    if zg < 0.0: #1 * state.gnorm:
        raise ValueError("Line search failed the direction is not along the gradient direction.")

    x1 = addmul(state.x, z, -rate)
    y1 = objective(x1)
    state.fev = state.fev + 1
    i = 0
    ymin = state.y
    xmin = state.x
    ratemin = rate
    while i < 10:
        #print('rate', rate, 'y', state.y, 'y1', y1, 'x', state.x, 'x1', x1, 'z', z)

        # watch out : do not check convergence ; avoid jumping too far to the other side
        # sufficient descent

        if y1 < ymin:
            ymin = y1
            xmin = x1
            ratemin = rate
        if y1 < state.y and abs(y1 - state.y) >= abs(rate * c * zg):
            break

        rate *= tau
        x1 = addmul(state.x, z, -rate)
        y1 = objective(x1)
        state.fev = state.fev + 1
        i = i + 1

    return xmin, ymin, None, rate

def exact_linesearch(vs, problem, state, z, rate, c=0.5):
    addmul = vs.addmul
    dot = vs.dot

    objective = problem.objective

    znorm = dot(z, z) ** 0.5

    from scipy.optimize import minimize_scalar

    best = [state.x, state.y, 1.0]

    def func(tau):
        if tau == 0: return state.y

        x1 = addmul(state.x, z, -tau * rate)
        state.fev = state.fev + 1
        y1 = objective(x1)
        if y1 < best[1]:
            best[0] = x1
            best[1] = y1
            best[2] = tau

        if y1 < state.y and abs(y1 - state.y) >= abs(c * znorm):
            raise StopIteration

        return y1

    try:
        r = minimize_scalar(func, (0, 1), bounds=(0, 1), method='brent', options={'maxiter':10}, )

        if not r.success or r.fun >= state.y:
            raise StopIteration

        x1 = addmul(state.x, z, -r.x * rate)
        return x1, r.fun, None, r.x * rate

    except StopIteration as e:
        x1, y1, tau = best
        return x1, y1, None, tau * rate


def check_convergence(state, y1, rtol, atol):
    valmax = max(abs(state.y), abs(y1))
    thresh = rtol * max(valmax, 1.0) + atol

    if y1 > state.y : return False
    if abs(state.y - y1) < thresh: return True

    return False

from scipy.optimize.linesearch import scalar_search_wolfe2

def minpack_linesearch(vs, problem, state, z, rate, c1=1e-4, c2=0.9, amax=50):
    """"
    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].

    """
    addmul = vs.addmul
    dot = vs.dot

    def phi(alpha):
        state.fev = state.fev + 1
        x1 = addmul(state.x, z, -alpha)
        y1 = problem.objective(x1)
        # print('phi', -alpha, y1, state.y)
        return y1

    gval = [state.g]

    def derphi(alpha):
        state.gev = state.gev + 1
        x1 = addmul(state.x, z, -alpha)
        g1 = problem.gradient(x1)
        gval[0] = g1
    #    print('derphi', x1, g1)
        return -dot(g1, z)

    derphi0 = -dot(state.g, z)

    y0 = state.y_[-1]
    if len(state.y_) > 1:
        y0_prev = state.y_[-2]
    else:
        y0_prev = None

    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
            phi, derphi, y0, y0_prev, derphi0, c1, c2, rate)

    if derphi_star is None:
        #raise ValueError('The line search algorithm did not converge')
        return None, None, None, None
    else:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derphi_star = gval[0]

    x1 = addmul(state.x, z, -alpha_star)
    return x1, phi_star, derphi_star, alpha_star

