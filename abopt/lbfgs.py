"""
    This module implements the schemes of L-BFGS initial matrix tested in

    Some numerical experiments with variable-storage quasi-Newton algorithms,
    1989,
    Gilbert, J.C. & Lemarechal, C. Mathematical Programming (1989) 45: 407.

    doi:10.1007/BF01589113

    https://link.springer.com/article/10.1007/BF01589113

    Another reference is Limited-Memory BFGS Diagonal Preconditioners by
    F. VEERSE D. AUROUX and M. FISHER, Optimization and Engineering, 1, 323-339, 2000

    M1QN3.B2 is recommended over the wikipedia L-BFGS; so we use it by default.

    Interesting quote from Gilbert and Lemarechal:

        Another point is that most QN-like methods modify the approximation of the
        Hessian by low rank corrections. As a result, w h e n / 4 , - Hk is a high rank matrix,
        a few QN updates have little effect on the improvement of Hk. On the other hand,
        scalings, which are full rank corrections, have a determining importance on the
        performance. In other words, least-change updates may be too shy in the present
        context and should perhaps be replaced by high rank stable updates.

"""

from .abopt2 import Optimizer

def scalar_diag(vs, state):
    """ M1QN2.A in Gilbert and Lemarechal 1989.  EQ 4.1; wikipedia version of L-BFGS """

    if len(state.S) == 0: return 1.0

    D0 = state.D
    s = state.S[-1]
    y = state.Y[-1]
    ys = state.YS[-1]
    yy = state.YY[-1]

    return  ys / yy

def inverse_bfgs_diag(vs, state):
    """ 
        M1QN3.A, Gilbert and Lemarechal 1989.
        Equation 4.6.
        This is bad.  D grows rapidly over iterations.

    """
    if len(state.S) == 0: return 1.0

    D0 = state.D
    s = state.S[-1]
    y = state.Y[-1]
    ys = state.YS[-1]
    yy = state.YY[-1]

    dot = vs.dot

    Dyy = dot(y * D0, y)

    D1 = D0 + 1 / ys * (
         (1 + Dyy / ys) * s ** 2 \
       - 2  * y * s * D0 )
    return D1

def direct_bfgs_diag(vs, state, scaled=False):
    """ 
        M1QN3.B, M1QN3.B2, Gilbert and Lemarechal 1989. 
        Equation 4.7, 4.9
    """
    if len(state.S) == 0: return 1.0

    D0 = state.D
    s = state.S[-1]
    y = state.Y[-1]
    ys = state.YS[-1]
    yy = state.YY[-1]

    dot = vs.dot

    if scaled:
        Dyy = dot(y * D0, y)
        sigma = ys / Dyy

        D0 = D0 * sigma

    invDss = dot(s * D0 ** -1, s)
    invD1 = D0 ** -1 + y** 2 / ys - (s / D0)**2 / invDss
    D1 = invD1 ** -1
    return D1

def scaled_direct_bfgs_diag(vs, state):
    """ M1QN3.B2 """
    return direct_bfgs_diag(vs, state, scaled=True)

def inverse_dfp_diag(vs, state, scaled=False):
    """ 
        M1QN3.C and M1QN3.C2, Gilbert and Lemarechal 1989. 
        Equation 4.8, 4.10
    """
    if len(state.S) == 0: return 1.0

    D0 = state.D
    s = state.S[-1]
    y = state.Y[-1]
    ys = state.YS[-1]
    yy = state.YY[-1]

    dot = vs.dot

    Dyy = dot(y * D0, y)

    if scaled:
        sigma = ys / Dyy
        D0 = D0 * sigma

    D1 = D0 + s**2 / ys - (D0 * y) ** 2 / Dyy

    return D1

def scaled_inverse_dfp_diag(vs, state):
    """ M1QN3.C2, bad """

    return inverse_dfp_diag(vs, state, scaled=True)

class LBFGS(Optimizer):
    problem_defaults = {'m' : 6}
    def __init__(self, vs, linesearch=Optimizer.backtrace, diag=scaled_direct_bfgs_diag):
        Optimizer.__init__(self, vs, linesearch=linesearch)
        self.diag = diag

    def move(self, problem, state, x1, y1, g1, r1):
        addmul = self.vs.addmul
        dot = self.vs.dot

        if state.nit == 0:
            state.Y = [] # Delta G
            state.S = [] # Delta S
            state.YS = []
            state.YY = []
            state.z = g1
        else:
            state.Y.append(addmul(g1, state.g, -1))
            state.S.append(addmul(x1, state.x, -1))
            state.YS.append(dot(state.Y[-1], state.S[-1]))
            state.YY.append(dot(state.Y[-1], state.Y[-1]))


            if len(state.Y) > problem.m:
                del state.Y[0]
                del state.S[0]
                del state.YS[0]

        state.D = self.diag(self.vs, state)

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

            if state.YY[-1] == 0 or state.YS[-1] == 0: # failed LBFGS
                raise StopIteration

            for i in range(len(state.Y) - 1, -1, -1):
                alpha[i] = dot(state.S[i], q) / state.YS[i]
                q = addmul(q, state.Y[i], -alpha[i])

            z = addmul(0, q, state.D)
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
            print('D', state.D)
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
            state.YY = []
            state.D = 1.0

            z = addmul(0, state.g, 1 / state.gnorm)

            x1, y1, g1, r1 = self.linesearch(self.vs, problem, state, state.g, 1.0 / state.gnorm)

            if x1 is None: raise ValueError("Line search failed.")

        if g1 is None:
            g1 = problem.gradient(x1)
            state.gev = state.gev + 1

        state.z = z
        self.post_single_iteration(problem, state, x1, y1, g1, r1)

        if len(state.Y) < problem.m and len(state.Y) > 1:
            # started building the hessian, then
            # must have a 'good' approximation before ending
            # Watch out: if > 0 is not protected we will not
            # terminated on a converged GD step.
            state.converged = False

        if state.gnorm <= problem.gtol:
            # but if gnorm is small, converged too
            state.converged = True
