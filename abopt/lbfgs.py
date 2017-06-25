"""
    This module implements the schemes of L-BFGS initial matrix tested in

    (GL)
    Some numerical experiments with variable-storage quasi-Newton algorithms,
    1989,
    Gilbert, J.C. & Lemarechal, C. Mathematical Programming (1989) 45: 407.

    doi:10.1007/BF01589113

    https://link.springer.com/article/10.1007/BF01589113

    Interesting quote from Gilbert and Lemarechal:

        Another point is that most QN-like methods modify the approximation of the
        Hessian by low rank corrections. As a result, w h e n / 4 , - Hk is a high rank matrix,
        a few QN updates have little effect on the improvement of Hk. On the other hand,
        scalings, which are full rank corrections, have a determining importance on the
        performance. In other words, least-change updates may be too shy in the present
        context and should perhaps be replaced by high rank stable updates.


    GL recommends using M1QN3.B2, which scales the diagonal D before the update.
    The scaling requires D to satisty the quasi-Cauchy condition, improving convergence.

        M1QN3_B2 = LBFGS(diag_update=pre_scaled_direct_bfgs, rescale_diag=False)

    In another reference, (VAF)  Limited-Memory BFGS Diagonal Preconditioners by
    F. VEERSE D. AUROUX and M. FISHER, Optimization and Engineering, 1, 323-339, 2000

    The authors showed scaling after D-update improves the convergence even more.

    We use post_scaled_direct_bfgs D-update as *default*:

        VAFFast = LBFGS(diag_update=post_scaled_direct_bfgs, rescale_diag=False)

    VAF recommended a "new approach", where they do not scale D during D-update, but only
    scale D before L-BFGS update. The "new approach" was supposed to improve hessian
    approximation without suffering convergence rate.

        VAFGoodHessian = LBFGS(diag_update=direct_bfgs, rescale_diag=True)

    I only find slower convergence in our linear + smoothing and LPT-nonlinear cases.
    Never cheched the hessian.

    One may tempt to use,

        M1QN3_B2_RESCALED = LBFGS(diag_update=pre_scaled_direct_bfgs, rescale_diag=True)

    but it is as slow as VAFGoodHessian, and due the prescaling it doesn't give good hessian.

    We shall do these tests more rigourously and write some notes.

    Yu Feng
"""

from .abopt2 import Optimizer

def scalar(vs, state):
    """ M1QN2.A in GL.  EQ 4.1;
        This is L-BFGS when people refers to it.
        wikipedia version of L-BFGS

    """

    if len(state.S) == 0: return vs.ones_like(state.x)

    D0 = state.D
    s = state.S[-1]
    y = state.Y[-1]
    ys = state.YS[-1]
    yy = state.YY[-1]

    D1 = vs.mul(vs.ones_like(state.x), ys / yy)
    return D1

def inverse_bfgs(vs, state):
    """ 
        M1QN3.A in GL Equation 4.6.
        This is bad.  D grows rapidly over iterations.

    """
    if len(state.S) == 0: return vs.ones_like(state.x)

    D0 = state.D
    s = state.S[-1]
    y = state.Y[-1]
    ys = state.YS[-1]
    yy = state.YY[-1]

    dot = vs.dot
    mul = vs.mul
    pow = vs.pow
    addmul = vs.addmul

    Dy = mul(D0, y)
    Dyy = dot(Dy, y)

    a3 = mul(mul(D0, y), s)

    D1 = addmul(D0, pow(s, 2), (1 / ys  + Dyy / ys ** 2))
    D1 = addmul(D1, mul(mul(D0, y), s), -2.0 / ys)

    return D1

def direct_bfgs(vs, state, pre_scaled=False, post_scaled=False):
    """ 
        M1QN3.B, M1QN3.B2, GL Equation 4.7, 4.9
        and VAF post update scaling.
    """
    if len(state.S) == 0: return vs.ones_like(state.x)

    D0 = state.D

    s = state.S[-1]
    y = state.Y[-1]
    ys = state.YS[-1]
    yy = state.YY[-1]

    dot = vs.dot
    addmul= vs.addmul
    mul = vs.mul
    pow = vs.pow

    D0yy = dot(mul(y, D0), y)

    if pre_scaled: D0 = mul(D0, ys / D0yy)

    invD0 = pow(D0, -1)

    t = addmul(invD0, pow(y, 2), 1 / ys)
    t = addmul(t, pow(mul(s, invD0), 2), -1 / dot(mul(invD0, s), s))

    D1 = pow(t, -1)

    if post_scaled:
        D1yy = dot(mul(y, D1), y)
        D1 = mul(D1, ys / D1yy)

    return D1

def pre_scaled_direct_bfgs(vs, state):
    """ M1QN3.B2 """
    return direct_bfgs(vs, state, pre_scaled=True)

def post_scaled_direct_bfgs(vs, state):
    """ Modified from M1QN3.B2; Recommended by 
        VEERSE, AUROUX & FISHER, for fewer iterations.
    """
    return direct_bfgs(vs, state, post_scaled=True)

def inverse_dfp(vs, state, pre_scaled=False, post_scaled=False):
    """ 
        M1QN3.C and M1QN3.C2 in GL Equation 4.8, 4.10;
        and VAF post update scaling.
        This is not applicable since we do not implement DFP.
    """
    D1 = vs.mul(state.x, 0.0)

    if len(state.S) == 0: return vs.ones_like(state.x)

    D0 = state.D
    s = state.S[-1]
    y = state.Y[-1]
    ys = state.YS[-1]
    yy = state.YY[-1]

    dot = vs.dot
    mul = vs.mul
    addmul = vs.addmul
    pow = vs.pow

    yD0 = mul(y, D0)
    D0yy = dot(yD0, y)

    if pre_scaled: D0 = mul(D0, ys / D0yy)

    t = addmul(D0, pow(s, 2), 1 / ys)
    t = addmul(t,  pow(yD0, 2), 1/ D0yy)

    D1 = t

    if post_scaled:
        yD1 = mul(y, D1)
        D1yy = dot(yD1, y)
        D1 = mul(D1, ys / D1yy)

    return D1

def pre_scaled_inverse_dfp(vs, state):
    """ M1QN3.C2, we didn't implement DFP."""

    return inverse_dfp(vs, state, pre_scaled=True)

def post_scaled_inverse_dfp(vs, state):
    """ post-update version of M1QN3.C2.  we didn't implement DFP."""

    return inverse_dfp(vs, state, post_scaled=True)

class LBFGS(Optimizer):
    problem_defaults = {'m' : 6}
    def __init__(self, vs=Optimizer.real_vector_space,
            linesearch=Optimizer.backtrace, diag_update=post_scaled_direct_bfgs, rescale_diag=False):
        Optimizer.__init__(self, vs, linesearch=linesearch)
        self.diag_update = diag_update
        self.rescale_diag = rescale_diag

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

        Optimizer.move(self, problem, state, x1, y1, g1, r1)

        state.D = self.diag_update(self.vs, state)

    def single_iteration(self, problem, state):
        """ Line by line translation of the LBFGS on wikipedia """
        addmul = self.vs.addmul
        dot = self.vs.dot
        mul = self.vs.mul

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

            D = state.D

            if self.rescale_diag:
                Dyy = dot(mul(state.Y[-1], D), state.Y[-1])
                D = mul(D, state.YS[-1]/ Dyy)

            z = addmul(0, q, D)
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
