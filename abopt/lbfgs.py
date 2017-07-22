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

def scalar(vs, hessian):
    """ M1QN2.A in GL.  EQ 4.1;
        This is L-BFGS when people refers to it.
        wikipedia version of L-BFGS

    """

    assert len(hessian.S) > 0

    D0 = hessian.D
    s = hessian.S[-1]
    y = hessian.Y[-1]
    ys = hessian.YS[-1]
    yy = hessian.YY[-1]

    D1 = vs.mul(vs.ones_like(D0), ys / yy)
    return D1

def inverse_bfgs(vs, hessian):
    """ 
        M1QN3.A in GL Equation 4.6.
        This is bad.  D grows rapidly over iterations.

    """
    assert len(hessian.S) > 0

    D0 = hessian.D
    s = hessian.S[-1]
    y = hessian.Y[-1]
    ys = hessian.YS[-1]
    yy = hessian.YY[-1]

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

def direct_bfgs(vs, hessian, pre_scaled=False, post_scaled=False):
    """ 
        M1QN3.B, M1QN3.B2, GL Equation 4.7, 4.9
        and VAF post update scaling.
    """
    assert len(hessian.S) > 0

    D0 = hessian.D

    s = hessian.S[-1]
    y = hessian.Y[-1]
    ys = hessian.YS[-1]
    yy = hessian.YY[-1]

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

def pre_scaled_direct_bfgs(vs, hessian):
    """ M1QN3.B2 """
    return direct_bfgs(vs, hessian, pre_scaled=True)

def post_scaled_direct_bfgs(vs, hessian):
    """ Modified from M1QN3.B2; Recommended by 
        VEERSE, AUROUX & FISHER, for fewer iterations.
    """
    return direct_bfgs(vs, hessian, post_scaled=True)

def inverse_dfp(vs, hessian, pre_scaled=False, post_scaled=False):
    """ 
        M1QN3.C and M1QN3.C2 in GL Equation 4.8, 4.10;
        and VAF post update scaling.
        This is not applicable since we do not implement DFP.
    """
    assert len(hessian.S) > 0

    D0 = hessian.D
    s = hessian.S[-1]
    y = hessian.Y[-1]
    ys = hessian.YS[-1]
    yy = hessian.YY[-1]

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

def pre_scaled_inverse_dfp(vs, hessian):
    """ M1QN3.C2, we didn't implement DFP."""

    return inverse_dfp(vs, hessian, pre_scaled=True)

def post_scaled_inverse_dfp(vs, hessian):
    """ post-update version of M1QN3.C2.  we didn't implement DFP."""

    return inverse_dfp(vs, hessian, post_scaled=True)

class LBFGSHessian(object):
    def __init__(self, vs, m, D, diag_update, rescale_diag):
        """ D is a vector represents the initial diagonal. """
        self.m = m
        self.Y = [] # Delta G
        self.S = [] # Delta S
        self.YS = []
        self.YY = []
        self.D = D
        self.vs = vs
        self.diag_update = diag_update
        self.rescale_diag = rescale_diag
    def __len__(self):
        return len(self.Y)

    def Hvp(self, v):
        """ Inverse of Hessian dot any vector """
        q = v

        dot = self.vs.dot
        addmul = self.vs.addmul
        mul = self.vs.mul

        if len(self.Y) == 0: # first step
            return mul(self.D, v)

        alpha = list(range(len(self.Y)))
        beta = list(range(len(self.Y)))

        if self.YY[-1] == 0 or self.YS[-1] == 0: # failed LBFGS
            return None

        for i in range(len(self.Y) - 1, -1, -1):
            alpha[i] = dot(self.S[i], q) / self.YS[i]
            q = addmul(q, self.Y[i], -alpha[i])

        D = self.D

        if self.rescale_diag:
            Dyy = dot(mul(self.Y[-1], D), self.Y[-1])
            D = mul(D, self.YS[-1]/ Dyy)

        z = addmul(0, q, D)
        for i in range(len(self.Y)):
            beta[i] = 1.0 / self.YS[i] * dot(self.Y[i], z)
            z = addmul(z, self.S[i], (alpha[i] - beta[i]))
        return z

    def update(self, state, prop):
        dot = self.vs.dot
        addmul = self.vs.addmul
        mul = self.vs.mul

        self.Y.append(addmul(prop.Pg, state.Pg, -1))
        self.S.append(addmul(prop.Px, state.Px, -1))
        self.YS.append(dot(self.Y[-1], self.S[-1]))
        self.YY.append(dot(self.Y[-1], self.Y[-1]))

        if len(self.Y) > self.m:
            del self.Y[0]
            del self.S[0]
            del self.YS[0]
            del self.YY[0]

        self.D = self.diag_update(self.vs, self)

class LBFGS(Optimizer):
    from .linesearch import backtrace

    optimizer_defaults = {
        'maxiter' : 1000,
        'm' : 6,
        'linesearch' : backtrace,
        'diag_update' : post_scaled_direct_bfgs,
        'rescale_diag' : False,
    }

    def _newLBFGSHessian(self, problem, Px):
        return LBFGSHessian(problem.vs, self.m, problem.vs.ones_like(Px), self.diag_update, self.rescale_diag)

    def assess(self, problem, state, prop):
        if len(state.B) < self.m and len(state.B) > 1:
            # started building the hessian, then
            # must have a 'good' approximation before ending
            # Watch out: if > 0 is not protected we will not
            # terminated on a converged GD step.
            return

        if prop.dxnorm <= problem.xtol:
            return True, "Solution stopped moving"

        if problem.check_convergence(state.y, prop.y):
            return True, "Objective stopped improving"

        if prop.gnorm <= problem.gtol:
            return True, "Gradient is sufficiently small"

        if prop.Pgnorm == 0:
            return False, "Preconditioned gradient vanishes"


    def move(self, problem, state, prop):
        addmul = problem.vs.addmul
        dot = problem.vs.dot

        prop.complete(state)

        if state.nit == 0:
            state.B = self._newLBFGSHessian(problem, prop.Px)
            state.z = prop.Pg
        else:
            state.B = prop.B
            state.z = prop.z
            state.B.update(prop, state)

        Optimizer.move(self, problem, state, prop)

    def single_iteration(self, problem, state):
        """ Line by line translation of the LBFGS on wikipedia """
        addmul = problem.vs.addmul
        dot = problem.vs.dot
        mul = problem.vs.mul

        if state.Pgnorm == 0:
            raise RuntimeError("gnorm is zero. This shall not happen because terminated() checks for this")

        try:
            # use old LBFGSHessian, and update it
            B = state.B
            z = B.Hvp(state.Pg)

            # Hvp cannot be computed, recover
            if z is None: raise StopIteration

            prop, r1 = self.linesearch(problem, state, z, 1.0)

            # failed line search, recover
            if prop is None: raise StopIteration

        except StopIteration:
            # LBFGS failed. Abandon LBFGS and restart with GD
            B = self._newLBFGSHessian(problem, state.Px)

            z = B.Hvp(state.Pg)

            prop, r1 = self.linesearch(problem, state, state.Pg, 1.0)

            # failed GD, die without a proposal
            if prop is None: return None

        prop.B = B
        prop.z = z

        return prop
