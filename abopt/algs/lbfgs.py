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

    We use the canonical scalar D-update as *default*. It seems to be more reliable than
    other methods for our cases, where the problem is quasi-linear, with complicated off
    diagonal structures. Perhaps in this case the low rank update is safer than the full
    rank update.

    Yu Feng
"""

from abopt.base import Optimizer
from abopt.base import Proposal

from abopt.linesearch import backtrace
from abopt.linesearch import simpleregulator

def scalar(vs, hessian):
    """ M1QN2.A in GL.  EQ 4.1;
        This is L-BFGS when people refers to it.
        wikipedia version of L-BFGS

    """

    assert len(hessian.S) > 0

    s = hessian.S[-1]
    y = hessian.Y[-1]
    ys = hessian.YS[-1]
    yy = hessian.YY[-1]

    D1 = ys / yy
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

    Dy = mul(y, D0)
    Dyy = dot(Dy, y)

    a3 = mul(mul(y, D0), s)

    D1 = addmul(D0, pow(s, 2), (1 / ys  + Dyy / ys ** 2))
    D1 = addmul(D1, mul(mul(y, D0), s), -2.0 / ys)

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

    if pre_scaled: D0 = mul(ys / D0yy, D0)

    invD0 = pow(D0, -1)

    t = addmul(invD0, pow(y, 2), 1 / ys)
    t = addmul(t, pow(mul(s, invD0), 2), -1 / dot(mul(s, invD0), s))

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

    if pre_scaled: D0 = mul(ys / D0yy, D0)

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
    def __init__(self, vs, m, diag_update=scalar, rescale_diag=False):
        """ D is a vector represents the initial diagonal. """
        self.m = m
        self.Y = [] # Delta G
        self.S = [] # Delta S
        self.YS = []
        self.YY = []
        self.D = 1.
        self.vs = vs
        self.diag_update = diag_update
        self.rescale_diag = rescale_diag

    def copy(self):
        r = LBFGSHessian(vs=self.vs, m=self.m, diag_update=self.diag_update, rescale_diag=self.rescale_diag)
        r.Y = self.Y[:]
        r.S = self.S[:]
        r.YS = self.YS[:]
        r.YY = self.YY[:]
        r.D = self.vs.copy(self.D)
        return r

    def hvp(self, v):
        """ Inverse of Hessian dot any vector; lowercase h indicates it is the inverse """
        q = v

        dot = self.vs.dot
        addmul = self.vs.addmul
        mul = self.vs.mul

        if len(self.Y) == 0: # first step
            return mul(v, self.D)

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
            D = mul(self.YS[-1]/ Dyy, D)

        z = addmul(0, q, D)
        for i in range(len(self.Y)):
            beta[i] = 1.0 / self.YS[i] * dot(self.Y[i], z)
            z = addmul(z, self.S[i], (alpha[i] - beta[i]))

        return z

    def update(self, Px0, Px1, Pg0, Pg1):
        dot = self.vs.dot
        addmul = self.vs.addmul
        mul = self.vs.mul

        y = addmul(Pg1, Pg0, -1)
        s = addmul(Px1, Px0, -1)

        ys = dot(y, s)
        yy = dot(y, y)

        if yy == 0 or ys == 0:
            # refuse to add a degenerate mode.
            return

        self.Y.append(y)
        self.S.append(s)
        self.YS.append(ys)
        self.YY.append(yy)

        if len(self.Y) > self.m:
            del self.Y[0]
            del self.S[0]
            del self.YS[0]
            del self.YY[0]

        self.D = self.diag_update(self.vs, self)

    def __repr__(self):
        return "LBFGSHessian(len(Y)=%d, m=%d)" % (len(self.Y), self.m)

class LBFGSFailure(StopIteration):
    def __init__(self, message):
        self.message = message
        StopIteration.__init__(self)

class LBFGS(Optimizer):
    optimizer_defaults = {
        'maxiter' : 100000,
        'conviter' : 6,
        'm' : 6,
        'linesearch' : backtrace,
        'linesearchiter' : 100,
        'regulator' : simpleregulator,
        'diag_update' : post_scaled_direct_bfgs,
        'rescale_diag' : False,
    }

    def start(self, problem, state, x0):
        prop = Optimizer.start(self, problem, state, x0)
        prop.B = None
        prop.z = prop.Pg
        # carry over the gradient descent search radius
        prop.r1 = getattr(state, 'r1', 1.0)
        return prop

    def move(self, problem, state, prop):
        addmul = problem.vs.addmul
        dot = problem.vs.dot

        prop.complete(state)

        state.B = prop.B
        state.z = prop.z
        state.r1 = prop.r1

        if state.B is None:
            state.B = LBFGSHessian(problem.vs, self.m, self.diag_update, self.rescale_diag)
        else:
            state.B.update(state.Px, prop.Px, state.Pg, prop.Pg)

        Optimizer.move(self, problem, state, prop)

    def single_iteration(self, problem, state):
        """ Line by line translation of the LBFGS on wikipedia """
        addmul = problem.vs.addmul
        dot = problem.vs.dot
        mul = problem.vs.mul

        r1 = state.r1

        if state.Pgnorm == 0:
            raise RuntimeError("gnorm is zero. This shall not happen because terminated() checks for this")

        try:
            # use old LBFGSHessian, and update it
            B = state.B

            # no hessian approximation yet, use GD
            if len(B.Y) == 0:
                raise LBFGSFailure("no lbfgs")

            z = B.hvp(state.Pg)

            # hvp cannot be computed, recover
            if z is None:
                raise LBFGSFailure("hvp failed")

            znorm = dot(z, z) ** 0.5
            theta = dot(z, state.Pg) / (state.Pgnorm * znorm)
            if theta < 0.0:
                # purge the hessian approximation
                B = LBFGSHessian(problem.vs, self.m, self.diag_update, self.rescale_diag)
                raise LBFGSFailure("lbfgs misaligned theta = %g" % (theta,))

            # LBFGS should have been good, so we shall not search too many times.
            prop, r2 = self.linesearch(problem, state, z, 1.0, maxiter=3)

            # failed line search, recover
            if prop is None:
                B = LBFGSHessian(problem.vs, self.m, self.diag_update, self.rescale_diag)
                raise LBFGSFailure("lbfgs linesearch failed theta=%g " % (theta, ))

            # print('BFGS Starting step = %0.2e, step moved = %0.2e'%(r2))

            #if LBFGS is not moving, try GD
            if problem.check_convergence(state.y, prop.y):
                raise LBFGSFailure("lbfgs no improvement")
                # print('LBFGS Starting step = %0.2e, abort LBGFS at step = %0.2e'%(r2, r1))

        except LBFGSFailure as e:

            z = state.Pg

            r1max = self.regulator(problem, state, z)
            r1max = min(r1max, state.r1 * 2)

            prop, r1 = self.linesearch(problem, state, state.Pg, r1max, maxiter=self.linesearchiter)

            # failed GD
            if prop is None:
                r1 = state.r1
                return None

            prop.message = e.message
            # print('GD Starting step = %0.2e, step moved = %0.2e'%(r1max, r1))


        prop.B = B
        prop.z = z
        prop.r1 = r1

        return prop
