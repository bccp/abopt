"""
    A general TrustRegion method.

    https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods

"""

from .abopt2 import Optimizer, Problem, Proposal
from .lbfgs import post_scaled_direct_bfgs, LBFGSHessian

class TrustRegionCG(Optimizer):
    optimizer_defaults = {'eta1' : 0.1,
                        'eta2' : 0.25,
                        'eta3' : 0.75,
                        't1' : 0.25,
                        't2' : 2.0,
                        'maxiter' : 1000,
                        'lbfgs_precondition' : True,
                        'm' : 6,
                        'diag_update' : post_scaled_direct_bfgs,
                        'rescale_diag' : False,
                        }

    problem_defaults = {
                        'cg_rtol' : 1e-2,
                        'maxradius' : 100.
                    }

    def _newHessianApprox(self, problem):
        return LBFGSHessian(problem.vs, m=self.m, diag_update=self.diag_update, rescale_diag=self.rescale_diag)

    def single_iteration(self, problem, state):
        mul = problem.vs.mul
        dot = problem.vs.dot
        addmul = problem.vs.addmul
        def Avp(v):
            state.hev = state.hev + 1
            return problem.PHvp(state.x, v)

        def cg_monitor(*kwargs):
            #print(*kwargs)
            pass

        if self.lbfgs_precondition:
            B1 = self._newHessianApprox(problem)
            mvp = state.B.hvp
        else:
            B1 = None
            mvp = None

        # solve H z = g constrained by the radius
        z = cg_steihaug(problem.vs, Avp, state.Pg, state.radius, problem.cg_rtol, monitor=cg_monitor, B=B1, mvp=mvp)

        mdiff = 0.5 * dot(z, Avp(z)) + dot(state.Pg, z)

        Px1 = addmul(state.Px, z, 1)
        x1 = problem.Px2x(Px1)
        y1 = problem.f(x1)
        state.fev = state.fev + 1

        fdiff = y1 - state.y

        if mdiff < 0:
            rho = fdiff / mdiff
        else:
            rho = 0

#        print(y1, x1)
#        print(state.y, state.x)
        #print('rho', rho, 'fdiff', fdiff, 'mdiff', mdiff, 'Avp(z)', Avp(z), 'Pg', state.Pg, 'znorm', dot(z, z) ** 0.5, 'radius', state.radius)

        interior = dot(z, z) ** 0.5 < 0.9 * state.radius

        if rho < self.eta2: # poor approximation
            # reinialize radius from the gradient norm if needed
            radius1 = min(self.t1 * state.radius, state.Pgnorm)
        elif rho > self.eta3 and not interior: # good and too conservative
            radius1 = min(state.radius * self.t2, problem.maxradius)
        else: # about right
            radius1 = state.radius

        if rho > self.eta1: # sufficient quadratic, move
            prop = Proposal(problem, Px=Px1, x=x1, y=y1)
        else: # poor, stay and adjust the radius
            prop = Proposal(problem, Px=state.Px, x=state.x, y=state.y)

        prop.radius = radius1
        prop.rho = rho
        prop.B = B1
        return prop

    def assess(self, problem, state, prop):
        #print("assess radius", state.radius, 'tol', problem.get_tol(state.y), 'gnorm', prop.gnorm, 'gtol', problem.gtol)
        if prop.radius >= state.radius:
            if problem.check_convergence(state.y, prop.y):
                return True, "Objective is not improving in trust region"
            if prop.dxnorm <= problem.xtol:
                return True, "Solution is not moving in trust region"

        if prop.gnorm <= problem.gtol:
            return True, "Gradient is sufficiently small"

        if prop.Pgnorm == 0:
            return False, "Preconditioned gradient vanishes"

    def move(self, problem, state, prop):
        if prop.init:
            # initial radius is the norm of the gradient.
            state.radius = prop.Pgnorm
            state.B = self._newHessianApprox(problem)
            state.rho = 1.0
        else:
            state.radius = prop.radius
            state.B = prop.B
            state.rho = prop.rho

        #print('move', prop.y)
        Optimizer.move(self, problem, state, prop)

def cg_steihaug(vs, Avp, g, Delta, rtol, monitor=None, B=None, mvp=None):
    """ best effort solving for y = - A^{-1} g with cg,
        given the trust-region constraint.

        This is roughly ported from Jeff Regier's

            https://github.com/jeff-regier/Celeste.jl/blob/master/src/cg_trust_region.jl

        the algorithm is almost identical to the one on NWU wiki,

            https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods

        if given B, update approximated Hessian, see Morale and Nocedal, 2000

            http://epubs.siam.org/doi/abs/10.1137/S1052623497327854

        mvp is the preconditioner operator. M^{-1}.
    """

    if mvp is None: mvp = lambda x:x
    dot = vs.dot
    mul = vs.mul
    addmul = vs.addmul

    z0 = vs.zeros_like(g)
    r0 = g
    mr0 = mvp(r0)
    d0 = mr0

    j = 0

    rho_init = dot(mr0, r0)

    rho0 = rho_init

    while True:
        Bd0 = Avp(d0)
        dBd0 = dot(d0, Bd0)

        alpha = rho0 / dBd0

        p0 = addmul(z0, d0, -alpha)

        if dBd0 == 0: # zero Hessian
            rho1 = 0 # will terminate
            z1 = z0
            r1 = r0
            mr1 = mr0
            d1 = d0

        elif dBd0 <= 0 or dot(p0, p0) ** 0.5 >= Delta:
            #print("dBd0", dBd0, "rad", dot(p0, p0) ** 0.5, Delta)
            # negative curvature or too fast
            # find tau such that p = z0 + tau d0 minimizes m(p)
            # and satisfies ||pk|| == \Delta_k.
            a_ = dot(d0, d0)
            b_ = -2 * dot(z0, d0)
            c_ = dot(z0, z0) - Delta ** 2
            tau = (- b_ + (b_ **2 - 4 * a_ * c_) ** 0.5) / (2 * a_)
            z1 = addmul(z0, d0, -tau)
            rho1 = 0 # will terminate
            r1 = r0
            mr1 = mr0
            d1 = d0
        else:
            z1 = addmul(z0, d0, -alpha)
            r1 = addmul(r0, Bd0, -alpha)
            mr1 = mvp(r1)

            rho1 = dot(mr1, r1)

            d1 = addmul(mr1, d0, rho1 / rho0)


            if B is not None:
                B.update(z0, z1, r0, r1)

        r0 = r1
        mr0 = mr1
        d0 = d1
        z0 = z1
        rho0 = rho1

        if monitor is not None:
            monitor(j, rho0, r0, d0, z0, Avp(z0), g, B)

        if rho1 / rho_init < rtol ** 2:
            #print("rho1 / rho_init", rho1 / rho_init, rtol ** 2)
            break

        j = j + 1

    return z0
