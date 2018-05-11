"""
    A general TrustRegion method.

    https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods

"""

from .abopt2 import Optimizer, Problem, Proposal, ContinueIteration, ConvergedIteration, FailedIteration
from .lbfgs import post_scaled_direct_bfgs, LBFGSHessian

class TrustRegionCG(Optimizer):
    from .linesearch import backtrace
    optimizer_defaults = {'eta1' : 0.1,
                        'eta2' : 0.25,
                        'eta3' : 0.75,
                        't1' : 0.25,
                        't2' : 2.0,
                        'maxiter' : 1000,
                        'lbfgs_precondition' : False,
                        'm' : 6,
                        'conviter' : 6, 
                        'diag_update' : post_scaled_direct_bfgs,
                        'rescale_diag' : False,
                        'linesearch' : backtrace,
                        'linesearchiter' : 100,
                        'cg_monitor' : None,
                        'cg_maxiter' : 50,
                        'cg_rtol' : 1e-2,
                        'maxradius' : 100.,
                        'initradius' : None,
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
            if self.cg_monitor is not None:
                self.cg_monitor(*kwargs)

        if self.lbfgs_precondition:
            raise RuntimeError("LBFGS Precondition is broken because I don't know how to apply LBFGS Hessian. The standard formula only does Hessian Inverse.")
            # cannot reuse state.B, must be a new B1 because
            # otherwise mvp will change during cg_steihaug!
            B1 = state.B
            Cinv = state.B.copy().hvp
            C = Unknown
        else:
            B1 = None
            Cinv = None
            C = None

        # solve - H z = g constrained by the radius
        radius1 = state.radius

        z = cg_steihaug(problem.vs, Avp, state.Pg, state.z, radius1,
                self.cg_rtol, self.cg_maxiter, monitor=cg_monitor, B=B1, Cinv=Cinv, C=C)

        mdiff = 0.5 * dot(z, Avp(z)) - dot(state.Pg, z)

        Px1 = addmul(state.Px, z, -1)
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
        #print('rho', rho, 'fdiff', fdiff, 'mdiff', mdiff, 'Avp(z)', Avp(z), 'Pg', state.Pg, 'znorm', dot(z, z) ** 0.5, 'radius', radius1)

        interior = dot(z, z) ** 0.5 < 0.9 * radius1

        if rho > self.eta1: # sufficient quadratic, move
            prop = Proposal(problem, Px=Px1, x=x1, y=y1, z=z)
        else: # poor, stay and use the shrunk radius
            # restart from the previus cg_steihaug result.
            prop = Proposal(problem, Px=state.Px, x=state.x, y=state.y, z=z)

        if rho < self.eta2: # poor approximation
            # reinialize radius from the gradient norm if needed
            radius1 = min(self.t1 * radius1, state.Pgnorm)
            prop.message = "poor approximation "
        elif rho > self.eta3 and not interior: # good and too conservative
            radius1 = min(radius1 * self.t2, self.maxradius)
            prop.message = "too conservative"
        else: # about right
            radius1 = radius1
            prop.message = "good approximation"


        prop.radius = radius1
        prop.rho = rho
        prop.B = B1

        #if TrustRegion is not moving, try GD
        if rho > self.eta1 and problem.check_convergence(state.y, prop.y):
            mul = problem.vs.mul

            z = mul(state.Pg, 1 / state.Pgnorm)

            prop, r1 = self.linesearch(problem, state, z, 2.0 * radius1, maxiter=self.linesearchiter)
            # reinit.
            prop.radius = r1
            prop.reinit = True
            prop.message = "Falling back to line search"

        return prop

    def assess(self, problem, state, prop):
        if prop is None:
            return FailedIteration("no proposal is made")

        prop = prop.complete(state)

        #print("assess radius", state.radius, 'tol', problem.get_tol(state.y), 'gnorm', prop.gnorm, 'gtol', problem.gtol)
        if hasattr(prop, 'reinit') and prop.reinit:
            if problem.check_convergence(state.y, prop.y):
                return ConvergedIteration("Objective is not improving in GD")

        else:
            if prop.radius >= state.radius:
                if problem.check_convergence(state.y, prop.y):
                    return ConvergedIteration("Objective is not improving in trust region")
                if prop.dxnorm <= problem.xtol:
                    return ConvergedIteration("Solution is not moving in trust region")

        if prop.gnorm <= problem.gtol:
            return ConvergedIteration("Gradient is sufficiently small")

        return ContinueIteration("normal iteration")

    def move(self, problem, state, prop):
        if prop.init or (hasattr(prop, 'reinit') and prop.reinit):
            # initial radius is the norm of the gradient.
            if self.initradius is None:
                state.radius = min(prop.Pgnorm, self.maxradius)
            else:
                state.radius = self.initradius
            state.B = self._newHessianApprox(problem)
            state.rho = 1.0
        else:
            state.radius = prop.radius
            state.B = prop.B
            state.rho = prop.rho

        #print('move', prop.y)
        Optimizer.move(self, problem, state, prop)

def cg_steihaug(vs, Avp, g, z0, Delta, rtol, maxiter=1000, monitor=None, B=None, Cinv=None, C=None):
    """ best effort solving for y = A^{-1} g with cg,
        given the trust-region constraint;

        This is roughly ported from Jeff Regier's

            https://github.com/jeff-regier/Celeste.jl/blob/316cae12efcc394a5253ee799056c5513c2efc67/src/cg_trust_region.jl

        the algorithm is almost identical to the one on NWU wiki,

            https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods

        if given B, update approximated Hessian, see Morale and Nocedal, 2000

            http://epubs.siam.org/doi/abs/10.1137/S1052623497327854


        Cinv and C are the preconditioner operator. C^{-1}, and C, 

        See Steihaug's paper. https://epubs.siam.org/doi/pdf/10.1137/0720042

    """
    if C is None: C = lambda x:x
    if Cinv is None: Cinv = lambda x:x
    dot = vs.dot
    mul = vs.mul
    addmul = vs.addmul

    if z0 is None:
        z0 = vs.zeros_like(g)

    # FIXME: how to seed cg with a different starting point?
    r0 = addmul(Avp(z0), g, -1)
    mr0 = Cinv(r0)
    d0 = mul(mr0, 1)

    j = 0

    rho_init = dot(mr0, r0)   # <r, mr>

    rho0 = rho_init

    while True:
        Bd0 = Avp(d0)
        dBd0 = dot(d0, Bd0)  # gamma

        alpha = rho0 / dBd0

        p0 = addmul(z0, d0, -alpha)

        message = ""

        if dBd0 == 0: # zero Hessian
            rho1 = 0 # will terminate
            z1 = z0
            r1 = r0
            mr1 = mr0
            d1 = d0
            message = "zero hessian"

        elif dBd0 <= 0 or dot(p0, C(p0)) ** 0.5 >= Delta:
            #print("dBd0", dBd0, "rad", dot(p0, p0) ** 0.5, Delta)
            # negative curvature or too fast
            # find tau such that p = z0 + tau d0 minimizes m(p)
            # and satisfies ||pk|| == \Delta_k.
            a_ = dot(d0, C(d0))
            b_ = 2 * dot(z0, C(d0))
            c_ = dot(z0, C(z0)) - Delta ** 2
            cond = (b_ **2 - 4 * a_ * c_)
            if a_ == 0 or cond < 0: # already at the solution, do not move.
                rho1 = -1
                tau = 1.0
                if c_ > 0:
                    tau = Delta / c_ ** 0.5
                z1 = mul(z0, tau)
                if a_ == 0:
                    message = "already at the right direction"
                    rho1 = -1
                else:
                    message = "no solution to second order equation, restarting "
                    z0 = vs.zeros_like(g)
                    r0 = addmul(Avp(z0), g, -1)
                    mr0 = Cinv(r0)
                    d0 = mul(mr0, 1)
                    rho1 = dot(mr0, r0)   # <r, mr>
            else:
                tau = (- b_ - cond ** 0.5) / (2 * a_)
                # do not assert because when d0 is close to zero
                # tau may be a large number
                # assert tau <= 0
                z1 = addmul(z0, d0, tau)

                if dBd0 <= 0:
                    rho1 = -1 # will terminate
                else:
                    rho1 = 0
                if dBd0 <= 0:
                    message = "negative curvature "
                else:
                    message = "truncation"

            r1 = r0
            mr1 = mr0
            d1 = d0
        else:
            z1 = addmul(z0, d0,  -alpha)
            r1 = addmul(r0, Bd0, -alpha)
            mr1 = Cinv(r1)

            rho1 = dot(mr1, r1)
            d1 = mul(mr1, 1)
            d1 = addmul(d1, d0, rho1 / rho0)


            if B is not None:
                B.update(z0, z1, r0, r1)

            message = "regular iteration"

        r0 = r1
        mr0 = mr1
        d0 = d1
        z0 = z1
        rho0 = rho1

        if monitor is not None:
#            monitor(j, rho0, r0, d0, z0, Avp(z0), g, B)
            z0norm = dot(z0, z0)
            zgnorm = dot(z0, g)
            gnorm = dot(g, g)
            monitor(j, message, rho0, rho_init, rtol, zgnorm / z0norm ** 0.5 / gnorm ** 0.5)

        if rho1 / rho_init < rtol ** 2:
            #print("rho1 / rho_init", rho1 / rho_init, rtol ** 2)
            break

        if j >= maxiter:
            break
        j = j + 1

    return z0
