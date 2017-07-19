"""
    A general TrustRegion method.

    https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods

"""

from .abopt2 import Optimizer, Problem

class TrustRegion(Optimizer):
    problem_defaults = {'eta1' : 0.1,
                        'eta2' : 0.25,
                        'eta3' : 0.75,
                        't1' : 0.25,
                        't2' : 2.0,
                        }


    def single_iteration(self, problem, state):
        mul = problem.vs.mul

        def Hvp(v):
            return problem.Hvp(state.x, v)

        z = cg_steihaug(problem.vs, Hvp, state.Pg, state.radius, rtol)

        mdiff = 0.5 * dot(z, Hvp(z)) + dot(g, z)
        assert mdiff < 0

        Px1 = addmul(state.Px, z, 1)
        x1 = problem.precond.vQp(Px1)

        fdiff = problem.f(x1) - state.f

        rho = fdiff / mdiff

        interior = dot(state.s, state.s) ** 0.5 < 0.9 * state.radius

        if rho < self.eta2:
            radius1 = self.t1 * state.radius
        elif rho > self.eta3 and interior:
            radius1 = min(state.radius * self.t2, self.max_radius)
        else:
            radius1 = state.radius

        if rho > self.eta3:
            accept
        else:
            decline

        self.post_single_iteration(problem, state, x1, Px1, y1, g1, Pg1, r1)

        if state.gnorm <= problem.gtol: 
            state.converged = True

def cg_steihaug(vs, Avp, z, Delta, rtol, monitor=None):
    """ best effort solving for y = A^{-1} g with cg,
        given by trust-region constraint.

        ported from Jeff Regier's

        https://github.com/jeff-regier/Celeste.jl/blob/master/src/cg_trust_region.jl

        the algorithm is identical to the one or NWU wiki.

    """
    dot = vs.dot
    mul = vs.mul
    addmul = vs.addmul

    z0 = vs.zeros_like(g)
    r0 = g
    d0 = g

    j = 0

    rho_init = dot(r0, r0)

    rho0 = rho_init

    while True:
        Bd0 = Avp(d0)
        dBd0 = dot(d0, Bd0)

        alpha = rho0 / dBd0

        p0 = addmul(z0, d0, -alpha)

        if dBd0 <= 0 or dot(p0, p0) >= Delta:
            # negative curvature or too fast
            # find tau such that p = z0 + tau d0 minimizes m(p)
            # and satisfies ||pk|| == \Delta_k.
            a_ = dot(d0, d0)
            b_ = -2 * dot(z0, d0)
            c_ = dot(z0, z0) - Delta ** 2
            tau = (- b_ + (b_ **2 - 4 * a_ * c_) ** 0.5) / (2 * a_)
            z1 = addmul(z0, d0, -tau)
            break

        if abs(dBd0) < 1e-15:
            break

        z1 = addmul(z0, d0, -alpha)
        r1 = addmul(r0, Bd0, -alpha)

        rho1 = dot(r1, r1)

        d1 = addmul(r1, d0, rho1 / rho0)

        r0 = r1
        d0 = d1
        z0 = z1
        rho0 = rho1

        if monitor is not None:
            monitor(j, rho0, r0, d0, z0)

        if rho1 / rho_init < rtol ** 2:
            break

        j = j + 1

    return z0
