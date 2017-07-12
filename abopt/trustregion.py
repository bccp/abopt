"""
    A general TrustRegion method.

    https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods

"""

from .abopt2 import Optimizer, Problem

class TrustRegionSubProblem(Problem):
    def __init__(self, vs, g, Hvp, Delta, atol, rtol, gtol):
        def objective(x):
            return vs.dot(g, x) + 0.5 * vs.dot(x, Hvp(x))

        def gradient(x):
            return vs.addmul(g, Hvp(x))

        Problem.__init__(self, objective, gradient, Hvp,
            atol=atol, gtol=gtol, rtol=rtol,
        )
        self.Delta = Delta


class TrustRegion(Optimizer):
    problem_defaults = {'eta1' : 0.75,
                        'eta2' : 0.25,
                        'eta3' : 0.1,
                        't1' : 0.25,
                        't2' : 2.0,
                        }


def cg_steihaug(vs, Bvp, g, Delta, rtol, monitor=None):
    """ best effort solving for y = A^{-1} b with cg,
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
        Bd0 = Bvp(d0)
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

    return 
