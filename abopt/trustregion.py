"""
    A general TrustRegion method.

    https://optimization.mccormick.northwestern.edu/index.php/Trust-region_methods

"""

from .abopt2 import Optimizer, Problem, Proposal

class TrustRegionCG(Optimizer):
    problem_defaults = {'eta1' : 0.1,
                        'eta2' : 0.25,
                        'eta3' : 0.75,
                        't1' : 0.25,
                        't2' : 2.0,
                        'cg_rtol' : 1e-4,
                        'maxradius' : 0.4
                        }


    def single_iteration(self, problem, state):
        mul = problem.vs.mul
        dot = problem.vs.dot
        addmul = problem.vs.addmul

        def Hvp(v):
            return problem.Hvp(state.x, v)

        z = cg_steihaug(problem.vs, Hvp, state.Pg, state.radius, self.cg_rtol, monitor=print)

        mdiff = 0.5 * dot(z, Hvp(z)) + dot(state.Pg, z)

        Px1 = addmul(state.Px, z, 1)
        x1 = problem.precond.vQp(Px1)
        y1 = problem.f(x1)
        state.fev = state.fev + 1

        fdiff = y1 - state.y

        if mdiff < 0:
            rho = fdiff / mdiff
        else:
            rho = 0

        interior = dot(z, z) ** 0.5 < 0.9 * state.radius

        print('rho', rho)
        if rho < self.eta2: # poor approximation
            radius1 = self.t1 * state.radius
        elif rho > self.eta3 and not interior:
            radius1 = min(state.radius * self.t2, self.maxradius)
        else:
            radius1 = state.radius

        if rho > self.eta3:
            prop = Proposal(problem, Px=Px1, x=x1, y=y1)
        else:
            prop = Proposal(problem, Px=state.Px, x=state.x, y=state.y)

        prop.radius = radius1
        return prop

    def assess(self, problem, state, prop):
        print("assess radius", state.radius, 'tol', problem.get_tol(state.y), 'gnorm', prop.gnorm, 'gtol', problem.gtol)
        if prop.radius <= problem.get_tol(state.y):
            return True

        if prop.gnorm <= problem.gtol:
            return True

    def move(self, problem, state, prop):
        if state.nit == 0:
            state.radius = self.maxradius
        else:
            state.radius = prop.radius

        Optimizer.move(self, problem, state, prop)

def cg_steihaug(vs, Avp, g, Delta, rtol, monitor=None):
    """ best effort solving for y = - A^{-1} g with cg,
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

        if abs(dBd0) < 1e-15:
            #print("bad dBd0")
            break

        if dBd0 <= 0 or dot(p0, p0) ** 0.5 >= Delta:
            #print("dBd0", dBd0, "rad", dot(p0, p0) ** 0.5, Delta)
            # negative curvature or too fast
            # find tau such that p = z0 + tau d0 minimizes m(p)
            # and satisfies ||pk|| == \Delta_k.
            a_ = dot(d0, d0)
            b_ = -2 * dot(z0, d0)
            c_ = dot(z0, z0) - Delta ** 2
            tau = (- b_ + (b_ **2 - 4 * a_ * c_) ** 0.5) / (2 * a_)
            z1 = addmul(z0, d0, -tau)
            rho1 = 0
            r1 = r0
            d1 = d0
        else:
            z1 = addmul(z0, d0, -alpha)
            r1 = addmul(r0, Bd0, -alpha)

            rho1 = dot(r1, r1)

            d1 = addmul(r1, d0, rho1 / rho0)

        r0 = r1
        d0 = d1
        z0 = z1
        rho0 = rho1

        if monitor is not None:
            monitor(j, rho0, r0, d0, z0, Avp(z0), g)

        if rho1 / rho_init < rtol ** 2:
            break

        j = j + 1

    return z0
