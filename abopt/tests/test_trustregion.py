from abopt.abopt2 import Problem, Preconditioner
from abopt.trustregion import cg_steihaug, TrustRegionCG
from abopt.vectorspace import real_vector_space
import numpy
from numpy.testing import assert_allclose
from scipy.optimize import rosen, rosen_der, rosen_hess_prod


def test_cg_steigaug():
    import numpy
    Hessian = numpy.diag([1, 2, 3, 4])
    g = numpy.zeros(4) + 0.5

    Delta = 10000.
    rtol = 1e-8
    def Bvp(v):
        return Hessian.dot(v)

    z = cg_steihaug(real_vector_space, Bvp, g, Delta, rtol, monitor=print)

    assert_allclose(Hessian.dot(z), -g)

def test_tr():
    trcg = TrustRegionCG(maxradius=10.)
    problem = Problem(objective=rosen, gradient=rosen_der, hessian_vector_product=rosen_hess_prod)

    x0 = numpy.zeros(2)
    trcg.minimize(problem, x0, monitor=print)

def test_tr_precond():
    trcg = TrustRegionCG(maxradius=10., maxiter=100)
    precond = Preconditioner(Pvp=lambda x: 2 * x, vQp=lambda x: 0.5 * x, Qvp=lambda x: 0.5 * x)
    problem = Problem(objective=rosen, gradient=rosen_der, hessian_vector_product=rosen_hess_prod, precond=precond)

    x0 = numpy.zeros(2)
    trcg.minimize(problem, x0, monitor=print)

