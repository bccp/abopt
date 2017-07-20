from abopt.abopt2 import Problem, Preconditioner
from abopt.trustregion import cg_steihaug, TrustRegionCG
from abopt.vectorspace import real_vector_space
import numpy
from numpy.testing import assert_allclose
from scipy.optimize import rosen, rosen_der, rosen_hess_prod


def test_cg_steihaug():
    import numpy
    Hessian = numpy.diag([1, 2, 3, 400.**2])
    g = numpy.zeros(4) + 1.0
    g[...] = [  1.67362208e-02,  3.10101278e-09,  2.50026433e-02,  5.14557305e-02]
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
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(r.x, 1.0)

def test_tr_precond():
    trcg = TrustRegionCG(maxradius=10., maxiter=100)
    precond = Preconditioner(Pvp=lambda x: 2 * x, vQp=lambda x: 0.5 * x, Qvp=lambda x: 0.5 * x)
    problem = Problem(objective=rosen, gradient=rosen_der, hessian_vector_product=rosen_hess_prod, precond=precond)

    x0 = numpy.zeros(2)
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(r.x, 1.0)

def test_gaussnewton():
    trcg = TrustRegionCG(maxradius=10., maxiter=10)
    JT = numpy.diag([1, 2, 3, 4e2])
    def f(x):
        return JT.dot(x)

    # watchout JT is differently indexed due to the poor name choice in vmad.
    def vjp(x, v):
        return JT.dot(v)

    def jvp(x, v):
        return v.dot(JT)

    def objective(x):
        y = f(x)
        return numpy.sum((y - 1.0) ** 2)

    def gradient(x):
        y = f(x)
        return vjp(x, y - 1.0) * 2

    def jTjvp(x, v):
        return jvp(x, vjp(x, v))

    problem = Problem(objective=objective, gradient=gradient, hessian_vector_product=jTjvp, cg_rtol=1e-2)

    x0 = numpy.zeros(4)
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    print(r.x)
