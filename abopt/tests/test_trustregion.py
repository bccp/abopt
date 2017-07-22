from abopt.abopt2 import Problem, Preconditioner
from abopt.trustregion import cg_steihaug, TrustRegionCG
from abopt.vectorspace import real_vector_space
import numpy
from numpy.testing import assert_allclose
from scipy.optimize import rosen, rosen_der, rosen_hess_prod


def test_cg_steihaug():
    import numpy
    #Hessian = numpy.diag([1, 2, 3, 400.**2])
    J = numpy.array([[0, 0, 0, 1],
                      [0, 0, 2, 0], 
                      [0, 3, 0, 0], 
                      [400, 0, 0, 0]])
    def f(x): return J.dot(x)
    def vjp(x, v): return v.dot(J)
    def jvp(x, v): return J.dot(v)

    def objective(x):
        y = f(x)
        return numpy.sum((y - 1.0) ** 2)

    def gradient(x):
        y = f(x)
        return vjp(x, y - 1.0) * 2

    def hessian(x, v):
        return vjp(x, jvp(x, v)) * 2

    g = numpy.zeros(4) + 1.0
    g[...] = [  -2.,   -4.,   -6., -800.]
    Delta = 8000.
    rtol = 1e-8

    z = cg_steihaug(real_vector_space, hessian, g, Delta, rtol, monitor=print)

    assert_allclose(Bvp(z), -g)

def test_tr():
    trcg = TrustRegionCG(maxradius=10.)
    problem = Problem(objective=rosen, gradient=rosen_der, hessian_vector_product=rosen_hess_prod)

    x0 = numpy.zeros(2)
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(r.x, 1.0, rtol=1e-4)

def test_tr_precond():
    trcg = TrustRegionCG(maxradius=10., maxiter=100)
    precond = Preconditioner(Pvp=lambda x: 20 * x, vQp=lambda x: 0.05 * x, Qvp=lambda x: 0.05 * x)
    problem = Problem(objective=rosen, gradient=rosen_der, hessian_vector_product=rosen_hess_prod, precond=precond)

    x0 = numpy.zeros(2)
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(r.x, 1.0, rtol=1e-4)

def test_gaussnewton():
    trcg = TrustRegionCG(maxiter=10)
    J = numpy.array([[0, 0, 0, 1],
                      [0, 0, 2, 0],
                      [0, 1, 0, 0],
                      [1, 0, 0, 0]])
    def f(x): return J.dot(x)
    def vjp(x, v): return v.dot(J)
    def jvp(x, v): return J.dot(v)

    def objective(x):
        y = f(x)
        return numpy.sum((y - 1.0) ** 2)

    def gradient(x):
        y = f(x)
        return vjp(x, y - 1.0) * 2

    def hessian(x, v):
        return vjp(x, jvp(x, v)) * 2

    problem = Problem(objective=objective, gradient=gradient, hessian_vector_product=hessian, cg_rtol=1e-4, maxradius=8000)

    print("Hessian")
    print(hessian(None, [1, 0, 0, 0]))
    print(hessian(None, [0, 1, 0, 0]))
    print(hessian(None, [0, 0, 1, 0]))
    print(hessian(None, [0, 0, 0, 1]))

    x0 = numpy.zeros(4)
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(f(r.x), 1.0, rtol=1e-4)
