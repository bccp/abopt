from __future__ import print_function
from abopt.abopt2 import Problem, Preconditioner
from abopt.lbfgs import LBFGSHessian
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

    def Avp(v):
        return hessian(0, v)

    z = cg_steihaug(real_vector_space, Avp, g, Delta, rtol, monitor=print)

    assert_allclose(Avp(z), -g)

def test_cg_steihaug_lbfgs():
    import numpy
    J = numpy.array([[0, 0, 4, 1],
                      [0, 2, 2, 0], 
                      [0, 3, 0, 0], 
                      [400, 0, 9, 0]])
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

    def Avp(v):
        return hessian(0, v)

    g = numpy.zeros(4) + 1.0
    g[...] = [  -2.,   -4.,   -6., -800.]
    Delta = 8000.
    rtol = 1e-8
    B = LBFGSHessian(real_vector_space, 5)
    z = cg_steihaug(real_vector_space, Avp, g, Delta, rtol, monitor=print, B=B)

    assert_allclose(B.hvp(g), -z, rtol=1e-3)
    assert_allclose(Avp(z), -g)

    print('-------', "run cg with the preconditioner")
    B2 = LBFGSHessian(real_vector_space, 5)
    z2 = cg_steihaug(real_vector_space, Avp, g, Delta, rtol, monitor=print, B=B2, mvp=B.hvp)
    assert_allclose(B2.hvp(g), -z, rtol=1e-3)
    #assert_allclose(z, z2)

    print('-------', "run cg with the new preconditioner")
    B3 = LBFGSHessian(real_vector_space, 5)
    z3 = cg_steihaug(real_vector_space, Avp, g, Delta, rtol, monitor=print, B=B3, mvp=B2.hvp)
    assert_allclose(B3.hvp(g), -z, rtol=1e-3)

def test_tr():
    trcg = TrustRegionCG(maxradius=10., maxiter=100, lbfgs_precondition=False)
    problem = Problem(objective=rosen, gradient=rosen_der, hessian_vector_product=rosen_hess_prod)

    x0 = numpy.zeros(20)
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(r.x, 1.0, rtol=1e-4)

def test_tr_precond():
    trcg = TrustRegionCG(maxradius=10., maxiter=100, lbfgs_precondition=False)
    precond = Preconditioner(Pvp=lambda x: 20 * x, vQp=lambda x: 0.05 * x, Qvp=lambda x: 0.05 * x)
    problem = Problem(objective=rosen, gradient=rosen_der, hessian_vector_product=rosen_hess_prod, precond=precond)

    x0 = numpy.zeros(2)
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(r.x, 1.0, rtol=1e-4)

def test_tr_lbfgs():
    trcg = TrustRegionCG(maxradius=10., maxiter=100, lbfgs_precondition=True)
    problem = Problem(objective=rosen, gradient=rosen_der, hessian_vector_product=rosen_hess_prod)

    x0 = numpy.zeros(20)
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(r.x, 1.0, rtol=1e-4)

def test_gaussnewton():
    trcg = TrustRegionCG(maxiter=10)
    J = numpy.array([ [0, 0,     2,  1],
                      [0,  10,   2,  0],
                      [40, 100,  0,  0],
                      [400, 0,   0,  0]])
    alpha = 0.5
    def phi(x): return x + alpha * x ** 2
    def phiprime(x): return 1 + 2 * alpha * x
    def f(x): return J.dot(phi(x))
    def vjp(x, v): return v.dot(J) * phiprime(x)
    def jvp(x, v): return J.dot(v * phiprime(x))

    def objective(x):
        y = f(x)
        return numpy.sum((y - 1.0) ** 2) + numpy.sum(x**2)

    def gradient(x):
        y = f(x)
        return vjp(x, y - 1.0) * 2 + 2 * x

    def hessian(x, v):
        v = numpy.array(v)
        return vjp(x, jvp(x, v)) * 2 + v * 2

    problem = Problem(objective=objective, gradient=gradient, hessian_vector_product=hessian, cg_rtol=1e-4, maxradius=80)

    x0 = numpy.zeros(4)
    print("Hessian")
    print(hessian(x0, [1, 0, 0, 0]))
    print(hessian(x0, [0, 1, 0, 0]))
    print(hessian(x0, [0, 0, 1, 0]))
    print(hessian(x0, [0, 0, 0, 1]))

    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
#    assert_allclose(f(r.x), 1.0, rtol=1e-4)
    assert_allclose(vjp(r.x, f(r.x) - 1.0), -r.x, rtol=1e-4)
