from __future__ import print_function
from abopt.abopt2 import Problem, Preconditioner
from abopt.trustregion import cg_steihaug, TrustRegionCG
from abopt.vectorspace import real_vector_space
import numpy
from numpy.testing import assert_allclose
import pytest

from abopt.testing import RosenProblem, ChiSquareProblem

def test_cg_steihaug():
    import numpy
    #Hessian = numpy.diag([1, 2, 3, 400.**2])
    J = numpy.array([[0, 0, 0, 1],
                      [0, 0, 2, 0], 
                      [0, 3, 0, 0], 
                      [400, 0, 0, 0]])
    C = numpy.array([[2, 0, 0, 0],
                      [0, 1, 0, 0], 
                      [0, 0, 3, 0], 
                      [0, 0, 0, 1]])

    problem = ChiSquareProblem(J=J)

    g = numpy.zeros(4) + 1.0
    g[...] = [  -2.,   -4.,   -6., -800.]
    Delta = 8000.
    rtol = 1e-8

    def Avp(v): return problem.Hvp(0, v)

    z = cg_steihaug(real_vector_space, Avp, g, g, Delta, rtol, monitor=print)

    assert_allclose(Avp(z), g)

    z = cg_steihaug(real_vector_space, Avp, g, g*0, Delta, rtol, monitor=print)

    assert_allclose(Avp(z), g)

    def precond(v, direction):
        if direction == -1:
            return numpy.linalg.inv(C).dot(v)
        else:
            return C.dot(v)

    z = cg_steihaug(real_vector_space, Avp, g, g*0, Delta, rtol, monitor=print, C=precond)

    assert_allclose(Avp(z), g)

def test_tr():
    trcg = TrustRegionCG(maxradius=10., maxiter=100, cg_monitor=print)
    problem = RosenProblem()

    x0 = numpy.zeros(20)
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(r.x, 1.0, rtol=1e-4)

def test_tr_precond():
    trcg = TrustRegionCG(maxradius=10., maxiter=100)
    precond = Preconditioner(
                        Pvp=lambda x: 20 * x,
                        vPp=lambda x: 20 * x,
                        vQp=lambda x: 0.05 * x,
                        Qvp=lambda x: 0.05 * x)
    problem = RosenProblem(precond=precond)

    x0 = numpy.zeros(2)
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(r.x, 1.0, rtol=1e-4)

@pytest.mark.parametrize("alpha,beta", 
    [
    [1.0, 0.0],
    [0.01, 1.0],
    [1.0, 1.0],
    ]
)
def test_gaussnewton(alpha, beta):
    trcg = TrustRegionCG(maxiter=100, cg_rtol=1e-9, rtol=1e-8, maxradius=80)

    J = numpy.array([ [0, 0,     2,  1],
                      [0,  10,   2,  0],
                      [40, 100,  0,  0],
                      [400, 0,   0,  0]])
    def phi(x): return alpha * x + beta * x ** 2
    def phiprime(x): return alpha + 2 * beta * x

    problem = ChiSquareProblem(J=J, phi=phi, phiprime=phiprime)

    x0 = numpy.zeros(4)
    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(problem.f(r.x), 0, atol=1e-7)

@pytest.mark.parametrize("alpha,beta", 
    [
    [1.0, 0.0],
    ]
)
def test_gaussnewton_prec(alpha, beta):
    trcg = TrustRegionCG(maxiter=100, cg_rtol=1e-9,
            rtol=1e-8, maxradius=80,
            cg_monitor=print)

    J = numpy.array([ [1e6, 0,     0,  1],
                      [0,  10,   0,  0],
                      [2,  0,  10,  0],
                      [1, 0,   0,  1]])

    # finding the diagonals and use it to precondition
    # the hessian inversion.
    def cg_precond(Avp):
        s = 0
        for v in numpy.eye(len(J)):
            Av = Avp(v)
            vAv = v * Av
            s = s + vAv
        def C(r, direction):
            if direction == -1:
                return r / s
            else:
                return r * s
        return C

    trcg.cg_preconditioner = cg_precond

    def phi(x): return alpha * x + beta * x ** 2
    def phiprime(x): return alpha + 2 * beta * x

    problem = ChiSquareProblem(J=J, phi=phi, phiprime=phiprime)

    x0 = numpy.zeros(len(J))

    r = trcg.minimize(problem, x0, monitor=print)
    assert r.converged
    assert r.nit < 10
    assert_allclose(problem.f(r.x), 0, atol=1e-7)

