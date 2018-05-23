
import pytest

from abopt.abopt2 import Preconditioner, minimize

from abopt.vectorspace import real_vector_space, complex_vector_space
from abopt.linesearch import minpack, backtrace, exact

from abopt.lbfgs import LBFGS
from abopt.lbfgs import inverse_bfgs, direct_bfgs, scalar, inverse_dfp
from abopt.lbfgs import pre_scaled_direct_bfgs, pre_scaled_inverse_dfp
from abopt.lbfgs import post_scaled_direct_bfgs, post_scaled_inverse_dfp

from abopt.testing import RosenProblem, ChiSquareProblem
import numpy
from numpy.testing import assert_allclose

@pytest.mark.parametrize("linesearch, diag_update, rescale_diag",
    [
        (backtrace, post_scaled_direct_bfgs, False), # VAFFast
#        (backtrace, direct_bfgs, True), # VAFGoodHessian many iterations!
#        (backtrace, pre_scaled_direct_bfgs, True), # VAFBadHessian many iterations!
        (minpack, post_scaled_direct_bfgs, False), # VAFFast
        (exact, post_scaled_direct_bfgs, False), # VAFFast
    ]
)
def test_abopt_lbfgs(linesearch, diag_update, rescale_diag):
    lbfgs = LBFGS(linesearch=linesearch,
            diag_update=diag_update,
            rescale_diag=rescale_diag,
            maxiter=1000)

    problem = RosenProblem()

    x0 = numpy.zeros(20)
    r = lbfgs.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(r.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_quad():
    lbfgs = LBFGS(linesearch=backtrace)

    J = numpy.array([ [0, 0,     2,  1],
                      [0,  10,   2,  0],
                      [40, 100,  0,  0],
                      [400, 0,   0,  0]])

    problem = ChiSquareProblem(J=J)

    x0 = numpy.zeros(4)
    r = lbfgs.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(problem.f(r.x), 0.0, atol=1e-7)

def test_abopt_lbfgs_quad_P():
    lbfgs = LBFGS(linesearch=backtrace)
    J = numpy.array([ [0, 0,     2,  1],
                      [0,  10,   2,  0],
                      [40, 100,  0,  0],
                      [400, 0,   0,  0]])
    precond = Preconditioner(
                        Pvp=lambda x: 20 * x,
                        vPp=lambda x: 20 * x,
                        vQp=lambda x: 0.05 * x,
                        Qvp=lambda x: 0.05 * x)
    problem = ChiSquareProblem(J=J, precond=precond)

    x0 = numpy.zeros(4)
    r = lbfgs.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(problem.f(r.x), 0.0, atol=1e-7)


