from __future__ import print_function

import pytest

from abopt.base import Preconditioner

from abopt.linesearch import minpack, backtrace, exact

from abopt.algs.lbfgs import LBFGS
from abopt.algs.lbfgs import inverse_bfgs, direct_bfgs, scalar, inverse_dfp
from abopt.algs.lbfgs import pre_scaled_direct_bfgs, pre_scaled_inverse_dfp
from abopt.algs.lbfgs import post_scaled_direct_bfgs, post_scaled_inverse_dfp

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


def diag_scaling(v, direction):
    if direction == -1:
        return 0.05 * v
    else:
        return 20 * v

precond = Preconditioner(Pvp=diag_scaling, vPp=diag_scaling)
@pytest.mark.parametrize("precond",
[None, precond])
def test_abopt_lbfgs_quad(precond):
    lbfgs = LBFGS(linesearch=backtrace)

    J = numpy.array([ [0, 0,     2,  1],
                      [0,  10,   2,  0],
                      [40, 100,  0,  0],
                      [400, 0,   0,  0]])

    problem = ChiSquareProblem(J=J, precond=precond)

    x0 = numpy.zeros(4)
    r = lbfgs.minimize(problem, x0, monitor=print)
    assert r.converged
    assert_allclose(problem.f(r.x), 0.0, atol=1e-7)
