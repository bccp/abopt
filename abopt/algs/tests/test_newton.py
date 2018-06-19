from __future__ import print_function
import numpy

from abopt.algs.newton import DirectNewton

from numpy.testing import assert_allclose
from abopt.testing import RosenProblem

import pytest

def diag_scaling(v, direction):
    if direction == -1:
        return 0.05 * v
    else:
        return 20 * v

@pytest.mark.parametrize("precond", [ True, False ])
def test_newton(precond):
    nt = DirectNewton()

    problem = RosenProblem(precond=precond)

    problem.atol = 1e-7 # ymin = 0

    x0 = numpy.zeros(20)
    r = nt.minimize(problem, x0, monitor=print)
    assert_allclose(r.x, 1.0, rtol=1e-4)
    assert r.converged
