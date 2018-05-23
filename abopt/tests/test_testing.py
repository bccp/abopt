from __future__ import print_function
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from abopt.abopt2 import Problem, LBFGS

import pytest

try:
    import autograd
    from abopt.testing.autograd_problems import get_all_nd, get_all_2d
except ImportError:
    autograd = None
    def get_all_nd(): return []
    def get_all_2d(): return []

pytestmark = pytest.mark.skipif(autograd is None, reason="skipping test functions due to lack of autograd")

@pytest.mark.parametrize("case", get_all_2d())
def test_cases_fmin(case):
    case = case(nd=2)
    assert_allclose(case.function(case.xmin), case.fmin)

@pytest.mark.parametrize("case", get_all_2d())
def test_cases_grad_min(case):
    case = case(nd=2)
    grad = case.gradient(case.xmin)

    assert_allclose(grad, 0, atol=1e-4)

@pytest.mark.parametrize("case", get_all_2d())
def test_cases_grad_start(case):
    case = case(nd=2)
    grad = case.gradient(case.start)

    eps = 1e-5
    grad2 = []
    for i in range(len(case.start)):
        x = 1.0 * case.start
        x[i] += eps
        y2 = case.function(x)
        x[i] -= 2 * eps
        y1 = case.function(x)
        grad2.append((y2 - y1) / (2 * eps))

    assert_allclose(grad, grad2)

@pytest.mark.parametrize("case", get_all_2d())
def test_cases_domain(case):
    case = case(nd=2)
    grad = case.gradient(case.xmin)

    left, right = case.domain

    assert (case.xmin <= right).all()
    assert (case.xmin >= left).all()

    assert (case.xmin != case.start).all()

    assert (case.start <= right).all()
    assert (case.start >= left).all()

@pytest.mark.parametrize("case", get_all_nd())
def test_cases_lbfgs_nd(case):
    case = case(nd=1000)
    opt = LBFGS()

    x0 = case.start
    p = Problem(objective=case.function, gradient=case.gradient)

    r = opt.minimize(p, x0, monitor=print)

    assert r['nit'] < 20 # Probably hitting slow convergence. Bug in optimizer?
    assert_allclose(r['y'], case.fmin, atol=case.yatol)
    assert_allclose(r['x'], case.xmin, atol=case.xatol)

@pytest.mark.parametrize("case", get_all_2d())
def test_cases_lbfgs_2d(case):
    case = case(nd=2)
    opt = LBFGS()

    x0 = case.start
    p = Problem(objective=case.function, gradient=case.gradient)

    r = opt.minimize(p, x0, monitor=print)

    assert r['nit'] < 20 # Probably hitting slow convergence. Bug in optimizer?
    assert_allclose(r['y'], case.fmin, atol=case.yatol)
    assert_allclose(r['x'], case.xmin, atol=case.xatol)
