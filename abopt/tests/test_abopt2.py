from __future__ import print_function

from abopt.abopt2 import GradientDescent, LBFGS, \
        minimize, minimize_p

from abopt.linesearch import minpack, backtrace, exact
from abopt.lbfgs import inverse_bfgs, direct_bfgs, scalar, inverse_dfp
from abopt.lbfgs import pre_scaled_direct_bfgs, pre_scaled_inverse_dfp
from abopt.lbfgs import post_scaled_direct_bfgs, post_scaled_inverse_dfp
from abopt.vectorspace import real_vector_space, complex_vector_space

from numpy.testing import assert_raises, assert_allclose

from scipy.optimize import rosen, rosen_der
import numpy

def quad(x):
    return (x[0] - .5)** 2 + (x[1] - .5) ** 2
def quad_der(x):
    return (x - 0.5) * 2

def crosen(x):
    v = numpy.concatenate([x.real, x.imag], axis=0)
    return rosen(v)

def crosen_der(x):
    s = x.shape[0]
    v = numpy.concatenate([x.real, x.imag], axis=0)
    v = rosen_der(v)
    r = v[:s] + v[s:] * 1j
    return r

x0 = numpy.zeros(2)


def test_abopt_gd_backtrace():
    gd = GradientDescent(linesearch=backtrace)

    s = minimize(gd, quad, quad_der, x0, monitor=print)
    print(s)
    assert s.converged
    assert_allclose(s.x, 0.5, rtol=1e-4)

def test_abopt_gd_minpack():
    gd = GradientDescent(linesearch=minpack)
    
    s = minimize(gd, quad, quad_der, x0)
    print(s)
    assert s.converged
    assert_allclose(s.x, 0.5, rtol=1e-4)

def test_abopt_gd_exact():
    gd = GradientDescent(linesearch=exact)
    
    s = minimize(gd, quad, quad_der, x0)
    print(s)
    assert s.converged
    assert_allclose(s.x, 0.5, rtol=1e-4)

def test_abopt_lbfgs_backtrace():
    lbfgs = LBFGS(linesearch=backtrace)
    
    s = minimize(lbfgs, rosen, rosen_der, x0)
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_minpack():
    lbfgs = LBFGS(linesearch=minpack)
    
    s = minimize(lbfgs, rosen, rosen_der, x0)
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_exact():
    lbfgs = LBFGS(linesearch=exact)
    
    s = minimize(lbfgs, rosen, rosen_der, x0)
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_quad():
    gd = LBFGS(linesearch=backtrace)
    s = minimize(gd, quad, quad_der, x0)
    print(s)
    assert s.converged
    assert_allclose(s.x, 0.5, rtol=1e-4)

def test_abopt_lbfgs_quad_P():
    gd = LBFGS(linesearch=backtrace)
    s = minimize_p(gd, quad, quad_der, x0, P=lambda x: 2 * x, PT=lambda x: 0.5 * x)
    print(s)
    assert s.converged
    assert_allclose(s.x, 0.5, rtol=1e-4)

def test_abopt_lbfgs_scaled_direct_bfgs():
    lbfgs = LBFGS(linesearch=exact, diag_update=pre_scaled_direct_bfgs)
    def monitor(state):
#        print(str(state), 'D', state.D)
        pass
    s = minimize(lbfgs, rosen, rosen_der, x0, monitor=monitor)
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_direct_bfgs():
    lbfgs = LBFGS(linesearch=exact, diag_update=direct_bfgs)
    def monitor(state):
#        print(str(state), 'D', state.D)
        pass
    s = minimize(lbfgs, rosen, rosen_der, x0, monitor=monitor)
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_scaled_inverse_dfp():
    lbfgs = LBFGS(linesearch=exact, diag_update=pre_scaled_inverse_dfp)
    def monitor(state):
#        print(str(state), 'D', state.D)
        pass
    s = minimize(lbfgs, rosen, rosen_der, x0, monitor=monitor)
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_inverse_dfp():
    lbfgs = LBFGS(linesearch=exact, diag_update=inverse_dfp)
    def monitor(state):
#        print(str(state), 'D', state.D)
        pass
    s = minimize(lbfgs, rosen, rosen_der, x0, monitor=monitor)
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_complex():
    for diag in [
            direct_bfgs, pre_scaled_direct_bfgs, post_scaled_direct_bfgs,
            inverse_dfp, pre_scaled_inverse_dfp, post_scaled_inverse_dfp,
            inverse_bfgs, scalar]:

        lbfgs_r = LBFGS(linesearch=exact, diag_update=diag)

        lbfgs_c = LBFGS(complex_vector_space, linesearch=exact, diag_update=diag)

        X = []
        Y = []
        def monitor_r(state):
            X.append(state.x[0] + state.x[1] * 1j)

        def monitor_c(state):
            Y.append(state.x[0])

        s = minimize(lbfgs_r, rosen, rosen_der, numpy.array([0., 0.]), monitor=monitor_r)
        s = minimize(lbfgs_c, crosen, crosen_der, numpy.array([0. + 0.j]), monitor=monitor_c)

        assert_allclose(X, Y, rtol=1e-4)

