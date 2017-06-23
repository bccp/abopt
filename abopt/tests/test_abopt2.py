from __future__ import print_function

from abopt.abopt2 import GradientDescent, LBFGS, \
        VectorSpace, minimize, minimize_p

from abopt.linesearch import minpack, backtrace, exact
from abopt.lbfgs import inverse_bfgs_diag, direct_bfgs_diag, scalar_diag, inverse_dfp_diag
from abopt.lbfgs import scaled_direct_bfgs_diag, scaled_inverse_dfp_diag

from numpy.testing import assert_raises, assert_allclose

from scipy.optimize import rosen, rosen_der
import numpy

def quad(x):
    return (x[0] - .5)** 2 + (x[1] - .5) ** 2
def quad_der(x):
    return (x - 0.5) * 2

def test_abopt_gd_backtrace():
    vs = VectorSpace()
    gd = GradientDescent(vs, linesearch=backtrace)
    
    s = minimize(gd, quad, quad_der, numpy.array([0, 0]), monitor=print)
    print(s)
    assert s.converged
    assert_allclose(s.x, 0.5, rtol=1e-4)

def test_abopt_gd_minpack():
    vs = VectorSpace()
    gd = GradientDescent(vs, linesearch=minpack)
    
    s = minimize(gd, quad, quad_der, numpy.array([0, 0]))
    print(s)
    assert s.converged
    assert_allclose(s.x, 0.5, rtol=1e-4)

def test_abopt_gd_exact():
    vs = VectorSpace()
    gd = GradientDescent(vs, linesearch=exact)
    
    s = minimize(gd, quad, quad_der, numpy.array([0, 0]))
    print(s)
    assert s.converged
    assert_allclose(s.x, 0.5, rtol=1e-4)

def test_abopt_lbfgs_backtrace():
    vs = VectorSpace()
    lbfgs = LBFGS(vs, linesearch=backtrace)
    
    s = minimize(lbfgs, rosen, rosen_der, numpy.array([0, 0]))
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_minpack():
    vs = VectorSpace()
    lbfgs = LBFGS(vs, linesearch=minpack)
    
    s = minimize(lbfgs, rosen, rosen_der, numpy.array([0, 0]))
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_exact():
    vs = VectorSpace()
    lbfgs = LBFGS(vs, linesearch=exact)
    
    s = minimize(lbfgs, rosen, rosen_der, numpy.array([0, 0]))
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_quad():
    vs = VectorSpace()
    gd = LBFGS(vs, linesearch=backtrace)
    s = minimize(gd, quad, quad_der, numpy.array([0, 0]))
    print(s)
    assert s.converged
    assert_allclose(s.x, 0.5, rtol=1e-4)

def test_abopt_lbfgs_quad_P():
    vs = VectorSpace()
    gd = LBFGS(vs, linesearch=backtrace)
    s = minimize_p(gd, quad, quad_der, numpy.array([0, 0]), P=lambda x: 2 * x, PT=lambda x: 0.5 * x)
    print(s)
    assert s.converged
    assert_allclose(s.x, 0.5, rtol=1e-4)

def test_abopt_lbfgs_scaled_direct_bfgs_diag():
    vs = VectorSpace()
    lbfgs = LBFGS(vs, linesearch=exact, diag=scaled_direct_bfgs_diag)
    def monitor(state):
#        print(str(state), 'D', state.D)
        pass
    s = minimize(lbfgs, rosen, rosen_der, numpy.array([0, 0]), monitor=monitor)
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_direct_bfgs_diag():
    vs = VectorSpace()
    lbfgs = LBFGS(vs, linesearch=exact, diag=direct_bfgs_diag)
    def monitor(state):
#        print(str(state), 'D', state.D)
        pass
    s = minimize(lbfgs, rosen, rosen_der, numpy.array([0, 0]), monitor=monitor)
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_scaled_inverse_dfp_diag():
    vs = VectorSpace()
    lbfgs = LBFGS(vs, linesearch=exact, diag=scaled_inverse_dfp_diag)
    def monitor(state):
#        print(str(state), 'D', state.D)
        pass
    s = minimize(lbfgs, rosen, rosen_der, numpy.array([0, 0]), monitor=monitor)
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)

def test_abopt_lbfgs_inverse_dfp_diag():
    vs = VectorSpace()
    lbfgs = LBFGS(vs, linesearch=exact, diag=inverse_dfp_diag)
    def monitor(state):
#        print(str(state), 'D', state.D)
        pass
    s = minimize(lbfgs, rosen, rosen_der, numpy.array([0, 0]), monitor=monitor)
    print(s)
    assert s.converged
    assert_allclose(s.x, 1.0, rtol=1e-4)
