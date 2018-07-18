from __future__ import print_function

from abopt.abopt2 import GradientDescent, LBFGS, Preconditioner, minimize

from abopt.linesearch import minpack, backtrace, exact
from abopt.vectorspace import real_vector_space, complex_vector_space

from numpy.testing import assert_raises, assert_allclose

from scipy.optimize import rosen, rosen_der
import numpy

def quad(x):
    return (x[0] - .5)** 2 + (x[1] - .5) ** 2
def quad_der(x):
    return (x - 0.5) * 2

def crosen(x):
    x = numpy.atleast_1d(x)
    v = numpy.concatenate([x.real, x.imag], axis=0)
    return rosen(v)

def crosen_der(x):
    x = numpy.atleast_1d(x)
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
    gd = GradientDescent(linesearch=exact, maxiter=10)
    
    s = minimize(gd, quad, quad_der, x0, monitor=print)
    assert s.converged
    assert_allclose(s.x, 0.5, rtol=1e-4)

def test_abopt_gd_complex():
    gd = GradientDescent(linesearch=exact, maxiter=10)

    X = []
    Y = []
    def monitor_r(state):
        X.append(state.x[0] + state.x[1] * 1j)

    def monitor_c(state):
        Y.append(state.x[0])

    s = minimize(gd, rosen, rosen_der, numpy.array([0., 0.]), monitor=monitor_r, vs=real_vector_space)
    s = minimize(gd, crosen, crosen_der, numpy.array([0. + 0.j]), monitor=monitor_c, vs=complex_vector_space)

    assert_allclose(X, Y, rtol=1e-4)

def test_state():
    from abopt.abopt2 import State
    s = State()
    print()
    print(s.format(header=True))
    print(s.format())
    print(s.format(columns=['nit', 'na']))
