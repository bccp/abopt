from __future__ import print_function
from abopt import GradientDescent, LBFGS
from numpy.testing import assert_raises, assert_allclose

def test_nothing():
    pass

def test_gradient_descent_customized():
    optimizer = GradientDescent(
            addmul=lambda a, b, s: a + b * s,
            dot=lambda dx1, dx2 : dx1 * dx2,
            )

    def df(x):
        y = 4 * x**3 - 9 * x**2
        return y
    def f(x):
        return x ** 4 - 3 * x ** 3 + 2

    called = [False]
    def notification(state):
        called[0] = True

    optimizer.maxsteps = 1000
    optimizer.tol = 1e-6
    optimizer.gtol = 1e-6
    optimizer.gamma = 0.001

    result = optimizer.minimize(objective=f, gradient=df, x0=6., monitor=notification)
    assert called[0]
    assert_allclose(result['x'], 2.25, rtol=0.05)
    assert_allclose(result['g'], 0.0, atol=0.05)

def test_gradient_descent():
    optimizer = GradientDescent()

    def df(x):
        y = 4 * x**3 - 9 * x**2
        return y
    def f(x):
        return x ** 4 - 3 * x ** 3 + 2

    called = [False]
    def notification(state):
        called[0] = True

    optimizer.maxsteps = 1000
    optimizer.tol = 1e-6
    optimizer.gtol = 1e-6
    optimizer.gamma = 0.001

    result = optimizer.minimize(objective=f, gradient=df, x0=6., monitor=notification)
    assert called[0]
    assert_allclose(result['x'], 2.25, rtol=0.05)
    assert_allclose(result['g'], 0.0, atol=0.05)

def test_lbfgs():
    optimizer = LBFGS(
            addmul=lambda a, b, s: a + b * s,
            dot=lambda dx1, dx2 : dx1 * dx2,
            )

    def df(x):
        y = 4 * x**3 - 9 * x**2
        return y
    def f(x):
        return x ** 4 - 3 * x ** 3 + 2

    called = [False]
    def notification(state):
        print("***NOTIFICATION***")
        print(state)
        called[0] = True

    optimizer.maxsteps=1000
    optimizer.tol=1e-5
    optimizer.gtol=1e-10
    optimizer.m = 10

    result = optimizer.minimize(objective=f, gradient=df, x0=6., monitor=notification)
    assert called[0]
    assert_allclose(result['x'], 2.25, rtol=0.05)
    assert_allclose(result['g'], 0.0, atol=0.05)


def test_lbfgs_default():
    optimizer = LBFGS()

    def df(x):
        y = 4 * x**3 - 9 * x**2
        return y
    def f(x):
        return x ** 4 - 3 * x ** 3 + 2

    called = [False]
    def notification(state):
        #print("***NOTIFICATION***")
        #print(state)
        called[0] = True

    optimizer.maxsteps=1000
    optimizer.tol=1e-5
    optimizer.gtol=1e-10
    optimizer.m = 10

    result = optimizer.minimize(objective=f, gradient=df, x0=6., monitor=notification)
    assert called[0]
    assert_allclose(result['x'], 2.25, rtol=0.05)
    assert_allclose(result['g'], 0.0, atol=0.05)

