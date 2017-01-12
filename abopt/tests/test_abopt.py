from __future__ import print_function
from abopt import GradientDescent, LBFGS
from numpy.testing import assert_raises

def test_nothing():
    pass

def test_gradient_descent():
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

    optimizer.configure(maxstep=1000, tol=1e-2, gtol=1e-2, stepsize=0.01)
    optimizer.configure(monitor=notification)
    result = optimizer.minimize(objective=f, gradient=df, x0=6.)
    assert called[0]
    assert abs(result['x'] - 2.25) < 0.05
    assert abs(result['gradient']) < 0.05

def test_gradient_descent_default():
    optimizer = GradientDescent()

    def df(x):
        y = 4 * x**3 - 9 * x**2
        return y
    def f(x):
        return x ** 4 - 3 * x ** 3 + 2

    called = [False]
    def notification(state):
        called[0] = True

    optimizer.configure(maxstep=1000, tol=1e-2, gtol=1e-2, stepsize=0.01)
    optimizer.configure(monitor=notification)
    result = optimizer.minimize(objective=f, gradient=df, x0=6.)
    assert called[0]
    assert abs(result['x'] - 2.25) < 0.05
    assert abs(result['gradient']) < 0.05

def test_lbfgs():
    optimizer = LBFGS()
    optimizer = LBFGS(
            addmul=lambda a, b, s: a + b * s,
            dot=lambda dx1, dx2 : dx1 * dx2,
            mul = lambda a, s: a * s
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

    optimizer.configure(maxstep=1000, tol=1e-5, gtol=1e-10, m = 10)
    optimizer.configure(monitor=notification)
    result = optimizer.minimize(objective=f, gradient=df, x0=6.)
    assert called[0]
    assert abs(result['x'] - 2.25) < 0.05
    assert abs(result['gradient']) < 0.05


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

    optimizer.configure(maxstep=1000, tol=1e-5, gtol=1e-10, m = 10)
    optimizer.configure(monitor=notification)
    result = optimizer.minimize(objective=f, gradient=df, x0=6.)
    assert called[0]
    assert abs(result['x'] - 2.25) < 0.05
    assert abs(result['gradient']) < 0.05

