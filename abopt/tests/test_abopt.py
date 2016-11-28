from __future__ import print_function
from abopt import GradientDescent

def test_nothing():
    pass

def test_gradient_descent():
    optimizer = GradientDescent(
            addmul=lambda a, b, s: a + b * s,
            create=lambda : 0,
            dot=lambda dx1, dx2 : dx1 * dx2,
            )

    def df(x):
        y = 4 * x**3 - 9 * x**2
        return y
    def f(x):
        return x ** 4 - 3 * x ** 3 + 2

    optimizer.configure(maxstep=1000, tol=1e-2, gtol=1e-2, stepsize=0.01)
    optimizer.configure(notification=print)
    result = optimizer.minimize(objective=f, gradient=df, x0=6.)
    print(result['x'], result['y'], result['gradient'])
