"""
    Functions of the test suite

    https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""

import autograd
from autograd import numpy
from autograd.numpy import cos, sin, exp, log, pi
from autograd import grad

class FunctionND:
    xatol = 2e-6
    yatol = 1e-7
    def __init__(self, nd):
        self.nd = nd

    def gradient(self, x):
        x = numpy.array(x, dtype='f8')
        return grad(self.function)(x)

    @property
    def start(self):
        return numpy.zeros(self.nd, dtype='f8')

    @property
    def start(self):
        return numpy.zeros(self.nd, dtype='f8')

    @property
    def fmin(self):
        return 0.0

class Function2D:
    xatol = 2e-6
    yatol = 1e-7
    def __init__(self, nd):
        assert nd == 2
        self.nd = 2

    def gradient(self, x):
        x = numpy.array(x, dtype='f8')
        return grad(self.function)(x)

    @property
    def start(self):
        return numpy.zeros(self.nd, dtype='f8')

    @property
    def start(self):
        return numpy.zeros(self.nd, dtype='f8')

    @property
    def fmin(self):
        return 0.0

class Ackley(Function2D):
    def function(self, x):
        x, y = x
        A = -20 * exp(-0.2 * (0.5 * (x**2 + y**2)) ** 0.5)
        B = - exp(0.5 *(cos(2 * pi * x) + cos(2 * pi * y)))
        C = exp(1) + 20

        return A + B + C

    @property
    def xmin(self):
        return numpy.zeros(self.nd)

    @property
    def start(self):
        return numpy.ones(self.nd, dtype='f8')

    @property
    def fmin(self):
        return 0.0

    @property
    def domain(self):
        return (-5. * numpy.ones(self.nd),
                 5. * numpy.ones(self.nd))

    def gradient(self, x):
        # This function has a nan gradient at 0, 0., not we want..
        x = numpy.array(x, dtype='f8')
        if (x == 0).all():
            return x * 0.0
        return grad(self.function)(x)

class Bukin6(Function2D):
    """ Very hard problem for quasi-newton:
        https://www.sfu.ca/~ssurjano/bukin6.html

    """

    xatol = 8e-2
    yatol = 5e-3

    def function(self, x):
        x, y = x
        return 100 * abs(y - 0.01 * x **2) ** 0.5 + 0.01 * abs(x + 10)

    @property
    def xmin(self):
        return numpy.array([-10, 1.])

    @property
    def start(self):
        return self.xmin + [0.1, 0.1]

    @property
    def fmin(self):
        return 0.0

    @property
    def domain(self):
        return numpy.array([-15. -5]), numpy.array([-3, 3])

    def gradient(self, x):
        # This function has a nan gradient at 0, 0., not we want..
        x = numpy.array(x, dtype='f8')
        if (x == self.xmin).all():
            return x * 0.0
        return grad(self.function)(x)

class Rastrigin(FunctionND):
    def function(self, x):
        x = numpy.array(x)

        A = 20
        r = A * self.nd
        r = r + numpy.sum(x ** 2 - A * cos(2 * pi * x))
        return r

    @property
    def xmin(self):
        return numpy.zeros(self.nd)

    @property
    def start(self):
        return numpy.ones(self.nd, dtype='f8')

    @property
    def fmin(self):
        return 0.0


    @property
    def domain(self):
        return (-5.12 * numpy.ones(self.nd),
                 5.12 * numpy.ones(self.nd))


class Sphere(FunctionND):
    def function(self, x):
        x = numpy.array(x)
        r = numpy.sum(x ** 2)
        return r

    @property
    def xmin(self):
        return numpy.zeros(self.nd)

    @property
    def start(self):
        return numpy.ones(self.nd, dtype='f8')

    @property
    def fmin(self):
        return 0.0

    @property
    def domain(self):
        return (-5. * numpy.ones(self.nd),
                 5. * numpy.ones(self.nd))

class StyblinskiTang(FunctionND):
    def function(self, x):
        x = numpy.array(x)
        A = numpy.sum(x ** 4)
        B = -16 * numpy.sum(x ** 2)
        C = 5 * numpy.sum(x)
        return (A + B + C) * 0.5

    @property
    def xmin(self):
        return numpy.ones(self.nd) * -2.903534

    @property
    def start(self):
        return self.xmin + 0.1 * numpy.ones(self.nd, dtype='f8')

    @property
    def fmin(self):
        return -39.1661655 * self.nd

    @property
    def domain(self):
        return (-5. * numpy.ones(self.nd),
                 5. * numpy.ones(self.nd))

class Rosenbrock(FunctionND):
    def function(self, x):
        x = numpy.array(x)
        r = 100 * numpy.sum((x[1:] - x[:-1] ** 2)**2) + numpy.sum((1 - x)**2)
        return r

    @property
    def xmin(self):
        return numpy.ones(self.nd)

    @property
    def start(self):
        return numpy.zeros(self.nd, dtype='f8')

    @property
    def fmin(self):
        return 0.0

    @property
    def domain(self):
        return (-5. * numpy.ones(self.nd),
                 5. * numpy.ones(self.nd))

class Beale(Function2D):
    xatol = 1e-5
    yatol = 5e-4 # very slow bottom!
    def function(self, x):
        x, y = x
        A = (1.5 - x + x * y) ** 2
        B = (2.25 -x + x * y**2)**2
        C = (2.625 - x + x * y ** 3) ** 2
        return A + B + C

    @property
    def xmin(self):
        return numpy.array([3, 0.5])

    @property
    def start(self):
        return numpy.ones(self.nd, dtype='f8')

    @property
    def fmin(self):
        return 0.0

    @property
    def domain(self):
        return (-4.5 * numpy.ones(self.nd),
                 4.5 * numpy.ones(self.nd))

class GoldsteinPrice(Function2D):
    def function(self, x):
        x, y = x

        A = 1 + (x + y + 1)**2 * (19 - 14 * x + 3 * x**2 - 114 * y + 6 * x*y  + 3 * y **2)
        B = 30 + (2  * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y**2)
        return A * B

    @property
    def xmin(self):
        return numpy.array([0., -1.])

    @property
    def start(self):
        # if too far will run beyond the constraints.
        return self.xmin + [0.1, 0.1]

    @property
    def fmin(self):
        return 3.0

    @property
    def domain(self):
        return (-10 * numpy.ones(self.nd),
                 10 * numpy.ones(self.nd))

class Himmelblau(Function2D):
    def function(self, x):
        x, y = x

        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    @property
    def xmin(self):
        return numpy.array([3.0, 2.0])

    @property
    def start(self):
        # if too far will run beyond the constraints.
        return numpy.ones(self.nd)

    @property
    def fmin(self):
        return 0.0

    @property
    def domain(self):
        return numpy.array([
                [-5, -5],
                [5, 5]])

    

class McCormick(Function2D):
    xatol = 1e-4
    yatol = 1e-5
    def function(self, x):
        x, y = x

        return sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1

    @property
    def xmin(self):
        return numpy.array([-0.54719, -1.54719])

    @property
    def start(self):
        # if too far will run beyond the constraints.
        return numpy.ones(self.nd)

    @property
    def fmin(self):
        return -1.913223

    @property
    def domain(self):
        return numpy.array([
                [-1.5, -3],
                [4, 4]])

def get_all_nd():
    r = []
    d = globals()
    for x in d:
        c = d[x]
        if not isinstance(c, type): continue
        if c is Function2D: continue
        if c is FunctionND: continue
        if(issubclass(c, FunctionND)):
            r.append(c)
    return r

def get_all_2d():
    r = []
    d = globals()
    for x in d:
        c = d[x]
        if not isinstance(c, type): continue
        if c is Function2D: continue
        if c is FunctionND: continue
        if issubclass(c, (Function2D, FunctionND)):
            r.append(c)
    return r
