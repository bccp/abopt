from __future__ import print_function
from pprint import pprint
from abopt.vmad3.lib import linalg
import numpy

from abopt.vmad3.testing import BaseScalarTest

class Test_to_scalar(BaseScalarTest):
    to_scalar = linalg.to_scalar

    x = numpy.arange(10)
    y = sum(x ** 2)
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 2 * x
    y_ = sum(2 * x)

    def model(self, x):
        return x

class Test_mul(BaseScalarTest):
    to_scalar = linalg.to_scalar
    x = numpy.arange(10)
    y = sum(x ** 2)
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 2 * x
    y_ = sum(2 * x)

    def model(self, x):
        return linalg.mul(x, 1.0)

class Test_pow(BaseScalarTest):
    to_scalar = linalg.to_scalar
    x = numpy.arange(10)
    y = sum(x ** 2)
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 2 * x
    y_ = sum(2 * x)

    def model(self, x):
        return linalg.pow(x, 1.0)

class Test_log(BaseScalarTest):
    to_scalar = linalg.to_scalar
    logx = numpy.arange(10)
    x = numpy.exp(logx)
    y = sum(logx ** 2)
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 2 * logx / x
    y_ = sum(2 * logx / x)

    def model(self, x):
        return linalg.log(x)

class Test_add(BaseScalarTest):
    to_scalar = linalg.to_scalar
    x = numpy.arange(10)
    y = sum(x ** 2)
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 2 * x
    y_ = sum(2 * x)

    def model(self, x):
        return linalg.add(x, 0.0)

class Test_copy(BaseScalarTest):
    to_scalar = linalg.to_scalar
    x = numpy.arange(10)
    y = sum(x ** 2)
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 2 * x
    y_ = sum(2 * x)

    def model(self, x):
        return linalg.copy(x)

class Test_stack(BaseScalarTest):
    to_scalar = linalg.to_scalar
    x = numpy.arange(10)
    y = sum(x ** 2) * 2
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 4 * x
    y_ = sum(4 * x)

    def model(self, x):
        return linalg.stack([x, x], axis=0)

class Test_take(BaseScalarTest):
    to_scalar = linalg.to_scalar

    x = numpy.arange(10)
    y = 2 ** 2
    _y = 1.0
    x_ = numpy.ones(10)

    _x = (x == 2) * 2 * 2.0
    y_ = 2 * x[2]

    def model(self, x):
        return linalg.take(x, 2, axis=0)

