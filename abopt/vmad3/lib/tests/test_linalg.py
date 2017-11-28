from __future__ import print_function
from abopt.vmad3 import Builder
from abopt.vmad3 import Context
from pprint import pprint

from abopt.vmad3.lib import linalg
from numpy.testing import assert_array_equal, assert_allclose
import numpy

class BaseScalarTest:
    def setup(self):
        with Builder() as m:
            x = m.input('x')
            x = self.model(x)
            y = linalg.to_scalar(x)
            m.output(y=y)
        self.m = m

    def test_opr(self):
        ctx = Context(x=self.x)
        y1 = ctx.compute(self.m, vout='y', return_tape=False)
        # correctness
        assert_allclose(y1, self.y)

    def test_vjp(self):
        ctx = Context(x=self.x)
        y1, tape = ctx.compute(self.m, vout='y', return_tape=True)

        vjp = tape.get_vjp()
        ctx = Context(_y=self._y)
        _x1 = ctx.compute(vjp, vout='_x')

        assert_allclose(self._x, _x1)

    def test_jvp(self):
        ctx = Context(x=self.x)
        y1, tape = ctx.compute(self.m, vout='y', return_tape=True)

        jvp = tape.get_jvp()
        ctx = Context(x_=self.x_)
        y_1, t1 = ctx.compute(jvp, vout='y_', return_tape=True)

        assert_allclose(self.y_, y_1)

class Test_to_scalar(BaseScalarTest):
    x = numpy.arange(10)
    y = sum(x ** 2)
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 2 * x
    y_ = sum(2 * x)

    def model(self, x):
        return x

class Test_mul(BaseScalarTest):
    x = numpy.arange(10)
    y = sum(x ** 2)
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 2 * x
    y_ = sum(2 * x)

    def model(self, x):
        return linalg.mul(x, 1.0)

class Test_pow(BaseScalarTest):
    x = numpy.arange(10)
    y = sum(x ** 2)
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 2 * x
    y_ = sum(2 * x)

    def model(self, x):
        return linalg.pow(x, 1.0)

class Test_log(BaseScalarTest):
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
    x = numpy.arange(10)
    y = sum(x ** 2)
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 2 * x
    y_ = sum(2 * x)

    def model(self, x):
        return linalg.add(x, 0.0)

class Test_copy(BaseScalarTest):
    x = numpy.arange(10)
    y = sum(x ** 2)
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 2 * x
    y_ = sum(2 * x)

    def model(self, x):
        return linalg.copy(x)

class Test_stack(BaseScalarTest):
    x = numpy.arange(10)
    y = sum(x ** 2) * 2
    _y = 1.0
    x_ = numpy.ones(10)

    _x = 4 * x
    y_ = sum(4 * x)

    def model(self, x):
        return linalg.stack([x, x], axis=0)
