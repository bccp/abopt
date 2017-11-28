from __future__ import print_function
from abopt.vmad3 import Builder
from abopt.vmad3 import Context
from pprint import pprint

from abopt.vmad3.lib.linalg import *
from numpy.testing import assert_array_equal
import numpy

def test_linalg_to_scalar():
    with Builder() as m:
        x = m.input('x')
        c = copy(x)
        y = to_scalar(x)
        m.output(y=y)

    x = numpy.arange(10)
    y = sum(x ** 2)
    _y = 1.0
    _x = 2 * x
    x_ = numpy.ones(10)
    y_ = 2 * x

    ctx = Context(x=x)
    y1, tape = ctx.compute(m, vout='y', return_tape=True)

    # correctness
    assert_array_equal(y1, y)

    vjp = tape.get_vjp()
    ctx = Context(_y=_y)
    _x1 = ctx.compute(vjp, vout='_x')

    assert_array_equal(_x, _x1)

    jvp = tape.get_jvp()
    ctx = Context(x_=x_)
    y_1 = ctx.compute(jvp, vout='y_')

    assert_array_equal(y_, y_1)
