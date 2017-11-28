from __future__ import print_function
from abopt.vmad3 import Builder
from abopt.vmad3 import Context
from pprint import pprint

from abopt.vmad3.lib.linalg import *
from numpy.testing import assert_array_equal
import numpy

def test_linalg_mul():
    with Builder() as m:
        x = m.input('x')
        #c = copy(x=x)
        y = to_scalar(x=mul(x1=x, x2=x))
        m.output(y=y)

    ctx = Context(x=numpy.arange(10))
    y, tape = ctx.compute(m, vout='y', return_tape=True)

    assert_array_equal(y, sum(numpy.arange(10) ** 4))


