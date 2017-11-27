from pprint import pprint
from abopt.vmad3.operator import add, to_scalar, operator
from abopt.vmad3.model import Builder
from abopt.vmad3.context import Context
import pytest

def test_operator_zero():
    @operator
    class op:
        ain = {'x' : '*'}
        aout = {'y' : '*'}

        def opr(self, x):
            return dict(y=x)

        def vjp(self, _y):
            raise AssertionError("shall not reach here")

        def jvp(self, x_):
            raise AssertionError("shall not reach here")

    with Builder() as m:
        a = m.input('a')
        t1 = op(x=a)
        m.output(c=t1)

    ctx = Context(a=3)

    c, tape = ctx.compute(m, vout='c', return_tape=True)
    assert c == 3

    vjp = tape.get_vjp()
    ctx = Context(_c=0)
    _a = ctx.compute(vjp, vout='_a', monitor=print)
    assert _a == 0

    jvp = tape.get_jvp()
    ctx = Context(a_=0)
    c_ = ctx.compute(jvp, vout='c_', monitor=print)
    assert c_ == 0

def test_operator_list():
    import numpy
    from numpy.testing import assert_array_equal

    @operator
    class op:
        ain = {('args', '*')}
        # for python 2.x need to use this syntax
        # to preserve orders
        aout = [('y', '*'),]

        def opr(self, args, axis):
            return dict(y=numpy.stack(args, axis=axis))

        def vjp(self, _y, args, axis):
            return dict(_args=[numpy.take(_y, i, axis=axis) for i in range(numpy.shape(_y)[axis])])

        def jvp(self, args_, args, axis):
            return dict(y_=numpy.stack(args_, axis))

    with Builder() as m:
        a = m.input('a')
        t = op(args=[a, a, a], axis=1)
        m.output(c=t)

    ctx = Context(a=[1, 2])

    c, tape = ctx.compute(m, vout='c', return_tape=True)
    assert_array_equal(c, [[1, 1, 1], [2, 2, 2]])

    vjp = tape.get_vjp()
    ctx = Context(_c=[[1, 1, 1], [1, 1, 1]])
    _a = ctx.compute(vjp, vout='_a', monitor=print)

    assert_array_equal(_a, [3, 3])

    jvp = tape.get_jvp()
    ctx = Context(a_=[1, 1])
    c_ = ctx.compute(jvp, vout='c_', monitor=print)

    assert_array_equal(c_, [[1, 1, 1], [1, 1, 1]])

def test_operator_multi_out():
    @operator
    class op:
        ain = {'x' : '*'}
        # for python 2.x need to use this syntax
        # to preserve orders
        aout = [('y1', '*'),
                ('y2', '*')]

        def opr(self, x):
            return dict(y1=x, y2=2 * x)
        def vjp(self, _y1, _y2):
            return dict(_x = _y1 + 2 * _y2)
        def jvp(self, x_):
            return dict(y1_=x_, y2_=2 * x_)

    with Builder() as m:
        a = m.input('a')
        t1, t2 = op(x=a)
        m.output(c=t1, d=t2)

    ctx = Context(a=3)

    (c, d), tape = ctx.compute(m, vout=('c', 'd'), return_tape=True)
    assert c == 3
    assert d == 6

    vjp = tape.get_vjp()
    ctx = Context(_c=1, _d=1)
    _a = ctx.compute(vjp, vout='_a', monitor=print)
    assert _a == 3

    jvp = tape.get_jvp()
    ctx = Context(a_=1)
    c_, d_ = ctx.compute(jvp, vout=('c_', 'd_'), monitor=print)
    assert c_ == 1
    assert d_ == 2
