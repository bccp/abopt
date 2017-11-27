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
