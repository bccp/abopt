from __future__ import print_function

from abopt.vmad3.context import Context
from abopt.vmad3.model import Builder

def test_model_nested():


    from abopt.vmad3.nested import example

    with Builder() as m:
        a = m.input('a')
        b = example(a, 2)
        m.output(b=b)

    ctx = Context(a = 1.0)
    b, tape = ctx.compute(m, vout='b', monitor=print, return_tape=True)

    assert b == 4.0

    vjp = tape.get_vjp()
    ctx = Context(_b = 1.0)
    _a = ctx.compute(vjp, vout='_a', monitor=print)
    assert _a == 4.0

    jvp = tape.get_jvp()
    ctx = Context(a_ = 1.0)
    b_ = ctx.compute(jvp, vout='b_', monitor=print)
    assert b_ == 4.0

def test_model_nested_build():

    from abopt.vmad3.nested import example

    m = example.build(n=2)
    ctx = Context(x = 1.0)
    y, tape = ctx.compute(m, vout='y', monitor=print, return_tape=True)
    assert y == 4.0
