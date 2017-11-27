from pprint import pprint
from abopt.vmad3.operator import add, to_scalar, operator
from abopt.vmad3.model import ModelBuilder
from abopt.vmad3.error import ModelError
from abopt.vmad3.context import Context
import pytest
def test_model():
    with ModelBuilder() as m:
        a, b = m.input('a', 'b')

        t1, = add(x1=a, x2=a)
        t2, = add(x1=b, x2=0)
        c, = add(x1=t1, x2=t2)

        m.output(c=c)

    print("----- model -----")
    pprint(m)
    pprint(m[:])

    print("----- compute -----")
    ctx = Context(a=3, b=4)

    c = ctx.compute(m, vout='c')
    print(ctx, c)

    print("----- with tape -----")
    ctx = Context(a=3, b=4)

    c, tape = ctx.compute(m, vout='c', return_tape=True)
    print(ctx, c)
    print("----- tape -----")
    pprint(tape)

    vjp = tape.get_vjp()

    print("----- vjp -----")
    pprint(vjp)
    pprint(vjp[:])

    ctx = Context(_c=1.0)
    _a, _b = ctx.compute(vjp, vout=['_a', '_b'], monitor=print)
    print('_a, _b = ', _a, _b)

def test_model_partial():
    with ModelBuilder() as m:
        a, = m.input('a')
        t1, = add(x1=a, x2=a)
        m.output(c=t1)

    ctx = Context(a=3)

    c, tape = ctx.compute(m, vout='c', return_tape=True)
    assert c == 6

    vjp = tape.get_vjp()

    ctx = Context(_c=1.0)
    _a = ctx.compute(vjp, vout='_a', monitor=print)
    assert _a == 2.0

def test_model_unused():
    with ModelBuilder() as m:
        a, b = m.input('a', 'b')
        m.output(c=1.0)

    ctx = Context(a=3, b=4)
    c, tape = ctx.compute(m, vout='c', return_tape=True)
    assert c == 1.0

    vjp = tape.get_vjp()
    ctx = Context(_c=1.0)
    _a, _b = ctx.compute(vjp, vout=['_a', '_b'], monitor=print)
    assert _a == 0
    assert _b == 0

def test_model_partial_out():
    with ModelBuilder() as m:
        a, = m.input('a')
        t1, = add(x1=a, x2=a)
        m.output(c=t1)
        m.output(a=a)

    ctx = Context(a=3)

    (a, c), tape = ctx.compute(m, vout=['a', 'c'], return_tape=True)
    assert c == 6
    assert a == 3

    vjp = tape.get_vjp()

    ctx = Context(_c=1.0, _a=0.0)
    _a = ctx.compute(vjp, vout='_a', monitor=print)
    assert _a == 2.0

    ctx = Context(_c=0.0, _a=1.0)
    _a = ctx.compute(vjp, vout='_a', monitor=print)
    assert _a == 1.0

def test_model_errors():
    with ModelBuilder() as m:
        a, = m.input('a')
        with pytest.raises(ModelError):
            m.output(a=a)
            m.output(a=a)

        with pytest.raises(ModelError):
            add(x1=1, x2=1)

        with pytest.raises(ModelError):
            add(x2=1)

def test_tape_unused():
    # assert unused extra args are not recorded on the tape.
    with ModelBuilder() as m:
        a, = m.input('a')
        b, = add(x1=a, x2=a, j=3)
        m.output(b=b)

    ctx = Context(a = 1.0)
    b, tape = ctx.compute(m, vout='b', monitor=print, return_tape=True)
    assert b == 2.0
    assert isinstance(tape[0].node, add.opr)
    assert 'j' not in tape[0].impl_kwargs

    pprint(tape[:])

def test_model_extra_args():
    # assert used extra args are recored on the tape
    @operator
    class extra_args:
        ain = {'x' : '*'}
        aout = {'y' : '*'}

        def opr(self, x, p):
            return dict(y = x * p)

        def vjp(self, x, _x, _y, p):
            return dict(_x = _y * p)

        def jvp(self, x, x_, y_, p):
            return dict(y_ = x_ * p)

    with ModelBuilder() as m:
        a, = m.input('a')
        b, = extra_args(x=a, p=2.0)
        m.output(b=b)

    ctx = Context(a = 1.0)
    b, tape = ctx.compute(m, vout='b', monitor=print, return_tape=True)

    assert b == 2.0
    assert isinstance(tape[0].node, extra_args.opr)
    assert 'p' in tape[0].impl_kwargs

