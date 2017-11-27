from pprint import pprint
from abopt.vmad3.operator import add, to_scalar, operator
from abopt.vmad3.nested import nestedoperator
from abopt.vmad3.model import ModelBuilder
from abopt.vmad3.error import ModelError
from abopt.vmad3.context import Context
import pytest

def test_model():
    with ModelBuilder() as m:
        a, b = m.input('a', 'b')

        t1 = add(x1=a, x2=a)
        t2 = add(x1=b, x2=0)
        c = add(x1=t1, x2=t2)

        m.output(c=c)

    print("----- model -----")
    pprint(m)
    pprint(m[:])

    print("----- compute -----")
    ctx = Context(a=3, b=4)

    c = ctx.compute(m, vout='c')
    print(ctx, c)

    print("----- tape -----")
    ctx = Context(a=3, b=4)
    c, tape = ctx.compute(m, vout='c', return_tape=True)
    print(ctx, c)
    pprint(tape)

    print("----- vjp -----")
    vjp = tape.get_vjp()
    pprint(vjp)
    pprint(vjp[:])

    ctx = Context(_c=1.0)
    _a, _b = ctx.compute(vjp, vout=['_a', '_b'], monitor=print)
    print('_a, _b = ', _a, _b)

    print("----- jvp -----")
    jvp = tape.get_jvp()
    pprint(jvp)
    pprint(jvp[:])

    ctx = Context(a_=1.0, b_=1.0)
    c_, = ctx.compute(jvp, vout=['c_'], monitor=print)
    print('c_ = ', c_)

def test_model_partial():
    with ModelBuilder() as m:
        a = m.input('a')
        t1 = add(x1=a, x2=a)
        m.output(c=t1)

    ctx = Context(a=3)

    c, tape = ctx.compute(m, vout='c', return_tape=True)
    assert c == 6

    vjp = tape.get_vjp()
    ctx = Context(_c=1.0)
    _a = ctx.compute(vjp, vout='_a', monitor=print)
    assert _a == 2.0

    jvp = tape.get_jvp()
    ctx = Context(a_=1.0)
    c_ = ctx.compute(jvp, vout='c_', monitor=print)
    assert c_ == 2.0

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

    jvp = tape.get_jvp()
    ctx = Context(a_=1.0, b_=1.0)
    c_ = ctx.compute(jvp, vout='c_', monitor=print)
    assert c_ == 0

def test_model_partial_out():
    with ModelBuilder() as m:
        a = m.input('a')
        t1 = add(x1=a, x2=a)
        m.output(c=t1)
        m.output(a=a)

    ctx = Context(a=3)

    (a, c), tape = ctx.compute(m, vout=['a', 'c'], return_tape=True)
    assert c == 6
    assert a == 3

    vjp = tape.get_vjp()

    # test two outputs individually
    ctx = Context(_c=1.0, _a=0.0)
    _a = ctx.compute(vjp, vout='_a', monitor=print)
    assert _a == 2.0

    ctx = Context(_c=0.0, _a=1.0)
    _a = ctx.compute(vjp, vout='_a', monitor=print)
    assert _a == 1.0

    jvp = tape.get_jvp()
    ctx = Context(a_=1.0)
    a_, c_ = ctx.compute(jvp, vout=['a_', 'c_'], monitor=print)
    assert c_ == 2.0
    assert a_ == 1.0

def test_model_errors():
    with ModelBuilder() as m:
        a = m.input('a')
        with pytest.raises(ModelError):
            m.output(a=a)
            m.output(a=a)

        with pytest.raises(ModelError):
            add(x1=1, x2=1)

        with pytest.raises(ModelError):
            add(x2=1)

        with pytest.raises(ModelError):
            add(x1=a, x2=a, y=a)

def test_tape_unused():
    # assert unused extra args are not recorded on the tape.
    with ModelBuilder() as m:
        a = m.input('a')
        b = add(x1=a, x2=a, j=3)
        m.output(b=b)

    ctx = Context(a = 1.0)
    b, tape = ctx.compute(m, vout='b', monitor=print, return_tape=True)
    assert b == 2.0
    assert isinstance(tape[0].node, add.opr)
    assert 'j' not in tape[0].resolved
    assert 'j' in tape[0].node.kwargs

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
        a = m.input('a')
        b = extra_args(x=a, p=2.0)
        m.output(b=b)

    ctx = Context(a = 1.0)
    b, tape = ctx.compute(m, vout='b', monitor=print, return_tape=True)

    assert b == 2.0
    assert isinstance(tape[0].node, extra_args.opr)
    assert 'p' not in tape[0].resolved
    assert 'p' in tape[0].node.kwargs

def test_model_nasty():
    # assert used extra args are recored on the tape
    n = 2
    with ModelBuilder() as m:
        x = m.input('x')
        for i in range(2):
            x = add(x1=x, x2=x)

        m.output(y=x)

    ctx = Context(x=1.0)
    y, tape = ctx.compute(m, vout='y', return_tape=True)
    assert y == 4.0

    vjp = tape.get_vjp()
    ctx = Context(_y = 1.0)
    _x = ctx.compute(vjp, vout='_x', monitor=print)
    assert _x == 4.0

def test_model_nested():
    @nestedoperator
    class nested:
        ain = {'x' : '*'}
        aout = {'y' : '*'}

        def model(self, x, n):
            with ModelBuilder() as m:
                x = m.input('x')
                for i in range(n):
                    x = add(x1=x, x2=x)

                m.output(y=x)

            return m

    with ModelBuilder() as m:
        a = m.input('a')
        b = nested(x=a, n=2)
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

