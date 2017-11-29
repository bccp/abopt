from __future__ import print_function
from pprint import pprint
from abopt.vmad3.core.operator import add, operator
from abopt.vmad3.core.model import Builder
import pytest

def test_model_partial():
    with Builder() as m:
        a = m.input('a')
        t1 = add(x1=a, x2=a)
        m.output(c=t1)

    init = dict(a=3)

    c, tape = m.compute(init=init, vout='c', return_tape=True)
    assert c == 6

    vjp = tape.get_vjp()
    init = dict(_c=1.0)
    _a = vjp.compute(init=init, vout='_a', monitor=print)
    assert _a == 2.0

    jvp = tape.get_jvp()
    init = dict(a_=1.0)
    c_ = jvp.compute(init=init, vout='c_', monitor=print)
    assert c_ == 2.0

def test_model_unused():
    with Builder() as m:
        a, b = m.input('a', 'b')
        m.output(c=1.0)

    init = dict(a=3, b=4)
    c, tape = m.compute(init=init, vout='c', return_tape=True)
    assert c == 1.0

    vjp = tape.get_vjp()
    init = dict(_c=1.0)
    _a, _b = vjp.compute(init=init, vout=['_a', '_b'], monitor=print)
    assert _a == 0
    assert _b == 0

    jvp = tape.get_jvp()
    init = dict(a_=1.0, b_=1.0)
    c_ = jvp.compute(init=init, vout='c_', monitor=print)
    assert c_ == 0

def test_model_partial_out():
    with Builder() as m:
        a = m.input('a')
        t1 = add(x1=a, x2=a)
        m.output(c=t1)
        m.output(a=a)

    init = dict(a=3)

    (a, c), tape = m.compute(init=init, vout=['a', 'c'], return_tape=True)
    assert c == 6
    assert a == 3

    vjp = tape.get_vjp()

    # test two outputs individually
    init = dict(_c=1.0, _a=0.0)
    _a = vjp.compute(init=init, vout='_a', monitor=print)
    assert _a == 2.0

    init = dict(_c=0.0, _a=1.0)
    _a = vjp.compute(init=init, vout='_a', monitor=print)
    assert _a == 1.0

    jvp = tape.get_jvp()
    init = dict(a_=1.0)
    a_, c_ = jvp.compute(init=init, vout=['a_', 'c_'], monitor=print)
    assert c_ == 2.0
    assert a_ == 1.0

def test_tape_unused():
    # assert unused extra args are not recorded on the tape.
    with Builder() as m:
        a = m.input('a')
        b = add(x1=a, x2=a, j=3)
        m.output(b=b)

    init = dict(a = 1.0)
    b, tape = m.compute(init=init, vout='b', monitor=print, return_tape=True)
    assert b == 2.0
    assert isinstance(tape[0].node, add._apl)
    assert 'j' not in tape[0].resolved
    assert 'j' in tape[0].node.kwargs

    pprint(tape[:])

def test_model_extra_args():
    # assert used extra args are recored on the tape
    @operator
    class extra_args:
        ain = {'x' : '*'}
        aout = {'y' : '*'}

        def apl(self, x, p):
            return dict(y = x * p)

        def vjp(self, x, _x, _y, p):
            return dict(_x = _y * p)

        def jvp(self, x, x_, y_, p):
            return dict(y_ = x_ * p)

    with Builder() as m:
        a = m.input('a')
        b = extra_args(x=a, p=2.0)
        m.output(b=b)

    init = dict(a = 1.0)
    b, tape = m.compute(init=init, vout='b', monitor=print, return_tape=True)

    assert b == 2.0
    assert isinstance(tape[0].node, extra_args._apl)
    assert 'p' not in tape[0].resolved
    assert 'p' in tape[0].node.kwargs

def test_model_many_rewrites():
    # this is a nasty model with many variable rewrites.
    n = 2
    with Builder() as m:
        x = m.input('x')
        for i in range(2):
            x = add(x1=x, x2=x)

        m.output(y=x)

    init = dict(x=1.0)
    y, tape = m.compute(init=init, vout='y', return_tape=True)
    assert y == 4.0

    vjp = tape.get_vjp()
    init = dict(_y = 1.0)
    _x = vjp.compute(init=init, vout='_x', monitor=print)
    assert _x == 4.0
