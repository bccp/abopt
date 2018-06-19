from __future__ import print_function
from abopt.legacy.vmad2 import CodeSegment, Engine, statement, programme, ZERO, logger, Literal
from numpy.testing import assert_raises, assert_array_equal, assert_allclose
from numpy.testing.decorators import skipif
import pytest
xfail = pytest.mark.xfail
import numpy
import logging

try: import graphviz
except ImportError: graphviz = None

logger.setLevel(level=logging.INFO)
class MySubEngine(Engine):
    @statement(ain=['x'], aout=['y'])
    def unitary(engine, x, y, factor):
        y[...] = x * factor

    @unitary.defvjp
    def _(engine, _x, _y, factor):
        _x[...] = _y * factor

    @unitary.defjvp
    def _(engine, x_, y_, factor):
        y_[...] = x_ * factor

class MyEngine(Engine):
    def __init__(self):
        self.subengine = MySubEngine()

    @statement(ain=['x'], aout=['y'])
    def unitary(engine, x, y, factor):
        y[...] = x * factor

    @unitary.defvjp
    def _(engine, _x, _y, factor):
        _x[...] = _y * factor

    @unitary.defjvp
    def _(engine, x_, y_, factor):
        y_[...] = x_ * factor

    @statement(ain=['x1', 'x2'], aout=['y'])
    def binary(engine, x1, x2, y):
        y[...] = x1 + x2

    @binary.defvjp
    def _(engine, _x1, _x2, _y):
        _x1[...] = _y
        _x2[...] = _y

    @binary.defjvp
    def _(engine, x1_, x2_, y_):
        y_[...] = x1_ + x2_

    @programme(ain=['u'], aout=['v'])
    def batch(engine, u, v):
        code = CodeSegment(engine)
        code.unitary(x=u, y=u, factor=1.0)
        code.binary(x1=u, x2=u, y=v)
        return code

    @programme(ain=['u'], aout=['v'])
    def batch_batch(engine, u, v):
        code = CodeSegment(engine)
        code.batch_with_exarg(u=u, v='t', factor=1)
        code.unitary(x='t', y=v, factor=1.0)
        return code

    @programme(ain=['u'], aout=['v'])
    def batch_with_sub(engine, u, v):
        code = CodeSegment(engine.subengine)
        code.unitary(x=u, y=v, factor=2.0)
        return code

    @programme(ain=['u'], aout=['v'])
    def batch_with_exarg(engine, u, v, factor):
        code = CodeSegment(engine)
        code.unitary(x=u, y=u, factor=factor)
        code.binary(x1=u, x2=u, y=v)
        return code

    @programme(ain=['u'], aout=['v'])
    def batch_unused(engine, u, v):
        code = CodeSegment(engine)
        code.unitary(x=Literal(1.0), y=v, factor=1.0)
        return code

def test_compute():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.unitary(x='a', y='b', factor=3.0)
    b = code.compute('b', {'a' : 1.0})
    assert_array_equal(b, 3.0)

def test_optimize():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.unitary(x='a', y='d', factor=3.0)
    code.unitary(x='a', y='b', factor=3.0)
    code.unitary(x='a', y='c', factor=3.0)

    opt = code.optimize(['b'])
    assert len(opt.nodes) == 1
    b = opt.compute('b', {'a' : 1.0})
    assert_array_equal(b, 3.0)

def test_optimized_execution():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.unitary(x='a', y='d', factor=3.0)
    code.unitary(x='a', y='b', factor=3.0)
    code.unitary(x='a', y='c', factor=3.0)

    opt = code.optimize(['b'])
    assert len(opt.nodes) == 1
    
    b, tape = code.compute('b', {'a' : 1.0}, return_tape=True)
    print(tape)

def test_nested_compute():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.unitary(x='a', y='b', factor=3.0)
    code.unitary(x='b', y='c', factor=3.0)

    c = code.compute('c', {'a' : 1.0})
    assert_array_equal(c, 9.0)

    c, _a = code.compute_with_gradient(['c', '_a'], {'a' : 1.0}, {'_c': 1.0})

    assert_array_equal(c, 9.0)
    assert_array_equal(_a, 9.0)

def test_tape_gradients():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.unitary(x='a', y='b1', factor=3.0)
    code.unitary(x='a', y='b2', factor=3.0)
    code.unitary(x='a', y='b3', factor=3.0)
    code.unitary(x='a', y='b4', factor=3.0)
    code.binary(x1='b1', x2='b2', y='c1')
    code.binary(x1='b3', x2='b4', y='c2')
    code.binary(x1='c1', x2='c2', y='d')

    d, tape = code.compute('d', {'a' : 1.0}, return_tape=True)
    assert_array_equal(d, 12.0)

    vjp = tape.get_vjp()
    _a  = vjp.compute(['_a'], {'_d': 1.0})
    assert_array_equal(_a, 12.0)

    jvp = tape.get_jvp()
    d_  = jvp.compute(['d_'], {'a_': 1.0})
    assert_array_equal(d_, 12.0)

    d, _a = code.compute_with_gradient(['d', '_a'], {'a' : 1.0}, {'_d': 1.0})

    assert_array_equal(d, 12.0)
    assert_array_equal(_a, 12.0)

    (c1, d), tape = code.compute(['c1', 'd'], {'a' : 1.0}, return_tape=True)

    assert_array_equal(d, 12.0)
    assert_array_equal(c1, 6.0)

    vjp = tape.get_vjp()
    jvp = tape.get_jvp()

    _a = vjp.compute(['_a'], {'_c1': 1.0})
    c1_, d_ = jvp.compute(['c1_', 'd_'], {'a_': 1.0})

    assert_array_equal(c1_, 6.0)
    assert_array_equal(d_, 12.0)

@xfail(reason="jvp from code doesn't work")
def test_jvp():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.unitary(x='a', y='b1', factor=3.0)
    code.unitary(x='a', y='b2', factor=3.0)
    code.unitary(x='a', y='b3', factor=3.0)
    code.unitary(x='a', y='b4', factor=3.0)
    code.binary(x1='b1', x2='b2', y='c1')
    code.binary(x1='b3', x2='b4', y='c2')
    code.binary(x1='c1', x2='c2', y='d')

    jvp = code.get_jvp(init={'a' : 1.0})

    d_ = jvp.compute('d_', {'a_' : 1.0})
    assert_array_equal(d_, 12.0)

    c1_, d_ = jvp.compute(['c1_', 'd_'], {'a_' : 1.0})
    assert_array_equal(d_, 12.0)
    assert_array_equal(c1_, 6.0)

@xfail(reason="jvp from code doesn't work")
def test_jvp_programme():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.batch(u='a', v='d')
    code.unitary(x='d', y='d', factor=2.0)
    code.batch_with_exarg(u='a', v='e', factor=3.0)

    jvp = code.get_jvp(init={'a' : 1.0})
    d_, e_ = jvp.compute(['d_', 'e_'], {'a_' : 1.0})
    assert_array_equal(d_, 4.0)
    assert_array_equal(e_, 6.0)

@xfail(reason="jvp from code doesn't work")
def test_jvp_programme_nested():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.batch_batch(u='a', v='d')

    jvp = code.get_jvp(init={'a' : 1.0})

    d_ = jvp.compute('d_', {'a_' : 1.0})
    assert_array_equal(d_, 2.0)

    d, tape = code.compute('d', init={'a' : 1.0}, return_tape=True)
    jvp = tape.get_jvp()

    d_ = jvp.compute('d_', {'a_' : 1.0})
    assert_array_equal(d_, 2.0)

@xfail(reason="jvp from code doesn't work")
def test_jvp_vector():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.batch(u='a', v='d')
    code.unitary(x='d', y='d', factor=2.0)
    code.batch_with_exarg(u='a', v='e', factor=3.0)

    A = numpy.array

    jvp = code.get_jvp(init={'a' : A([1.0, 1.0])})
    d_, e_ = jvp.compute(['d_', 'e_'], {'a_' : A([1.0, 1.0])})
    assert_array_equal(d_, [4.0, 4.0])
    assert_array_equal(e_, [6.0, 6.0])

def test_inplace():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.unitary(x='a', y='a', factor=3.0)
    code.unitary(x='a', y='b1', factor=3.0)
    code.unitary(x='a', y='b2', factor=3.0)
    code.binary(x1='b1', x2='b2', y='b1')
    code.unitary(x='b1', y='d', factor=3.0)

    d = code.compute('d', {'a' : 1.0})
    assert_array_equal(d, 54.0)

    d, _a = code.compute_with_gradient(['d', '_a'], {'a' : 1.0}, {'_d': 1.0})

    assert_array_equal(d, 54.0)
    assert_array_equal(_a, 54.0)

@skipif(graphviz == None, "graphviz is not properly installed")
def test_to_graph():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.batch_with_sub(u='a', v='e')
    code.unitary(x='a', y='a', factor=3.0)
    code.unitary(x='a', y='b1', factor=3.0)
    code.unitary(x='a', y='b2', factor=3.0)
    code.binary(x1='b1', x2='b2', y='b1')
    code.unitary(x='b1', y='d', factor=3.0)
    code.batch(u='b2', v='f')

    d, tape = code.compute(('e', 'a', 'f', 'd'), {'a' : 1.0}, return_tape=True)
    vjp = tape.get_vjp()

    graph1 = code.to_graph()
    graph2 = vjp.to_graph()

#    graph2.render('temp.png', view=True)

def test_zeros():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.unitary(x='a', y='a', factor=3.0)
    code.unitary(x='a', y='b1', factor=3.0)
    code.unitary(x='a', y='b2', factor=3.0)
    a, b1, b2, _a = code.compute_with_gradient(['a', 'b1', 'b2', '_a'], {'a' : 1.0}, {'_b1': ZERO, '_b2' : ZERO, '_a' : ZERO})
    assert _a is ZERO

def test_literal():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.binary(x1='a', x2=Literal(2.0), y='a')
    code.batch(u=Literal(2.0), v='d')
    d, a, _a = code.compute_with_gradient(['d', 'a', '_a'], {'a' : 1.0}, {'_a': 1.0, '_d' : 0.0})
    assert_array_equal(a, 3.0)
    assert_array_equal(_a, 1.0)

def test_copy():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.batch_with_sub(u='a', v='e')
    code.unitary(x='a', y='a', factor=3.0)
    code.unitary(x='a', y='b1', factor=3.0)
    code.unitary(x='a', y='b2', factor=3.0)
    code.binary(x1='b1', x2='b2', y='b1')
    code.unitary(x='b1', y='d', factor=3.0)
    code.batch(u='b2', v='f')

    code2 = code.copy()

def test_programme():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.batch(u='a', v='d')
    code.batch_with_exarg(u='a', v='e', factor=3.0)
    (d, e), tape = code.compute(('d', 'e'), {'a' : 1.0}, return_tape=True)
    assert_array_equal(d, 2.0)
    assert_array_equal(e, 6.0)
    e, d, _a = code.compute_with_gradient(['e', 'd', '_a'], {'a' : 1.0}, {'_d': 1.0, '_e' : 0.0})
    assert_array_equal(d, 2.0)
    assert_array_equal(e, 6.0)
    assert_array_equal(_a, 2.0)
    e, d, _a = code.compute_with_gradient(['e', 'd', '_a'], {'a' : 1.0}, {'_d': 0.0, '_e' : 1.0})
    assert_array_equal(d, 2.0)
    assert_array_equal(e, 6.0)
    assert_array_equal(_a, 6.0)

def test_programme_unused():
    engine = MyEngine()
    code = CodeSegment(engine)
    code.batch_unused(u='a', v='d')
    d, _a = code.compute_with_gradient(['d', '_a'], {'a' : 1.0}, {'_d': 1.0})
    assert_array_equal(d, 1.0)
    assert_array_equal(_a, 0.0)

