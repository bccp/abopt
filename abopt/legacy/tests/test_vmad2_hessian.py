from __future__ import print_function
from abopt.legacy.vmad2 import CodeSegment, Engine, statement, programme, ZERO, logger, Literal
from numpy.testing import assert_raises, assert_array_equal, assert_allclose
from numpy.testing.decorators import skipif
import numpy
import logging

try: import graphviz
except ImportError: graphviz = None

class ScalarEngine(Engine):
    def __init__(self):
        pass    

    @statement(ain=['x'], aout=['y'])
    def unitary(engine, x, y, factor):
        y[...] = x * factor

    @unitary.defvjp
    def _(engine, _x, _y, factor):
        _x[...] = _y * factor

    @unitary.defjvp
    def _(engine, x_, y_, factor):
        y_[...] = x_ * factor

    @unitary.jvp.defvjp
    def _(engine, factor, _x_, _y_):
        _x_[...] = _y_ * factor

    @statement(ain=['x'], aout=['y'])
    def power(engine, x, y, factor):
        y[...] = x ** factor

    @power.defvjp
    def _(engine, _x, _y, x, factor):
        if factor > 1:
            _x[...] = _y * ( x ** (factor - 1) * factor if factor > 1 else 1 )
        else:
            x_[...] = _y

    @power.defjvp
    def _(engine, x_, y_, x, factor):
        y_[...] = x_ * x ** (factor - 1) * factor

    @power.jvp.defvjp
    def _(engine, x, factor, _x_, _y_, _x, x_):
        _x[...] = _y_ * x_ * (x ** (factor - 2) * factor * (factor - 1) if factor > 2 else 1)
        _x_[...] = _y_ * ( x ** (factor - 1) * factor if factor > 1 else 1)

def test_scalar():
    engine = ScalarEngine()
    code = CodeSegment(engine)
    code.unitary(x='a', y='r', factor=9.0)
    code.power(x='r', y='b', factor=2.0)
    b = code.compute('b', {'a' : 2.0})
    b, tape = code.compute('b', {'a' : 2.0}, return_tape=True)

    backward_gradient = tape.get_vjp()
    forward_gradient = code.get_jvp()

    b_ = forward_gradient.compute('b_', {'a' : 2.0, 'a_' : 1.0})
    _a = backward_gradient.compute('_a', {'_b' : 1.0})
    assert_array_equal(b_, _a)

    # to do the hessian, first augment the compute with a forward pass 

    # then do the backward gradient on the forward tape
    # we need us terms like x_, etc.

    print(forward_gradient)
    (b_, b), tape = forward_gradient.compute(['b_', 'b'], {'a' : 2.0, 'a_' : 1.0}, return_tape=True)
    print('b_', 'b', b_, b)
    hessian_dot = tape.get_vjp()
    print(hessian_dot)

    _a = hessian_dot.compute('_a', {'_b_' : 1.0, '_b' : 0.0}, monitor=lambda node, frontier, r:print('---', node, frontier, r))
    print(_a)

class VectorEngine(Engine):
    def __init__(self):
        pass    

    @statement(ain=['x'], aout=['y'])
    def multiply(engine, x, y, factor):
        y[...] = x * factor

    @multiply.defvjp
    def _(engine, _x, _y, factor):
        _x[...] = _y * factor

    @multiply.defjvp
    def _(engine, x_, y_, factor):
        y_[...] = x_ * factor

    @multiply.jvp.defvjp
    def _(engine, factor, _y_, _x_):
        _x_[...] = _y_ * factor

    @statement(ain=['x1', 'x2'], aout=['y'])
    def dot(engine, x1, x2, y):
        y[...] = numpy.einsum('i,i->', x1, x2)

    @dot.defvjp
    def _(engine, _x1, _x2, x1, x2, _y):
        _x1[...] = _y * x2
        _x2[...] = _y * x1

    @dot.defjvp
    def _(engine, x1_, x2_, x1, x2, y_):
        y_[...] = numpy.einsum('i,i->', x1, x2_) \
                + numpy.einsum('i,i->', x2, x1_)

    @dot.jvp.defvjp
    def _(engine, x1_, x2_, x1, x2, _x1_, _x2_, _x1, _x2, _y_):
        _x1_[...] = x2 * _y_
        _x2_[...] = x1 * _y_
        _x1[...] = _y_ * x2
        _x2[...] = _y_ * x1

    @statement(ain=['x'], aout=['y'])
    def power(engine, x, y, factor):
        y[...] = x ** factor

    @power.defvjp
    def _(engine, _x, _y, x, factor):
        _x[...] = _y * (x ** (factor - 1) * factor if factor != 1 else 1)

    @power.defjvp
    def _(engine, x_, y_, x, factor):
        y_[...] = x_ * (x ** (factor - 1) * factor if factor != 1 else 1)

    @power.jvp.defvjp
    def _(engine, x_, x, factor, _x_, _y_, _x):
        _x_[...] = _y_ * (x ** (factor - 1) * factor if factor != 1 else 1)
        _x[...] = _y_ * x_ * (x ** (factor - 2) * (factor  - 1) * factor if factor != 2 else 1)

def impulse(shape, ind):
    x = numpy.zeros(shape)
    x[ind] = 1.0
    return x

def bases(shape, return_index=False):
    for ind in numpy.ndindex(*shape):
        if return_index:
            yield impulse(shape, ind), ind
        else:
            yield impulse(shape, ind)

def test_vector():
    engine = VectorEngine()
    code = CodeSegment(engine)

    code.multiply(x='x1', y='r1', factor=3.0)
    code.multiply(x='x2', y='r2', factor=3.0)
    code.dot(x1='r1', x2='r2', y='r3')
    code.power(x='r3', y='y', factor=2)

    init = {'x1' : numpy.array([1.0, 1.0]),
            'x2' : numpy.array([1.0, 1.0])}
    b, tape = code.compute('y', init, return_tape=True)

    backward_gradient = tape.get_vjp()
    forward_gradient = code.get_jvp()

    _x1, _x2 = backward_gradient.compute(['_x1', '_x2'], {'_y' : 1.0})

    for (x1_, x2_), ind in bases((2,2), return_index=True):
        d = {'x1_' : x1_, 'x2_' : x2_}
        d.update(init)
        y_ = forward_gradient.compute('y_', d)
        assert_array_equal(y_ , numpy.array([_x1, _x2])[ind])

    x1_, x2_ = impulse((2, 2), (0, 1))

    d = {'x1_' : x1_, 'x2_' : x2_}
    d.update(init)
    (y, y_), tape = forward_gradient.compute(['y', 'y_'], d, return_tape=True)
    hessian_dot = tape.get_vjp()
    _x1, _x2 = hessian_dot.compute({'_x1', '_x2'}, {'_y_' : 1.0, '_y' : 0.0})

