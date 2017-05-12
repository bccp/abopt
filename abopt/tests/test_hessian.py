from __future__ import print_function
from abopt.vmad2 import CodeSegment, Engine, statement, programme, ZERO, logger, Literal
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

    @statement(ain=['x'], aout=['y'])
    def power(engine, x, y, factor):
        y[...] = x ** factor

    @power.defvjp
    def _(engine, _x, _y, x, factor):
        _x[...] = _y * x ** (factor - 1) * factor

    @power.defjvp
    def _(engine, x_, y_, x, factor):
        y_[...] = x_ * x ** (factor - 1) * factor

def test_scalar():
    engine = ScalarEngine()
    code = CodeSegment(engine)
    code.unitary(x='a', y='r', factor=3.0)
    code.power(x='r', y='b', factor=2.0)
    b = code.compute('b', {'a' : 1.0})
    b, tape = code.compute('b', {'a' : 1.0}, return_tape=True)

    backward_gradient = tape.gradient()
    forward_gradient = code.gradient()

    b_ = forward_gradient.compute('b_', {'a' : 1.0, 'a_' : 1.0})
    _a = backward_gradient.compute('_a', {'_b' : 1.0})
    assert_array_equal(b_, _a)

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

    @statement(ain=['x'], aout=['y'])
    def power(engine, x, y, factor):
        y[...] = x ** factor

    @power.defvjp
    def _(engine, _x, _y, x, factor):
        if factor > 1:
            _x[...] = _y * x ** (factor - 1) * factor
        else:
            _x[...] = _y

    @power.defjvp
    def _(engine, x_, y_, x, factor):
        if factor > 1:
            y_[...] = x_ * x ** (factor - 1) * factor
        else:
            y_[...] = x_

def impulse(shape, ind):
    x = numpy.zeros(shape)
    x[ind] = 1.0
    return x

def bases(shape):
    for ind in numpy.ndindex(*shape):
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

    backward_gradient = tape.gradient()
    forward_gradient = code.gradient()

    for x1_, x2_ in bases((2,2)):
            d = {'x1_' : x1_, 'x2_' : x2_}
            d.update(init)
            y_ = forward_gradient.compute('y_', d)
            print(x1_, x2_, y_)

    _x1, _x2 = backward_gradient.compute(['_x1', '_x2'], {'_y' : 1.0})
    print(_x1)
    print(_x2)
