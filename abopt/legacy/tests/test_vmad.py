from __future__ import print_function
from abopt.legacy.vmad import VM, microcode, programme, Zero, Tape
from numpy.testing import assert_raises, assert_array_equal, assert_allclose
import numpy

class TestVM(VM):
    @microcode(ain=['x'], aout=['y'])
    def unitary(self, x, y, factor):
        y[...] = x * factor
    @unitary.defvjp
    def _(self, _x, _y, factor):
        _x[...] = _y * factor
    @microcode(ain=['x1', 'x2'], aout=['y'])
    def binary(self, x1, x2, y):
        y[...] = x1 + x2
    @binary.defvjp
    def _(self, _x1, _x2, _y):
        _x1[...] = _y
        _x2[...] = _y

def test_single_compute():
    vm = TestVM()
    code = vm.code()
    code.unitary(x='a', y='b', factor=3.0)
    b = code.compute('b', {'a' : 1.0})
    assert_array_equal(b, 3.0)

def test_single_gradient():
    vm = TestVM()
    code = vm.code()
    code.unitary(x='a', y='b', factor=3.0)
    b, _a = code.compute_with_gradient(['b', '_a'], {'a' : 1.0}, {'_b': 1.0})
    assert_array_equal(b, 3.0)
    assert_array_equal(_a, 3.0)

def test_nested_gradient():
    vm = TestVM()
    code = vm.code()
    code.unitary(x='a', y='b', factor=3.0)
    code.unitary(x='b', y='c', factor=3.0)
    c = code.compute('c', {'a' : 1.0})
    assert_array_equal(c, 9.0)

    _a = code.compute_with_gradient('_a', {'a' : 1.0}, {'_c': 1.0})
    c, _a = code.compute_with_gradient(['c', '_a'], {'a' : 1.0}, {'_c': 1.0})

    assert_array_equal(c, 9.0)
    assert_array_equal(_a, 9.0)

def test_partial_gradient():
    vm = TestVM()
    code = vm.code()
    code.unitary(x='a', y='b1', factor=3.0)
    code.unitary(x='a', y='b2', factor=3.0)
    code.unitary(x='a', y='b3', factor=3.0)
    code.unitary(x='a', y='b4', factor=3.0)
    code.binary(x1='b1', x2='b2', y='c1')
    code.binary(x1='b3', x2='b4', y='c2')
    code.binary(x1='c1', x2='c2', y='d')

    d, tape = code.compute('d', {'a' : 1.0}, return_tape=True)
    assert_array_equal(d, 12.0)
    gradient = vm.gradient(tape)

    d, _a = code.compute_with_gradient(['d', '_a'], {'a' : 1.0}, {'_d': 1.0})

    assert_array_equal(d, 12.0)
    assert_array_equal(_a, 12.0)


def test_inplace_gradient():
    vm = TestVM()
    code = vm.code()
    code.unitary(x='a', y='a', factor=3.0)
    code.unitary(x='a', y='b1', factor=3.0)
    code.unitary(x='a', y='b2', factor=3.0)
    code.binary(x1='b1', x2='b2', y='b1')
    code.unitary(x='b1', y='d', factor=3.0)

    d, tape = code.compute('d', {'a' : 1.0}, return_tape=True)
    assert_array_equal(d, 54.0)
    gradient = vm.gradient(tape)

    d, _a = code.compute_with_gradient(['d', '_a'], {'a' : 1.0}, {'_d': 1.0})

    assert_array_equal(d, 54.0)
    assert_array_equal(_a, 54.0)

