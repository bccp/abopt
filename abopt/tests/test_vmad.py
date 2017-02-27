from __future__ import print_function
from abopt.vmad import VM
from numpy.testing import assert_raises, assert_array_equal, assert_allclose
import numpy

def test_booster():
    class Booster(VM):
        def __init__(self, q):
            self.q = q

        @VM.microcode(ain=['x'], aout=['y'])
        def boost(self, x, y, factor):
            y[...] = x * factor * self.q

        @boost.grad
        def gboost(self, _y, _x, factor):
            _x[...] = _y * factor

    vm = Booster(q=1.0)
    code = vm.code()
    code.boost(x='i', y='r1', factor=1.0)
    code.boost(x='r1', y='r2', factor=2.0)
    code.boost(x='r2', y='y', factor=3.0)
    code = code.copy()
    print('code', code)

    tape = vm.tape()
    y = code.compute('y', {'i' : numpy.ones(1)}, tape)
    assert_array_equal(y, 6.0)
    print('tape', tape)
    gcode = vm.gradient(tape, add=Booster.Add)
    print('gcode', gcode)
    _i = gcode.compute('_i', {'_y' : numpy.ones(1)}, monitor=print)
    assert_array_equal(_i, 6.0)

    tape = vm.tape()
    y = code.compute('y', {'i' : 1, 'q' : 1234}, tape)
    assert_array_equal(y, 6.0)

    gcode = vm.gradient(tape, add=Booster.Add)
    _i = gcode.compute('_i', {'_y' : 1})
    assert_array_equal(_i, 6.0)

def test_integrator():
    class Integrator(VM):
        @VM.microcode(ain=['v', 'a'], aout=['v'])
        def kick(self, v, a):
            v[...] += a * 0.01

        @kick.grad
        def _(self, _v, _a):
            _a[...] = 0.01 * _v

        @VM.microcode(ain=['x', 'v'], aout=['x'])
        def drift(self, x, v):
            x[...] += v * 0.01

        @drift.grad
        def _(self, _x, _v):
            _v[...] = 0.01 * _x

        @VM.microcode(ain=['x'], aout=['a'])
        def force(self, x, a):
            a[...] = -x

        @force.grad
        def _(self, _a, _x):
            _x[...] = - _a

        @VM.microcode(ain=['x'], aout=['chi2'])
        def reduce(self, x, chi2):
            chi2[...] = (x ** 2).sum()

        @reduce.grad
        def _(self, x, _chi2, _x):
            _x[...] = 2 * x * _chi2

    vm = Integrator()
    tape = vm.tape()
    code = vm.code()
    code.force()
    for i in range(2):
        code.kick()
        code.drift()
        code.drift()
        code.force()
        code.kick()

    #vm.push('reduce')
    print(code)
    def objective(x, v):
        init = {'x' : x, 'v' : v}
        tape = vm.tape()
        x = code.compute('x', init, tape)
        chi2 = (x ** 2).sum()
        return chi2

    def gradient(x, v):
        init = {'x' : x, 'v' : v}
        tape = vm.tape()
        x = code.compute('x', init, tape, monitor=print)
        print(tape)
        gcode = vm.gradient(tape, add=Integrator.Add)
        print(gcode)

    #    ginit = {'^chi2' : 1.}
        ginit = {'_x' : 2 * x}
        r = gcode.compute(['_x', '_v'], ginit, monitor=print)
        return r

    x0 = numpy.ones(1024) 
    v0 = numpy.zeros_like(x0)
    eps = numpy.zeros_like(x0)

    g_x, g_v = gradient(x0, v0)

    eps[0] = 1e-7
    chi0 = objective(x0 - eps, v0)
    chi1 = objective(x0 + eps, v0)
    numgx = (chi1 - chi0) / (2 * eps[0])
    chi0 = objective(x0, v0 - eps)
    chi1 = objective(x0, v0 + eps)
    numgv = (chi1 - chi0) / (2 * eps[0])
    assert_allclose(
        [g_x[0], g_v[0]],
        [numgx, numgv], rtol=1e-3)
