from __future__ import print_function
from abopt.vmad import VM
from numpy.testing import assert_raises
import numpy

class Integrator(VM):
    @VM.microcode(fin=['v', 'a'], fout=['v'])
    def kick(self, frontier):
        frontier['v'] = frontier['v'] + frontier['a'] * 0.01

    @kick.grad
    @VM.microcode(fin=[])
    def gkick(self, frontier):
        frontier['^v'] = frontier['^v']
        frontier['^a'] = 0.01 * frontier['^v']

    @VM.microcode(fin=['x', 'v'], fout=['x'])
    def drift(self, frontier):
        frontier['x'] = frontier['x'] + frontier['v'] * 0.01

    @drift.grad
    @VM.microcode(fin=[])
    def gdrift(self, frontier):
        frontier['^x'] = frontier['^x']
        frontier['^v'] = 0.01 * frontier['^x']


    @VM.microcode(fin=['x'], fout=['a'])
    def force(self, frontier):
        frontier['a'] = - frontier['x']

    @force.grad
    @VM.microcode(fin=[])
    def gforce(self, frontier):
        frontier['^x'] = - frontier['^a']

    @VM.microcode(fin=['x'], fout=['chi2'])
    def reduce(self, frontier):
        frontier['chi2'] = (frontier['x'] ** 2).sum()

    @reduce.grad
    @VM.microcode(fin=['x'])
    def greduce(self, frontier):
        frontier['^x'] = (frontier['x'] * 2)

def test_myvm():
    vm = Integrator()
    tape = []

    vm.push('force')
    for i in range(2):
        vm.push('kick')
        vm.push('drift')
        vm.push('drift')
        vm.push('force')
        vm.push('kick')
    #vm.push('reduce')

    def objective(x, v):
        init = {'x' : x, 'v' : v}
        tape = []
        #r = vm.compute(['chi2'], init, tape)
        r = vm.compute(['x'], init, tape)
        r['chi2'] = (r['x'] ** 2).sum()
        return r['chi2']

    def gradient(x, v):
        init = {'x' : x, 'v' : v}
        tape = []
        r = vm.compute(['x'], init, tape)
    #    ginit = {'^chi2' : 1.}
        ginit = {'^x' : 2 * r['x']}
        r = vm.gradient(['^x', '^v'], ginit, tape)
        return r

    x0 = numpy.ones(1024) 
    v0 = numpy.zeros_like(x0)
    eps = numpy.zeros_like(x0)

    g = gradient(x0, v0)
    eps[0] = 1e-7
    chi0 = objective(x0 - eps, v0)
    chi1 = objective(x0 + eps, v0)
    print ('num ^x', (chi1 - chi0) / (2 * eps[0]))
    chi0 = objective(x0, v0 - eps)
    chi1 = objective(x0, v0 + eps)
    print ('num ^v', (chi1 - chi0) / (2 * eps[0]))

    print('gradient = ', g)
