from __future__ import print_function
from pprint import pprint
from abopt.vmad3.lib import pmeshvmad, linalg
import numpy

from abopt.vmad3.testing import BaseScalarTest
from mpi4py import MPI

pm = pmeshvmad.ParticleMesh(Nmesh=[4, 4], BoxSize=8.0, comm=MPI.COMM_SELF)

from pmesh.pm import RealField, ComplexField

def create_bases(x):
    bases = numpy.eye(x.size).reshape([-1] + list(x.shape))
    if isinstance(x, RealField):
        # FIXME: remove this after pmesh 0.1.36
        def create_field(pm, data):
            real = pm.create(mode='real')
            real[...] = data
            return real
        return [create_field(pm, i) for i in bases]
    else:
        return [i for i in bases]

class Test_r2c_c2r(BaseScalarTest):
    to_scalar = pmeshvmad.to_scalar

    x = pm.generate_whitenoise(seed=300, unitary=True, mode='real')
    y = x.cnorm()
    x_ = create_bases(x)

    def model(self, x):
        c = pmeshvmad.r2c(x)
        r = pmeshvmad.c2r(c)
        return r

#    def teardown(self):
#        print(self.y_)

def transfer(k):
    k2 = sum(ki ** 2 for ki in k)
    mask = k2 == 0
    k2[mask] = 1
    r = 1 / k2
    r[mask] = 0
    return r

class Test_r2c_transfer_c2r(BaseScalarTest):
    to_scalar = pmeshvmad.to_scalar

    x = pm.generate_whitenoise(seed=300, unitary=True, mode='real')
    y = x.r2c().apply(lambda k, v: transfer(k) * v).c2r().cnorm()
    x_ = create_bases(x)

    def model(self, x):
        c = pmeshvmad.r2c(x)
        c = pmeshvmad.apply_transfer(c, tf=transfer)
        r = pmeshvmad.c2r(c)
        return r

#    def teardown(self):
#        print(self.y_)

class Test_paint_x(BaseScalarTest):
    to_scalar = pmeshvmad.to_scalar

    mesh = pm.generate_whitenoise(seed=300, unitary=True, mode='real')

    x = pm.generate_uniform_particle_grid(shift=0.1)
    y = (1 + mesh).cnorm()
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
        y = pmeshvmad.paint(x, layout=None, mass=1.0, pm=pm)
        y = linalg.add(y, self.mesh) # biasing a bit to get non-zero derivatives.
        return y

class Test_decompose_paint_x(BaseScalarTest):
    to_scalar = pmeshvmad.to_scalar

    mesh = pm.generate_whitenoise(seed=300, unitary=True, mode='real')

    x = pm.generate_uniform_particle_grid(shift=0.1)
    y = (1 + mesh).cnorm()
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
        layout = pmeshvmad.decompose(x, pm=pm)
        y = pmeshvmad.paint(x, layout=layout, mass=1.0, pm=pm)
        y = linalg.add(y, self.mesh) # biasing a bit to get non-zero derivatives.
        return y

class Test_paint_mass(BaseScalarTest):
    to_scalar = pmeshvmad.to_scalar

    mesh = pm.generate_whitenoise(seed=300, unitary=True, mode='real')

    pos = pm.generate_uniform_particle_grid(shift=0.1)
    x = numpy.ones(len(pos))
    y = x.sum()
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
        y = pmeshvmad.paint(self.pos, layout=None, mass=x, pm=pm)
        return y

class Test_readout_x(BaseScalarTest):
    to_scalar = linalg.to_scalar

    mesh = pm.generate_whitenoise(seed=300, unitary=True, mode='real')

    x = pm.generate_uniform_particle_grid(shift=0.5)
    y = (mesh.readout(x) ** 2).sum()
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
        y = pmeshvmad.readout(self.mesh, x, layout=None, pm=pm)
        return y

class Test_readout_mesh(BaseScalarTest):
    to_scalar = linalg.to_scalar

    x = pm.generate_whitenoise(seed=300, unitary=True, mode='real')

    pos = pm.generate_uniform_particle_grid(shift=0.5)
    y = (x.readout(pos) ** 2).sum()
    x_ = create_bases(x)

    epsilon = 1e-3

    def model(self, x):
        y = pmeshvmad.readout(x, self.pos, layout=None, pm=pm)
        return y
