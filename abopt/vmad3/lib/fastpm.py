from abopt.vmad3 import operator, autooperator
from abopt.vmad3.core.model import Literal
from pmesh.pm import ParticleMesh
from abopt.vmad3.lib import linalg
import numpy

@operator
class to_scalar:
    ain = {'x' : 'RealField'}
    aout = {'y' : '*'}

    def apl(self, x):
        return dict(y = x.cnorm())

    def vjp(self, _y, x):
        return dict(_x= x * (2 * _y))

    def jvp(self, x_, x):
        return dict(y_ = x.cdot(x_) * 2)

# FIXME: this is probably not correct.
"""
@operator
class to_scalar_co:
    ain = {'x' : 'ndarray'}
    aout = {'y' : '*'}

    def apl(self, x, comm):
        return dict(y = comm.allreduce((x * numpy.conj(x)).sum()))

    def vjp(self, _y, x, comm):
        return dict(_x = 2 * numpy.conj(_y) * x)

    def jvp(self, x_, x, comm):
        return dict(y_ = comm.allreduce((x_ * numpy.conj(x) + numpy.conj(x_) * x).sum()))
"""

@operator
class as_complex_field:
    ain = {'x' : '*'}
    aout = {'y' : 'ComplexField'}

    def apl(self, x, pm):
        y = pm.create(mode='complex')
        y.real[...] = x[..., 0]
        y.imag[...] = x[..., 1]
        return dict(y=y)

    def vjp(self, _y, pm):
        _x = numpy.stack([_y.real, _y.imag], axis=-1)
        return dict(_x=_x)

    def jvp(self, x_):
        y_ = pm.create(mode='complex')
        y_.real[...] = x_[..., 0]
        y_.imag[...] = x_[..., 1]
        return dict(y_=y_)

@operator
class r2c:
    ain = {'x' : 'RealField'}
    aout = {'y' : 'ComplexField'}

    def apl(self, x):
        return dict(y=x.r2c())
    def vjp(self, _y):
        return dict(_x=_y.r2c_vjp())
    def jvp(self, x_):
        return dict(y_=x_.r2c())

@operator
class c2r:
    ain = {'x' : 'ComplexField'}
    aout = {'y' : 'RealField'}

    def apl(self, x):
        return dict(y=x.c2r())
    def vjp(self, _y):
        return dict(_x=_y.c2r_vjp())
    def jvp(self, x_):
        return dict(y_=x_.c2r())

def nyquist_mask(v, i, Nmesh):
    mask = ~numpy.bitwise_and.reduce([(ii == 0) | (ii == ni // 2) for ii, ni in zip(i, Nmesh)])
    v.imag[...] *= mask
    return v

@operator
class apply_transfer:
    ain = {'x' : 'ComplexField'}
    aout = {'y' : 'ComplexField'}

    def apl(self, x, tf):
        filter = lambda k, v: nyquist_mask(v * tf(k), v.i, v.Nmesh)
        return dict(y=x.apply(filter))

    def vjp(self, _y, tf):
        filter = lambda k, v: nyquist_mask(v * numpy.conj(tf(k)), v.i, v.Nmesh)
        return dict(_x=_y.apply(filter))

    def jvp(self, x_, tf):
        filter = lambda k, v: nyquist_mask(v * tf(k), v.i, v.Nmesh)
        return dict(y_=x_.apply(filter))

@operator
class paint:
    aout = {'mesh' : 'RealField'}
    ain =  {'x': 'ndarray',
           'layout': 'Layout',
           'mass' : 'ndarray'
           }

    def apl(self, x, mass, layout, pm):
        N = pm.comm.allreduce(len(x))
        mesh = pm.paint(x, mass=mass, layout=layout, hold=False)
        # 1 + delta
        mesh[...] *= 1.0 * pm.Nmesh.prod() / N
        return dict(mesh=mesh)

    def vjp(self, _mesh, x, mass, layout, pm):
        N = pm.comm.allreduce(len(x))
        _x, _mass = pm.paint_vjp(_mesh, x, layout=layout, mass=mass)
        _x[...] *= 1.0 * pm.Nmesh.prod() / N
        _mass[...] *= 1.0 * pm.Nmesh.prod() / N
        return dict(
            _layout = 0,
            _x=_x,
            _mass=_mass)

    def jvp(self, x_, x, layout, mass, layout_, mass_, pm):
        if x_ is 0: x_ = None
        if mass_ is 0: mass_ = None # force cast it to a scalar 0, so make it None
        mesh_ = pm.paint_jvp(x, v_mass=mass_, mass=mass, v_pos=x_, layout=layout)

        return dict(mesh_=mesh_)

@operator
class readout:
    aout = {'value' : 'ndarray'}

    ain = {'x': 'ndarray',
        'mesh': 'RealField',
      'layout' : 'Layout'}

    def apl(self, mesh, x, layout, resampler=None):
        N = mesh.pm.comm.allreduce(len(x))
        value = mesh.readout(x, layout=layout, resampler=resampler)
        return dict(value=value)

    def vjp(self, _value, x, layout, mesh, resampler=None):
        _mesh, _x = mesh.readout_vjp(x, _value, layout=layout, resampler=resampler)
        return dict(_mesh=_mesh, _x=_x, _layout=0)

    def jvp(self, x_, mesh_, x, layout, layout_, mesh, resampler=None):
        if mesh_ is 0: mesh_ = None
        if x_ is 0: x_ = None
        value_ = mesh.readout_jvp(x, v_self=mesh_, v_pos=x_, layout=layout, resampler=resampler)
        return dict(value_=value_)

@operator
class decompose:
    aout={'layout' : 'Layout'}
    ain={'x': 'ndarray'}

    def apl(self, x, pm):
        return dict(layout=pm.decompose(x))

    def vjp(engine, _layout):
        return dict(_x=0)

    def jvp(engine, x_):
        return dict(layout_=0)

def fourier_space_gradient(dir):
    def filter(k):
        return 1j * k[dir]
    return filter

def fourier_space_laplace(k):
    k2 = sum(ki **2 for ki in k)
    bad = k2 == 0
    k2[bad] = 1
    k2 = - 1 / k2
    k2[bad] = 0
    return k2

@autooperator
class lpt1:
    ain = [
            ('rhok',  'ComplexField'),
          ]
    aout = [
            ('dx1', '*'),
         #, ('dx2', '*'),
           ]

    def main(self, rhok, q, pm):
        p = apply_transfer(rhok, fourier_space_laplace)
        q = Literal(self, q)

        layout = decompose(q, pm)

        r1 = []
        for d in range(pm.ndim):
            dx1_c = apply_transfer(p, fourier_space_gradient(d))
            dx1_r = c2r(dx1_c)
            dx1 = readout(dx1_r, q, layout)
            r1.append(dx1)

        dx1 = linalg.stack(r1, axis=-1)

        return dict(dx1 = dx1)

@autooperator
class lpt2src:
    ain = [
            ('rhok',  'ComplexField'),
          ]
    aout = [
            ('rho_lpt2', 'RealField'),
           ]

    def main(self, rhok, pm):
        if pm.ndim != 3:
            raise ValueError("LPT 2 only works in 3d")

        D1 = [1, 2, 0]
        D2 = [2, 0, 1]

        potk = apply_transfer(rhok, fourier_space_laplace)

        Pii = []
        for d in range(pm.ndim):
            t = apply_transfer(potk, fourier_space_gradient(d))
            Pii1 = apply_transfer(t, fourier_space_gradient(d))
            Pii1 = c2r(Pii1)
            Pii.append(Pii1)

        source = None
        for d in range(pm.ndim):
            source1 = linalg.mul(Pii[D1[d]], Pii[D2[d]])
            if source is None:
                source = source1
            else:
                source = linalg.add(source, source1)

        for d in range(pm.ndim):
            t = apply_transfer(potk, fourier_space_gradient(D1[d]))
            Pij1 = apply_transfer(t, fourier_space_gradient(D2[d]))
            Pij1 = c2r(Pij1)
            neg = linalg.mul(Pij1, -1)
            source1 = linalg.mul(Pij1, neg)
            source = linalg.add(source, source1)

        source = linalg.mul(source, 3.0/7 )

        return dict(rho_lpt2=source)

@autooperator
class induce_correlation:
    ain = [
            ('wnk',  'ComplexField'),
          ]
    aout = [
            ('c', 'ComplexField'),
           ]

    def main(self, wnk, powerspectrum, pm):
        def tf(k):
            k = sum(ki ** 2 for ki in k) ** 0.5
            return (powerspectrum(k) / pm.BoxSize.prod()) ** 0.5

        c = apply_transfer(wnk, tf)
        return dict(c = c)

@autooperator
class lpt:
    ain = [
            ('rhok',  'RealField'),
          ]

    aout = [
            ('dx1', '*'),
            ('dx2', '*'),
           ]

    def main(self, rhok, q, pm):

        dx1 = lpt1(rhok, q, pm)
        source2 = lpt2src(rhok, pm)
        rhok2 = r2c(source2)
        dx2 = lpt1(rhok2, q, pm)

        return dict(dx1=dx1, dx2=dx2)

