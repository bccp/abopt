from abopt.vmad3 import operator
from pmesh.pm import ParticleMesh
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
