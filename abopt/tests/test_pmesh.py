from __future__ import print_function
from abopt.vmad2 import CodeSegment, Engine, statement, programme, ZERO, logger

from numpy.testing import assert_raises, assert_array_equal, assert_allclose
from numpy.testing.decorators import skipif
import numpy
import logging

logger.setLevel(level=logging.WARNING)

try:
    from abopt.engines.pmesh import ParticleMeshEngine, ParticleMesh, RealField, ComplexField
    pm = ParticleMesh(BoxSize=1.0, Nmesh=(4, 4), dtype='f8')
except ImportError:
    pm = None

@skipif(pm == None)
def test_compute():
    def transfer(k): return 2.0
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='r', complex='c')
    code.transfer(complex='c', tf=transfer)
    code.c2r(complex='c', real='r')
    code.norm(field='r', r='sum', metric=None)

    field = pm.generate_whitenoise(seed=1234).c2r()

    norm = code.compute('sum', init={'r': field})
    assert_allclose(norm, field.cnorm() * 4)

@skipif(pm == None)
def test_gradient():
    def transfer(k):
        k2 = sum(ki **2 for ki in k)
        k2[k2 == 0] = 1.0
#        return 1 / k2
        return 2.0
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='r', complex='c')
    code.transfer(complex='c', tf=transfer)
    code.c2r(complex='c', real='r')
    code.norm(field='r', r='sum', metric=None)

    field = pm.generate_whitenoise(seed=1234).c2r()

    norm = code.compute('sum', init={'r': field})
    assert_allclose(norm, field.cnorm() * 4)
    norm, _r = code.compute_with_gradient(('sum', '_r'), init={'r': field}, ginit={'_sum': 1.0})
    assert_allclose(_r, field * 4 * 2)

@skipif(pm == None)
def test_force():
    def transfer(k): return 2.0
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    code.r2c(real='r', complex='c')
    code.force(density='c', s='s', force='force')

    s = engine.q * 0.0 + 0.1
    field = pm.generate_whitenoise(seed=1234).c2r()

    cshape = engine.pm.comm.allreduce(engine.q.shape[0]), engine.q.shape[1]

    check_grad(code, 'force', 's', init={'r': field, 's': s}, eps=1e-4,
                cshape=cshape, cperturb=perturb_pos, cget=get_pos,
                rtol=1e-2)

    check_grad(code, 'force', 'r', init={'r': field, 's': s}, eps=1e-4,
                cshape=field.cshape, cperturb=perturb_field, cget=get_field,
                rtol=1e-2)

@skipif(pm == None)
def test_paint():
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    s = numpy.random.uniform(size=engine.q.shape) * 0.1

    cshape = engine.pm.comm.allreduce(engine.q.shape[0]), engine.q.shape[1]
    code.decompose(s='s', layout='layout')
    code.paint(s='s', mesh='density', layout='layout')

    check_grad(code, 'density', 's', init={'s': s}, eps=1e-4,
                cshape=cshape, cperturb=perturb_pos, cget=get_pos,
                rtol=1e-2)

@skipif(pm == None)
def test_readout():
    engine = ParticleMeshEngine(pm)
    code = CodeSegment(engine)
    s = numpy.random.uniform(size=engine.q.shape) * 0.1
    cshape = engine.pm.comm.allreduce(engine.q.shape[0]), engine.q.shape[1]

    field = pm.generate_whitenoise(seed=1234).c2r()

    code.decompose(s='s', layout='layout')
    code.readout(s='s', mesh='density', layout='layout', value='value')

    check_grad(code, 'value', 's', init={'density' : field, 's': s}, eps=1e-4,
                cshape=cshape, cperturb=perturb_pos, cget=get_pos,
                rtol=1e-2)

    check_grad(code, 'value', 'density', init={'density' : field, 's': s}, eps=1e-4,
                cshape=field.cshape, cperturb=perturb_field, cget=get_field,
                rtol=1e-2)

def check_grad(code, yname, xname, init, eps, cshape, cperturb, cget, rtol):
    code = code.copy()
    code.to_scalar(x=yname, y='y')

    y, tape = code.compute('y', init=init, return_tape=True)
    gradient = code.gradient(tape)
    _x = gradient.compute('_' + xname, init={'_y' : 1.0})

    center = init[xname]
    init2 = init.copy()
    for index in numpy.ndindex(*cshape):
        x1 = cperturb(code.engine, center, index, eps)
        x0 = cperturb(code.engine, center, index, -eps)
        analytic = cget(code.engine, _x, index)
        init2[xname] = x1
        y1 = code.compute('y', init2)
        init2[xname] = x0
        y0 = code.compute('y', init2)
        #print(y1, y0, y1 - y0, get_pos(code.engine, _x, index) * 2 * eps)
        assert_allclose(y1 - y0, get_pos(code.engine, _x, index) * 2 * eps, rtol=rtol)


def get_field(engine, real, index):
    return real.cgetitem(index)

def perturb_field(engine, real, index, eps):
    old = real.cgetitem(index)
    r1 = real.copy()
    r1.csetitem(index, old + eps)
    return r1

def perturb_pos(engine, pos, ind, eps):
    comm = engine.pm.comm
    pos = pos.copy()
    start = sum(comm.allgather(pos.shape[0])[:comm.rank])
    end = sum(comm.allgather(pos.shape[0])[:comm.rank + 1])
    if ind[0] >= start and ind[0] < end:
        old = pos[ind[0] - start, ind[1]]
        coord = pos[ind[0]-start].copy()
        pos[ind[0] - start, ind[1]] = old + eps
        new = pos[ind[0] - start, ind[1]]
    else:
        old = 0
        new = 0
        coord = 0
    diff = comm.allreduce(new - old)

    return pos

def get_pos(engine, pos, ind):
    comm = engine.pm.comm
    start = sum(comm.allgather(pos.shape[0])[:comm.rank])
    end = sum(comm.allgather(pos.shape[0])[:comm.rank + 1])
    if ind[0] >= start and ind[0] < end:
        old = pos[ind[0] - start, ind[1]]
    else:
        old = 0
    return comm.allreduce(old)
