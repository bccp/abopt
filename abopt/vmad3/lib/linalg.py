from abopt.vmad3 import operator
from abopt.vmad3 import modeloperator

@operator
class mul:
    ain = {'x1' : '*',
           'x2' : '*',
          }
    aout = {'y' : '*'}

    def opr(self, x1, x2):
        return dict(y = x1 * x2)

    def vjp(self, _y, x1, x2):
        return dict(_x1 = _y * x2,
                    _x2 = _y * x1)

    def jvp(self, x1_, x2_, x1, x2):
        return dict(y_ = x1_* x2 + x1 * x2_)

@operator
class copy:
    ain = {'x' : 'ndarray'}
    aout = {'y' : 'ndarray'}

    def opr(self, x, ):
        return dict(y = 1.0 * x)

    def vjp(self, _y):
        return dict(_x = 1.0 * _y)

    def jvp(self, x_):
        return dict(y_ = 1.0 * x_)

@operator
class put:
    ain = {'x' : 'ndarray',}
           'r' : 'ndarray',}
    aout = {'y' : 'ndarray'}


    def opr(self, x, r, axis, d):
        y = 1.0 * x

        shape = numpy.shape(y)

        slices = [slice() for i in range(len(shape))]
        slices[axis] = d
        slices = tuple(slices)
        y[slices] = r
        return dict(y = y)

    def vjp(self, _y, axis, d):
        opr = type(self).operator.opr

        _x = opr(_y, 0, axis, d)['y']
        _r = 1.0 * _y.take(d, axis=axis)

        return dict(_x = _x, _r =_r)

    def jvp(self, x_, r_, axis, d):
        opr = type(self).operator.opr
        r = opr(self, x_, r_, axis, d)
        return dict(y_ = r['y'])


