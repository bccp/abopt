"""
    Vectors.

    define addmul, which allows transporting vectors;
       and dot, which defines the inner product and distance.

    ABOPT sees complex numbers as a tuple of real numbers.
    and thus the vector space is self-dual;
    :math:`x + \lambda g` being well defined.

    `autograd` sees complex numbers as complex numbers,
    thus the vector space is not self-dual; gradient lives in the dual
    space and :math:`x + \lambda g^\dagger` is well defined.

"""
from abopt.base import VectorSpace

class RealVectorSpace(VectorSpace):
    def addmul(self, a, b, c, p=1):
        """ a + b * c ** p, follow the type of b """
        if p is not 1: c = c ** p
        c = b * c
        if a is not 0: c = c + a
        return c

    def dot(self, a, b):
        """ einsum('i,i->', a, b) """
        if hasattr(a, 'dot'):
            return (a * b).sum()
        try:
            return sum(a * b)
        except TypeError:
            return float(a * b)

# helper functions to pack and unpack complex numbers.
def _c2r(a):
    # complex vector space needs numpy
    import numpy
    # in abopt, a scalar almost always means Identity * scalar
    # it must be real; use numpy's broadcast
    if numpy.isscalar(a):
        assert numpy.imag(a) == 0
        return a
    a = numpy.concatenate([numpy.real(a), numpy.imag(a)], axis=0)
    return a


def _r2c(a):
    import numpy
    # in abopt, a scalar almost always means Identity * scalar
    # it must be real; use numpy's broadcast
    if numpy.isscalar(a): 
        assert numpy.imag(a) == 0
        return a
    h = a.shape[0] // 2
    return a[:h] + a[h:] * 1j

class ComplexVectorSpace(VectorSpace):
    def addmul(self, a, b, c, p=1):
        a = _c2r(a)
        b = _c2r(b)
        c = _c2r(c)
        return _r2c(RealVectorSpace.addmul(RealVectorSpace(), a, b, c, p))

    def dot(self, a, b):
        return RealVectorSpace.dot(RealVectorSpace(), _c2r(a), _c2r(b))

real_vector_space = RealVectorSpace()
complex_vector_space = ComplexVectorSpace()
