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
class VectorSpace(object):
    def __init__(self, addmul=None, dot=None):
        if addmul:
            self.addmul = addmul
        if dot:
            self.dot = dot

    def copy(self, a):
        r = self.addmul(0, a, 1)
        assert type(r) is type(a)
        return r

    def ones_like(self, b):
        r = self.addmul(1, b, 0)
        assert type(r) is type(b)
        return r

    def zeros_like(self, b):
        r = self.addmul(0, b, 0)
        assert type(r) is type(b)
        return r

    def mul(self, b, c, p=1):
        return self.addmul(0, b, c, p)

    def pow(self, c, p):
        i = self.ones_like(c)
        return self.addmul(0, i, c, p)

    def addmul(self, a, b, c, p=1):
        """ Defines the addmul operation.

            either subclass this method or supply a method in the constructor, __init__

            addmul(a, b, c, p) := a + b * c ** p

            The result shall be a vector like b.

            b is always a vector for this VectorSpace; though be aware
            that there can be multiple valid Python types defined on the same
            VectorSpace. For example, particle positions are straight numpy.ndarray,
            An overdensity field may be a ComplexField or a RealField object.
        """

        raise NotImplementedError

    def dot(self, a, b):
        """ defines the inner product operation. 

            dot(a, b) := a @ b

            The result shall be a scalar floating point number.

            a and b are always vector for this VectorSpace, and are guarenteed
            to be of the same Python type -- if not there is a bug from upstream
            callers.

        """
        raise NotImplementedError

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
            return a.dot(b)
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
