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
        return self.addmul(a, 0, 0)

    def ones_like(self, a):
        o = self.copy(a)
        return self.addmul(o, 1, 1)

    def mul(self, b, c, p=1):
        return self.addmul(0, b, c, p)

    def pow(self, c, p):
        return self.addmul(0, 1, c, p)

def addmul(a, b, c, p=1):
    """ a + b * c ** p """
    if b is 0 and c is 0:
        return 1.0 * a # always a new instance is created
    if p is not 1: c = c ** p
    if b is not 1: c = b * c
    if a is not 0: c = a + c
    return c

def dot(a, b):
    """ einsum('i,i->', a, b) """
    if hasattr(a, 'dot'):
        return a.dot(b)
    try:
        return sum(a * b)
    except TypeError:
        return float(a * b)

def c2r(a):
    # complex vector space needs numpy
    import numpy
    if numpy.isscalar(a): return a
    a = numpy.concatenate([numpy.real(a), numpy.imag(a)], axis=0)
    return a

def r2c(a):
    h = a.shape[0] // 2
    return a[:h] + a[h:] * 1j

def caddmul(a, b, c, p=1):
    a = c2r(a)
    b = c2r(b)
    c = c2r(c)
    return r2c(addmul(a, b, c, p))

def cdot(a, b):
    return dot(c2r(a), c2r(b))

real_vector_space = VectorSpace(addmul=addmul, dot=dot)
complex_vector_space = VectorSpace(addmul=caddmul, dot=cdot)

__all__ = ['VectorSpace', 'real_vector_space', 'complex_vector_space']
