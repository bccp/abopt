from abopt.trustregion import cg_steihaug
from abopt.vectorspace import real_vector_space

def test_cg_steigaug():
    import numpy
    Hessian = numpy.diag([1, 2, 3, 4])
    g = numpy.zeros(4) + 0.5

    Delta = 10000.
    rtol = 1e-8
    def Bvp(v):
        return Hessian.dot(v)

    def monitor(it, rho0, r0, d0, z0):
        print(it, rho0, r0, d0, z0)

    cg_steihaug(real_vector_space, Bvp, g, Delta, rtol, monitor=monitor)

