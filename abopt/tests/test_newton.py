from __future__ import print_function
from abopt.abopt2 import Problem, Preconditioner
from abopt.lbfgs import LBFGSHessian
from abopt.newton import Newton
from abopt.vectorspace import real_vector_space
import numpy
from scipy.optimize import rosen, rosen_der, rosen_hess_prod, rosen_hess
from scipy.linalg import inv
import pytest
from numpy.testing import assert_allclose

def rosen_inverse_hess(x):
    H = rosen_hess(x)
    return inv(H)

def test_newton():
    nt = Newton()

    problem = Problem(objective=rosen,
                      gradient=rosen_der,
                      inverse_hessian_vector_product=lambda x, v:
                            rosen_inverse_hess(x).dot(v)
                     )

    problem.atol = 1e-7 # ymin = 0

    x0 = numpy.zeros(20)
    r = nt.minimize(problem, x0, monitor=print)
    assert_allclose(r.x, 1.0, rtol=1e-4)
    assert r.converged

def test_newton_pre():
    nt = Newton()

    precond = Preconditioner(
                            Pvp=lambda x: 20 * x,
                            vPp=lambda x: 20 * x,
                            vQp=lambda x: 0.05 * x,
                            Qvp=lambda x: 0.05 * x,
                            )
    problem = Problem(objective=rosen,
                      gradient=rosen_der,
                      inverse_hessian_vector_product=lambda x, v:
                            rosen_inverse_hess(x).dot(v),
                      precond=precond
                     )

    problem.atol = 1e-7 # ymin = 0

    x0 = numpy.zeros(20)
    r = nt.minimize(problem, x0, monitor=print)
    assert_allclose(r.x, 1.0, rtol=1e-4)
    assert r.converged
