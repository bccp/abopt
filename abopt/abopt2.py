# expose optimization algorithms
from .algs.lbfgs import LBFGS
from .algs.gradient import GradientDescent
from .algs.newton import DirectNewton
from .algs.trustregion import TrustRegionCG

# expose common vector spaces
from .vectorspace import real_vector_space
from .vectorspace import complex_vector_space

# providing base classes here
# for external subclassing

from .base import VectorSpace
from .base import State
from .base import Preconditioner
from .base import Problem

def minimize(optimizer, objective, gradient, x0, hessian_vector_product=None,
    monitor=None, vs=real_vector_space, precond=None):

    problem = Problem(objective, gradient, hessian_vector_product=hessian_vector_product, vs=vs, precond=precond)

    return optimizer.minimize(problem, x0, monitor=monitor)
