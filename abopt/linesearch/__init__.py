from abopt.base import Proposal

# line search methods:
from .backtrace import backtrace
from .minpack import minpack
from .exact import exact

def simpleregulator(problem, state, z):
    # In LBFGS, the purpose of GD is to estimate the Hessian,
    # thus we do not want to move too far yet
    # limit it to 10 x of the original proposal
    dot = problem.vs.dot
    znorm = dot(z, z) ** 0.5

    rmax = 1.0

    if state.Pxnorm != 0:
        rmax = min(10 * state.Pxnorm / znorm, rmax)

    return rmax

