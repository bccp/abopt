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

def nullsearch(problem, state, z, rate, maxiter):
    """ A null line search that does not change the rate;

        Useful for e.g. stochastic gradient descent.

    """

    Px1 = addmul(state.Px, z, -rate)
    prop = Proposal(problem, Px=Px1, z=z).complete_y(state)
    return prob, rate
