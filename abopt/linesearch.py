from .abopt2 import Proposal

def backtrace(problem, state, z, rate, c=1e-5, tau=0.5):
    vs = problem.vs

    addmul = vs.addmul
    dot = vs.dot

    zz = dot(z, z)
    zg = dot(z, state.Pg) / zz ** 0.5

    if zg < 0.0: #1 * state.Pgnorm:
        return None, None

    Px1 = addmul(state.Px, z, -rate)
    prop = Proposal(problem, Px=Px1).complete_y(state)

    i = 0
    propmin = prop
    ratemin = rate
    while i < 100:
        # print('rate', rate, 'y', state.y, 'y1', y1, 'x', state.Px, 'x1', Px1, 'z', z)

        # watch out : do not check convergence ; avoid jumping too far to the other side
        # sufficient descent

        if prop.y < propmin.y:
            propmin = prop
            ratemin = rate
        if prop.y < state.y and abs(prop.y - state.y) >= abs(rate * c * zg):
            # sufficient
            return prop, rate

        rate *= tau
        Px1 = addmul(state.Px, z, -rate)
        prop = Proposal(problem, Px=Px1).complete_y(state)
        i = i + 1
    return None, None

def exact(problem, state, z, rate, c=0.5):
    vs = problem.vs
    addmul = vs.addmul
    dot = vs.dot

    znorm = dot(z, z) ** 0.5

    from scipy.optimize import minimize_scalar

    Px1 = addmul(state.Px, z, - rate)
    prop = Proposal(problem, Px=Px1).complete_y(state)

    best = [prop, 1.0]

    def func(tau):
        if tau == 0: return state.y

        Px1 = addmul(state.Px, z, -tau * rate)
        prop = Proposal(problem, Px=Px1).complete_y(state)

        if prop.y < best[0].y:
            best[0] = prop
            best[1] = tau

        if prop.y < state.y and abs(prop.y - state.y) >= abs(c * znorm):
            raise StopIteration

        return prop.y

    try:
        r = minimize_scalar(func, (0, 1), bounds=(0, 1), method='brent', options={'maxiter':10}, )

        if not r.success or r.fun >= state.y:
            raise StopIteration

        Px1 = addmul(state.Px, z, -r.x * rate)
        return Proposal(problem, Px=Px1, y=r.fun), r.x * rate

    except StopIteration as e:
        return best[0], best[1] * rate

from .scipywolfe2 import scalar_search_wolfe2

def minpack(problem, state, z, rate, c1=1e-4, c2=0.9, amax=50):
    """"
    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].

    """
    vs = problem.vs
    addmul = vs.addmul
    dot = vs.dot

    def phi(alpha):
        state.fev = state.fev + 1
        Px1 = addmul(state.Px, z, -alpha)
        x1 = problem.precond.vQp(Px1)
        y1 = problem.f(x1)
        # print('phi', -alpha, y1, state.y)
        return y1

    Pgval = [(state.g, state.Pg), (state.x, state.Px)]

    def derphi(alpha):
        state.gev = state.gev + 1
        Px1 = addmul(state.Px, z, -alpha)
        x1 = problem.Px2x(Px1)
        g1 = problem.g(x1)
        Pg1 = problem.g2Pg(g1)
        Pgval[0] = (g1, Pg1)
        Pgval[1] = (x1, Px1)
    #    print('derphi', x1, g1)
        return -dot(Pg1, z)

    derphi0 = -dot(state.Pg, z)

    y0 = state.y_[-1]
    if len(state.y_) > 1:
        y0_prev = state.y_[-2]
    else:
        y0_prev = None

    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
            phi, derphi, y0, y0_prev, derphi0, c1, c2, amax * rate)

    if derphi_star is None:
        #raise ValueError('The line search algorithm did not converge')
        return None, None
    else:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        (g1, Pg1), (x1, Px1) = Pgval
        prop = Proposal(problem, g=g1, Pg=Pg1, x=x1, Px=Px1, y=phi_star)
        return prop, alpha_star

