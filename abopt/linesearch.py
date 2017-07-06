def backtrace(vs, problem, state, z, rate, c=1e-5, tau=0.5):
    addmul = vs.addmul
    dot = vs.dot

    objective = problem.objective

    zz = dot(z, z)
    zg = dot(z, state.g) / zz ** 0.5

    if zg < 0.0: #1 * state.gnorm:
        return None, None, None, None

    x1 = addmul(state.x, z, -rate)
    y1 = objective(x1)
    state.fev = state.fev + 1
    i = 0
    ymin = state.y
    xmin = state.x
    ratemin = rate
    while i < 100:
        #print('rate', rate, 'y', state.y, 'y1', y1, 'x', state.x, 'x1', x1, 'z', z)

        # watch out : do not check convergence ; avoid jumping too far to the other side
        # sufficient descent

        if y1 < ymin:
            ymin = y1
            xmin = x1
            ratemin = rate
        if y1 < state.y and abs(y1 - state.y) >= abs(rate * c * zg):
            return xmin, ymin, None, rate

        rate *= tau
        x1 = addmul(state.x, z, -rate)
        y1 = objective(x1)
        state.fev = state.fev + 1
        i = i + 1
    return None, None, None, None

def exact(vs, problem, state, z, rate, c=0.5):
    addmul = vs.addmul
    dot = vs.dot

    objective = problem.objective

    znorm = dot(z, z) ** 0.5

    from scipy.optimize import minimize_scalar

    best = [state.x, state.y, 1.0]

    def func(tau):
        if tau == 0: return state.y

        x1 = addmul(state.x, z, -tau * rate)
        state.fev = state.fev + 1
        y1 = objective(x1)
        if y1 < best[1]:
            best[0] = x1
            best[1] = y1
            best[2] = tau

        if y1 < state.y and abs(y1 - state.y) >= abs(c * znorm):
            raise StopIteration

        return y1

    try:
        r = minimize_scalar(func, (0, 1), bounds=(0, 1), method='brent', options={'maxiter':10}, )

        if not r.success or r.fun >= state.y:
            raise StopIteration

        x1 = addmul(state.x, z, -r.x * rate)
        return x1, r.fun, None, r.x * rate

    except StopIteration as e:
        x1, y1, tau = best
        return x1, y1, None, tau * rate

from .scipywolfe2 import scalar_search_wolfe2

def minpack(vs, problem, state, z, rate, c1=1e-4, c2=0.9, amax=50):
    """"
    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions.  See Wright and Nocedal, 'Numerical Optimization',
    1999, pg. 59-60.

    For the zoom phase it uses an algorithm by [...].

    """
    addmul = vs.addmul
    dot = vs.dot

    def phi(alpha):
        state.fev = state.fev + 1
        x1 = addmul(state.x, z, -alpha)
        y1 = problem.objective(x1)
        # print('phi', -alpha, y1, state.y)
        return y1

    gval = [state.g]

    def derphi(alpha):
        state.gev = state.gev + 1
        x1 = addmul(state.x, z, -alpha)
        g1 = problem.gradient(x1)
        gval[0] = g1
    #    print('derphi', x1, g1)
        return -dot(g1, z)

    derphi0 = -dot(state.g, z)

    y0 = state.y_[-1]
    if len(state.y_) > 1:
        y0_prev = state.y_[-2]
    else:
        y0_prev = None

    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
            phi, derphi, y0, y0_prev, derphi0, c1, c2, amax * rate)

    if derphi_star is None:
        #raise ValueError('The line search algorithm did not converge')
        return None, None, None, None
    else:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derphi_star = gval[0]

    x1 = addmul(state.x, z, -alpha_star)
    return x1, phi_star, derphi_star, alpha_star

