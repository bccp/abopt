from abopt.base import Proposal

def exact(problem, state, z, rate, maxiter, c=0.5):
    vs = problem.vs
    addmul = vs.addmul
    dot = vs.dot

    znorm = dot(z, z) ** 0.5

    from scipy.optimize import minimize_scalar

    Px1 = addmul(state.Px, z, - rate)
    prop = Proposal(problem, Px=Px1, z=z).complete_y(state)

    best = [prop, 1.0]

    def func(tau):
        if tau == 0: return state.y

        Px1 = addmul(state.Px, z, -tau * rate)
        prop = Proposal(problem, Px=Px1, z=z).complete_y(state)

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
        return Proposal(problem, Px=Px1, y=r.fun, z=z), r.x * rate

    except StopIteration as e:
        return best[0], best[1] * rate

