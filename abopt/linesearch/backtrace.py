from abopt.base import Proposal

def backtrace(problem, state, z, rate, maxiter, c=1e-5, tau=0.5):
    vs = problem.vs

    addmul = vs.addmul
    dot = vs.dot

    zz = dot(z, z)
    zg = dot(z, state.Pg) / zz ** 0.5

    if zg < 0.0: #1 * state.Pgnorm:
        return None, None

    Px1 = addmul(state.Px, z, -rate)
    prop = Proposal(problem, Px=Px1, z=z).complete_y(state)

    i = 0
    propmin = prop
    ratemin = rate
    while i < maxiter:
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
        prop = Proposal(problem, Px=Px1, z=z).complete_y(state)
        i = i + 1
    return None, None

