def UndefinedDot(a, b):
    raise NotImplementedError('Inner product operator is required but undefined')

def UndefinedCreate():
    raise NotImplementedError('Allocation operator is required but undefined')

def UndefinedAddMul(a, b, scale):
    raise NotImplementedError('AddMul operator is required but undefined')

class Optimizer(object):
    def __init__(self,
                 dot=UndefinedDot,
                 create=UndefinedCreate,
                 addmul=UndefinedAddMul):
        # FIXME: use check the function signature is correct.
        self.dot = dot
        self.create = create
        self.addmul = addmul
        self.config = {}
        self.configure()

    def configure(self, **kwargs):
        self.config.update(kwargs)

    def minimize(self, objective, gradient, x0):
        raise NotImplementedError

class State(dict):
    pass

class GradientDescent(Optimizer):
    def __init__(self, dot=UndefinedDot, create=UndefinedCreate, addmul=UndefinedAddMul):
        Optimizer.__init__(self, dot, create, addmul)

    def configure(self,
                tol=1e-6,
                gtol=1e-6,
                maxsteps=1000,
                gamma=1e-3,
                notification=None,
                **kwargs):

        c = {}
        c['tol'] = tol
        c['gtol'] = gtol
        c['maxsteps'] = maxsteps
        c['gamma'] = gamma
        c['notification'] = notification
        c.update(kwargs)
        
        Optimizer.configure(self, **c)

    def minimize(self, objective, gradient, x0):
        tol = self.config.get('tol')
        gtol = self.config.get('gtol')
        maxsteps = self.config.get('maxsteps')
        gamma = self.config.get('gamma')
        notification = self.config.get('notification')

        # FIXME: line search 
        step = 0
        dy = None # initial step
        y0 = objective(x0)
        while step < maxsteps:
            dx0 = gradient(x0)
            gradnorm = self.dot(dx0, dx0) ** 0.5
            state = State(x=x0, y=y0, dy=dy, gradient=dx0, gradnorm=gradnorm, step=step)
            notification(state)

            if gradnorm < gtol: break
            if dy is not None and dy < tol: break

            # move to the next point
            x1 = self.addmul(x0, dx0, -gamma)
            y1 = objective(x1)
            dy = abs(y0 - y1)
            x0 = x1
            y0 = y1
            step = step + 1

        return state

