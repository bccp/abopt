def default_addmul(a, b, s):
    if s == 0:
        return 1.0 * a # always a new instance is created
    return a + b * s

def default_dot(a, b):
    if hasattr(a, 'dot'):
        return a.dot(b)
    try:
        return sum(a * b)
    except TypeError:
        return float(a * b)

class Optimizer(object):
    def __init__(self,
                 addmul=default_addmul,
                 dot=default_dot,
                 ):
        """
            Parameters
            ----------
            addmul : callable
                addmul(a, b, s) returns a + b * s as a new vector from vector a, b and a Python scalar s.
                when s is 0 (not 0.0), it returns a copy of a, serving as a constructor of new vectors.

            dot : callable
                dot(a, b) returns the inner product of vectors a and b as a Python scalar
        """
        # FIXME: use check the function signature is correct.
        self.dot = dot
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

    def configure(self,
                tol=1e-6,
                gtol=1e-6,
                maxsteps=1000,
                gamma=1e-3,
                monitor=None,
                **kwargs):

        c = {}
        c['tol'] = tol
        c['gtol'] = gtol
        c['maxsteps'] = maxsteps
        c['gamma'] = gamma
        c['monitor'] = monitor
        c.update(kwargs)
        
        Optimizer.configure(self, **c)

    def minimize(self, objective, gradient, x0):
        tol = self.config.get('tol')
        gtol = self.config.get('gtol')
        maxsteps = self.config.get('maxsteps')
        gamma = self.config.get('gamma')
        monitor = self.config.get('monitor')

        # FIXME: line search 
        step = 0
        dy = None # initial step
        y0 = objective(x0)
        while step < maxsteps:
            dx0 = gradient(x0)
            gradnorm = self.dot(dx0, dx0) ** 0.5
            state = State(x=x0, y=y0, dy=dy, gradient=dx0, gradnorm=gradnorm, step=step)
            monitor(state)

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

