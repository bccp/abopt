class Optimizer(object):
    def __init__(self,
                 dot,
                 create,
                 addmul):
        # FIXME: use check the function signature is correct.
        self.dot = dot
        self.create = create
        self.addmul = addmul

    def configure(self, **kwargs):
        self.config.update(kwargs)

    def minimize(self, objective, gradient, x0):
        raise NotImplementedError

class Result(object):
    def __init__(self, xs, ys, gradient):
        self.xs = xs
        self.ys = ys
        self.gradient = gradient

class GradientDescent(Optimizer):
    def __init__(self, dot, create, addmul):
        Optimizer.__init__(self, dot, create, addmul)

    def configure(self,
                tol=1e-6,
                gtol=1e-6,
                maxsteps=1000,
                **kwargs):

        c = {}
        c['tol'] = tol
        c['gtol'] = gtol
        c['maxsteps'] = maxsteps
        c.update(kwargs)
        
        Optimizer.configure(**c)

    def minimize(self, objective, gradient, x0, stepsize):
        tol = self.config.get('tol')
        gtol = self.config.get('gtol')
        maxsteps = self.config.get('maxsteps')

        y0 = objective(x0)
        # FIXME: line search 
        steps = 0
        while steps < maxsteps:
            dx0 = gradient(x0)
            x1 = self.addmul(x0, dx0, stepsize)
            y1 = objective(y1)
            x0 = x1
            y0 = y1
            if abs(y0 - y1) < tol: break
            if self.dot(x1, x1) < gtol: break
            steps = steps + 1
        dx0 = gradient(x0)
        return Result(x0, y0, dx0)

