class Optimizer(object):
    def __init__(self,
                 dot,
                 create,
                 addmul):
        # FIXME: use check the function signature is correct.
        self.dot = dot
        self.create = create
        self.addmul = addmul

    def minimize(self, objective, gradient, x0):
        raise NotImplementedError

class Result(object):
    def __init__(self, xs, ys, gradient):
        self.xs = xs
        self.ys = ys
        self.gradient = gradient

class GradientDecent(Optimizer):
    def __init__(self, dot, create, addmul):
        Optimizer.__init__(self, dot, create, addmul)

    def minimize(self, objective, gradient, x0, stepsize, maxsteps):
        y0 = objective(x0)
        # FIXME: line search 
        steps = 0
        while steps < maxsteps:
            dx0 = gradient(x0)
            x1 = self.addmul(x0, dx0, stepsize)
            y1 = objective(y1)
            x0 = x1
            y0 = y1
            steps = steps + 1
        dx0 = gradient(x0)
        return Result(x0, y0, dx0)

