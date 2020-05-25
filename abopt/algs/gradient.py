from abopt.base import Optimizer
from abopt.linesearch import backtrace
from abopt.linesearch import nullsearch

class LineSearchGradientDescent(Optimizer):

    optimizer_defaults = {
        'maxiter' : 100000,
        'conviter' : 1,
        'linesearch' : backtrace,
        'linesearchiter' : 100,
    }

    def start(self, problem, state, x0):
        prop = Optimizer.start(self, problem, state, x0)
        prop.rate = 1.0
        return prop

    def accept(self, problem, state, prop):
        state.rate = prop.rate
        Optimizer.accept(self, problem, state, prop)

    def propose(self, problem, state):
        mul = problem.vs.mul

        z = mul(state.Pg, 1 / state.Pgnorm)

        prop, r1 = self.linesearch(problem, state, z,
            rate=state.rate * 2.0,
            maxiter=self.linesearchiter)

        prop.rate = r1
        return prop

class GradientDescent(Optimizer):
    """ Simple Gradient Descent without linear search. """
    optimizer_defaults = {
        'maxiter' : 100000,
        'conviter' : 1,
        'rate' : 1,
    }

    def start(self, problem, state, x0):
        prop = Optimizer.start(self, problem, state, x0)
        return prop

    def accept(self, problem, state, prop):
        Optimizer.accept(self, problem, state, prop)

    def propose(self, problem, state):
        mul = problem.vs.mul

        prop, r1 = nullsearch(problem, state, state.Pg,
            rate=self.rate,
            maxiter=1)
        return prop
