from abopt.base import Optimizer
from abopt.linesearch import backtrace

class GradientDescent(Optimizer):

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

    def move(self, problem, state, prop):
        state.rate = prop.rate
        Optimizer.move(self, problem, state, prop)

    def single_iteration(self, problem, state):
        mul = problem.vs.mul

        z = mul(state.Pg, 1 / state.Pgnorm)

        prop, r1 = self.linesearch(problem, state, z, state.rate * 2, maxiter=self.linesearchiter)

        prop.rate = r1
        return prop
