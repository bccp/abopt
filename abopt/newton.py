from .abopt2 import Optimizer, Problem, Proposal, ContinueIteration, ConvergedIteration, FailedIteration

class Newton(Optimizer):
    """
        A general Newton method.

        This method assumes the problem also knows how to compute the inverse of hessian, without actually
        inverting it.

        A line search along the newton proposal direction, starting from the newton step size
        is used.
    """

    from .linesearch import backtrace
    optimizer_defaults = {
                        'maxiter' : 1000,
                        'conviter' : 2,
                        'linesearch' : backtrace,
                        'linesearchiter' : 100,
                        }

    def single_iteration(self, problem, state):
        mul = problem.vs.mul
        dot = problem.vs.dot
        addmul = problem.vs.addmul

        state.hev = state.hev + 1
        z = problem.Phvp(state.x, state.Pg)

        prop, r1 = self.linesearch(problem, state, z, 1.0, maxiter=self.linesearchiter)

        return prop
