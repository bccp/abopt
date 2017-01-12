def default_mul(a, s):
    if s == 0:
        return 0.0 * a # always a new instance is created
    return a * s

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
                 mul=default_mul,
                 addmul=default_addmul,
                 dot=default_dot,
                 ):
        """
            Parameters
            ----------
            mul : callable
                mul(a, s) returns a * s as a new vector from vector a and a Python scalar s.
            addmul : callable
                addmul(a, b, s) returns a + b * s as a new vector from vector a, b and a Python scalar s.
                when s is 0 (not 0.0), it returns a copy of a, serving as a constructor of new vectors.

            dot : callable
                dot(a, b) returns the inner product of vectors a and b as a Python scalar
        """
        # FIXME: use check the function signature is correct.
        self.dot = dot
        self.addmul = addmul
        self.mul = mul
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

class LBFGS(Optimizer):

    def configure(self,
                tol=1e-6,
                gtol=1e-6,
                maxsteps=1000,
                m = 10,
                monitor=None,
                **kwargs):

        c = {}
        c['tol'] = tol
        c['gtol'] = gtol
        c['maxsteps'] = maxsteps
        c['m'] = m
        c['monitor'] = monitor
        c.update(kwargs)

        Optimizer.configure(self, **c)

    def minimize(self, objective, gradient, x0):
        tol = self.config.get('tol')
        gtol = self.config.get('gtol')
        maxsteps = self.config.get('maxsteps')
        m = self.config.get('m')
        monitor = self.config.get('monitor')

        assert tol > 0
        assert gtol >= 0

        def mycopy(x):
            return self.addmul(x, x, 0)

        it = 0
        x = mycopy(x0)
        val = objective(x)
        g = gradient(x)
        gradnorm = self.dot(g, g) ** 0.5

        rho = [0.0] * m
        alpha = [0.0] * m
        zero = self.mul(x0, 0.0)
        s = [zero] * m
        y = [zero] * m

        xprev = mycopy(x)
        gprev = mycopy(g)
        H0k = 1.0
        step = 0
        rate = 1.0

        function_eval = 0

        converged_iters = 0

        dy = None
        use_steepest_descent = False
        converged_state = "NO"

        state = State(x = x, y = val, dy = dy, gradient = g, gradnorm = gradnorm, step = it, function_evaluations = function_eval, steepest_descent = use_steepest_descent, converged = converged_state)
        monitor(state)

        while True:
            q = mycopy(g)
            thism = min(m, step)

            for i in range(m):
                dotproduct = self.dot(s[i], q)
                alpha[i] = rho[i] * dotproduct
                q = self.addmul(q, y[i], -alpha[i])

            z = self.mul(q, H0k)

            for i in range(m - 1, -1, -1):
                dotproduct = self.dot(y[i], z)
                beta = rho[i] * dotproduct
                z = self.addmul(z, s[i], alpha[i] - beta)

            use_steepest_descent = False
            znorm = self.dot(z, z) ** 0.5
            zg = 0.0 if znorm == 0 else self.dot(z, g) / znorm
            zg_grannorm = 0.0 if gradnorm == 0 else zg / gradnorm

            if zg_grannorm < 0.01:
                z = mycopy(g)
                zg = 1.0
                use_steepest_descent = True

            oldval = val

            rate = 1.0 / gradnorm if (it == 0 or use_steepest_descent) else 1.0

            # doing only backtracking line search
            # FIXME: implement more-thuente
            tau = 0.5
            c = 1e-5
            search_x = self.addmul(x, z, -rate)
            newval = objective(search_x)
            function_eval += 1
            while True:
                valmax = max(abs(val), abs(newval))
                if abs(val - newval) / max(valmax, 1.0) < tol and val >= newval:
                    break
                if val - newval >= rate * c * zg:
                    break

                rate *= tau
                search_x = self.addmul(x, z, -rate)
                newval = objective(search_x)
                function_eval += 1

            # now move
            x = mycopy(search_x)
            val = newval
            g = gradient(x)
            gradnorm = self.dot(g, g) ** 0.5

            # end of backtracking line search

            it += 1
            dy = abs(val - oldval)
            valmax = max(abs(val), abs(oldval))
            ratio = dy / max(valmax, 1.0)
            min_iter = 10

            if ratio < tol and it >= min_iter:
                converged_iters += 1
                if converged_iters >= 3:
                    converged_state = "YES: Tolerance achieved"
                    break
            else:
                converged_iters = 0

            if gradnorm < gtol:
                converged_state = "YES: Gradient tolerance achieved"
                break

            # move everything down
            del s[-1]
            del y[-1]
            del rho[-1]
            s.insert(0, self.addmul(x, xprev, -1))
            y.insert(0, self.addmul(g, gprev, -1))
            ys = self.dot(s[0], y[0])
            yy = self.dot(y[0], y[0])

            if ys == 0.0 or yy == 0.0:
                converged_state = "NO: LBFGS didn't move for some reason, QUITTING"
                break

            rho.insert(0, 1.0 / ys)
            H0k = ys / yy

            xprev = mycopy(x)
            gprev = mycopy(g)

            if it > maxsteps:
                converged_state = "NO: but maximum number of iterations reached. QUITTING."
                break

            state = State(x = x, y = val, dy = dy, gradient = g, gradnorm = gradnorm, step = it, function_evaluations = function_eval, steepest_descent = use_steepest_descent, converged = converged_state)
            monitor(state)

        # update the state one last time
        state = State(x = x, y = val, dy = dy, gradient = g, gradnorm = gradnorm, step = it, function_evaluations = function_eval, steepest_descent = use_steepest_descent, converged = converged_state)
        monitor(state)

        return state

