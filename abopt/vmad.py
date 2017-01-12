import numpy
import warnings

class VM(object):

    def copy_var(self, a):
        """ override for copying a variable """
        try:
            return a.copy()
        except:
            return 1.0 * a

    def iadd_var(self, a, b):
        """ override for inplace adding to a variable """
        a[...] += b

    @staticmethod
    def inspect(record):
        return str(list((k, "0x%0X" % id(v)) for k, v in record.items()))

    @staticmethod
    def microcode(fout=[], fin=[], lout=[]):
        """ Declare a subclass member function as a microcode.

            A microcode is a function with names of input and names of output
            It always takes a frontier argument and a list of *args.

            lout is the list of 'literal' outputs (do not participate in gradients)

            >>> @VM.microcode(fin=['x'], fout=['x'])
            >>> def mul(self, frontier, factor):
            >>>    frontier['x'] = factor * frontier['x']

            The backtrace gradient microcode shall be
            >>> @mul.grad
            >>> def gmul(self, frontier, factor):
            >>>    frontier['^x'] = factor * frontier['^x']

            Notice that `^` denotes the backtraced gradient against a variable.
            The same name can be both lvalue and rvalue
        """
        def decorator(func):
            def gradient_decorator(func1):
                func1.gout = ['^' + v for v in fin]
                func1.gin  = ['^' + v for v in fout]
                func1.fin = func.fin
                func1.fout = []
                func1.lout = []
                func.g = func1
                return func1

            func.fout = fout
            func.fin = fin
            func.lout = lout
            func.grad = gradient_decorator
            return func
        # no parameters, just a plain decorator
        if hasattr(fout, "__call__"):
            func = fout
            return decorator(func)
        return decorator

    def __init__(self):
        self._microcodes = []

    def __len__(self):
        return len(self._microcodes) + 1

    @property
    def microcodes(self):
        # always append a sentinal for final connection to the output.
        r = self._microcodes + [(None, ())]
        return r

    def push(self, op, v, args):
        """ Append to the microcode list of the VM

            Use this to build complicated microcode sequences.

            >>> vm.push('mul', 3.0)
            >>> vm.push('mul', 2.0)
        """
        for name, impl in self.__class__.__dict__.items():
            if name == op:
                self._microcodes.append((impl, args))
                break 
        else:
            raise AttributeError("code %s is not found" % op)

    def compute(self, fout, init, tape=None, monitor=None):
        """
            Run the list of microcodes with `init` dictionary as input
            Record the results on `tape` (list), and return a dictionary
            contains variables named in fout (list).
            The items in the init shall support [...] or the '+' operator

            >>> vm.compute(['x'], {'x': numpy.ones(10)})

        """

        frontier = {}
        frontier.update(init)
        literals = set()

        for step, (impl, args) in enumerate(self.microcodes):
            if impl is None:
                # sentinal,
                # add the final fout into the tape
                # to properly update refcount of the input gradients

                impl = lambda self, frontier: None
                # skip the literal from fin, even if it is requested from fout.
                impl.fin = [v for v in fout if v not in literals]
                impl.fout = []
                impl.lout = []

            for name in impl.fin:
                if name in literals:
                    raise RuntimeError(
                    "An input of microcode `%s`, `%s` is marked as literal by previous steps. The machine is likely buggy." % (name, impl.__name__))

            record = {}

            for var in set(impl.fin) | literals:
                record[var] = frontier[var]

            if monitor:
                monitor(step, len(self), impl, args, VM.inspect(record))

            if tape is not None:
                # replace the frontier with a copy -- such that the one on tape is preserved.
                for var in record:
                    if var in impl.fout:
                        # save a copy of the variables for in-place operations.
                        copy = self.copy_var(record[var])
                        assert id(copy) != id(record[var])
                        frontier[var] = copy

                tape.append(record)

            literals |= set(impl.lout)
            impl(self, frontier, *args)

        if tape is not None:
            # record fout literals
            # for future gradient
            tape.append(fout)
            tape.append(literals)

        r = {}
        for name in fout:
            r[name] = frontier[name]
        return r

    @staticmethod
    def _refcount(tape):
        """ count number of references to any variables on a tape """
        d = {}
        for record in tape:
            for name, value in record.items():
                # note that all variables on the record are alive,
                # in this case id uniquely determines the values.
                uid = id(value)
                d[uid] = d.get(uid, 0) + 1
        return d

    def gradient(self, gout, ginit, tape, monitor=None):
        """ Backtrace the gradient from ginit (dict).

            tape is obtained from a prevoius call to `compute`.

            Returns a dict contains the requested gradients in gout (list).

            >>> tape = []
            >>> vm.compute(['x'], {'x': numpy.ones(10)})
            >>> vm.gradient(['^x'], {'^x', numpy.ones(10)}, tape)
        """

        sentinal = [(None, ())]

        fout = tape[-2]
        literals = tape[-1]

        refcount = self._refcount(tape[:-2])

        #for i, n in refcount.items():
            #print ('0X%X' %i, n)

        # holding the partially computed gradient of variables
        partial = {}

        frontier = {}
        for (impl, args), record in reversed(list(zip(self.microcodes, tape))):
            d = {}
            d.update(record)
            d.update(frontier)

            if impl is None:
                # sentinal -- initialization
                # pump up d with the initial and be aware of missing items
                # in ginit -- if they are in gin we later set them to zero.

                impl = lambda : None
                impl.g = lambda self, d: None
                # literals are skipped from gradients, even if they are
                # requested by fout; if we keep them here
                # we will end up having warnings about unfinished partial
                # gradients. -- this is the only path literals gets mixed.
                # otherwise the some of the microcodes has inproperly decleared
                # literals in fin.
                impl.g.gin = ['^' + v for v in fout if v not in literals]
                impl.g.gout = impl.g.gin
                d.update(ginit)

            if monitor:
                monitor('gradient', impl, 'before', VM.inspect(record))

            
            for name in impl.g.gin:
                # non existing inputs are implied to start with gradients of VM.Zero
                if name not in d:
                    d[name] = VM.Zero

            impl.g(self, d, *args)

            # reduce the result
            for name in impl.g.gout:
                vname = name[1:]
                uid = id(record[vname])
                if monitor:
                    monitor('partial gradient', name, refcount[uid], d[name])
                pg = d[name]
                if uid not in partial or partial[uid] is VM.Zero:
                    # this is the first channel, create the gradient storage
                    partial[uid] = pg
                elif pg is not VM.Zero:
                    # this is an additional channel. cummulate the gradient.
                    self.iadd_var(partial[uid], pg)
                refcount[uid] = refcount[uid] - 1

                if refcount[uid] == 0:
                    # update the frontier with the new gradients
                    # we no longer need to save it on partial since cummulation is done.
                    frontier[name] = partial.pop(uid)
                    if monitor:
                        monitor('finalized gradient', name, frontier[name])

        for i, v in partial.items():
            warnings.warn('problem: remainging partial : 0X%X %d %s' % (i, refcount[i], str(v)), RuntimeWarning)

        r = {}
        for name in gout:
            r[name] = frontier[name]
        return r

#####################
#
# ZeroType.
#
# Zero * anything = Zero; Zero + anything = anything
#
# VM.Zero is used to suppress gradient computation
# backtracing from VM.Zero shall always give VM.Zero
# regardless of the microcode function; this may 
# get more complicated if a microcode function
# has multiple outputs -- e.g. if the backtrace gradient
# of some are not VM.Zero.
#
# In case it occurs in a regular Python expression,
# these operator overides ensure we can 
# propagate VM.Zero properly.
#
# In other cases the microcode functions may need
# to special case on variables that are VM.Zero.

def ZeroType():
    """ creates a special type of ZeroType; """
    def self(self, *args): return self
    def other(self, other): return other
    def zde(self, a): raise ZeroDivisionError
    def __sub__(self, a): return -a
    def __xor__(self, a): return ~a
    def __int__(self): return 0
    def __float__(self): return 0.0
    def __round__(self): return 0
    def __array__(self): return numpy.array(0)

    dict = {}
    for name, value in locals().items():
         if name.startswith("__"): dict[name] = value

    for name in [
        "neg", "pos", "abs", "invert", "complex",
        "mul", "rmul", "matmul", "rmatmul", 
        "mod", "divmod", "div", "truediv", "floordiv",
        "pow",
        "and", "rand", "lshift", "rlshift", "rshift", "rrshift",
        "getitem", "reversed"]:
        dict["__" + name + "__"] = self

    for name in [
        "rmod", "rdivmod", "rdiv", "rtruediv", "rfloordiv",
        "rpow", "rsub", "rxor"]:
        dict["__" + name + "__"] = zde

    for name in [
        "add", "radd", "or", "ror"]:
        dict["__" + name + "__"] = other

    return type("ZeroType", (object,), dict)

ZeroType = ZeroType()
Zero = ZeroType()

# now add Zero to VM for easier access.
VM.Zero = Zero
