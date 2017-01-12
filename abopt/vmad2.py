from __future__ import print_function
import numpy
import warnings
def microcode(ain, aout):
    class decorator(object):
        def __init__(self, function):
            self.function = function
            self.instance = None
            self.ain = ain
            self.aout = aout
            self.argnames = function.__code__.co_varnames[1:function.__code__.co_argcount]
            for an in ain:
                if not an in self.argnames:
                    raise ValueError(
        "argument `%s` of ain in microcode decorator is not declared by function `%s`"
                           % (an, str(self.function))
                    )
            self.gradient = NotImplemented

        def grad(self, function):
            gout = ['_' + a for a in ain]
            gin  = ['_' + a for a in aout]

            self.gradient = microcode(gin, gout)(function)
            # allow the gradient with the same name as the original function.
            return self

        def __get__(self, vm, owner):
            if vm is not None:
                def method(**kwargs):
                    vm.append(self, kwargs)
                return method
            return self

        def __call__(self, **kwargs):
            self.instance.append(self, kwargs)

        def __repr__(self):
            return self.function.__name__

        def invoke(self, instance, frontier, kwargs, tape, monitor=None):
            din = {}

            # copy in all arguments
            for an in self.argnames:
                din[an] = kwargs[an]

            # replace argument with variable name
            # then fetch it from the frontier.
            vin = []
            for an in self.ain:
                vn = kwargs.get(an, an)
                din[an] = frontier[vn]
                vin.append(vn)
            if tape is not None:
                for an in self.aout:
                    vn = kwargs.get(an, an)
                    if vn in vin and an in din:
                        # overwriting, need backup
                        din[an] = instance.copy_var(din[an])

                tape.append(self, kwargs, din)

            vin = [ din[an] for an in self.argnames ]

            out = self.function(instance, *vin)

            # zip the output arguments
            if len(self.aout) == 1: out = [out]
            dout = {}
            for an, value in zip(self.aout, out):
                dout[an] = value

            if monitor:
                monitor(self, din, dout)

            r = {}
            for an in self.aout:
                vn = kwargs.get(an, an)
                r[vn] = dout[an]
            return r

    return decorator

class Tape(list):
    def __init__(self):
        list.__init__(self)
        self.init = {}
        self._refcount = {}

    def __str__(self):
        def format(code, kwargs, d):
            r = str(code)
            r += ' '
            r += str(kwargs)
            r += ' '
            r += ', '.join([ '%s(%08X) : %s ' % (name, id(value), str(value)[:17])
                    for name, value in d.items()])
            return r
        r = '-- Inputs ----\n'
        r += '\n'.join([format(code, kwargs, d) for code, kwargs, d in self ])
        r += '\n'
        r += '-- Refcounts ----\n'
        r += ' '.join(["%08X : %d" % refcount for refcount in self._refcount.items()])
        return r

    def get_uid(self, value):
        return id(value)

    def get_refcount(self, value):
        uid = id(value)
        return self._refcount.get(uid)

    def append(self, code, kwargs, din):
        """ add a record to the tape. Record is the argument name and the value. """
        for an, value in din.items():
            uid = id(value)
            self._refcount[uid] = self._refcount.get(uid, 0) + 1
        list.append(self, (code, kwargs, din))


class VM(object):
    def copy_var(self, a):
        return 1.0 * a

    def __init__(self):
        self._microcodes = []

    @property
    def microcodes(self):
        return self._microcodes

    def append(self, code, kwargs):
        self._microcodes.append((code, kwargs))

    @microcode(ain=['x'], aout=['y'])
    def func(self, x, factor):
        y = factor * x
        return y

    @func.grad
    def gfunc(self, x, factor, _y):
        _x = factor * _y
        return _x

    @microcode(ain=['a', 'b'], aout=['c'])
    def add(self, a, b):
        return a + b

    @add.grad
    def gadd(self, _c, _a, _b):
        return _c, _c

    def _gc(self, frontier, future, vout):
        """ remove variables that are never used again """
        used = []
        used.extend(vout)
        for code, kwargs in future:
            for an in code.ain:
                vn = kwargs.get(an, an)
                used.append(vn)

        used = set(used)
        for vn in list(frontier.keys()):
            if vn not in used: frontier.pop(vn)

    def _terminate(self, future, vout):
        """ No variables in vout are mentioned in the future, we can terminate. """
        used = []
        for code, kwargs in future:
            for an in code.aout:
                vn = kwargs.get(an, an)
                used.append(vn)

        used = set(used)
        for vn in vout:
            if vn in used: return False
        return True

    def compute(self, vout, init, tape=None, monitor=None):
        frontier = {}
        frontier.update(init)
        if tape is not None:
            tape.init.update(init)

        started = False
        for i, (code, kwargs) in enumerate(self.microcodes):
            r = code.invoke(self, frontier, kwargs, tape, monitor)
            frontier.update(r)
            future = self.microcodes[i+1:]
            self._gc(frontier, future, vout)
            if self._terminate(future, vout):
                break

        if not isinstance(vout, (tuple, list)):
            r = frontier[vout]
        else:
            r = [frontier[vn] for vn in vout]
        return r

    def gradient_of(self, tape, add):
        """ set up the VM as the gradient of the objective VM.
            with tape and a microcode for adding partial gradients.
        """
        occurances = {}

        for code, kwargs, record in tape[::-1]:
            din = {}

            din.update(kwargs)
            din.update(record)

            for an in code.aout:
                din['_' + an] = '_' + kwargs.get(an, an)

            for an in code.ain:
                value = record[an]
                uid = tape.get_uid(value)
                oc = occurances.get(uid, 0)
                din['_' + an] = '_' + kwargs.get(an, an)
                if oc > 0:
                    din['_' + an] += '_%d' % oc

            self.append(code.gradient, din)

            # now append a reduction operation for the partial gradients
            for an in code.ain:
                value = record[an]
                uid = tape.get_uid(value)
                oc = occurances.get(uid, 0)

                if oc > 0:
                    reduceto = '_' + kwargs.get(an, an)
                    partial = din['_' + an]
                    din = {}
                    din[add.ain[0]] = reduceto
                    din[add.ain[1]] = partial
                    din[add.aout[0]] = reduceto
                    self.append(add, din)

                occurances[uid] = oc + 1

        return self

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

VM.Zero = Zero
VM.microcode = microcode

vm = VM()

vm.func(x='a', y='a1', factor=10)
vm.add(a='a1', b='a', c='a')
vm.func(x='b', y='b', factor=2)
vm.add(a='a', b='b', c='c')
vm.func(x='c', y='d', factor=2)
tape = Tape()
vm.compute('c', init={'a' : numpy.array(10), 'b' : numpy.array(1)}, tape=tape, monitor=print)
print(tape)
print(tape._refcount)

gvm = VM()
gvm.gradient_of(tape, add=VM.add)
print(gvm.microcodes)
print(gvm.compute(['_a', '_b'], init={'_c' : numpy.array(1)}, monitor=print))
