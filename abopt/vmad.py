from __future__ import print_function
import warnings
import functools

class MicroCode(object):
    """ An object that represents a microcode.

        Mostly a python function with some additional introspection, marking
        the input and output variables.
    """
    def __init__(self, function, ain, aout, literals):
        self.function = function
        self.ain = ain
        self.aout = aout
        self.literals = literals
        self.argnames = function.__code__.co_varnames[1:function.__code__.co_argcount]
        for an in ain:
            if not an in self.argnames:
                raise ValueError(
    "argument `%s` of ain in microcode decorator is not declared by function `%s`"
                       % (an, str(self.function))
                )
        self.gradient = NotImplemented
        functools.update_wrapper(self, function)

    def grad(self, function):
        gout = ['_' + a for a in self.ain]
        gin  = ['_' + a for a in self.aout]

        function.__name__ = "G:" + self.function.__name__
        self.gradient = microcode(gin, gout)(function)
        # allow the gradient with the same name as the original function.
        return self.gradient

    def __get__(self, instance, owner):
        """ As a class member, return the microcode,
            As an instance member of VM, returns the function as a method,
            As an instance member of Code, returns a method to add to the code.
        """
        if instance is not None:
            if isinstance(instance, Code):
                @functools.wraps(self.function)
                def method(**kwargs):
                    instance.append(self, kwargs)
                return method
            return self.function.__get__(instance, owner)
        return self

    def __repr__(self):
        return self.function.__name__

    def to_label(self, kwargs, html=False):
        if html:
            delim = '<BR ALIGN="CENTER"/>'
        else:
            delim = " "
        def format(kwargs):
            r = []
            for an in self.argnames:
                vn = kwargs.get(an, an)
                if an not in self.ain + self.aout + self.literals:
                    vn = "Ext:%s" % type(vn).__name__
                    io = 'p'
                else:
                    io = ''
                    if an in self.ain:
                        io += 'i'
                    if an in self.aout:
                        io += 'o'
                    if an in self.literals:
                        io += 'l'
                r.append("%s:%s = %s" % (io, an, vn))
            return delim.join(r)
        name = str(self)
        if html:
            name = "<b>" + name + "</b>"

        s = delim.join([name, format(kwargs)])
        if html:
            return "<" + s + ">"
        else:
            return s

    def invoke(self, vm, frontier, kwargs, tape, monitor=None):
        din = {}

        # copy in all arguments
        for an in self.argnames:
            if self.gradient is not NotImplemented and an not in self.gradient.argnames:
                # skip if thi is not used in gradient backtracing.
                pass
            if an in self.ain + self.literals:
                # replace argument with variable name
                # then fetch it from the frontier.
                vn = kwargs.get(an, an)
                if vn not in frontier:
                    raise ValueError("Argument `%s' of `%s` resolves to an undefined varabile `%s'" % (an, self, vn))
                din[an] = frontier[vn]
            else:
                # use the kwargs -- raise error if not found!
                if an not in kwargs:
                    raise ValueError("Argument `%s' of `%s' could not be bound to a keyword argument" % (an, self))
                din[an] = kwargs.get(an)

        if tape is not None:
            tape.append(self, kwargs, din)

        vin = []
        for an in self.argnames:
            data = din[an]
            if an in self.aout and an in self.ain:
                vn = kwargs.get(an, an)
                # create a copy because we will taint the values.
                # FIXME: only needed if we are recording to a tape.
                if vn in din:
                    data = vm.CopyVariable(data)
            vin.append(data)

        out = self.function(vm, *vin)

        # zip the output arguments
        if len(self.aout) == 1: out = [out]
        dout = {}
        for an, value in zip(self.aout, out):
            dout[an] = value

        r = {}
        for an in self.aout:
            vn = kwargs.get(an, an)
            r[vn] = dout[an]

        if monitor:
            monitor(self, din, dout, frontier, r)
        return r

def microcode(ain, aout, literals=[]):
    """ Declares a VM member function as a 'microcode'.
        microcode is the building block for Code objects,
        which can be computed and differentiated.

        See MicroCode. 
    """
    def decorator(func):
        return MicroCode(func, ain, aout, literals)
    return decorator

class VM(object):
    """ A virtual machine that interprets and runs Code objects
        consisting of microcodes.

        Subclass VM to add more microcodes.
        Override `CopyVariable` and `Add` to support different types of
        variables.

        Convention for gradient is to prepend `_`, e.g. `_x` is the gradient
        backtraced to `x`.

        The Each microcode carries a list of input and output arguments (
        `ain` and `aout`) that can be assigned as variable names (string).

        The rest of arguments in the function signature
        are external parameters who do not go into gradients.

        Example
        -------

        >>> vm = VM()
        >>> code = vm.code()
        >>> code.Add(a='x', b='x', c='y')
        >>> code.compute('y', {'x' : 10})

    """

    @microcode(ain=['a'], aout=['b'])
    def CopyVariable(self, a): return 1.0 * a

    @CopyVariable.grad
    def _(self, _b): return _b

    @microcode(ain=['a', 'b'], aout=['c'])
    def Add(self, a, b):
        if a is VM.Zero: return b
        if b is VM.Zero: return a
        return a + b

    @Add.grad
    def _(self, _c, _a, _b): return _c, _c

    def tape(self):
        return Tape()

    def code(self):
        """ Creates a Code object for this VM.

            Build model with this.
        """
        d = {}
        t = type(self)
        for name in dir(t):
            method = getattr(t, name)
            if isinstance(method, MicroCode):
                d[name] = method
        MyCode = type("Code%s" % (type(self).__name__), (Code, ), d)
        return MyCode(self)

    def gradient(self, tape, add=None):
        """ Create a code object that backtraces the gradient of a previously
            recorded execution in `tape`.

            The `Add` microcode (None means `type(self).Add`) reduces partial
            derivatives.

        """
        newinst = self.code()

        if add is None:
            add = type(self).Add

        occurances = {}

        def emit_add(a, b, c):
            din = {}
            din[add.ain[0]] = a
            din[add.ain[1]] = b
            din[add.aout[0]] = c
            newinst.append(add, din)

        for microcode, kwargs, record in tape[::-1]:
            din = {}

            din.update(kwargs)
            din.update(record)

            # inputs
            for an in microcode.aout:
                din['_' + an] = '_' + kwargs.get(an, an)

            for an in microcode.ain:
                # need to rename to avoid accidentally overwrite
                # a finished gradient
                din['_' + an] = '#partial#_' + kwargs.get(an, an)
                value = record[an]
                if tape.get_refcount(value) == 1:
                    # direct overwriting is OK. no partials.
                    din['_' + an] = '_' + kwargs.get(an, an)
                elif occurances.get(tape.get_uid(value), 0) == 0:
                    # first occurance, OK to overwrite?
                    # if the tape has extra operations this may overwrite a calculated gradient with
                    # a incomplete gradient.
                    din['_' + an] = '_' + kwargs.get(an, an)
                else:
                    if an in microcode.aout:
                        # rename the input.
                        emit_add("", '_' + kwargs.get(an, an), din['_' + an])

            newinst.append(microcode.gradient, din)

            # outputs
            for an in microcode.ain:
                value = record[an]
                uid = tape.get_uid(value)
                oc = occurances.get(uid, 0)
                occurances[uid] = oc + 1

                if tape.get_refcount(value) > 1:
                    reduceto = '_' + kwargs.get(an, an)
                    partial = din['_' + an]
                    if oc > 0:
                        emit_add(reduceto, partial, reduceto)
                    else:
                        # move the partial to reduceto
                        # OK to use a move because a partial is only used once.
                        # thus we ignore the op if partial is already reduceto.
                        if partial != reduceto:
                            emit_add("", partial, reduceto)
                else:
                    # result already in right place
                    pass
        for id in occurances:
            if tape._refcount[id] != occurances[id]:
                raise RuntimeError("FIXME: make this error more informative. Some gradients remain not fully computed.")
        inputs = newinst._find_inputs()
        for vin in inputs:
            newinst.defaults[vin] = VM.Zero
        return newinst

    @staticmethod
    def _add_to_graph(graph, init, list):
        """
            add a list of microcodes to a graph. The init node is duplicated as needed
            (because it may be used many times and mess up the diagram. It hurts to have
            very long edges like that.

        """
        source = {}
        dest = {}
        sans = {}
        for vn in init:
            source[vn] = "#INIT"
            dest[vn] = "#OUT"

        for i, record in enumerate(list):
            microcode, kwargs = record[0], record[1]

            graph.node(str(i) + str(id(microcode)), label=microcode.to_label(kwargs, html=True), )
            for an in microcode.ain:
                vn = kwargs.get(an, an)
                from_init = []
                if vn in source:
                    if vn == "": continue
                    if source[vn] == "#INIT":
                        from_init.append(vn)
                    else:
                        san = sans[vn]
                        dan = an
                        attrs = {}
                        if san != vn:
                            attrs['taillabel'] = san + ":"
                        if dan != vn:
                            attrs['headlabel'] = ":" + dan
                        attrs['label'] = vn
                        graph.edge(source[vn], str(i) + str(id(microcode)), **attrs)

                    dest[vn] = str(i) + str(id(microcode))

                if len(from_init) > 0:
                    graph.node(str(i) + "#INIT", label="<<b>#</b>>")
                    for vn in from_init:
                        graph.edge(str(i) + "#INIT", str(i) + str(id(microcode)), label=vn)

            for an in microcode.aout:
                vn = kwargs.get(an, an)
                source[vn] = str(i) + str(id(microcode))
                sans[vn] = an
                dest[vn] = "#OUT"

        # mark the potential output edges.
        for vn, node in dest.items():
            if node != "#OUT": continue
            graph.node(source[vn] + "#OUT", label="<<b>#</b>>")
            graph.edge(source[vn], source[vn] + "#OUT", label=vn)

    @microcode(ain=['x'], aout=['y'])
    def func(self, x, factor):
        """ this is a function """
        y = factor * x
        return y

    @func.grad
    def gfunc(self, x, factor, _y):
        _x = factor * _y
        return _x

class Tape(list):
    """ A tape records the computation of a code object.
        The tape object can then be used by the VM to build
        a gradient code object.
    """
    def __init__(self):
        list.__init__(self)
        self.init = {}
        self._refcount = {}

    def __str__(self):
        r = '-- Inputs (%08X)----\n' % id(self)
        r += '\n'.join([microcode.to_label(kwargs) for microcode, kwargs, d in self ])
        r += '\n'
        r += '-- Refcounts ----\n'
        r += ' '.join(["%08X : %d" % refcount for refcount in self._refcount.items()])
        return r

    def get_uid(self, value):
        return id(value)

    def get_refcount(self, value):
        uid = id(value)
        return self._refcount.get(uid, 0)

    def append(self, microcode, kwargs, din):
        """ add a record to the tape. Record is the argument name and the value. """
        for an, value in din.items():
            uid = id(value)
            self._refcount[uid] = self._refcount.get(uid, 0) + 1
        list.append(self, (microcode, kwargs, din))

    def to_graph(self, **kwargs):
        """ create a graphviz Digraph"""
        import graphviz
        graph = graphviz.Digraph(**kwargs)
        VM._add_to_graph(graph, self.init, self)
        return graph

    def _repr_png_(self):
        return self.to_graph(engine='dot', graph_attr=dict(rankdir="LR")).pipe(format="png")

    def _repr_svg_(self):
        return self.to_graph(engine='dot', graph_attr=dict(rankdir="LR"))._repr_svg_()

class Code(list):
    """ A code object is a sequence of microcodes with input, output variables and parameters.
    """
    def __init__(self, vm):
        self.microcodes = []
        self.vm = vm
        self.defaults = {}

    def copy(self):
        r = self.vm.code()
        r.defaults.update(self.defaults)
        for microcode, kwargs in self.microcodes:
            r.append(microcode, kwargs)
        return r

    def append(self, microcode, kwargs):
        self.microcodes.append( (microcode, kwargs))

    def __repr__(self):
        r = "--Code---\n"
        r += '\n'.join(code.to_label(kwargs) for code, kwargs in self.microcodes)
        r += '\n'
        return r

    def _find_inputs(self):
        """ find inputs of a code segment """
        live = set()
        for microcode, kwargs in reversed(self.microcodes):
            for an in microcode.aout:
                vn = kwargs.get(an, an)
                if vn in live:
                    live.remove(vn)
            for an in microcode.ain:
                vn = kwargs.get(an, an)
                live.add(vn)
        return list(live)

    def _optimize(self, vout):
        live = set(vout)
        code = self.vm.code()
        optimized = []
        for microcode, kwargs in reversed(self.microcodes):
            keep = False
            for an in microcode.aout:
                vn = kwargs.get(an, an)
                if vn in live:
                    keep = True
                    live.remove(vn)
            if keep:
                for an in microcode.ain:
                    vn = kwargs.get(an, an)
                    live.add(vn)

                optimized.append((microcode, kwargs))

        for microcode, kwargs in reversed(optimized):
            code.append(microcode, kwargs)
        return code

    def __iter__(self):
        return iter(self.microcodes)

    def compute(self, vout, init, tape=None, monitor=None):
        if not isinstance(vout, (tuple, list)):
            vout = [vout]
            squeeze = True
        else:
            squeeze = False

        code = self._optimize(vout)

        inputs = code._find_inputs()
        frontier = {}

        init2 = {}
        init2.update(self.defaults)
        init2.update(init)

        for key, value in init2.items():
            # up cast 0 to VM.Zero for convenience
            if value is 0:
                value = VM.Zero
            frontier[key] = value

        frontier[""] = VM.Zero

        for vn in inputs:
            if vn not in frontier:
                raise ValueError("`%s` not defined in inputs" % vn)

        if tape is not None:
            tape.init.update(init2)

        started = False
        for i, (microcode, kwargs) in enumerate(code):
            try:
                r = microcode.invoke(self.vm, frontier, kwargs, tape, monitor)
            except Exception as e:
                print("Failure in running `%s`" % microcode)
                raise
            frontier.update(r)
            future = code.microcodes[i+1:]
            code._gc(frontier, future, vout, monitor)

        r = [frontier[vn] for vn in vout]
        if squeeze:
            r = r[0]
        return r


    def _gc(self, frontier, future, vout, monitor=None):
        """ remove variables that are never used again """
        used = []
        used.extend(vout)
        for microcode, kwargs in future:
            for an in microcode.ain + microcode.literals:
                vn = kwargs.get(an, an)
                used.append(vn)

        used = set(used)
        for vn in list(frontier.keys()):
            if vn not in used:
                if monitor:
                    monitor("freeing", vn)
                frontier.pop(vn)

    def to_graph(self, **kwargs):
        """ create a graphviz Digraph"""
        import graphviz
        graph = graphviz.Digraph(**kwargs)

        VM._add_to_graph(graph, self._find_inputs(), self.microcodes)

        return graph

    def _repr_png_(self):
        return self.to_graph(engine='dot', graph_attr=dict(rankdir="LR")).pipe(format="png")

    def _repr_svg_(self):
        return self.to_graph(engine='dot', graph_attr=dict(rankdir="LR"))._repr_svg_()


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
    def __array__(self):
        import numpy
        return numpy.array(0)
    def __repr__(self): return "<ZERO>"
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

    dict["__repr__"] = __repr__
    return type("ZeroType", (object,), dict)

ZeroType = ZeroType()
Zero = ZeroType()

VM.Zero = Zero
VM.microcode = staticmethod(microcode)

