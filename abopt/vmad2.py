from __future__ import print_function
import warnings
import functools
import logging

logger = logging.getLogger("VMAD")
_logging_handler = logging.StreamHandler()
logger.addHandler(_logging_handler)

# TODO:
# Add visualization

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
    def __array__(self, dtype=None):
        import numpy
        return numpy.array(0, dtype=dtype)
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
ZERO = ZeroType()

# decorators
def statement(ain, aout): return lambda body: Statement(body, ain, aout)
def programme(ain, aout): return lambda body: Programme(body, ain, aout)

# create a model using a given computing engine
def model(engine): return CodeSegment(engine)

class LValue(object):
    def __init__(self, name, ns):
        self.ns = ns
        self.name = name

    def __getattr__(self, attr): return getattr(self[...], attr)
    def __repr__(self): return "LValue:%s" % self.name
    def __getitem__(self, index): return self.ns[self.name]
    def __setitem__(self, index, value): self.ns[self.name] = value

class Literal(object):
    def __init__(self, value):
        self.value = value
    def __repr__(self): return "Literal:%s" % _short_repr(self.value)

class Primitive(object):
    def __init__(self, body, ain, aout, argnames=None):
        self.body = body
        self.ain = ain
        self.aout = aout
        if argnames is None:
            argnames = body.__code__.co_varnames[1:body.__code__.co_argcount]

        if getattr(body, '__kwdefaults__', None) is not None:
            self.defaults = dict(body.__kwdefaults__)
        elif getattr(body, '__defaults__', None) is not None:
            self.defaults = dict(zip(argnames[-len(body.__defaults__):], body.__defaults__))
        else:
            self.defaults = {}

        self.argnames = argnames
        for an in ain:
            if not an in self.argnames:
                raise ValueError(
    "argument `%s` of ain in microcode decorator is not declared by function `%s`"
                       % (an, str(self.body))
                )
        functools.update_wrapper(self, body)

        args = Arguments()
        for argname in argnames:
            if argname in ain:
                if argname in aout:
                    arg = IOArgument(argname)
                else:
                    arg = IArgument(argname)
            elif argname in aout:
                arg = OArgument(argname)
            else:
                arg = EXArgument(argname)
            args.append(arg)

        self.args = args
        body.args = args

    def __repr__(self):
        return self.body.__name__

    def create_node(self, engine):
        nodetype = type(self).NodeType
        return nodetype(engine, self)

class Variable(object):
    """ if the same variable name is modified we use a different postifx
        this happens as Variable mentioned in Code as O/IOArgument.
    """
    def __init__(self, name, postfix):
        self.name = name
        self.postfix = postfix

    @property
    def name_vjp(self): return '_' + self.name

    @property
    def name_jvp(self): return self.name + '_'

    def __hash__(self): return hash(self.name + '-%s' % self.postfix)
    def __eq__(self, other): return self.name == other.name and self.postfix == other.postfix

    def __repr__(self):
        if self.postfix is not None:
            return "%s/%d" % (self.name, self.postfix)
        else:
            return "%s" % (self.name)

class Argument(object):
    def __init__(self, name):
        self.name = name
        self.value = None
        self.ovalue = None

    def copy(self):
        arg = type(self)(self.name)
        arg.value = self.value
        arg.ovalue = self.ovalue
        return arg

    @property
    def name_vjp(self): return '_' + self.name

    @property
    def name_jvp(self): return self.name + '_'


    def dereference(self, context):
        """ returns the value of an argument by its value

            if context is None, returns the name of the variable
        """
        if isinstance(self.value, Literal):
            if context is None:
                return self.value
            else:
                return self.value.value
        elif isinstance(self.value, Variable):
            if context is None:
                return self.value.name
            else:
                return context[self.value.name]
        else:
            return self.value

    def __repr__(self):
        if isinstance(self, IOArgument):
            return "%s:%s=%s=>%s" % (type(self).__name__, self.name, self.value, self.ovalue)
        else:
            return "%s:%s=%s" % (type(self).__name__, self.name, _short_repr(self.value))

class IArgument(Argument): pass
class OArgument(Argument): pass
class IOArgument(Argument): pass
class EXArgument(Argument): pass
class Arguments(list):
    def copy(self):
        args = Arguments()
        for arg in self:
            args.append(arg.copy())
        return args

    def find(self, argname):
        for arg in self:
            if arg.name == argname: return arg
        else:
            raise KeyError

    def get_kwargs(self):
        kwargs = {}
        for arg in self:
            if isinstance(arg.value, Variable):
                kwargs[arg.name] = arg.value.name
            else:
                kwargs[arg.name] = arg.value
        return kwargs

    def set_values(self, kwargs, defaults, code):
        _kwargs = defaults.copy()
        _kwargs.update(kwargs)
        kwargs = _kwargs

        for arg in self:
            if isinstance(arg, EXArgument):
                variable = kwargs.pop(arg.name)
                arg.value = variable
        for arg in self:
            if isinstance(arg, IArgument):
                varname = kwargs.pop(arg.name, arg.name)
                variable = code.get_latest_variable(varname, expired=False)
                arg.value = variable
        for arg in self:
            if isinstance(arg, IOArgument):
                varname = kwargs.pop(arg.name, arg.name)
                variable = code.get_latest_variable(varname, expired=True)
                arg.value = variable
                arg.ovalue = code.create_latest_variable(varname)
        for arg in self:
            if isinstance(arg, OArgument):
                varname = kwargs.pop(arg.name, arg.name)
                variable = code.create_latest_variable(varname)
                arg.ovalue = variable
                arg.value = variable

        if len(kwargs) > 0:
            raise ValueError("additional kwargs are found: %s" % list(kwargs.keys()))

class Node(object):
    # if true, invoke will directly return ZERO when all inputs are ZERO
    ZERO_BYPASS = False 

    def __init__(self, engine, primitive):
        self.primitive = primitive
        self.engine = engine
        self.args = primitive.args.copy()

    def copy(self):
        return type(self)(self.engine, self.primitive)

    def bind(self, frontier, results):
        """ bind args to objects in frontier, or LValues """
        bound = []
        primitive = self.primitive
        for arg in self.args:
            if isinstance(arg, (IArgument, IOArgument)):
                bound.append(arg.dereference(frontier))
            elif isinstance(arg, OArgument):
                bound.append(LValue(arg.value.name, results))
            else:
                bound.append(arg.value)
        return bound

    def __repr__(self):
        return "%s(%s)" % (self.primitive, self.args)

    def call(self, bound, return_tape=False):
        # a simple node doesn't know how to return a Tape
        assert not return_tape
        r = self.primitive.body(self.engine, *bound)
        return None

    def invoke(self, frontier):
        #logger.info("Invoke %s" % (self))
        out = {}
        bound = self.bind(frontier, out)

        if self.ZERO_BYPASS and (
            all([value is ZERO for arg, value in zip(self.args, bound) 
                if isinstance(arg, (IArgument, IOArgument))])):

            for arg, value in zip(self.args, bound):
                if isinstance(arg, OArgument):
                    # IOArguments are already ZEROs
                    value[...] = ZERO
        else:
            self.call(bound)

        for arg, value in zip(self.args, bound):
            if isinstance(arg, IOArgument):
                out[arg.value.name] = value

        return out

    def invoke_for_tape(self, frontier):
        bound = self.bind(frontier, {})
        tape = self.call(bound, return_tape=True)
        # The call didn't return a Tape. Likley calling the method on a statement's Node?
        assert isinstance(tape, Tape)
        return tape

class CodeSegNode(Node):
    def get_codeseg(self):
        raise NotImplementedError

    def call(self, bound, return_tape=False):
        init = {}
        lvalues = {}
        for arg, value in zip(self.args, bound):
            if isinstance(arg, (IArgument, IOArgument)):
                init[arg.name] = value
            elif isinstance(arg, (EXArgument,)):
                init[arg.name] = value
            else:
                lvalues[arg.name] = value

        aout = [ arg.name for arg in self.args
                if isinstance(arg, (OArgument, IOArgument))]

        codeseg = self.get_codeseg()

        # compute doesn't taint init.
        out = codeseg.compute(aout, init, return_tape=return_tape)
        #logger.info("CodeSegment results %s %s" % (aout, _short_repr(out)))
        if return_tape:
            out, tape = out
        else:
            tape = None

        for argname, value in zip(aout, out):
            lvalues[argname][...] = value

        return tape

class Statement(Primitive):
    NodeType = Node

    def __init__(self, body, ain, aout):
        Primitive.__init__(self, body, ain, aout)

    def defvjp(self, body):
        """ Define the back-propagation gradient operator. """
        gout = ['_' + a for a in self.ain]
        gin  = ['_' + a for a in self.aout]

        argnames = body.__code__.co_varnames[1:body.__code__.co_argcount]
        ain = [a for a in self.ain if a in argnames]

        body.__name__ = "BG:" + self.body.__name__
        self.vjp = StatementVJP(body, gin + ain, gout)

        # allow the gradient with the same name as the original body.
        return self.vjp

    def defjvp(self, body):
        """ Define the forward-propagation gradient operator. """
        gin = [a + '_' for a in self.ain]
        gout  = [a + '_' for a in self.aout]
        argnames = body.__code__.co_varnames[1:body.__code__.co_argcount]
        ain = [a for a in self.ain if a in argnames]
        body.__name__ = "FG:" + self.body.__name__

        self.jvp = StatementJVP(body, gin + ain, gout)

        # allow the gradient with the same name as the original body.
        return self.jvp

class StatementVJP(Statement):
    class NodeType(Node):
        ZERO_BYPASS = True

class StatementJVP(Statement):
    class NodeType(Node):
        ZERO_BYPASS = True

class Programme(Primitive):
    def __init__(self, body, ain, aout):
        Primitive.__init__(self, body, ain, aout)
        self.vjp = ProgrammeVJP(self)
        self.jvp = ProgrammeJVP(self)

    class NodeType(CodeSegNode):
        def get_codeseg(self):
            return self.primitive.body(self.engine,
                *[arg.value if isinstance(arg, EXArgument)
                else arg.name for arg in self.args])

class ProgrammeVJP(Primitive):
    def __init__(self, programme):
        gout = ['_' + a for a in programme.ain]
        gin  = ['_' + a for a in programme.aout]
        ex = [a for a in programme.argnames if a not in programme.aout + programme.ain]
        extra = ['#replay-record']
        body = lambda : None
        body.__name__ = "VJP:" + programme.body.__name__
        argnames = list(set(gin + gout + programme.ain + ex + extra))
        Primitive.__init__(self, body, gin, gout, argnames=argnames)

    class NodeType(CodeSegNode):
        ZERO_BYPASS = True

        def get_codeseg(self):
            # replay then obtain the gradient codeseg
            node, d = self.args.find('#replay-record').value

            tape = node.invoke_for_tape(d)

            vjpcode = tape.get_vjp()
            # if an output variable is not mentioned in the code
            # then the vjp code segment doesn't set the default to ZERO
            # we fix it here.
            for arg in self.args:
                if isinstance(arg, OArgument):
                    vjpcode.defaults[arg.name] = ZERO

            return vjpcode

class ProgrammeJVP(Primitive):
    def __init__(self, programme):
        gin  = [a + '_' for a in programme.ain]
        gout = [a + '_' for a in programme.aout]
        ex = [a for a in programme.argnames if a not in programme.aout + programme.ain]
        extra = ['#replay-record']
        body = lambda : None
        body.__name__ = "JVP:" + programme.body.__name__
        argnames = list(set(gin + gout + programme.ain + ex + extra))
        Primitive.__init__(self, body, gin, gout, argnames=argnames)

    class NodeType(CodeSegNode):
        ZERO_BYPASS = True

        def get_codeseg(self):
            node, d = self.args.find('#replay-record').value
            tape = node.invoke_for_tape(d)
            # Watch out: We use the tape version
            # of VJP because with the code version of VJP
            # we do not know how to pass in the arguments
            # these arguments are marked as EXArgument in the VJP
            # but we need to resolve them from the frontier.

            jvpcode = tape.get_jvp()
            return jvpcode

class Tape(object):
    def __init__(self, engine, init):
        self.records = []
        self.init = {}
        self.init.update(init)
        self.engine = engine

    def append(self, node, frontier):
        d = {}
        for arg in node.args:
            # remember all input variable as their values
            if isinstance(arg, (IArgument, IOArgument)) \
                 and isinstance(arg.value, Variable):
                d[arg.value.name] = arg.dereference(frontier)

        self.records.append((node, d))

    def __repr__(self):
        return '\n'.join('%s | %s' % (node, list(d.keys())) for node, d in self.records)

    def get_vjp(self):
        """ Create a code segment that computes the vector jacobian product for a tape,
            with backward gradient propagation.

            A vector jacobian product is J_ij v_j where j is the output variable index.

            The input variable of the returned CodeSegment is '_a', '_b', ... where a, b,
            ... are the output variables.
        """

        code = CodeSegment(self.engine)

        add = self.engine.add

        first_time = {} 
        for node, d in self.records[::-1]:
            vjp = node.primitive.vjp
            kwargs = {}
            partials = []
            for arg in node.args:
                if isinstance(arg, OArgument) \
                and arg.name_vjp in vjp.argnames:
                    kwargs[arg.name_vjp] = arg.value.name_vjp
                if isinstance(arg, (IArgument, IOArgument)) \
                and arg.name in vjp.argnames:
                    kwargs[arg.name] = Literal(arg.dereference(d))

                if isinstance(arg, (IArgument, IOArgument)) and \
                   arg.name_vjp in vjp.argnames:
                    if isinstance(arg.value, Literal):
                        kwargs[arg.name_vjp] = '###abandon###'
                    else:
                        if first_time.get(arg.value, True):
                            # directly write to the gradient, it is used
                            kwargs[arg.name_vjp] = arg.value.name_vjp
                            first_time[arg.value] = False
                        else:
                            newname = arg.value.name_vjp + '#partial'
                            kwargs[arg.name_vjp] = newname
                            partials.append((newname, arg.value.name_vjp))

                if isinstance(arg, EXArgument) and arg.name in vjp.argnames:
                    kwargs[arg.name] = arg.value

            if isinstance(node.primitive, Programme):
                # the vjp of a Programme requires more arguments
                # to build the vjp codesegment on the fly
                kwargs['#replay-record'] = (node, d)

            code.append(vjp, kwargs)
            for p, r in partials:
                kwargs = {}
                kwargs['x1'] = p
                kwargs['x2'] = r
                kwargs['y'] = r
                code.append(add, kwargs)

        for variable in code._input_variables.values():
            code.defaults[variable.name] = ZERO

        #logger.info("GRADIENT code.defaults: %s " % code.defaults)
        return code

    def get_jvp(self):
        """ creates a CodeSegment that computes the jacobian vector product, from a tape,
            via forward gradient propagation.

            A jacobian vector product is J_ij v_i where i is index of the input variables.

            The returned CodeSegment input is 'a_', 'b_', ... where 'a', 'b', ...
            are the input variables of the original code segment.

            The advantage of starting from a tape is that we do not need to compute
            the original code together with the forward pass. Useful if we need to do
            vjp and jvp same time.
        """
        code = CodeSegment(self.engine)

        for node, d in self.records:
            jvp = node.primitive.jvp
            kwargs = {}
            for arg in node.args:
                if isinstance(arg.value, Variable) and arg.name_jvp in jvp.argnames:
                    kwargs[arg.name_jvp] = arg.value.name_jvp
                if isinstance(arg, IArgument) and arg.name in jvp.argnames:
                    kwargs[arg.name] = Literal(arg.dereference(d))
                    # for literal inputs, we shall set the input gradient to zero
                    if isinstance(arg.value, Literal):
                        kwargs[arg.name_jvp] = ZERO
                if isinstance(arg, IOArgument) and arg.name in jvp.argnames:
                    kwargs[arg.name] = Literal(arg.dereference(d))
                if isinstance(arg, EXArgument) and arg.name in jvp.argnames:
                    kwargs[arg.name] = arg.value

            if isinstance(jvp, ProgrammeJVP):
                kwargs['#replay-record'] = node, d

            code.append(jvp, kwargs)

        for variable in code._input_variables.values():
            code.defaults[variable.name] = ZERO

        return code

    def to_graph(self, **kwargs):
        nodes = [node for node, kwargs in self.records]
        return nodes_to_graph(nodes, **kwargs)[0]

class CodeSegment(object):
    def __init__(self, engine):
        self.engine = engine
        self.nodes = []
        self.defaults = {} # use these if not provided in init

        self._liveset = {} # the set of variables ready to be used as input (latest variables).
                           # If a variable is destroyed the value replaced with None.

        self._postfix = 0 # a unique postfix added to every variable;

        # if a variable is used as input but not yet been mentioned on the liveset.
        self._input_variables = {} # input variables

    def __getattr__(self, name):
        """ Allow looking up primitives and programmes from the engine namespace """
        try:
            item = getattr(self.engine, name)
        except AttributeError:
            raise AttributeError("%s is not a declared primitiveuction in %s" % (name, type(self.engine)))

        if isinstance(item, Primitive):
            primitive = item
            def func(**kwargs):
                self.append(primitive, kwargs)
            functools.update_wrapper(func, item)
            return func
        else:
            raise TypeError

    def __dir__(self):
        l = []
        for name in dir(self.engine):
            item = getattr(self.engine, name)
            if isinstance(item, Primitive):
                l.append(name)
        return l + dir(type(self))

    def copy(self):
        code = CodeSegment(self.engine)
        code.nodes.extend(self.nodes)
        code.defaults.update(self.defaults)
        code._liveset.update(self._liveset)
        code._postfix = self._postfix
        code._input_variables = self._input_variables
        return code

    def get_latest_variable(self, varname, expired=False):
        if isinstance(varname, Literal):
            return varname
        if varname not in self._liveset:
            self._postfix = self._postfix + 1
            variable = Variable(varname, self._postfix)
            self._input_variables[varname] = variable
        else:
            variable = self._liveset.get(varname)
            if variable is None:
                self._postfix = self._postfix + 1
                variable = Variable(varname, self._postfix)

        if not expired:
            self._liveset[varname] = variable
        else:
            self._liveset[varname] = None

        return variable

    def create_latest_variable(self, varname):
        self._postfix = self._postfix + 1
        variable = Variable(varname, self._postfix)
        self._liveset[varname] = variable
        return variable

    def append(self, primitive, kwargs):
        node = primitive.create_node(self.engine)
        node.args.set_values(kwargs, primitive.defaults, self)
        self.nodes.append(node)

    def optimize(self, vout):
        out = [ self.get_latest_variable(varname) for varname in vout]
        nodes = _optimize(self.nodes, out)
        segment = self.copy()
        segment.nodes = nodes
        return segment

    def compute(self, vout, init, return_tape=False, monitor=None):
        assign = self.engine.assign.body

        if not isinstance(vout, (tuple, list, set)):
            vout = [vout]
            squeeze = True
        else:
            squeeze = False

        frontier = {}

        for var, value in self.defaults.items():
            frontier[var] = value
        for var, value in init.items():
            frontier[var] = value

        if return_tape:
            tape = Tape(self.engine, frontier)

            # XXX This is not nice. But requires too much 
            # refactoring to get it right.
            self = self.copy()
            # we need to connect the outputs to a 'terminal'
            # node such that their input gradients are not
            # overwritten by the partial gradients of
            # subsequential operations.
            @statement(ain=['x'], aout=['x'])
            def mark(engine, x): pass
            @mark.defvjp
            def _(engine, _x): pass
            @mark.defjvp
            def _(engine, x_): pass
            @mark.vjp.defjvp
            def _(engine, _x_): pass
            for varname in vout:
                self.append(mark, {'x' : varname})
        else:
            tape = None

        out = [ self.get_latest_variable(varname) for varname in vout]

        nodes = _optimize(self.nodes, out)
        freeables = _get_freeables(nodes, out)

        for i, (node, abandon) in enumerate(zip(nodes, freeables)):
            if tape:
                tape.append(node, frontier)
                if node.primitive is not mark:
                    for arg in node.args:
                        if not isinstance(arg, IOArgument): continue
                        # FIXME: use copy
                        assign(self.engine, x=frontier[arg.value.name], y=LValue(arg.value.name, frontier))
            try:
                r = node.invoke(frontier)
            except Exception as e:
                print("Failure in running `%s`" % node)
                raise

            if monitor is not None:
                monitor(node, frontier, r)
            for var in abandon:
                frontier.pop(var.name)

#            if len(abandon):
#                logger.info("Removed from frontier %s", abandon)
            frontier.update(r)
            #logger.info("Frontier %s", list(frontier.keys()))

        r = [frontier[vn] for vn in vout]
        if squeeze:
            r = r[0]
        if return_tape:
            r = r, tape
        return r

    def get_jvp(self, init={}):
        """ creates a CodeSegment that computes the jacobian vector product, with forward
            gradient propagation.

            A jacobian vector product is J_ij v_i where i is index of the input variables.

            The returned CodeSegment input is 'a_', 'b_', ... where 'a', 'b', ...
            are the input variables of the original code segment.

            This will compute the original code together with the forward gradient pass.

        """
        code = CodeSegment(self.engine)

        for node in self.nodes:
            jvp = node.primitive.jvp
            kwargs = {}
            for arg in node.args:
                if isinstance(arg.value, Variable) and arg.name_jvp in jvp.argnames:
                    kwargs[arg.name_jvp] = arg.value.name_jvp
                if isinstance(arg, (IArgument, IOArgument)) and arg.name in jvp.argnames:
                    kwargs[arg.name] = arg.dereference(None)
                if isinstance(arg, EXArgument) and arg.name in jvp.argnames:
                    kwargs[arg.name] = arg.value

            if isinstance(jvp, ProgrammeJVP):
                # empty init because all variables are on the frontier.
                kwargs['#replay-record'] = node, {}

            code.append(jvp, kwargs)
            code.append(node.primitive, node.args.get_kwargs())

        for variable in code._input_variables.values():
            code.defaults[variable.name] = ZERO

        # merge in the defaults of self
        code.defaults.update(self.defaults)

        # initialize with the defaults
        code.defaults.update(init)
        return code

    def compute_with_gradient(self, vout, init, ginit, return_tape=False):
        if not isinstance(vout, (tuple, list, set)):
            vout = [vout]
            squeeze = True
        else:
            squeeze = False

        cnout = [vn for vn in vout if not vn.startswith('_')]
        # if gradient request are requested, they must be computed
        cnout_g = [ vn[1:] for vn in ginit]

        gnout = [vn for vn in vout if vn.startswith('_')]

        cout, tape = self.compute(cnout + cnout_g, init, return_tape=True)
        cout = cout[:len(cnout)]

        vjpcode = tape.get_vjp()

        gout = vjpcode.compute(gnout, ginit)

        d = {}
        d.update(zip(cnout, cout))
        d.update(zip(gnout, gout))

        out = [d[vn] for vn in vout]
        if squeeze:
            out = out[0]
        return out

    def __repr__(self):
        nodes = '\n'.join('%s' % node for node in self.nodes)
        return '\n'.join([nodes])

    def to_graph(self, **kwargs):
        return nodes_to_graph(self.nodes, **kwargs)[0]

# an optional base class for computing engines
class Engine(object):
    @statement(ain=['x1', 'x2'], aout=['y'])
    def add(engine, x1, x2, y):
        y[...] = x1 + x2

    @statement(ain=['x'], aout=['y'])
    def assign(engine, x, y):
        y[...] = x * 1.0

def _optimize(nodes, out):
    """ return an optimized codeseg for computing vout. irrelevant nodes are pruned. """

    deps = set(out)

    newnodes = []
    for node in nodes[::-1]:
        keep = False
        for arg in node.args:
            if isinstance(arg, OArgument) and arg.value in deps:
                keep = True
                deps.remove(arg.value)
            if isinstance(arg, IOArgument) and arg.ovalue in deps:
                keep = True
                deps.remove(arg.ovalue)
        if not keep: continue
        newnodes.append(node)
        for arg in node.args:
            if isinstance(arg, (IArgument, IOArgument)):
                deps.add(arg.value)

    return list(reversed(newnodes))

def _get_freeables(nodes, out):
    refcounts = {}
    for var in out:
        refcounts[var] = refcounts.get(var, 0) + 1

    for node in nodes:
        for arg in node.args:
            if isinstance(arg, (IArgument, IOArgument)):
                refcounts[arg.value] = refcounts.get(arg.value, 0) + 1

    free_list = []

    for node in nodes:
        item = []
        for arg in node.args:
            if isinstance(arg, (IArgument, IOArgument)):
                if isinstance(arg.value, Literal): continue
                refcounts[arg.value] = refcounts[arg.value] - 1
                if refcounts[arg.value] == 0:
                    item.append(arg.value)
        free_list.append(item)
    return free_list

def nodes_to_graph(nodes, depth=0, **kwargs):
    """
        Graph representation of nodes, kwargs are sent to graphviz

        depth controls the behavior of programme nodes. only 
        depth level of sub graphs are made.
    """
    import graphviz
    graph = graphviz.Digraph(**kwargs)

    def unique(obj):
        return '%08X%08X' % (id(nodes), id(obj))

    subgraphs = {}

    for i, node in enumerate(nodes):
        label = '%s<BR/>' % str(node.primitive)
        ex = []
        for arg in node.args:
            if isinstance(arg, EXArgument):
                # bypass aux arguments starting with sharp
                if not arg.name.startswith('#'):
                    ex.append(str(arg))
        label = label + '<BR/>'.join(ex)
        label = '<' + label + '>'

        if depth > 0 and isinstance(node, CodeSegNode):
            # render the programme nodes as subgraphs
            
            codeseg = node.get_codeseg()
            subgraph, inputs, outputs = nodes_to_graph(codeseg.nodes, depth - 1)
            subgraph.name = 'cluster_' + str(node.primitive)
            subgraph.attr('graph', label=label)
            subgraph.attr('graph', color='blue')
            subgraph.attr('graph', style='dotted')
            graph.subgraph(subgraph)
            subgraphs[unique(node)] = (inputs, outputs)
        else:
            graph.node(unique(node), label=label, shape='box')

    source = {}

    inputs, outputs = {}, {}

    def process_in_arg(arg, node):
        attrs = {}
        attrs['label'] = '<' + str(arg.value) + '>'
        nodeid = unique(node)
        if isinstance(arg, (IArgument, IOArgument)):
            if arg.value in source:
                from_nodeid, from_arg = source[arg.value]
                attrs['taillabel'] = '<' + str(from_arg.name) + '>'
                attrs['tail_lp'] = "12"
            else:
                from_nodeid = nodeid + unique(arg.value)
                graph.node(from_nodeid, label=str(arg.value))

                if not isinstance(arg.value, Literal):
                    inputs[arg.value.name] = from_nodeid
                          
            if nodeid in subgraphs:
                nodeid = subgraphs[nodeid][0][arg.name]

            attrs['headlabel'] = '<' + str(arg.name) + '>'
            attrs['head_lp'] = "12"
            graph.edge(from_nodeid, nodeid, **attrs)

    def process_out_arg(arg, node):
        if isinstance(arg, (OArgument, IOArgument)):
            nodeid = unique(node)
            if nodeid in subgraphs:
                nodeid = subgraphs[nodeid][1][arg.name]

            source[arg.ovalue] = (nodeid, arg)
            outputs[arg.value.name] = nodeid

    for i, node in enumerate(nodes):
        for arg in node.args:
            process_in_arg(arg, node)
            process_out_arg(arg, node)

    return graph, inputs, outputs


def _short_repr(obj):
    if isinstance(obj, (list, tuple)):
        return [_short_repr(i) for i in obj]
    else:
        s = '%s' % obj
        if len(s) > 30:
            s = '[%s]' % type(obj).__name__
        return s
