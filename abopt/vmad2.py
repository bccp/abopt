from __future__ import print_function
import warnings
import functools
import logging

logger = logging.getLogger("VMAD")
_logging_handler = logging.StreamHandler()
logger.addHandler(_logging_handler)

# TODO:
# Add visualization
# Optimization, removing unused instructions from code segment

class LValue(object):
    def __init__(self, name, ns):
        self.ns = ns
        self.name = name

    def __getattr__(self, attr): return getattr(self[...], attr)
    def __repr__(self): return "LValue:%s" % self.name
    def __getitem__(self, index): return self.ns[self.name]
    def __setitem__(self, index, value): self.ns[self.name] = value

class Instruction(object):
    def __init__(self, body, ain, aout, argnames=None):
        self.body = body
        self.ain = ain
        self.aout = aout
        if argnames is None:
            argnames = body.__code__.co_varnames[1:body.__code__.co_argcount]
        self.argnames = argnames
        for an in ain:
            if not an in self.argnames:
                raise ValueError(
    "argument `%s` of ain in microcode decorator is not declared by function `%s`"
                       % (an, str(self.body))
                )
        functools.update_wrapper(self, body)

    def defvjp(self, body):
        """ Define the back-propagation gradient operator. """
        gout = ['_' + a for a in self.ain]
        gin  = ['_' + a for a in self.aout]

        body.__name__ = "G:" + self.body.__name__
        self.vjp = Primitive(body, gin, gout)
        # allow the gradient with the same name as the original body.
        return self.vjp


    def __repr__(self):
        return self.body.__name__

    def create_node(self, engine, kwargs):
        nodetype = type(self).NodeType
        node = nodetype(engine, self, kwargs)
        return node

class Variable(object):
    """ if the same variable name is modified we use a different postifx
        this happens as Variable mentioned in Code as O/IOArgument.
    """
    def __init__(self, name, postfix):
        self.name = name
        self.postfix = postfix

    def __hash__(self): return hash(self.name + '-%s' % self.postfix)
    def __eq__(self, other): return self.name == other.name and self.postfix == other.postfix

    def __repr__(self):
        if self.postfix is not None:
            return "%s/%d" % (self.name, self.postfix)
        else:
            return "%s" % (self.name)

class Argument(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return "%s:%s=%s" % (type(self).__name__, self.name, self.value)

class IArgument(Argument):
    pass
class OArgument(Argument):
    pass
class IOArgument(Argument):
    pass
class EXArgument(Argument):
    pass

class Node(object):
    def __init__(self, engine, instr, kwargs):
        self.instr = instr
        self.engine = engine
        self.args = []
        kwargs = kwargs.copy()
        for arg in instr.argnames:
            if arg in instr.ain:
                if arg in instr.aout:
                    var = IOArgument(arg, kwargs.pop(arg, arg))
                else:
                    var = IArgument(arg, kwargs.pop(arg, arg))
            elif arg in instr.aout:
                var = OArgument(arg, kwargs.pop(arg, arg))
            else:
                var = EXArgument(arg, kwargs.pop(arg))
            self.args.append(var)
        if len(kwargs) > 0:
            raise ValueError("additional kwargs are found: %s" % list(kwargs.keys()))
    def copy(self):
        return type(self)(self.engine, self.instr, self.kwargs)

    def bind(self, frontier, results):
        """ bind args to objects in frontier, or LValues """
        bound = []
        instr = self.instr
        for arg in self.args:
            if isinstance(arg, (IArgument, IOArgument)):
                bound.append(frontier[arg.value.name])
            elif isinstance(arg, OArgument):
                bound.append(LValue(arg.value.name, results))
            else:
                bound.append(arg.value)
        return bound

    def __repr__(self):
        return "%s(%s)" % (self.instr, self.args)

class CodeSegNode(Node):
    def __init__(self, engine, instr, kwargs):
        Node.__init__(self, engine, instr, kwargs)

    def copy(self):
        return Node.copy(self)

    def _invoke(self, codeseg, frontier, return_tape=False):
        aout = [ arg.name for arg in self.args
                if isinstance(arg, (OArgument, IOArgument))]
        vout = [ arg.value.name for arg in self.args
                if isinstance(arg, (OArgument, IOArgument))]
        init = {}
        if return_tape:
            print('----', self.args, frontier)
        for arg in self.args:
            if isinstance(arg, (IArgument, IOArgument)):
                init[arg.name] = frontier[arg.value.name]

        out = codeseg.compute(aout, init, return_tape=return_tape)
        if return_tape:
            out, tape = out
            return dict(zip(vout, out)), tape
        else:
            return dict(zip(vout, out))

class Primitive(Instruction):
    def __init__(self, body, ain, aout):
        Instruction.__init__(self, body, ain, aout)

    class NodeType(Node):
        def __init__(self, engine, instr, kwargs):
            Node.__init__(self, engine, instr, kwargs)

        def copy(self):
            node = Node.copy(self)
            return node

        def invoke(self, frontier):
            logger.info("Invoke %s" % (self))
            out = {}
            bound = self.bind(frontier, out)
            self.instr.body(self.engine, *bound)
            return out

def primitive(ain, aout): return lambda body: Primitive(body, ain, aout)

class ProgrammeVJP(Instruction):
    def __init__(self, programme):
        gout = ['_' + a for a in programme.ain]
        gin  = ['_' + a for a in programme.aout]
        ex = [a for a in programme.argnames if a not in programme.aout + programme.ain]

        body = lambda : None
        body.__name__ = "G:" + programme.body.__name__
        argnames = list(set(gin + gout + programme.ain + ex))
        Instruction.__init__(self, body, gin, gout, argnames=argnames)

    def create_vjp_node(self, programme_node, d, kwargs):
        node = self.create_node(programme_node.engine, kwargs)
        node.create_codeseg = lambda : programme_node.gradient(d)
        return node

    class NodeType(CodeSegNode):
        def __init__(self, engine, instr, kwargs):
            Node.__init__(self, engine, instr, kwargs)

        def copy(self):
            node = CodeSegNode.copy(self)
            node.create_codeseg = create_codeseg
            return

        def invoke(self, frontier):
            logger.info("Invoke %s" % (self))
            codeseg = self.create_codeseg()
            return self._invoke(codeseg, frontier, False)

class Programme(Instruction):
    def __init__(self, body, ain, aout):
        Instruction.__init__(self, body, ain, aout)
        self.vjp = ProgrammeVJP(self)

    class NodeType(CodeSegNode):
        def __init__(self, engine, instr, kwargs):
            Node.__init__(self, engine, instr, kwargs)
            codeseg = CodeSegment(self.engine)
            self.instr.body(codeseg, *[arg.name for arg in self.args])
            self.codeseg = codeseg

        def copy(self):
            node = CodeSegNode.copy(self)
            node.codeseg = codeseg
            return node

        def invoke(self, frontier):
            logger.info("Invoke %s" % (self))
            return self._invoke(self.codeseg, frontier, False)

        def gradient(self, d):
            logger.info("Gradient %s" % (self))
            r, tape = self._invoke(self.codeseg, d, True)
            gradient = self.codeseg.gradient(tape)
            return gradient



def programme(ain, aout): return lambda body: Programme(body, ain, aout)

class Tape(object):
    def __init__(self, init):
        self.records = []
        self.init = {}
        self.init.update(init)

    def append(self, node, frontier):
        d = {}
        for arg in node.args:
            if isinstance(arg, (IArgument, IOArgument)):
                d[arg.value.name] = frontier[arg.value.name]

        self.records.append((node, d))

    def __repr__(self):
        return '\n'.join('%s | %s' % (node, list(d.keys())) for node, d in self.records)

class Engine(object):
    pass

class CodeSegment(object):
    def __init__(self, engine):
        self.engine = engine
        self.nodes = []
        self.defaults = {} # use these if not provided in init

        self.liveset = {} # stores the version of variable with the same name
                          # each overwrite will increase this number
        self.refs = {} # stores the numbers a versioned Variable used as input
        self._postfix = 0 # a unique postfix added to every variable.

    @property
    def postfix(self):
        self._postfix = self._postfix + 1
        return self._postfix

    def copy(self):
        code = CodeSegment(self.engine)
        code.nodes.extend(self.nodes)
        code.defaults.update(self.defaults)
        code.liveset.update(self.liveset)
        code.refs.update(self.refs)
        code.postfix = self.postfix

    def _get_ref(self, var):
        return self.refs.get(var, 0)

    def _inc_ref(self, var):
        ref = self._get_ref(var) + 1
        self.refs[var] = ref
        return ref

    def append(self, node):
        for arg in node.args:
            if isinstance(arg, IArgument):
                variable = self.liveset.get(arg.value, Variable(arg.value, self.postfix))
                self.liveset[arg.value] = variable
                arg.value = variable
            elif isinstance(arg, IOArgument):
                # remove the variable from liveset; next time if it becomes a IArgument
                # it must contains new value.
                variable = self.liveset.pop(arg.value, Variable(arg.value, self.postfix))
                self.liveset[arg.value] = variable
                arg.value = variable
            elif isinstance(arg, OArgument):
                variable = Variable(arg.value, self.postfix)
                self.liveset[arg.value] = variable
                arg.value = variable
        self.nodes.append(node)

    def _build_free_list(self):
        ocd = {}
        free_list = []
        for node in self.nodes:
            item = []
            for arg in node.args:
                if isinstance(arg, (IArgument, IOArgument)):
                    ocd[arg.value] = ocd.get(arg.value, 0) + 1
                    if ocd[arg.value] == self._get_ref(arg.value):
                        item.append(arg.value)
            free_list.append(item)
        return free_list

    def __getattr__(self, name):
        try:
            item = getattr(self.engine, name)
        except AttributeError:
            raise AttributeError("%s is not a declared instruction in %s" % (name, type(self.engine)))

        if isinstance(item, Instruction):
            instr = item
            def func(**kwargs):
                self.append(instr.create_node(self.engine, kwargs))
            return func
        else:
            raise TypeError

    def compute(self, vout, init, return_tape=False):
        if hasattr(self.engine, "Copy"):
            copy = self.engine.Copy.body
        else:
            copy = lambda x: x * 1.0

        if not isinstance(vout, (tuple, list, set)):
            vout = [vout]
            squeeze = True
        else:
            squeeze = False

        frontier = {}
        free_list = self._build_free_list()

        for var, value in self.defaults.items():
            frontier[var] = value
        for var, value in init.items():
            frontier[var] = value

        if return_tape:
            tape = Tape(frontier)
        else:
            tape = None

        for i, (node, abandon) in enumerate(zip(self.nodes, free_list)):
            if tape:
                tape.append(node, frontier)
                for arg in node.args:
                    if not isinstance(arg, IOArgument): continue
                    # FIXME: use copy
                    frontier[arg.value.name] = copy(frontier[arg.value.name])
            try:
                r = node.invoke(frontier)
            except Exception as e:
                print("Failure in running `%s`" % node)
                raise
            for var in abandon:
                frontier.pop(var.name)
            if len(abandon):
                logger.info("Removed from frontier %s, new size %d", abandon, len(frontier))
            frontier.update(r)

        r = [frontier[vn] for vn in vout]
        if squeeze:
            r = r[0]
        if return_tape:
            r = r, tape
        return r

    def gradient(self, tape):
        """ Create a code segment that computes the gradient from tape for the current
            code segment """

        code = CodeSegment(self.engine)

        if hasattr(self.engine, "Add"):
            add = self.engine.Add
        else:
            @primitive(ain=['x1', 'x2'], aout=['y'])
            def add(engine, x1, x2, y):
                y[...] = x1 + x2

        ocd = {} # number of times seen
        for node, d in tape.records[::-1]:
            vjp = node.instr.vjp
            kwargs = {}
            partials = []
            for arg in node.args:
                if isinstance(arg, OArgument) \
                and '_' + arg.name in vjp.argnames:
                    kwargs['_' + arg.name] = '_' + arg.value.name
                if isinstance(arg, (IArgument, IOArgument)) \
                and arg.name in vjp.argnames:
                    value = d[arg.value.name]
                    kwargs[arg.name] = value

                if isinstance(arg, (IArgument, IOArgument)) \
                and '_' + arg.name in vjp.argnames:
                    occ = ocd.get(arg.value, 0)
                    ocd[arg.value] = occ + 1
                    if occ == 0:
                        # directly write to the gradient, it is used
                        kwargs['_' + arg.name] = '_' + arg.value.name
                    else:
                        newname = '_' + arg.value.name + '#partial'
                        kwargs['_' + arg.name] = newname
                        partials.append((newname, '_' + arg.value.name))
                if isinstance(arg, EXArgument):
                    kwargs[arg.name] = arg.value

            if isinstance(vjp, ProgrammeVJP):
                node = vjp.create_vjp_node(node, d, kwargs)
            else:
                node = vjp.create_node(self.engine, kwargs)

            code.append(node)
            for p, r in partials:
                kwargs = {}
                kwargs['x1'] = p
                kwargs['x2'] = r
                kwargs['y'] = r
                node = add.create_node(self.engine, kwargs)
                code.append(node)
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

        gradient = self.gradient(tape)

        _init = init.copy()
        _init.update(ginit)

        gout = gradient.compute(gnout, _init)
        d = {}
        d.update(zip(cnout, cout))
        d.update(zip(gnout, gout))

        out = [d[vn] for vn in vout]
        if squeeze:
            out = out[0]
        return out

    def __repr__(self):
        nodes = '\n'.join('%s' % node for node in self.nodes)
        refs = '%s' % self.refs
        return '\n'.join([nodes, refs])
from .zero import ZERO
