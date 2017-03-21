from __future__ import print_function
import warnings
import functools
import logging

logger = logging.getLogger("VMAD")
_logging_handler = logging.StreamHandler()
logger.addHandler(_logging_handler)

# TODO:
# Add visualization

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
    def gradname(self):
        return '_' + self.name

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

    def __repr__(self):
        if isinstance(self, IOArgument):
            return "%s:%s=%s=>%s" % (type(self).__name__, self.name, self.value, self.ovalue)
        else:
            return "%s:%s=%s" % (type(self).__name__, self.name, self.value)

class IArgument(Argument): pass
class OArgument(Argument): pass
class IOArgument(Argument): pass
class EXArgument(Argument): pass
class Arguments(list):
    def find(self, argname):
        for arg in self:
            if arg.name == argname: return arg
        else:
            raise KeyError

    def set_values(self, kwargs, code):
        kwargs = kwargs.copy()
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
    def __init__(self, engine, instr, args=None):
        self.instr = instr
        self.engine = engine
        if args is None:
            args = Arguments()
        self.args = args
        for argname in instr.argnames:
            if argname in instr.ain:
                if argname in instr.aout:
                    arg = IOArgument(argname)
                else:
                    arg = IArgument(argname)
            elif argname in instr.aout:
                arg = OArgument(argname)
            else:
                arg = EXArgument(argname)
            self.args.append(arg)

    def copy(self):
        return type(self)(self.engine, self.instr, self.args)

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
    def copy(self):
        return Node.copy(self)

    @property
    def codeseg(self):
        raise NotImplementedError

    def _invoke(self, codeseg, frontier, return_tape=False):
        aout = [ arg.name for arg in self.args
                if isinstance(arg, (OArgument, IOArgument))]
        vout = [ arg.value.name for arg in self.args
                if isinstance(arg, (OArgument, IOArgument))]
        init = {}
        #if return_tape:
        #    print('----', self.args, frontier)
        for arg in self.args:
            if isinstance(arg, (IArgument, IOArgument)):
                init[arg.name] = frontier[arg.value.name]

        out = codeseg.compute(aout, init, return_tape=return_tape)
        if return_tape:
            out, tape = out
            return dict(zip(vout, out)), tape
        else:
            return dict(zip(vout, out))

    def invoke(self, frontier):
        logger.info("Invoke %s" % (self))
        return self._invoke(self.codeseg, frontier, False)

    def gradient(self, d):
        logger.info("Gradient %s" % (self))
        return gradient

class Primitive(Instruction):
    def __init__(self, body, ain, aout):
        Instruction.__init__(self, body, ain, aout)

    class NodeType(Node):
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
        extra = ['#programme_node', '#frontier']
        body = lambda : None
        body.__name__ = "G:" + programme.body.__name__
        argnames = list(set(gin + gout + programme.ain + ex + extra))
        Instruction.__init__(self, body, gin, gout, argnames=argnames)

    class NodeType(CodeSegNode):
        @property
        def codeseg(self):
            node = self.args.find('#programme_node').value
            d = self.args.find('#frontier').value
            codeseg = node.codeseg
            r, tape = node._invoke(codeseg, d, True)
            gradient = codeseg.gradient(tape)
            return gradient

        def copy(self):
            node = CodeSegNode.copy(self)
            node.vjp_args = vjp_args
            return

class Programme(Instruction):
    def __init__(self, body, ain, aout):
        Instruction.__init__(self, body, ain, aout)
        self.vjp = ProgrammeVJP(self)

    class NodeType(CodeSegNode):
        @property
        def codeseg(self):
            return self.instr.body(self.engine, *[arg.name for arg in self.args])

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

        self._liveset = {} # stores the version of variable with the same name
                          # each overwrite will increase this number
        self._postfix = 0 # a unique postfix added to every variable.

    def __getattr__(self, name):
        """ Allow looking up primitives and programmes from the engine namespace """
        try:
            item = getattr(self.engine, name)
        except AttributeError:
            raise AttributeError("%s is not a declared instruction in %s" % (name, type(self.engine)))

        if isinstance(item, Instruction):
            instr = item
            def func(**kwargs):
                self.append(instr, kwargs)
            return func
        else:
            raise TypeError

    def copy(self):
        code = CodeSegment(self.engine)
        code.nodes.extend(self.nodes)
        code.defaults.update(self.defaults)
        code._liveset.update(self._liveset)
        code._postfix = self._postfix
        return code

    def get_latest_variable(self, varname, expired=False):
        if varname not in self._liveset:
            self._postfix = self._postfix + 1
            variable = Variable(varname, self._postfix)
        else:
            variable = self._liveset.pop(varname)
        if not expired:
            self._liveset[varname] = variable
        return variable

    def create_latest_variable(self, varname):
        self._postfix = self._postfix + 1
        variable = Variable(varname, self._postfix)
        self._liveset[varname] = variable
        return variable

    def append(self, instr, kwargs):
        node = instr.create_node(self.engine)
        node.args.set_values(kwargs, self)
        self.nodes.append(node)

    def get_refcounts(self, vout):
        refcounts = {}
        for varname in vout:
            var = self.get_latest_variable(varname)
            refcounts[var] = refcounts.get(var, 0) + 1
        for node in self.nodes:
            for arg in node.args:
                if isinstance(arg, (IArgument, IOArgument)):
                    refcounts[arg.value] = refcounts.get(arg.value, 0) + 1
        return refcounts

    def get_freeables(self, vout):
        refcounts = self.get_refcounts(vout)

        free_list = []

        for node in self.nodes:
            item = []
            for arg in node.args:
                if isinstance(arg, (IArgument, IOArgument)):
                    refcounts[arg.value] = refcounts[arg.value] - 1
                    if refcounts[arg.value] == 0:
                        item.append(arg.value)
            free_list.append(item)
        return free_list

    def optimize(self, vout):
        """ return an optimized codeseg for computing vout. irrelevant nodes are pruned. """
        deps = []
        for varname in vout:
            deps.append(self.get_latest_variable(varname))

        deps = set(deps)

        nodes = []
        for node in self.nodes[::-1]:
            keep = False
            for arg in node.args:
                if isinstance(arg, OArgument) and arg.value in deps:
                    keep = True
                    deps.remove(arg.value)
                if isinstance(arg, IOArgument) and arg.ovalue in deps:
                    keep = True
                    deps.remove(arg.ovalue)
            if not keep: continue
            nodes.append(node)
            for arg in node.args:
                if isinstance(arg, (IArgument, IOArgument)):
                    deps.add(arg.value)

        segment = self.copy()
        segment.nodes = list(reversed(nodes))
        return segment

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

        for var, value in self.defaults.items():
            frontier[var] = value
        for var, value in init.items():
            frontier[var] = value

        if return_tape:
            tape = Tape(frontier)
        else:
            tape = None

        for i, (node, abandon) in enumerate(zip(self.nodes, self.get_freeables(vout))):
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
                    kwargs['_' + arg.name] = arg.value.gradname
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
                        kwargs['_' + arg.name] = arg.value.gradname
                    else:
                        newname = arg.value.gradname + '#partial'
                        kwargs['_' + arg.name] = newname
                        partials.append((newname, arg.value.gradname))
                if isinstance(arg, EXArgument):
                    kwargs[arg.name] = arg.value

            if isinstance(node.instr, Programme):
                # the vjp of a Programme requires more arguments
                # to build the gradient codesegment on the fly
                kwargs['#programme_node'] = node
                kwargs['#frontier'] = d

            code.append(vjp, kwargs)
            for p, r in partials:
                kwargs = {}
                kwargs['x1'] = p
                kwargs['x2'] = r
                kwargs['y'] = r
                code.append(add, kwargs)
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
        return '\n'.join([nodes])

    def to_graph(self, **kwargs):
        return nodes_to_graph(self.nodes, **kwargs)[0]

from .zero import ZERO

def nodes_to_graph(nodes, **kwargs):
    """
        add a list of microcodes to a graph. The init node is duplicated as needed
        (because it may be used many times and mess up the diagram. It hurts to have
        very long edges like that.

    """
    import graphviz
    graph = graphviz.Digraph(**kwargs)

    def unique(obj):
        return '%08X%08X' % (id(nodes), id(obj))

    dests = {}
    source = {}
    subgraphs = {}
    for i, node in enumerate(nodes):
        for arg in node.args:
            if isinstance(arg, (OArgument, IOArgument)):
                source[arg.ovalue] = (node, arg)
            if isinstance(arg, (IArgument, IOArgument)):
                l = dests.pop(arg.value, [])
                l.append((node, arg))
                dests[arg.value] = l

    for i, node in enumerate(nodes):
        label = '%s<BR/>' % str(node.instr)
        ex = []
        for arg in node.args:
            if isinstance(arg, EXArgument):
                # bypass aux arguments starting with sharp
                if not arg.name.startswith('#'):
                    ex.append(str(arg))
        label = label + '<BR/>'.join(ex)
        label = '<' + label + '>'

        if not isinstance(node, CodeSegNode):
            graph.node(unique(node), label=label, shape='box')
        else:
            codeseg = node.codeseg
            subgraph, inputs, outputs = nodes_to_graph(codeseg.nodes)
            subgraph.name = 'cluster_' + unique(node)
            subgraph.attr('graph', label=label)
            subgraph.attr('graph', color='blue')
            subgraph.attr('graph', style='dotted')
            graph.subgraph(subgraph)
            subgraphs[node] = (inputs, outputs)

    inputs = {}
    outputs = {}
    for var in set(list(source.keys()) + list(dests.keys())):
        attrs = {}
        attrs['label'] = '<' + str(var) + '>'

        if var in source:
            from_node, from_arg = source[var]
            if from_node not in subgraphs:
                from_node = unique(from_node)
            else:
                from_node = subgraphs[from_node][1][from_arg.name]

            attrs['taillabel'] = '<' + str(from_arg.name) + '>'
            attrs['tail_lp'] = "12"
        else:
            from_node = unique(var)
            graph.node(from_node, label=str(var))
            inputs[var.name] = from_node
        if var in dests:
            for to_node, to_arg in dests[var]:
                if to_node not in subgraphs:
                    to_node = unique(to_node)
                else:
                    to_node = subgraphs[to_node][0][to_arg.name]
                attrs['headlabel'] = '<' + str(to_arg.name) + '>'
                attrs['head_lp'] = "12"
                graph.edge(from_node, to_node, **attrs)
        else:
            to_node = unique(var)
            graph.node(to_node, label=str(var))
            outputs[var.name] = to_node
            graph.edge(from_node, to_node, **attrs)

    return graph, inputs, outputs
