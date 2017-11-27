import weakref

class Symbol(object):
    def __init__(self, model, name=None):
        from .model import Model

        if name is not None:
            assert isinstance(name, str)
        assert isinstance(model, Model)

        self._model = weakref.ref(model)
        self.name = name

        # a list of nodes that makes use of the symbol
        self.references = []

    def add_reference(self, node):
        self.references.append(weakref.ref(node))

    @property
    def vjp_name(self):
        return '_' + self.name

    @property
    def jvp_name(self):
        return self.name + '_'

    @property
    def model(self):
        return self._model()

    def __repr__(self):
        return "[%s:]" % self.name

    def resolve(self, context):
        return context[self.name]

class Literal(Symbol):
    def __init__(self, model, value):
        Symbol.__init__(self, model, None)
        self.value = value

    def __repr__(self):
        return "%s" % (str(self.value))

    def resolve(self, context):
        return self.value

class ZeroLiteral(Literal):
    def __init__(self, model):
        Symbol.__init__(self, model, None)

    def __repr__(self):
        return "[ZERO]"

    def resolve(self, context):
        return 0

