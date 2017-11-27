class Node(object):
    def __init__(self, primitive, kwargs):
        self.primitive = primitive
        self.kwargs = kwargs

class Variable(object):
    def __init__(self, varname, value):
        self.varname = varname
        self.value = value
        self.version = 0

    def update(self, value):
        self.version += 1
        self.value = value

    @property
    def fullname(self):
        return self.varname + '/%d' % self.version

class Context(object):
    def __init__(self, init):
        self._scope = {}
        for key, value in init.items():
            self.update(key, value)

    def update(self, varname, value):
        if varname in self._scope:
            var = self._scope[varname]
            var.update(value)
        else:
            var = Variable(varname, value)

        self._scope.update(varname, var)

    def read(self, varname):
        assert varname in self._scope
        return self._scope[varname]

    def execute(self, model):
        for node in model:
            node.primitive.execute()
