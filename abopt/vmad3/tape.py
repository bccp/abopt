from .autodiff import vjp, jvp

class Record(object):
    def __init__(self, node, resolved):
        self.node = node
        self.resolved = resolved
    def __repr__(self):
        return '%s / %s' % (self.node, self.resolved)

class Tape(list):
    def __init__(self, model):
        self.model = model

    def append(self, node, resolved):
        list.append(self, Record(node, resolved))

    def get_vjp(self):
        return vjp(self)

    def get_jvp(self):
        return jvp(self)
