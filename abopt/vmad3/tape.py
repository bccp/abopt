from .autodiff import vjp, jvp

class Record(object):
    def __init__(self, node, impl_kwargs):
        self.node = node
        self.impl_kwargs = impl_kwargs
    def __repr__(self):
        return '%s / %s' % (self.node, self.impl_kwargs)

class Tape(list):
    def __init__(self, model):
        self.model = model

    def append(self, node, impl_kwargs):
        list.append(self, Record(node, impl_kwargs))

    def get_vjp(self):
        return vjp(self)

    def get_jvp(self):
        return jvp(self)
