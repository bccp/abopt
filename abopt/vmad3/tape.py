class Record(object):
    """ A record on the tape. 

        A record contains the node and the resolved arg symbols to the node.

    """
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
        # to avoid cicurlar reference; this is not a strong dependency
        from .autodiff import vjpmodel
        return vjpmodel(self)

    def get_jvp(self):
        # to avoid cicurlar reference; this is not a strong dependency
        from .autodiff import jvpmodel
        return jvpmodel(self)
