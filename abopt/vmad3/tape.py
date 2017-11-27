from .autodiff import vjp

class Tape(list):
    def __init__(self, model):
        self.model = model

    def append(self, node, impl_kwargs):
        list.append(self, (node, impl_kwargs))

    def get_vjp(self):
        return vjp(self)

