from .operator import terminal
from .tape import Tape

class Context(dict):
    """ A context is a collection of python objects referred by symbol names

    """
    def __init__(self, **kwargs):
        self.update(kwargs)

    def remove_unused(self, plist):
        used = set()
        for p in plist:
            for argname, var in p.varin.items():
                used.add(var.name)

        toremove = []
        for key in self:
            if not key in used:
                toremove.append(key)

        for key in toremove:
            self.pop(key)

    def compute(self, model, vout, return_tape=False, monitor=None):
        """
            compute a model in the current context (self).

        """
        tape = Tape(model)

        if isinstance(vout, str):
            single_return = True
        else:
            single_return = False

        r = {}
        for i, p in enumerate(model):

            p.execute(self, tape)

            if isinstance(p, terminal.opr):
                for argname, var in p.varout.items():
                    r[var.name] = self[var.name]

            self.remove_unused(model[i+1:])

            if monitor is not None:
                monitor(p, self)

        if single_return:
            r = r[vout]
        else:
            r = [r[varname] for varname in vout]

        if return_tape:
            r = r, tape

        return r
