class Primitive(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs.copy()

        self.process_ain()
        self.process_aout()

    @classmethod
    def get_type_patterns(kls):
        return [a[1] for a in list(kls.ain) + list(kls.aout)]

    def return(self, dict):
        pass

@registry
class add(Primitive):
    ain  = [('x1', '*'),
            ('x2', '*')]
    aout = [('y', '*')]

    def operator(self, x1, x2, y):
        return dict(y = x1 + x2)

    def vjp(self, _x1, _x2, _y):
        return dict(_x1 = _y, _x2 = _y)

    def jvp(self, x1_, x2_, y_):
        return dict(y_ = x1_ + x2_)

