from .operator import _make_primitive, Operator
from .context import Context
from .model import ModelBuilder

def nestedoperator(kls):
    """ Create a nested operator

        ain : input arguments
        aout : output arguments

        model : function(model, ...) building the model; returns the dict of output arguments.

        see the example below in this file.
    """
    impl = kls.model

    # use the argnames of the model
    argnames = impl.__code__.co_varnames[1:impl.__code__.co_argcount]
    argnames_vjp = list(argnames)
    argnames_jvp = list(argnames)

    # add v arguments
    for argname in kls.aout:
        argnames_vjp.append('_' + argname)
    for argname in kls.ain:
        argnames_jvp.append(argname + '_')

    def model(kwargs):
        model_args = {}
        for argname in argnames:
            model_args[argname] = kwargs[argname]

        with ModelBuilder() as m:
            for argname in kls.ain:
                model_args[argname] = m.input(argname)
            r = impl(m, **model_args)
            m.output(**r)
        return m

    def compute(self, m, return_tape, **kwargs):
        vout = [var.name for var in m._vout]
        vin  = [var.name for var in m._vin]

        init = {}
        for varname in vin:
            init[varname] = kwargs[varname]

        ctx = Context(**init)
        if return_tape:
            r1, tape = ctx.compute(m, vout=vout, return_tape=True)
        else:
            r1 = ctx.compute(m, vout=vout, return_tape=False)

        r = {}
        for varname, value in zip(vout, r1):
            r[varname] = value

        if return_tape:
            r = r, tape
        return r

    def opr(self, **kwargs):
        m = model(kwargs)
        y = compute(self, m, False, **kwargs)
        return y

    def vjp(self, **kwargs):
        m = model(kwargs)
        y, tape = compute(self, m, True, **kwargs)

        m = tape.get_vjp()
        y = compute(self, m, False, **kwargs)
        return y

    def jvp(self, **kwargs):
        m = model(kwargs)
        y, tape = compute(self, m, True, **kwargs)

        m = tape.get_jvp()
        y = compute(self, m, False, **kwargs)
        return y

    kls.opr = _make_primitive(kls, 'opr', opr, argnames=argnames)
    kls.vjp = _make_primitive(kls, 'vjp', vjp, argnames=argnames_vjp)
    kls.jvp = _make_primitive(kls, 'jvp', jvp, argnames=argnames_jvp)

    return type(kls.__name__, (Operator, kls, kls.opr), {})

from .operator import add
@nestedoperator
class example:
    ain  = {'x': '*'}
    aout = {'y': '*'}

    # must declare all arguments
    def model(self, x, n):
        for i in range(n):
            x = add(x1=x, x2=x)
        return dict(y=x)

