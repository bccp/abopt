from .operator import _make_primitive, Operator, unbound
from .context import Context
from .model import Builder

def modeloperator(kls):
    """ Create a nested operator

        ain : input arguments
        aout : output arguments

        model : function(model, ...) building the model; returns the dict of output arguments.

        see the example below in this file.
    """

    impl = unbound(kls.model)

    # use the argnames of the model
    argnames = impl.__code__.co_varnames[1:impl.__code__.co_argcount]
    argnames_vjp = list(argnames)
    argnames_jvp = list(argnames)

    # add v arguments
    for argname in kls.aout:
        argnames_vjp.append('_' + argname)
    for argname in kls.ain:
        argnames_jvp.append(argname + '_')

    def _build(kwargs):

        model_args = {}
        # copy extra args of the model function
        for argname in argnames:
            if argname not in kls.ain:
                model_args[argname] = kwargs[argname]

        with Builder() as m:
            # add input args as variables
            for argname in kls.ain:
                model_args[argname] = m.input(argname)
            r = impl(m, **model_args)
            # assert outputs are generated
            for argname in kls.aout:
                if argname not in r:
                    raise ModelError("output arg '%s' is not produced by the model" % argname)
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
        m = _build(kwargs)
        y = compute(self, m, False, **kwargs)
        return y

    def vjp(self, **kwargs):
        m = _build(kwargs)
        y, tape = compute(self, m, True, **kwargs)

        m = tape.get_vjp()
        y = compute(self, m, False, **kwargs)
        return y

    def jvp(self, **kwargs):
        m = _build(kwargs)
        y, tape = compute(self, m, True, **kwargs)

        m = tape.get_jvp()
        y = compute(self, m, False, **kwargs)
        return y

    kls._opr = _make_primitive(kls, 'opr', opr, argnames=argnames)
    kls._vjp = _make_primitive(kls, 'vjp', vjp, argnames=argnames_vjp)
    kls._jvp = _make_primitive(kls, 'jvp', jvp, argnames=argnames_jvp)

    # FIXME: add docstring / argnames
    # shall be the list of extra args
    def build(**kwargs):
        for argname in kwargs:
            if argname in kls.ain:
                raise ModelError("argname %s is an input, shall not be used to produce a model" % argname)

        return _build(kwargs)

    return type(kls.__name__, (Operator, kls, kls._opr), {'build':build})

@modeloperator
class example:
    ain  = {'x': '*'}
    aout = {'y': '*'}

    # must take both extra parameters and input parameters
    def model(self, x, n):
        from .operator import add

        for i in range(n):
            x = add(x1=x, x2=x)
        return dict(y=x)

