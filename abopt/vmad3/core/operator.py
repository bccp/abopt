"""
    Routines to define an operator

    use @operator decorator on a class to define an operator.

    Example: see the source code of :class:`add`

    use @nested.modeloperator to define a nested operator, where
    you only need to define a model and the ain, aout

"""

class Operator(object): pass

def record_copy_all(self, **kwargs):
    """ A default rcd implementation that copies all kwargs to the tape.

        the impl is used for the apl primitives if no rcd is given;
        the impl is also used for the vjp and vjp primitives.
    """
    return kwargs


def unbound(method):
    if hasattr(method, 'im_func'):
        # python 2.7 has this unbound method thing
        return method.im_func
    # python 3, cool
    return method

def operator(kls):
    """ Decorator to declare an operator. 

        The decorator is similar to a meta-class. It produces a
        new class with Operator as a baseclass, and apl, jvp and vjp are
        converted to primitives.

        An operator must define `ain, aout` and apl, vjp, jvp functions.

        ain : dict(name => type_pattern) describes the input arguments of
              the operator
        aout : dict(name => type_pattern) describes the output arguments of
              the operator

        Currently the type_pattern is not used; the plan is to add multi-dispatch
        if it is proven to be useful.

        apl : function(self, ...) the application of the operator;
              it shall return a dictionary
              of the evaluated values (exactly the same number of aout);
              except when there is only one output argument, then
              the result can be returned directly.

              all input arguments are resolved to python objects;
              it can have extra arguments in addition to ain.
              self is the node object that is used in the model

        rcd : function(self, ...) recording the arguments for
              invoking jvp and vjp. It shall return a dict.
              the only items included in the dict are available
              to vjp and vjp; if not defined, all arguments to apl are recorded.

        jvp : function(self, ...) the jacobian vector product. The convention
              is to use '_' + argname as the name of vectors. used for back-prop.
              self is the node object that is used in the model

        vjp : function(self, ...) the vector jacobian product. The convention
              is to use argname + '_' as the name of vectors. used for foward-prop.
              self is the node object that is used in the model

    """

    if hasattr(kls, 'rcd'):
        record_impl = unbound(kls.rcd)
    else:
        record_impl = record_copy_all

    kls._apl = _make_primitive(kls, 'apl', unbound(kls.apl),
        record_impl=record_impl)

    kls._vjp = _make_primitive(kls, 'vjp', unbound(kls.vjp))
    kls._jvp = _make_primitive(kls, 'jvp', unbound(kls.jvp))

    return type(kls.__name__, (Operator, kls, kls._apl), {})

def zerobypass(impl):
    def zerobypassimpl(self, **kwargs):
        ain = type(self).ain
        aout = type(self).aout
        if all(kwargs[argname] is 0 for argname in ain):
            d = {}
            for argname in aout:
                d[argname] = 0
            return d
        return impl(self, **kwargs)
    return zerobypassimpl

def _make_primitive(operator, func, impl, argnames=None, record_impl=record_copy_all):
    """ create primitives for the operator.

        This is used to define a primitive based on the unbound method
        defined in the operator class.

    """
    from .primitive import Primitive
    from .symbol import Symbol
    from collections import OrderedDict

    assert func in ('apl', 'vjp', 'jvp')

    kls = operator

    kls.ain = OrderedDict(kls.ain)
    kls.aout = OrderedDict(kls.aout)

    aout = {}
    ain = {}
    if argnames is None:
        argnames = impl.__code__.co_varnames[1:impl.__code__.co_argcount]

    if func == 'apl':
        ain = kls.ain
        aout = kls.aout
    elif func == 'vjp' : # in and out are prefixed.
        for arg in kls.ain:
            aout['_' + arg] = kls.ain[arg]

        for arg in kls.aout:
            ain['_' + arg] = kls.aout[arg]
        impl = zerobypass(impl)

    elif func == 'jvp' : # in and out are prefixed.
        for arg in kls.ain:
            ain[arg + '_'] = kls.ain[arg]

        for arg in kls.aout:
            aout[arg + '_'] = kls.aout[arg]
        impl = zerobypass(impl)

    members =  dict(
                impl     = impl,
                record_impl     = record_impl,
                func     = func,
                ain      = ain,
                aout     = aout,
                argnames = argnames,
                operator = operator,
                )

    bases = (Primitive,)

    primitive = type(operator.__name__ + '-' + func,
            bases,
            members
            )
    return primitive

# special operator used for partial gradient summation
@operator
class add:
    ain  = {'x1': '*',
            'x2': '*',
           }
    aout = {'y': '*'}

    def apl(self, x1, x2):
        return dict(y = x1 + x2)

    def vjp(self, _y):
        return dict(_x1 = _y, _x2 = _y)

    def jvp(self, x1_, x2_):
        return dict(y_ = x1_ + x2_)

# special operator for marking an output
@operator
class terminal:
    ain  = {'x': '*'}
    aout = {'y': '*'}

    def apl(self, x):
        return dict(y=x)
    def vjp(self, _y):
        return dict(_x=_y)
    def jvp(self, x_):
        return dict(y_=x_)

