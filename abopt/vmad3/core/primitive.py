import weakref

from .error import InferError, UnpackError, OverwritePrecaution, MissingArgument, BrokenPrimitive, BadArgument
from .symbol import Symbol, Literal, List

def make_symbol(model, var):
    if isinstance(var, Primitive):
        # unpack the primitive result
        #
        # see __iter__ if explict unpack (a, b = primitive(...))
        # is used.

        if len(var.varout) > 1:
            raise UnpackError("More than one output variable, need to unpack them")
        var = next(iter(var.varout.values()))

    if isinstance(var, (list, tuple)):
        var = List(model, [make_symbol(model, v) for v in var])

    if not isinstance(var, Symbol):
        var = Literal(model, var)

    return var

def _infer_model(var):
    if isinstance(var, Primitive):
        var = next(iter(var.varout.values()))

    if isinstance(var, Symbol):
        model = var.model
        return model

    if isinstance(var, (list, tuple)):
        for v in var:
            model = _infer_model(v)
            if model is not None:
                return model

    return None

def check_var_references(var):
    if isinstance(var, List):
        for v in var.value:
            check_var_references(v)
        return

    if len(var.references) != 0:
        raise OverwritePrecaution("Overwritting used symbols is not supported. Because it breaks vjp.")

class Primitive(object):
    """ Primitives are building blocks of models.

        Instantiation of a primitive creates a node on a model.

        This is the base class for all operators. Primitive classes are generated
        and attached to the operators classes via the `operator` decorator.

    """

    def _scan_kwargs(self, kwargs):
        kls = type(self)
        model = None

        for argname in kls.ain:
            if not argname in kwargs: raise MissingArgument("input argument '%s' not provided" % argname)

            var = kwargs[argname]
            model = _infer_model(var)
            if model is not None:
                self._model = weakref.ref(model)
                return model

        for argname in kls.aout:
            if argname not in kwargs: continue
            var = kwargs[argname]

            model = _infer_model(var)
            if model is not None:
                self._model = weakref.ref(model)
                return model

        raise InferError("Cannot infer model from variables -- try to mark at least one literal argument explicitly as Literal")

    def __init__(self, *args, **kwargs):

        kls = type(self)

        # assert the primitive is properly defined.
        for attr in ['ain', 'aout', 'impl', 'func', 'argnames', 'operator']:
            if not hasattr(kls, attr):
                raise BrokenPrimitive("primitive class attribute '%s' is not defined" % attr)

        self.varin = {}
        self.varout = {}
        self.hyper_args = {}

        kwargs = kwargs.copy() # will modify

        # first attempt to map args into kwargs
        if len(args) > len(kls.argnames):
            raise BadArgument("Argument list longer than total number of args")

        for argname, arg in zip(kls.argnames, args):
            if argname in kwargs:
                raise BadArgument("argument %s already provided as keyword" % argname)

            kwargs[argname] = arg

        model = self._scan_kwargs(kwargs)
        self._name = model.unique_name(kls.__name__)

        for argname in kls.ain:
            var = make_symbol(model, kwargs[argname])

            # checking symbol references
            #print(self._name, var.name, id(var), id(model.get(var.name)))

            ref = var.add_reference(self)
            self.varin[argname] = ref

        for argname in kls.aout:
            if not argname in kwargs:
                # if a name is not supplied, generate a name
                varname = self.name + '-' + argname
                var = model.define(varname)
            else:
                var = kwargs[argname]
                var = make_symbol(model, kwargs[argname])
                # already given a symbol, overwrite it
                # but this doesn't work for gradients
                check_var_references(var)
                # make a new symbol of the same name
                # var = model.define(var.name)

            self.varout[argname] = var

        # record all `hyper` arguments that do not go into derivatives.
        for k, v in kwargs.items():
            if k not in kls.ain and k not in kls.aout:
                self.hyper_args[k] = v

        model.append(self)

    def __iter__(self):
        """ for unpacking the results in model building """
        for argname in self.aout:
            yield self.varout[argname]

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return "%s(%s=>%s)" % (self._name, self.varin, self.varout)

    def call(self, **kwargs):
        """ call the implementation function of the primitive """
        r = type(self).impl(self, **kwargs)

        # allow returning without using a dict
        # if there is only a single output argument
        if not isinstance(r, dict) and len(self.varout) == 1:
            argname = next(iter(self.varout.keys()))
            r = {argname:r}

        return r

    def record(self, kwargs):
        """ generate the kwargs that goes into the tape """
        if type(self).record_impl is not None:
            return type(self).record_impl(self, **kwargs)
        else:
            return kwargs

