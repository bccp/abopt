import weakref

from .error import ModelError
from .symbol import Symbol, Literal

class Primitive(object):
    """ Instantiation of a primitive creates a node on a model """

    def _infer_model(self, **kwargs):
        kls = type(self)
        model = None

        for argname in kls.ain:
            if not argname in kwargs: raise ModelError("input argument '%s' not provided" % argname)

            var = kwargs[argname]
            if isinstance(var, Symbol):
                model = var.model
                self._model = weakref.ref(model)

        if model is None:
            for argname in kls.aout:
                if argname not in kwargs: continue
                var = kwargs[argname]
                if isinstance(var, Symbol):
                    model = var.model
                    self._model = weakref.ref(model)

        if model is None:
            raise ModelError("Cannot infer model from variables -- avoid using all Literal node")

        return model

    def __init__(self, **kwargs):

        kls = type(self)

        self.varin = {}
        self.varin_info = {} # currently only the reference id of the symbol
        self.varout = {}
        self.kwargs = {}

        model = self._infer_model(**kwargs)
        self._name = model.unique_name(kls.__name__)

        for argname in kls.ain:
            var = kwargs[argname]

            if not isinstance(var, Symbol):
                # automaticlaly make literal symbols
                var = Literal(model, var)

            # checking symbol references
            #if not isinstance(var, Literal):
                #print(self._name, var.name, id(var), id(model.get(var.name)))

            var.add_reference(self)
            self.varin[argname] = var
            self.varin_info[argname] = len(var.references)

        for argname in kls.aout:
            if not argname in kwargs:
                # if a name is not supplied, generate a name
                varname = self.name + '-' + argname
                var = model.define(varname)
            else:
                # already given a symbol, overwrite it
                var = kwargs[argname]
                # make a new symbol of the same name
                var = model.define(var.name)

            self.varout[argname] = var

        # record all 'side' arguments that do not go into derivatives.
        for k, v in kwargs.items():
            if k not in kls.ain and k not in kls.aout:
                self.kwargs[k] = v
            setattr(self, k, v)

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

    def execute(self, context, tape):
        kwargs = {}
        for argname, var in self.varin.items():
            kwargs[argname] = var.resolve(context)

        tape.append(self, kwargs)

        r = type(self).impl(self, **kwargs)
        for argname, var in self.varout.items():
            context[var.name] = r[argname]

