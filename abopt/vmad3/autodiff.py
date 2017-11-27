from .symbol import ZeroLiteral, Literal, Symbol
from .model import Model
from .operator import terminal, add

def find_primitive_type(p, func):
    # we will only do this on the opr primitives
    # because otherwise this is undefined
    # the algebra of autodiff in vmad3 is explicitly not closed!
    assert isinstance(p, type(p).operator.opr)

    assert func in ['vjp', 'jvp', 'opr']

    if func == 'jvp': return p.operator.jvp
    if func == 'vjp': return p.operator.vjp
    if func == 'opr': return p.operator.opr

def vjp(tape):
    model = Model()
    for var in tape.model._vout:
        model.input(var.vjp_name)

    for i, record in enumerate(tape[::-1]):
        p = record.node
        impl_kwargs = record.impl_kwargs

        vjp_of_p = find_primitive_type(p, func='vjp')

        kwargs = {}
        kwargs.update(p.kwargs)

        # convert original arguments to literals
        for k, v in impl_kwargs.items():
            kwargs[k] = Literal(model, v)

        # initialize 'v'
        for argname, var in p.varout.items():
            kwargs['_' + argname] = model.get(var.vjp_name)

        # create output vjps
        for argname, var in p.varin.items():
            # bypass literal arguments
            if isinstance(var, Literal): continue

            reference_id = p.varin_info[argname]
            if reference_id == len(var.references):
                # largest reference_id, must be the
                # first time seeing the partial derivative
                # define the symbol for the full derivative
                var_p = model.define(var.vjp_name)
            else:
                var_p = model.define(var.vjp_name + '#%d' % reference_id)

            kwargs['_' + argname] = var_p

        node = vjp_of_p(**kwargs)

        # combine partial derivatives.
        for argname, var in p.varin.items():
            # bypass literal arguments
            if isinstance(var, Literal): continue
            reference_id = p.varin_info[argname]
            # cummulate the partials
            if reference_id != len(var.references):

                var_f = model.get(var.vjp_name)
                var_p = model.get(var.vjp_name + '#%d' % reference_id)

                add(x1=var_f, x2=var_p, y=var_f)

    # mark outputs
    for var in tape.model._vin:
        if not model.has(var.vjp_name):
            varout = ZeroLiteral(model)
        else:
            varout = model.get(var.vjp_name)
        model.output(**{var.vjp_name : varout})

    return model

