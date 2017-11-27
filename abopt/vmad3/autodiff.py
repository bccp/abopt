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

def prepare_opr_kwargs(record, model):
    """ generate a first guess of kwargs based on the record.

    """
    p = record.node
    resolved = record.resolved

    kwargs = {}

    kwargs.update(p.kwargs)

    # add resolved symbols as literals
    for k, v in resolved.items():
        # v is a python object
        assert k in p.varin
        # if we expect an input, convert it to a literal
        kwargs[k] = Literal(model, v)

    return kwargs

def vjp(tape):
    """ generate a vector jacobian product model based on a tape """
    model = Model()
    for var in tape.model._vout:
        model.input(var.vjp_name)

    for i, record in enumerate(tape[::-1]):
        p = record.node
        resolved = record.resolved

        vjp_of_p = find_primitive_type(p, func='vjp')

        kwargs = prepare_opr_kwargs(record, model)

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
            # accummulate the partials
            if reference_id != len(var.references):
                var_f = model.get(var.vjp_name)
                var_p = model.get(var.vjp_name + '#%d' % reference_id)
                # create a new symbol for the result, with the same name
                # because we intent to overwrite it.
                var_f2 = model.define(var.vjp_name)

                add(x1=var_f, x2=var_p, y=var_f2)

    # mark outputs
    for var in tape.model._vin:
        if not model.has(var.vjp_name):
            varout = ZeroLiteral(model)
        else:
            varout = model.get(var.vjp_name)
        model.output(**{var.vjp_name : varout})

    return model

def jvp(tape):
    """ generate a jacobian vector product model based on a tape """
    model = Model()
    for var in tape.model._vin:
        model.input(var.jvp_name)

    for i, record in enumerate(tape):
        p = record.node
        resolved = record.resolved

        jvp_of_p = find_primitive_type(p, func='jvp')

        kwargs = prepare_opr_kwargs(record, model)

        # initialize 'v'
        for argname, var in p.varin.items():
            if isinstance(var, Literal):
                jvp_var = ZeroLiteral(model)
            else:
                jvp_var = model.get(var.jvp_name)
            kwargs[argname + '_'] = jvp_var

        # create output symbols
        for argname, var in p.varout.items():
            jvp_var = model.define(var.jvp_name)
            kwargs[argname + '_'] = jvp_var

        jvp_of_p(**kwargs)

    # mark outputs
    for var in tape.model._vout:
        if not model.has(var.jvp_name):
            varout = ZeroLiteral(model)
        else:
            varout = model.get(var.jvp_name)
        model.output(**{var.jvp_name : varout})

    return model
