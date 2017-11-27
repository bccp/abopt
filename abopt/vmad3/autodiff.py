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
        model.input('_' + var.name)

    for i, (p, impl_kwargs) in enumerate(tape[::-1]):
        vjp_of_p = find_primitive_type(p, func='vjp')

        kwargs = {}
        kwargs.update(p.kwargs)

        for k, v in impl_kwargs.items():
            kwargs[k] = Literal(model, v)

        for argname, var in p.varout.items():
            kwargs['_' + argname] = model.get('_' + var.name)

        for argname, var in p.varin.items():
            # bypass literal arguments
            if isinstance(var, Literal): continue

            reference_id = p.varin_info[argname]
            if reference_id == len(var.references):
                # largest reference_id, must be the
                # first time seeing the partial derivative
                # define the symbol for the full derivative
                var_p = model.define('_' + var.name)
            else:
                var_p = model.define('_' + var.name + '#%d' % reference_id)

            kwargs['_' + argname] = var_p

        node = vjp_of_p(**kwargs)

        for argname, var in p.varin.items():
            # bypass literal arguments
            if isinstance(var, Literal): continue
            reference_id = p.varin_info[argname]
            # cummulate the partials
            if reference_id != len(var.references):

                var_f = model.get('_' + var.name)
                var_p = model.get('_' + var.name + '#%d' % reference_id)

                add(x1=var_f, x2=var_p, y=var_f)

    for var in tape.model._vin:
        if not model.has('_' + var.name):
            varout = ZeroLiteral(model)
        else:
            varout = model.get('_' + var.name)
        model.output(**{'_' + var.name : varout})

    return model

