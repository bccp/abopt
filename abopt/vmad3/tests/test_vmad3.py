from abopt.vmad3.model import Model, Literal

from abopt.vmad3.basics import add


def test_vmad3():

    with Model() as model:
        x, y = model.input('x', 'y')

        z = add.opr(x, y)
        z = add.opr(z, Literal(1.0))

        model.output('z' = z)


    z, t = model.compute('z', init=dict(x=3.0, y=1.0), return_tape=True)


    vjp_model = tape.get_vjp()

    vjp_model.compute(('_x', '_y'), init=dict(_z=1.0))

    jvp_model = tape.get_jvp()

    jvp_model.compute('_z', init=dict(_x=1.0, _y=1.0))



