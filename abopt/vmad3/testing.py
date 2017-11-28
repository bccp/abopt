from abopt.vmad3 import Builder
from abopt.vmad3 import Context
from abopt.vmad3.operator import to_scalar as default_to_scalar
from numpy.testing import assert_array_equal, assert_allclose

class BaseScalarTest:
    to_scalar = default_to_scalar
    def setup(self):
        with Builder() as m:
            x = m.input('x')
            x = self.model(x)
            y = self.to_scalar(x)
            m.output(y=y)
        self.m = m

    def test_opr(self):
        ctx = Context(x=self.x)
        y1 = ctx.compute(self.m, vout='y', return_tape=False)
        # correctness
        assert_allclose(y1, self.y)

    def test_vjp(self):
        ctx = Context(x=self.x)
        y1, tape = ctx.compute(self.m, vout='y', return_tape=True)

        vjp = tape.get_vjp()
        ctx = Context(_y=self._y)
        _x1 = ctx.compute(vjp, vout='_x')

        assert_allclose(self._x, _x1)

    def test_jvp(self):
        ctx = Context(x=self.x)
        y1, tape = ctx.compute(self.m, vout='y', return_tape=True)

        jvp = tape.get_jvp()
        ctx = Context(x_=self.x_)
        y_1, t1 = ctx.compute(jvp, vout='y_', return_tape=True)

        assert_allclose(self.y_, y_1)

