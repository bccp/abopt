from __future__ import print_function
from abopt.vmad3 import Builder
from abopt.vmad3 import Context
from abopt.vmad3 import modeloperator
from abopt.vmad3.operator import add
from pprint import pprint

def test_vmad3_functional():
    """ this test demonstrates building a model directly """
    with Builder() as m:
        a, b = m.input('a', 'b')

        t1 = add(x1=a, x2=a)
        t2 = add(x1=b, x2=0)
        c = add(x1=t1, x2=t2)

        m.output(c=c)

    print("----- model -----")
    pprint(m)
    pprint(m[:])

    print("----- compute -----")
    ctx = Context(a=3, b=4)

    c = ctx.compute(m, vout='c')
    print(ctx, c)

    print("----- tape -----")
    ctx = Context(a=3, b=4)
    c, tape = ctx.compute(m, vout='c', return_tape=True)
    print(ctx, c)
    pprint(tape)

    print("----- vjp -----")
    vjp = tape.get_vjp()
    pprint(vjp)
    pprint(vjp[:])

    ctx = Context(_c=1.0)
    _a, _b = ctx.compute(vjp, vout=['_a', '_b'], monitor=print)
    print('_a, _b = ', _a, _b)

    print("----- jvp -----")
    jvp = tape.get_jvp()
    pprint(jvp)
    pprint(jvp[:])

    ctx = Context(a_=1.0, b_=1.0)
    c_, = ctx.compute(jvp, vout=['c_'], monitor=print)
    print('c_ = ', c_)

def test_modeloperator():
    """ this test demonstrates using modeloperator
        to build a model that can also be used as an operator.
    """
    @modeloperator
    class mymodel:
        ain = {'a' : '*',
               'b' : '*'}
        aout = {'c' : '*'}

        def model(model, a, b, n):
            for i in range(n):
                a = add(x1=a, x2=a)

            t2 = add(x1=b, x2=0)
            return dict(c=add(x1=a, x2=t2))

    m3 = mymodel.build(n=3)
    print("----- model 2-----")
    pprint(m3)
    ctx = Context(a=3, b=4)
    c = ctx.compute(m3, vout='c')
    assert c == 3 * 2 ** 3 + 4

    m2 = mymodel.build(n=2)
    print("----- model 3-----")
    pprint(m2)
    ctx = Context(a=3, b=4)
    c = ctx.compute(m2, vout='c')
    assert c == 3 * 2 ** 2 + 4

    # complicated model is longer
    assert len(m3[:]) > len(m2[:])


