from abopt.vmad3 import Builder
from abopt.vmad3 import Context
from abopt.vmad3.operator import add
from pprint import pprint

def test_model():
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

