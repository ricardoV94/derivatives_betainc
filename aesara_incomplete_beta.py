import numpy as np
import aesara.tensor as at
from aesara.scan import until, scan
from aesara.tensor import gammaln

def incomplete_beta_cfe(a, b, x, small):
    """Incomplete beta continued fraction expansions
    based on Cephes library by Steve Moshier (incbet.c).
    small: Choose element-wise which continued fraction expansion to use.
    """
    BIG = at.constant(4.503599627370496e15, dtype="float64")
    BIGINV = at.constant(2.22044604925031308085e-16, dtype="float64")
    THRESH = at.constant(3.0 * np.MachAr().eps, dtype="float64")

    zero = at.constant(0.0, dtype="float64")
    one = at.constant(1.0, dtype="float64")
    two = at.constant(2.0, dtype="float64")

    r = one
    k1 = a
    k3 = a
    k4 = a + one
    k5 = one
    k8 = a + two

    k2 = at.switch(small, a + b, b - one)
    k6 = at.switch(small, b - one, a + b)
    k7 = at.switch(small, k4, a + one)
    k26update = at.switch(small, one, -one)
    x = at.switch(small, x, x / (one - x))

    pkm2 = zero
    qkm2 = one
    pkm1 = one
    qkm1 = one
    r = one

    def _step(i, pkm1, pkm2, qkm1, qkm2, k1, k2, k3, k4, k5, k6, k7, k8, r, a, b, x, small):
        xk = -(x * k1 * k2) / (k3 * k4)
        pk = pkm1 + pkm2 * xk
        qk = qkm1 + qkm2 * xk
        pkm2 = pkm1
        pkm1 = pk
        qkm2 = qkm1
        qkm1 = qk

        xk = (x * k5 * k6) / (k7 * k8)
        pk = pkm1 + pkm2 * xk
        qk = qkm1 + qkm2 * xk
        pkm2 = pkm1
        pkm1 = pk
        qkm2 = qkm1
        qkm1 = qk

        old_r = r
        r = at.switch(at.eq(qk, zero), r, pk / qk)

        k1 += one
        k2 += k26update
        k3 += two
        k4 += two
        k5 += one
        k6 -= k26update
        k7 += two
        k8 += two

        big_cond = at.gt(at.abs_(qk) + at.abs_(pk), BIG)
        biginv_cond = at.or_(at.lt(at.abs_(qk), BIGINV), at.lt(at.abs_(pk), BIGINV))

        pkm2 = at.switch(big_cond, pkm2 * BIGINV, pkm2)
        pkm1 = at.switch(big_cond, pkm1 * BIGINV, pkm1)
        qkm2 = at.switch(big_cond, qkm2 * BIGINV, qkm2)
        qkm1 = at.switch(big_cond, qkm1 * BIGINV, qkm1)

        pkm2 = at.switch(biginv_cond, pkm2 * BIG, pkm2)
        pkm1 = at.switch(biginv_cond, pkm1 * BIG, pkm1)
        qkm2 = at.switch(biginv_cond, qkm2 * BIG, qkm2)
        qkm1 = at.switch(biginv_cond, qkm1 * BIG, qkm1)

        return (
            (pkm1, pkm2, qkm1, qkm2, k1, k2, k3, k4, k5, k6, k7, k8, r),
            until(at.abs_(old_r - r) < (THRESH * at.abs_(r))),
        )

    (pkm1, pkm2, qkm1, qkm2, k1, k2, k3, k4, k5, k6, k7, k8, r), _ = scan(
        _step,
        sequences=[at.arange(0, 300)],
        outputs_info=[
            e
            for e in at.cast((pkm1, pkm2, qkm1, qkm2, k1, k2, k3, k4, k5, k6, k7, k8, r), "float64")
        ],
        non_sequences=[a, b, x, small],
    )

    return r[-1]


def incomplete_beta_ps(a, b, value):
    """Power series for incomplete beta
    Use when b*x is small and value not too close to 1.
    Based on Cephes library by Steve Moshier (incbet.c)
    """
    one = at.constant(1, dtype="float64")
    ai = one / a
    u = (one - b) * value
    t1 = u / (a + one)
    t = u
    threshold = np.MachAr().eps * ai
    s = at.constant(0, dtype="float64")

    def _step(i, t, s, a, b, value):
        t *= (i - b) * value / i
        step = t / (a + i)
        s += step
        return ((t, s), until(at.abs_(step) < threshold))

    (t, s), _ = scan(
        _step,
        sequences=[at.arange(2, 302)],
        outputs_info=[e for e in at.cast((t, s), "float64")],
        non_sequences=[a, b, value],
    )

    s = s[-1] + t1 + ai

    t = gammaln(a + b) - gammaln(a) - gammaln(b) + a * at.log(value) + at.log(s)
    return at.exp(t)


def incomplete_beta(a, b, value):
    """Incomplete beta implementation
    Power series and continued fraction expansions chosen for best numerical
    convergence across the board based on inputs.
    """
    machep = at.constant(np.MachAr().eps, dtype="float64")
    one = at.constant(1, dtype="float64")
    w = one - value

    ps = incomplete_beta_ps(a, b, value)

    flip = at.gt(value, (a / (a + b)))
    aa, bb = a, b
    a = at.switch(flip, bb, aa)
    b = at.switch(flip, aa, bb)
    xc = at.switch(flip, value, w)
    x = at.switch(flip, w, value)

    tps = incomplete_beta_ps(a, b, x)
    tps = at.switch(at.le(tps, machep), one - machep, one - tps)

    # Choose which continued fraction expansion for best convergence.
    small = at.lt(x * (a + b - 2.0) - (a - one), 0.0)
    cfe = incomplete_beta_cfe(a, b, x, small)
    w = at.switch(small, cfe, cfe / xc)

    # Direct incomplete beta accounting for flipped a, b.
    t = at.exp(
        a * at.log(x) + b * at.log(xc) + gammaln(a + b) - gammaln(a) - gammaln(b) + at.log(w / a)
    )

    t = at.switch(flip, at.switch(at.le(t, machep), one - machep, one - t), t)
    return at.switch(
        at.and_(flip, at.and_(at.le((b * x), one), at.le(x, 0.95))),
        tps,
        at.switch(at.and_(at.le(b * value, one), at.le(value, 0.95)), ps, t),
    )

