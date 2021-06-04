import numpy as np
import scipy.special

import aesara.scalar as aes
from aesara.scalar import ScalarOp, upgrade_to_float_no_complex
from aesara.tensor.elemwise import Elemwise

def _betainc_a_n(f, p, q, n):
    """
    Numerator (a_n) of the nth approximant of the continued fraction
    representation of the regularized incomplete beta function
    """

    if n == 1:
        return p * f * (q - 1) / (q * (p + 1))

    p2n = p + 2 * n
    F1 = p ** 2 * f ** 2 * (n - 1) / (q ** 2)
    F2 = (p + q + n - 2) * (p + n - 1) * (q - n) / ((p2n - 3) * (p2n - 2) ** 2 * (p2n - 1))

    return F1 * F2


def _betainc_b_n(f, p, q, n):
    """
    Offset (b_n) of the nth approximant of the continued fraction
    representation of the regularized incomplete beta function
    """
    pf = p * f
    p2n = p + 2 * n

    N1 = 2 * (pf + 2 * q) * n * (n + p - 1) + p * q * (p - 2 - pf)
    D1 = q * (p2n - 2) * p2n

    return N1 / D1


def _betainc_da_n_dp(f, p, q, n):
    """
    Derivative of a_n wrt p
    """

    if n == 1:
        return -p * f * (q - 1) / (q * (p + 1) ** 2)

    pp = p ** 2
    ppp = pp * p
    p2n = p + 2 * n

    N1 = -(n - 1) * f ** 2 * pp * (q - n)
    N2a = (-8 + 8 * p + 8 * q) * n ** 3
    N2b = (16 * pp + (-44 + 20 * q) * p + 26 - 24 * q) * n ** 2
    N2c = (10 * ppp + (14 * q - 46) * pp + (-40 * q + 66) * p - 28 + 24 * q) * n
    N2d = 2 * pp ** 2 + (-13 + 3 * q) * ppp + (-14 * q + 30) * pp
    N2e = (-29 + 19 * q) * p + 10 - 8 * q

    D1 = q ** 2 * (p2n - 3) ** 2
    D2 = (p2n - 2) ** 3 * (p2n - 1) ** 2

    return (N1 / D1) * (N2a + N2b + N2c + N2d + N2e) / D2


def _betainc_da_n_dq(f, p, q, n):
    """
    Derivative of a_n wrt q
    """
    if n == 1:
        return p * f / (q * (p + 1))

    p2n = p + 2 * n
    F1 = (p ** 2 * f ** 2 / (q ** 2)) * (n - 1) * (p + n - 1) * (2 * q + p - 2)
    D1 = (p2n - 3) * (p2n - 2) ** 2 * (p2n - 1)

    return F1 / D1


def _betainc_db_n_dp(f, p, q, n):
    """
    Derivative of b_n wrt p
    """
    p2n = p + 2 * n
    pp = p ** 2
    q4 = 4 * q
    p4 = 4 * p

    F1 = (p * f / q) * ((-p4 - q4 + 4) * n ** 2 + (p4 - 4 + q4 - 2 * pp) * n + pp * q)
    D1 = (p2n - 2) ** 2 * p2n ** 2

    return F1 / D1


def _betainc_db_n_dq(f, p, q, n):
    """
    Derivative of b_n wrt to q
    """
    p2n = p + 2 * n
    return -(p ** 2 * f) / (q * (p2n - 2) * p2n)


def _betainc_derivative(x, p, q, wrtp=True):
    """
    Compute the derivative of regularized incomplete beta function wrt to p (alpha) or q (beta)

    Reference: Boik, R. J., & Robison-Cox, J. F. (1998). Derivatives of the incomplete beta function.
    Journal of Statistical Software, 3(1), 1-20.
    """

    # Input validation
    if not (0 <= x <= 1) or p < 0 or q < 0:
        return np.nan

    if x > (p / (p + q)):
        return -_betainc_derivative(1 - x, q, p, not wrtp)

    min_iters = 3
    max_iters = 200
    err_threshold = 1e-12

    derivative_old = 0

    Am2, Am1 = 1, 1
    Bm2, Bm1 = 0, 1
    dAm2, dAm1 = 0, 0
    dBm2, dBm1 = 0, 0

    f = (q * x) / (p * (1 - x))
    K = np.exp(p * np.log(x) + (q - 1) * np.log1p(-x) - np.log(p) - scipy.special.betaln(p, q))
    if wrtp:
        dK = np.log(x) - 1 / p + scipy.special.digamma(p + q) - scipy.special.digamma(p)
    else:
        dK = np.log1p(-x) + scipy.special.digamma(p + q) - scipy.special.digamma(q)

    for n in range(1, max_iters + 1):
        a_n_ = _betainc_a_n(f, p, q, n)
        b_n_ = _betainc_b_n(f, p, q, n)
        if wrtp:
            da_n = _betainc_da_n_dp(f, p, q, n)
            db_n = _betainc_db_n_dp(f, p, q, n)
        else:
            da_n = _betainc_da_n_dq(f, p, q, n)
            db_n = _betainc_db_n_dq(f, p, q, n)

        A = a_n_ * Am2 + b_n_ * Am1
        B = a_n_ * Bm2 + b_n_ * Bm1
        dA = da_n * Am2 + a_n_ * dAm2 + db_n * Am1 + b_n_ * dAm1
        dB = da_n * Bm2 + a_n_ * dBm2 + db_n * Bm1 + b_n_ * dBm1

        Am2, Am1 = Am1, A
        Bm2, Bm1 = Bm1, B
        dAm2, dAm1 = dAm1, dA
        dBm2, dBm1 = dBm1, dB

        if n < min_iters - 1:
            continue

        F1 = A / B
        F2 = (dA - F1 * dB) / B
        derivative = K * (F1 * dK + F2)

        errapx = abs(derivative_old - derivative)
        d_errapx = errapx / max(err_threshold, abs(derivative))
        derivative_old = derivative

        if d_errapx <= err_threshold:
            break

        if n >= max_iters:
            return np.nan

    return derivative


class TernaryScalarOp(ScalarOp):
    nin = 3


class BetaIncDda(TernaryScalarOp):
    """
    Gradient of the regularized incomplete beta function wrt to the first argument (a)
    """

    def impl(self, a, b, z):
        return _betainc_derivative(z, a, b, wrtp=True)


class BetaIncDdb(TernaryScalarOp):
    """
    Gradient of the regularized incomplete beta function wrt to the second argument (b)
    """

    def impl(self, a, b, z):
        return _betainc_derivative(z, a, b, wrtp=False)


betainc_dda_scalar = BetaIncDda(upgrade_to_float_no_complex, name="betainc_dda")
betainc_ddb_scalar = BetaIncDdb(upgrade_to_float_no_complex, name="betainc_ddb")


class BetaInc(TernaryScalarOp):
    """
    Regularized incomplete beta function
    """

    nfunc_spec = ("scipy.special.betainc", 3, 1)

    def impl(self, a, b, x):
        return scipy.special.betainc(a, b, x)

    def grad(self, inp, grads):
        a, b, z = inp
        (gz,) = grads

        return [
            gz * betainc_dda_scalar(a, b, z),
            gz * betainc_ddb_scalar(a, b, z),
            gz
            * aes.exp(
                aes.log1p(-z) * (b - 1)
                + aes.log(z) * (a - 1)
                - (aes.gammaln(a) + aes.gammaln(b) - aes.gammaln(a + b))
            ),
        ]


betainc_scalar = BetaInc(upgrade_to_float_no_complex, "betainc")
betainc = Elemwise(betainc_scalar, name="Elemwise{betainc,no_inplace}")