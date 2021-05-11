import numpy as np
import scipy.special as sp


def _inv(x):
    return 1 / x


def _fma(x, y, z):
    return x * y + z


def _grad_2F1(a1, a2, b1, z, precision=1e-14, max_steps=1e6):
    # Implementation from https://github.com/stan-dev/math/blob/master/stan/math/prim/fun/grad_2F1.hpp
    g_a1 = 0.0
    g_b1 = 0.0

    log_g_old = [-np.inf, -np.inf]

    log_t_old = 0.0
    log_t_new = 0.0

    log_z = np.log(z)

    log_precision = np.log(precision)
    log_t_new_sign = 1.0
    log_t_old_sign = 1.0
    log_g_old_sign = [1.0, 1.0]

    for k in range(0, int(max_steps)):
        p = (a1 + k) * (a2 + k) / ((b1 + k) * (1 + k))
        if p == 0:
            return g_a1, g_b1

        log_t_new += np.log(np.abs(p)) + log_z
        log_t_new_sign = log_t_new_sign if p >= 0 else -log_t_new_sign

        term = log_g_old_sign[0] * log_t_old_sign * np.exp(log_g_old[0] - log_t_old) + _inv(a1 + k)
        log_g_old[0] = log_t_new + np.log(np.abs(term))
        log_g_old_sign[0] = log_t_new_sign if term >= 0 else -log_t_new_sign

        term = log_g_old_sign[1] * log_t_old_sign * np.exp(log_g_old[1] - log_t_old) - _inv(b1 + k)
        log_g_old[1] = log_t_new + np.log(np.abs(term))
        log_g_old_sign[1] = log_t_new_sign if term >= 0 else - log_t_new_sign

        g_a1 += np.exp(log_g_old[0]) if log_g_old_sign[0] > 0 else -np.exp(log_g_old[0])
        g_b1 += np.exp(log_g_old[1]) if log_g_old_sign[1] > 0 else -np.exp(log_g_old[1])

        if (
                log_g_old[0] <= max(np.log(np.abs(g_a1)) + log_precision, log_precision)
                and log_g_old[1] <= max(np.log(np.abs(g_b1)) + log_precision, log_precision)
            ):
            return g_a1, g_b1

        log_t_old = log_t_new
        log_t_old_sign = log_t_new_sign

    raise ValueError('Gradient of 2F1 did not converge')


def _grad_inc_beta(a, b, z):
    # Gradients of the incomplete beta function wrt to a and b
    # https://github.com/stan-dev/math/blob/master/stan/math/prim/fun/grad_inc_beta.hpp
    c1 = np.log(z)
    c2 = np.log1p(-z)
    c3 = sp.beta(a, b) * sp.betainc(a, b, z)
    C = np.exp(a * c1 + b * c2) / a
    if C:
        dF1, dF2 = _grad_2F1(a+b, 1.0, a+1.0, z)
    else:
        dF1, dF2 = 0, 0
    g1 = _fma((c1 - _inv(a)), c3, C * (dF1 + dF2))
    g2 = _fma(c2, c3, C * dF1)
    return g1, g2


def grad_reg_inc_beta(a, b, z, digammaA=None, digammaB=None, digammaSum=None, betaAB=None):
    # if digammaA is None:
    #     digammaA = sp.digamma(a)
    # if digammaB is None:
    #     digammaB = sp.digamma(b)
    # if digammaSum is None:
    #     digammaSum = sp.digamma(a + b)
    # if betaAB is None:
    #     betaAB = sp.beta(a, b)

    # Gradients of the regularized incomplete beta wrt to a and b
    # https://github.com/stan-dev/math/blob/master/stan/math/prim/fun/grad_reg_inc_beta.hpp
    dBda, dBdb = _grad_inc_beta(a, b, z)
    b1 = betaAB * sp.betainc(a, b, z)
    g1 = (dBda - b1 * (digammaA - digammaSum)) / betaAB
    g2 = (dBdb - b1 * (digammaB - digammaSum)) / betaAB
    return g1, g2


def inc_beta_dda(a, b, z, digamma_a, digamma_ab, threshold=1e-10, conv_threshold=1e5):
    # Gradient of the regularized incomplete beta wrt to a
    # https://github.com/stan-dev/math/blob/master/stan/math/prim/fun/inc_beta_dda.hpp
    if (b > a):
        if (
                (0.1 < z and z <= 0.75 and b > 500)
                or (0.01 < z and z <= 0.1 and b > 2500)
                or (0.001 < z and z <= 0.01 and b > 1e5)
        ):
            return -inc_beta_ddb(b, a, 1 - z, digamma_a, digamma_ab, threshold, conv_threshold)
    if (
            (z > 0.75 and a < 500)
            or (z > 0.9 and a < 2500)
            or (z > 0.99 and a < 1e5)
            or (z > 0.999)
    ):
        return -inc_beta_ddb(b, a, 1 - z, digamma_a, digamma_ab, threshold, conv_threshold)


    a_plus_b = a + b
    a_plus_1 = a + 1

    digamma_a += _inv(a)

    prefactor = (a_plus_1 / a_plus_b) ** 3

    sum_numer = (digamma_ab - digamma_a) * prefactor
    sum_denom = prefactor

    summand = prefactor * z * a_plus_b / a_plus_1

    k = 1
    digamma_ab += _inv(a_plus_b)
    digamma_a += _inv(a_plus_1)

    while (abs(summand) > threshold):
        sum_numer += (digamma_ab - digamma_a) * summand
        sum_denom += summand

        summand *= (1 + a_plus_b / k) * (1 + k) / (1 + a_plus_1 / k)
        digamma_ab += _inv(a_plus_b + k)
        digamma_a += _inv(a_plus_1 + k)
        k += 1
        summand *= z / k

        if k > conv_threshold:
            raise ValueError('inc_beta_dda did not converge')

    return sp.betainc(a, b, z) * (np.log(z) + sum_numer / sum_denom)


def inc_beta_ddb(a, b, z, digamma_b, digamma_ab, threshold=1e-10, conv_threshold=1e5):
    # Gradient of the regularized incomplete beta wrt to b
    # https://github.com/stan-dev/math/blob/master/stan/math/prim/fun/inc_beta_ddb.hpp
    if (b > a):
        if (
                (0.1 < z and z <= 0.75 and b > 500)
                or (0.01 < z and z <= 0.1 and b > 2500)
                or (0.001 < z and z <= 0.01 and b > 1e5)
        ):
            return -inc_beta_dda(b, a, 1 - z, digamma_b, digamma_ab, threshold, conv_threshold)
    if (
            (z > 0.75 and a < 500)
            or (z > 0.9 and a < 2500)
            or (z > 0.99 and a < 1e5)
            or (z > 0.999)
    ):
        return -inc_beta_dda(b, a, 1 - z, digamma_b, digamma_ab, threshold, conv_threshold)


    a_plus_b = a + b
    a_plus_1 = a + 1

    prefactor = (a_plus_1 / a_plus_b) ** 3

    sum_numer = digamma_ab * prefactor
    sum_denom = prefactor

    summand = prefactor * z * a_plus_b / a_plus_1

    k = 1
    digamma_ab += _inv(a_plus_b)

    while (abs(summand) > threshold):
        sum_numer += digamma_ab * summand
        sum_denom += summand

        summand *= (1 + a_plus_b / k) * (1 + k) / (1 + a_plus_1 / k)
        digamma_ab += _inv(a_plus_b + k)
        k += 1
        summand *= z / k

        if k > conv_threshold:
            raise ValueError('inc_beta_ddb did not converge')

    return sp.betainc(a, b, z) * (np.log1p(-z) - digamma_b + sum_numer / sum_denom)


def inc_beta_ddz(a, b, z):
    # Gradient of the regularized incomplete beta wrt to z
    return np.exp(
        (b-1) * np.log(1-z) + (a-1) * np.log(z) + sp.loggamma(a + b) - sp.loggamma(a) - sp.loggamma(b)
    )


def stan_grad_reg_inc_beta(z, a, b):
    return grad_reg_inc_beta(a, b, z, sp.digamma(a), sp.digamma(b), sp.digamma(a+b), sp.beta(a, b))


def stan_inc_beta(z, a, b):
    try:
        dda = inc_beta_dda(a, b, z, sp.digamma(a), sp.digamma(a+b))
    except ValueError:
        dda = np.nan

    try:
        ddb = inc_beta_ddb(a, b, z, sp.digamma(b), sp.digamma(a+b))
    except ValueError:
        ddb = np.nan

    return (dda, ddb)
   

def stan_inc_beta_strict(z, a, b):
    try:
        dda = inc_beta_dda(a, b, z, sp.digamma(a), sp.digamma(a+b), threshold=1e-18)
    except ValueError:
        dda = np.nan

    try:
        ddb = inc_beta_ddb(a, b, z, sp.digamma(b), sp.digamma(a+b), threshold=1e-18)
    except ValueError:
        ddb = np.nan

    return (dda, ddb)

