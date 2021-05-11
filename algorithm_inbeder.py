# Reference:
#   Robinson-Cox, J. F., & Boik, R. J. (1998). Derivatives of the Incomplete Beta
#   Function. Journal of Statistical Software, 3(01).
# Adapted from https://github.com/canerturkmen/betaincder

from scipy.special import digamma, betaln
from numpy import log, log1p, exp

def _a_n(x, p, q, n):

    f = q * x / (p * (1 - x))

    if (n == 1):
        return p * f * (q - 1) / (q * (p + 1))

    p2n = p + 2*n
    F1 = p**2 * f**2 *(n-1) / (q**2)
    F2 = (p+q+n-2)*(p+n-1)*(q-n)/((p2n-3) * (p2n-2) ** 2 * (p2n-1))

    return F1 * F2


def _b_n(x, p, q, n):
    f = q * x / (p * (1 - x))

    pf = p*f
    p2n = p + 2*n

    N1 = 2 * (pf + 2*q) * n * (n+p-1) + p*q*(p-2-pf)
    D1 = q*(p2n - 2)* p2n

    return N1/D1


def _da_n_dp(x, p, q, n):
    f = (q * x) / (p * (1-x))

    if (n == 1):
        return - p*f*(q-1) / (q*(p+1)*(p+1))

    nn = n**2
    nnn = nn * n
    pp = p**2
    ppp = pp*p
    N2a = (-8 + 8*p + 8*q) * nnn
    N2b = (16*pp + (-44+20*q)*p + 26 - 24*q) * nn
    N2c = (10*ppp + (14*q - 46)*pp + (-40*q + 66)*p - 28 + 24*q)*n
    N2d = 2*pp*pp + (-13 + 3*q)*ppp + (-14*q + 30)*pp
    N2e = (-29 + 19*q)*p + 10 - 8*q

    p2n = p + 2*n
    D = (p2n - 2)**3 * (p2n - 1)**2
    N1 = -(n-1) * f**2 * pp * (q-n)  / (q**2 * (p2n - 3)**2)

    return (N2a + N2b + N2c + N2d + N2e) / D * N1


def _da_n_dq(x, p, q, n):

    f = (q * x) / (p * (1-x))

    if (n == 1):
        return p * f / (q * (p + 1))

    p2n = p + 2*n
    N1 = (p**2 * f**2 / (q**2)) * (n-1) * (p+n-1) * (2*q+p-2)
    D = (p2n-3) * (p2n-2)**2 * (p2n-1)
    return N1 / D


def _db_n_dp(x, p, q, n):
    f = (q * x) / (p * (1-x))

    p2n = p + 2*n
    pp = p**2
    q4 = 4 * q
    p4 = 4 * p
    N1 = (p*f/q) * ((-p4 - q4 + 4) * n**2 + (p4 - 4 + q4 - 2*pp)*n + pp * q)
    D = (p2n-2)**2 * p2n**2
    return N1 / D


def _db_n_dq(x, p, q, n):
    f = (q * x) / (p * (1-x))

    p2n = p + 2*n
    return -(p**2 * f) / (q * (p2n - 2) * (p2n))


def betaincderp(x, p, q, min_iters=3, max_iters=200, err_threshold=1e-12, debug=False):

    if (x == 0):
        return 0

    if (x > p/(p+q)):
        if debug:
            print('Switching to betaincderq')
        return -betaincderq(1-x, q, p, min_iters, max_iters, err_threshold, debug)

    derp_old = 0
    Am2 = 1; Am1 = 1 
    Bm2 = 0; Bm1 = 1
    dAm2 = 0; dAm1 = 0
    dBm2 = 0; dBm1 = 0

    C1 = exp(p * log(x) + (q - 1) * log(1 - x) - log(p) - betaln(p, q))
    C2 = log(x) - 1/p + digamma(p+q) - digamma(p)

    for n in range(1, max_iters+1):
        a_n_ = _a_n(x, p, q, n)
        b_n_ = _b_n(x, p, q, n)
        da_n_dp = _da_n_dp(x, p, q, n)
        db_n_dp = _db_n_dp(x, p, q, n)

        A = Am2 * a_n_ + Am1 * b_n_
        dA = da_n_dp * Am2 + a_n_ * dAm2 + db_n_dp * Am1 + b_n_ * dAm1
        B = Bm2 * a_n_ + Bm1 * b_n_
        dB = da_n_dp * Bm2 + a_n_ * dBm2 + db_n_dp * Bm1 + b_n_ * dBm1

        Am2 = Am1;  Am1 = A;  dAm2 = dAm1;  dAm1 = dA
        Bm2 = Bm1;  Bm1 = B;  dBm2 = dBm1;  dBm1 = dB

        if n < min_iters - 1: 
            continue

        dr1 = A / B
        dr2 = (dA - dr1 * dB) / B

        derp = C1 * (dr1 * C2 + dr2)

        # Check for convergence
        errapx = abs(derp_old - derp)
        d_errapx = errapx / max(err_threshold, abs(derp))
        derp_old = derp

        if d_errapx <= err_threshold:
            break

        if n >= max_iters:
            raise RuntimeError('Derivative did not converge')

    if debug:
        # TODO: Add approx error
        print(f'Converged in {n+1} iterations, appx error = {errapx}')
        print(f'Estimated betainc = {C1 * dr1}')

    return derp


def betaincderq(x, p, q, min_iters=3, max_iters=200, err_threshold=1e-12, debug=False):  

    if (x == 0):
        return 0

    if (x > p/(p+q)):
        if debug:
            print('Switching to betaincderp')
        return -betaincderp(1-x, q, p, min_iters, max_iters, err_threshold, debug)

    derq_old = 0
    Am2 = 1; Am1 = 1 
    Bm2 = 0; Bm1 = 1
    dAm2 = 0; dAm1 = 0
    dBm2 = 0; dBm1 = 0

    C1 = exp(p * log(x) + (q - 1) * log1p(-x) - log(p) - betaln(p, q))
    C2 = log1p(-x) + digamma(p+q) - digamma(q)

    for n in range(1, max_iters+1):
        a_n_ = _a_n(x, p, q, n)
        b_n_ = _b_n(x, p, q, n)
        da_n_dq = _da_n_dq(x, p, q, n)
        db_n_dq = _db_n_dq(x, p, q, n)

        A = Am2 * a_n_ + Am1 * b_n_
        B = Bm2 * a_n_ + Bm1 * b_n_
        dA = da_n_dq * Am2 + a_n_ * dAm2 + db_n_dq * Am1 + b_n_ * dAm1
        dB = da_n_dq * Bm2 + a_n_ * dBm2 + db_n_dq * Bm1 + b_n_ * dBm1

        Am2 = Am1;  Am1 = A;  dAm2 = dAm1;  dAm1 = dA
        Bm2 = Bm1;  Bm1 = B;  dBm2 = dBm1;  dBm1 = dB
        
        if n < min_iters - 1: 
            continue

        dr1 = A / B
        dr2 = (dA - dr1 * dB) / B
    
        derq = C1 * (dr1 * C2 + dr2)

        # Check for convergence
        errapx = abs(derq_old - derq)
        d_errapx = errapx / max(err_threshold, abs(derq))  
        derq_old = derq            

        if d_errapx <= err_threshold:
            break

        if n >= max_iters:
            raise RuntimeError('betaincderq did not converge')
            

    if debug:
        print(f'betaincderq converged in {n+1} iterations, appx error = {errapx}')
        print(f'Estimated betainc = {C1 * dr1}')

    return derq


def inbeder(z, a, b):
    return (
        betaincderp(z, a, b), 
        betaincderq(z, a, b),
    )
