import math
import renumpying

def my_frexp(x):
    """Decompose x into mantissa and exponent so that x = mantissa * 2**exponent."""
    if x == 0.0:
        return 0.0, 0

    exponent = 0
    mantissa = abs(x)

    # Scale mantissa up until it's in [0.5, 1)
    while mantissa < 0.5:
        mantissa *= 2.0
        exponent -= 1

    # Scale mantissa down if >= 1
    while mantissa >= 1.0:
        mantissa /= 2.0
        exponent += 1

    # Restore sign
    if x < 0:
        mantissa = -mantissa

    return mantissa, exponent

def my_log(x):
    """Approximate natural log (ln) using a C-style approach."""
    # Handle invalid input
    if x <= 0.0:
        return float('nan')

    # Decompose x into mantissa (m) and exponent (e)
    m, e = my_frexp(x)  # x = m * 2**e, with m in [0.5, 1)

    # Adjust range so m is in [sqrt(0.5), sqrt(2))
    if m < renumpying.sqroot(0.5):
        m *= 2.0
        e -= 1

    # f = m - 1
    f = m - 1.0
    f2 = f * f
    f3 = f2 * f

    # Polynomial approximation for log(1 + f)
    r = f - 0.5 * f2 + (1.0 / 3.0) * f3

    # Combine with e * ln(2)
    return r + e * 0.6931471805599453  # ln(2)
