def prime_factors(n):
    factors = {}
    i = 2
    while i * i <= n:
        while n % i == 0:
            factors[i] = factors.get(i, 0) + 1
            n //= i
        i += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors

def sqroot(x):
    if x < 0:
        raise ValueError("Cannot compute square root of negative number")
    if x == 0:
        return 0.0

    factors = prime_factors(x)
    outside = 1
    inside = 1
    for prime, power in factors.items():
        outside *= prime ** (power // 2)
        if power % 2 != 0:
            inside *= prime

    return outside * (inside ** 0.5)
def sum_func(iterable):
    total = 0
    for item in iterable:
        total += item   
    return total
def zero(size):
    if isinstance(size, int):
        return [0.0] * size 
def exp(x):
    # approximate e^x using the Taylor series expansion
    # e^x = 1 + x + x²/2! + x³/3! + ...
    n_terms = 50  # more terms = higher precision
    result = 1.0
    term = 1.0
    for n in range(1, n_terms):
        term *= x / n
        result += term
    return result
