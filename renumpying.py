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
def dot(a, b):
    """
    Compute the dot product between vectors or matrices.
    Works for:
        - 1D lists (vectors)
        - 2D lists (matrices)
    """
    # Check if vectors (1D)
    if all(not isinstance(i, list) for i in a) and all(not isinstance(i, list) for i in b):
        if len(a) != len(b):
            raise ValueError("Vectors must be the same length")
        return sum(a[i] * b[i] for i in range(len(a)))
    
    # Check if matrix-vector multiplication (a = 2D, b = 1D)
    if all(isinstance(row, list) for row in a) and all(not isinstance(i, list) for i in b):
        result = []
        for row in a:
            if len(row) != len(b):
                raise ValueError("Matrix columns must match vector length")
            result.append(sum(row[i] * b[i] for i in range(len(b))))
        return result

    # Check if matrix-matrix multiplication (a = 2D, b = 2D)
    if all(isinstance(row, list) for row in a) and all(isinstance(row, list) for row in b):
        # Number of columns in a must equal number of rows in b
        if len(a[0]) != len(b):
            raise ValueError("Matrix A columns must equal Matrix B rows")
        result = []
        for row in a:
            new_row = []
            for col in range(len(b[0])):
                s = sum(row[i] * b[i][col] for i in range(len(row)))
                new_row.append(s)
            result.append(new_row)
        return result

    raise ValueError("Unsupported input shapes")

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
def outer_array(a, b):
    """
    Compute the outer product of two vectors.
    Works if a and b are instances of Array or plain lists.
    """
    # Extract data if input is an Array
    if isinstance(a, Array):
        a = a.data
    if isinstance(b, Array):
        b = b.data

    # Make sure they are 1D
    if any(isinstance(x, list) for x in a) or any(isinstance(x, list) for x in b):
        raise ValueError("Only 1D vectors are supported")

    # Compute outer product
    result = []
    for ai in a:
        row = []
        for bj in b:
            row.append(ai * bj)
        result.append(row)

    return Array(result)  # return as Array instance
def argmax(lst):
    """
    Return the index of the maximum element in a list.
    """
    if len(lst) == 0:
        raise ValueError("argmax() called on empty list")
    
    max_index = 0
    max_value = lst[0]
    
    for i in range(1, len(lst)):
        if lst[i] > max_value:
            max_value = lst[i]
            max_index = i
            
    return max_index
