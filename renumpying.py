from redoing_numpy_arrays import Array


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
        - 1D lists/Arrays (vectors)
        - 2D lists/Arrays (matrices)
    """
    def get_shape(x):
        if not isinstance(x, list):
            return (1,)
        if not isinstance(x[0], list):
            return (len(x),)
        return (len(x), len(x[0]))
    # Extract data if inputs are Array objects
    if hasattr(a, 'data'):
        a = a.data
    if hasattr(b, 'data'):
        b = b.data

    # Handle empty inputs
    if not a or not b:
        raise ValueError("Empty input arrays")
        
    print(f"Input shapes: a{get_shape(a)}, b{get_shape(b)}")
        
    # For vectors, convert single numbers to lists
    if not isinstance(a, list):
        a = [a]
    if not isinstance(b, list):
        b = [b]
    
    # Check if vectors (1D)
    if (not isinstance(a[0], list)) and (not isinstance(b[0], list)):
        if len(a) != len(b):
            raise ValueError("Vectors must be the same length")
        return sum(a[i] * b[i] for i in range(len(a)))
    
    # Check if vector-matrix multiplication (a = 1D vector, b = 2D matrix)
    if not isinstance(a[0], list) and isinstance(b[0], list):
        # For vector-matrix multiplication:
        # result[j] = sum(a[i] * b[i][j]) for all i
        # result length should be number of columns in b
        result = [0] * len(b[0])  # initialize with correct length
        print(f"Creating result vector of length {len(b[0])}")  # Debug print
        for j in range(len(b[0])):  # for each column
            s = 0  # use temporary sum
            for i in range(len(b)):  # iterate over rows
                s += a[i] * b[i][j]
            result[j] = s  # store final sum
        print(f"Result vector length: {len(result)}")  # Debug print
        return result
    
    # Check if matrix-vector multiplication (a = 2D, b = 1D)
    if isinstance(a[0], list) and not isinstance(b[0], list):
        result = []
        for row in a:
            if len(row) != len(b):
                raise ValueError(f"Matrix columns ({len(row)}) must match vector length ({len(b)})")
            result.append(sum(row[i] * b[i] for i in range(len(b))))
        return result

    # Check if matrix-matrix multiplication (a = 2D, b = 2D)
    if isinstance(a[0], list) and isinstance(b[0], list):
        if len(a[0]) != len(b):
            raise ValueError(f"Matrix A columns ({len(a[0])}) must equal Matrix B rows ({len(b)})")
        result = []
        for row in a:
            new_row = []
            for col in range(len(b[0])):
                s = sum(row[i] * b[i][col] for i in range(len(row)))
                new_row.append(s)
            result.append(new_row)
        return result

    raise ValueError(f"Unsupported input shapes. a[0] type: {type(a[0])}, b[0] type: {type(b[0])}")

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
    elif isinstance(size, tuple) and len(size) == 2:
        return [[0.0] * size[1] for _ in range(size[0])]
    else:
        raise ValueError("Size must be an integer or a 2-tuple")
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

    # Compute outer product (return as plain list-of-lists)
    result = []
    for ai in a:
        row = []
        for bj in b:
            row.append(ai * bj)
        result.append(row)

    return result
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
