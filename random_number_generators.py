def lcg(seed):
    """
    Linear Congruential Generator (LCG)
    Returns: (random_int, next_seed)
    """
    m = 2**32
    a = 1664525
    c = 1013904223
    next_seed = (a * seed + c) % m
    return next_seed, next_seed  # return value and new seed

def uniformity(low, high, size=None, seed_start=1):
    """
    Generate pseudo-random numbers between low and high.
    - size=None -> returns a single float
    - size=(rows, cols) -> returns a 2D list
    """
    def single_random(seed):
        r, next_seed = lcg(seed)
        return low + (high - low) * r / (2**32), next_seed

    if size is None:
        val, _ = single_random(seed_start)
        return val

    rows, cols = size
    matrix = []
    seed = seed_start
    for i in range(rows):
        row = []
        for j in range(cols):
            val, seed = single_random(seed)
            row.append(val)
        matrix.append(row)
    return matrix

def randint(x, y, z=None):
    """
    Generate random integers between x and y (inclusive).
    - If z is None, return a single integer.
    - If z is provided, return a list of z integers.
    """
    def single_int(seed=1):
        # Simple LCG-based random integer
        r, _ = lcg(seed)  # Unpack the tuple, we only need the first value
        r = r % (y - x + 1) + x
        return r

    if z is None:
        return single_int()
    else:
        result = []
        seed = 1
        for _ in range(z):
            result.append(single_int(seed))
            seed += 1
        return result
