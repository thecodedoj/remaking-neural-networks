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

def randint(x, y, z, seed_start=1):
    """
    Generate z pseudo-random integers between x and y inclusive.
    """
    result = []
    seed = seed_start
    while len(result) < z:
        r, seed = lcg(seed)
        val = x + r % (y - x + 1)  # scale into [x, y]
        result.append(val)
    return result
