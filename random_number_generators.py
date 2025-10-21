def lcg(seed):
    m = 2**32
    a = 1664525
    c = 1013904223
    n = 21
    so_much_math = (a**n * seed + c * (a**n - 1) // (a - 1))
    return so_much_math % m
def uniformity(low, high, size=None, seed_start=1):
    """
    Generate pseudo-random numbers between low and high.
    
    - size=None -> returns a single float
    - size=(rows, cols) -> returns a 2D list
    """
    def single_random(seed):
        r, m = lcg(seed)
        return low + (high - low) * r / m

    if size is None:
        return single_random(seed_start)
    
    rows, cols = size
    matrix = []
    seed = seed_start
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(single_random(seed))
            seed += 1  # increment seed for next number
        matrix.append(row)
    return matrix
def randint(x,y,z):
    the_list = []
    while len(the_list) < z:
        for i in range(z):
            the_lcg = int(lcg(34))
            if x <= the_lcg <= y:
                the_list.append(the_lcg)
            else:
                del the_lcg
    return the_list 