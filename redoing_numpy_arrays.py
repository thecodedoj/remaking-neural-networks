class Array:
    def __init__(self, data):
        if isinstance(data, list):
            # If elements are Arrays, get their data
            self.data = [x.data if isinstance(x, Array) else x for x in data]
        else:
            self.data = data

    def __str__(self):
        return str(self.data)
        
    def __getitem__(self, index):
        if isinstance(self.data[index], list):
            return Array(self.data[index])
        return self.data[index]
        
    def __setitem__(self, index, value):
        self.data[index] = value.data if isinstance(value, Array) else value

    def get_shape(self):
        """Return the shape of the array as a tuple."""
        def shape_helper(d):
            if not isinstance(d, list):
                return ()
            if len(d) == 0:
                return (0,)
            # recursively get shape of first element
            return (len(d),) + shape_helper(d[0])
        
        return shape_helper(self.data)

    def dimension(self):
        """Return the number of dimensions."""
        def dim_helper(d):
            if not isinstance(d, list):
                return 0
            if len(d) == 0:
                return 1
            return 1 + dim_helper(d[0])
        
        return dim_helper(self.data)
    def dot(self , other):
        a = self.data
        b = other.data
        if len(a) != len(b):
            raise ValueError("Vectors must be the same length")
        if len(a) == len(b):
            if len(a) == 0:
                return 0
            total = 0
            for i in range(len(a)):
                total += a[i] * b[i]
            return total
