class Vector :

    def __init__ (self, data : list[float] | str | list[int]):
        self.data = list(data)
        self.size = len(self.data)
        
    def __repre__ (self) -> str:
        return f'vector : {self.data}'

    def __add__ (self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)]);
    def __sub__ (self, other):
        return Vector([a - b for a,b in zip(self.data, other.data)]);
    def __mul__ (self, scaler):
        return sum(a * scaler for a in self.data)
    def __dot__ (self, other):
        return sum(a * b for a, b in zip(self.data, other.data))
    def __magnitude__ (self):
        return sum(a ** 2 for a in self.data) ** 0.5

## Now to create a matrix

class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.columns = len(self.data[0])
        self.shape = (self.rows, self.columns)

    def __repre__ (self):
        rows = "\n".join(str(row) for row in self.data)
        return f"Matrix{self.shape} : {rows}"
    def __add__ (self, other):
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.columns)] for i in range(self.rows)])

    def __sub__ (self, other):
        return Matrix([[self.data[i][j] - other.data[i][j] for j in range(self.columns)] for i in range(self.rows)])

    def scaler_multiply (self, scaler):
        return Matrix (
            [[self.data[i][j] * scaler for j in range(self.columns)] for i in range(self.rows)]
        )

    def element_wise_mul(self, other):
        return Matrix (
            [[self.data[i][j] * other.data[i][j] for j in range(self.columns)] for i in range(self.rows)]
        )

    def matmul(self, other):
        return Matrix (
            [
                [sum(self.data[i][j] * other.data[j][k] for j in range(self.columns))
                    for k in range(other.columns)]
                for i in range(self.rows)
            ]
        )
    def transpose(self):
        return Matrix(
            [[self.data[j][i] for j in range(self.rows)] for i in range(self.columns)]
        )

    # to be continued.....
    def determinant :

