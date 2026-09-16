



class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def backward(self):
        vst = set()
        topo = []
        def build_topo(v):
            if v not in vst:
                vst.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
            

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __pow__ (self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __add__ (self, other):
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__ (self, other):
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def relu(self):
        out = Value (max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward
        return out
#expression = relu(w1x1 + w2x2 + b)
x1 = Value(4)
x2 = Value(5)
w1 = Value(6)
w2 = Value(7)
b = Value(10)
exp1 = x1 * w1
exp2 = x2 * w2 
exp3 = exp1 + exp2 + b
ans = exp3.relu()
ans.backward()

print(f"x1 : {x1}")
print(f"x2 : {x2}")
print(f'ans: {ans}')




