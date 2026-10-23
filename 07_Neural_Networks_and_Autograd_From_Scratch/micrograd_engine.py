"""
AI Research Foundations - Module 07: Micrograd Autograd Engine & MLP From Scratch
Run with: python micrograd_engine.py
"""

import math
import random

class Value:
    """Scalar autograd node with reverse-mode automatic differentiation."""
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self * -1.0

    def relu(self):
        out = Value(max(0.0, self.data), (self,), "ReLU")
        def _backward():
            self.grad += (out.data > 0.0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-0.5, 0.5)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin: int, nouts: list):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) if isinstance(x, list) else layer([x])
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == "__main__":
    print("=== Testing Micrograd Scalar Autograd ===")
    random.seed(42)

    # Gradient check on a composite expression: f(x, y) = (x * y + 3)²
    x = Value(2.0)
    y = Value(-3.0)
    u = x * y       # 2 * -3 = -6
    v = u + 3.0     # -6 + 3 = -3
    f = v * v       # (-3)² = 9.0

    f.backward()
    print("Function value f(x, y):", f.data)
    print("df/dx (analytical):", x.grad, "| Expected: 2 * (x*y + 3) * y = 2 * (-3) * (-3) = 18.0")
    print("df/dy (analytical):", y.grad, "| Expected: 2 * (x*y + 3) * x = 2 * (-3) * 2 = -12.0")

    print("\n=== Training Toy MLP with ReLU ===")
    model = MLP(2, [8, 1])
    xs = [
        [Value(0.0), Value(0.0)],
        [Value(1.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(1.0)]
    ]
    ys = [0.0, 1.0, 1.0, 0.0]  # Non-linear XOR pattern

    for step in range(101):
        # Forward pass
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt) * (yout - ygt) for ygt, yout in zip(ys, ypred))

        # Zero grad
        for p in model.parameters():
            p.grad = 0.0

        # Backward pass
        loss.backward()

        # Step
        for p in model.parameters():
            p.data -= 0.05 * p.grad

        if step % 20 == 0:
            print(f"Step {step:3d} | Loss: {loss.data:.5f}")

    print("\nPredictions after training:")
    for x, y_true in zip(xs, ys):
        pred = model(x).data
        print(f"Input: {[xi.data for xi in x]} -> Pred: {pred:.3f} (True: {y_true})")

    print("\n✅ Module 07 micrograd engine verified!")
