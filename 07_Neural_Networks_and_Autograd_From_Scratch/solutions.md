# 📝 Worked Solutions — Module 07: Neural Networks & Autograd From Scratch

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. Why `grad` Must Accumulate (`+=`) Rather Than Assign (`=`)
In calculus, the multivariable chain rule states that if variable $x$ influences the final output $L$ through multiple distinct pathways, the total derivative is the **sum** of the derivatives along all pathways:
$$\frac{dL}{dx} = \sum_{i} \frac{dL}{du_i} \frac{\partial u_i}{\partial x}$$
For example, in $f(x) = x \cdot x$, $x$ feeds into both arguments of the multiplication. If you used `=`, the second branch would overwrite the first branch's gradient, resulting in $\frac{df}{dx} = x$ instead of the true derivative $2x$. Using `+=` correctly accumulates gradients across all computational branches.

### 2. Topological Sort in a DAG
A **topological sort** orders vertices of a Directed Acyclic Graph such that for every directed edge $u \to v$, $u$ comes before $v$ in the ordering.
In backpropagation, node $v$ cannot compute its final gradient until all downstream nodes that depend on $v$ have already computed and pushed back their gradients. Reversing the topological order ensures that we process nodes from output backwards to inputs, guaranteeing all incoming gradients have arrived before computing parameter updates.

### 3. Derivative of ReLU
$$\text{ReLU}(x) = \max(0, x) \implies \frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$
*(At $x = 0$, the mathematical derivative is undefined; deep learning frameworks define the sub-gradient as $0.0$).*

---

## Level 2: Hands-On Implementation

### 1 & 2. Value Class with ReLU, Division, and Exponentiation

```python
import math

class Value:
    def __init__(self, data: float, _children=(), _op=""):
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

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other: float):
        assert isinstance(other, (int, float)), "only int/float powers supported"
        out = Value(self.data ** other, (self,), f"**{other}")
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

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
```

### 3. Solving Non-Linear XOR Problem

```python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        return [n(x) for n in self.neurons]
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

if __name__ == "__main__":
    random.seed(42)
    # Inputs: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
    xs = [[Value(0), Value(0)], [Value(0), Value(1)], [Value(1), Value(0)], [Value(1), Value(1)]]
    ys = [0.0, 1.0, 1.0, 0.0]

    model = MLP(2, [4, 4, 1])

    for epoch in range(150):
        # Forward pass
        ypred = [model(x) for x in xs]
        loss = sum((yout - Value(ygt))**2 for ygt, yout in zip(ys, ypred))

        # Zero grad
        for p in model.parameters():
            p.grad = 0.0

        # Backward pass
        loss.backward()

        # Update
        for p in model.parameters():
            p.data -= 0.1 * p.grad

    print("Final XOR Loss:", loss.data)
    for x, ygt in zip(xs, ys):
        pred = model(x).data
        print(f"Input: {[xi.data for xi in x]} | Target: {ygt} | Pred: {pred:.3f}")
    assert loss.data < 0.05
    print("✅ XOR problem solved successfully from scratch!")
```

---

## Level 3: Research Deep Dive (Approach & Conceptual Guidance)

### Vanishing Gradients in Tanh vs ReLU
- **When $|x| = 5.0$**: $\tanh(5.0) \approx 0.9999$. Its derivative is $\frac{d}{dx}\tanh(x) = 1 - \tanh(x)^2 \approx 1 - 0.9998 = 0.0002$.
- **Chain of 10 Layers**: During backpropagation, the gradients are multiplied across all 10 layers via the chain rule:
  $$\frac{\partial L}{\partial w_1} \propto \prod_{k=1}^{10} \sigma'(z_k) \approx (0.0002)^{10} \approx 10^{-37}$$
- **Result**: The gradient arriving at the earliest layers completely vanishes, rendering early feature extraction layers unable to learn. ReLU overcomes this because for all positive activations, its derivative is identically $1.0$, allowing gradients to flow backward through arbitrary depths without attenuation.
