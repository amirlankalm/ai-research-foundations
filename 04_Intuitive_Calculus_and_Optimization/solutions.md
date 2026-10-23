# 📝 Worked Solutions — Module 04: Intuitive Calculus & Optimization

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises.

---

## Level 1: Recall & Conceptual

### 1. Difference Between a Derivative and a Gradient
- **Derivative**: The instantaneous rate of change of a function with respect to a **single** scalar variable (e.g., $f'(x) = \frac{df}{dx}$). It tells you whether increasing $x$ pushes $f(x)$ up or down on a 2D line.
- **Gradient**: A **vector** collecting all partial derivatives of a multivariable function ($\nabla f(\vec{w}) = \left[\frac{\partial f}{\partial w_1}, \dots, \frac{\partial f}{\partial w_n}\right]^T$). Geometrically, it points in the direction of **steepest increase** in multi-dimensional space, and its magnitude equals the slope in that direction.

### 2. If Gradient is $+4.5$, Should You Increase or Decrease $w_1$?
You must **decrease** $w_1$.
A positive gradient ($\frac{\partial L}{\partial w_1} = +4.5$) means that increasing $w_1$ increases the loss. In gradient descent, we update weights in the negative gradient direction:
$$w_1 \leftarrow w_1 - \eta \cdot (+4.5)$$
Subtracting a positive number decreases $w_1$, thereby reducing the loss.

### 3. Why Adam Divides by $\sqrt{\hat{v}_t} + \epsilon$ & Role of $\epsilon$
- **Dividing by $\sqrt{\hat{v}_t}$**: $\hat{v}_t$ tracks the exponentially smoothed second moment (uncentered variance) of past gradients. Dividing by $\sqrt{\hat{v}_t}$ normalizes updates: parameters that receive frequent massive gradients take smaller, safer steps; parameters with rare or tiny gradients are amplified.
- **Role of $\epsilon$ (typically $10^{-8}$)**: It is a numerical stability damping factor that prevents division by zero when gradients are zero (e.g. at initialization or inactive neurons).

---

## Level 2: Hands-On Implementation

### 1. Finite-Difference Numerical Gradient Function

```python
import numpy as np
from typing import Callable

def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Computes numerical gradient of scalar function f at point x using two-sided central differences:
    df/dx_i = (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps)
    """
    grad = np.zeros_like(x, dtype=float)
    x_float = x.astype(float)

    for i in range(len(x)):
        x_plus = x_float.copy()
        x_minus = x_float.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        
        grad[i] = (f(x_plus) - f(x_minus)) / (2.0 * eps)
        
    return grad

# Verification on f(x, y) = x^2 + 3*y^2 -> analytical grad at (2, 1) is [4, 6]
if __name__ == "__main__":
    test_fn = lambda v: v[0]**2 + 3.0 * v[1]**2
    pt = np.array([2.0, 1.0])
    num_g = numerical_gradient(test_fn, pt)
    print("Numerical Gradient:", num_g)
    assert np.allclose(num_g, [4.0, 6.0], atol=1e-4)
    print("✅ Numerical gradient verified against analytical calculus!")
```

### 2. SGD, Momentum, and Adam on Rosenbrock

```python
import numpy as np

def rosenbrock(v: np.ndarray) -> float:
    x, y = v[0], v[1]
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

def rosenbrock_grad(v: np.ndarray) -> np.ndarray:
    x, y = v[0], v[1]
    df_dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
    df_dy = 200.0 * (y - x**2)
    return np.array([df_dx, df_dy])

def optimize_rosenbrock(opt: str, steps: int = 1000, lr: float = 0.002):
    p = np.array([-1.5, 2.0]) # Starting point
    vel = np.zeros(2)
    m = np.zeros(2)
    v = np.zeros(2)
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, steps + 1):
        g = np.clip(rosenbrock_grad(p), -100.0, 100.0)
        if opt == "sgd":
            p -= lr * g
        elif opt == "momentum":
            vel = 0.9 * vel + g
            p -= lr * vel
        elif opt == "adam":
            m = b1 * m + (1.0 - b1) * g
            v = b2 * v + (1.0 - b2) * (g ** 2)
            m_hat = m / (1.0 - b1 ** t)
            v_hat = v / (1.0 - b2 ** t)
            p -= (lr / (np.sqrt(v_hat) + eps)) * m_hat
    return p, rosenbrock(p)

if __name__ == "__main__":
    for name in ["sgd", "momentum", "adam"]:
        coords, loss = optimize_rosenbrock(name)
        print(f"[{name.upper():8s}] Final point: ({coords[0]:.3f}, {coords[1]:.3f}) | Loss: {loss:.4f}")
```
