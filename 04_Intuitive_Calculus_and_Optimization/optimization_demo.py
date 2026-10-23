"""
AI Research Foundations - Module 04: Optimizers From Scratch (SGD, Momentum, Adam)
Run with: python optimization_demo.py
"""

import numpy as np

def rosenbrock(x: float, y: float) -> float:
    """The Rosenbrock valley function: global minimum at (1, 1) where f(x, y) = 0."""
    return (1.0 - x)**2 + 100.0 * (y - x**2)**2

def rosenbrock_grad(x: float, y: float) -> np.ndarray:
    """Analytical gradient of Rosenbrock function with respect to [x, y]."""
    df_dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x**2)
    df_dy = 200.0 * (y - x**2)
    return np.array([df_dx, df_dy])

def run_optimizer(opt_name: str, steps: int = 500, lr: float = 0.002):
    # Starting point far from minimum (1, 1)
    params = np.array([-1.5, 2.0])
    
    # Optimizer internal states
    velocity = np.zeros(2)
    m = np.zeros(2)
    v = np.zeros(2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    trajectory = [params.copy()]

    for t in range(1, steps + 1):
        g = rosenbrock_grad(params[0], params[1])
        
        # Clip extreme gradients for stability
        g = np.clip(g, -100.0, 100.0)

        if opt_name == "sgd":
            params = params - lr * g

        elif opt_name == "momentum":
            velocity = 0.9 * velocity + g
            params = params - lr * velocity

        elif opt_name == "adam":
            m = beta1 * m + (1.0 - beta1) * g
            v = beta2 * v + (1.0 - beta2) * (g ** 2)
            m_hat = m / (1.0 - beta1 ** t)
            v_hat = v / (1.0 - beta2 ** t)
            params = params - (lr / (np.sqrt(v_hat) + eps)) * m_hat

        trajectory.append(params.copy())

    final_loss = rosenbrock(params[0], params[1])
    return params, final_loss

if __name__ == "__main__":
    print("=== Optimizing Rosenbrock Valley: Minimum at (1.0, 1.0) ===")
    for opt in ["sgd", "momentum", "adam"]:
        final_coords, loss = run_optimizer(opt, steps=1000, lr=0.002)
        print(f"[{opt.upper():8s}] Final coords: ({final_coords[0]:.3f}, {final_coords[1]:.3f}) | Final Loss: {loss:.6f}")

    print("\n✅ Module 04 optimizer experiment finished!")
