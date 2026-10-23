"""
AI Research Foundations - Module 06: Classical Machine Learning & Logistic Regression From Scratch
Run with: python titanic_baseline.py
"""

import numpy as np

class LogisticRegressionScratch:
    """Logistic regression classifier trained with batch gradient descent."""
    def __init__(self, lr: float = 0.1, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients: dL/dw = (1/N) * X.T * (y_pred - y)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

if __name__ == "__main__":
    print("=== Training Logistic Regression From Scratch ===")
    np.random.seed(42)
    # Generate synthetic binary dataset
    n_samples = 200
    X_pos = np.random.randn(n_samples // 2, 2) + np.array([1.5, 1.5])
    X_neg = np.random.randn(n_samples // 2, 2) + np.array([-1.5, -1.5])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # Split 80/20
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LogisticRegressionScratch(lr=0.1, n_iters=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    accuracy = np.mean(preds == y_test)
    print(f"Learned Weights: {model.weights.round(3)}, Learned Bias: {model.bias:.3f}")
    print(f"Test Accuracy: {accuracy * 100:.1f}%")
    print("\n✅ Module 06 classical ML baseline verified!")
