# 📝 Worked Solutions — Module 06: Classical Machine Learning & Scikit-Learn

These worked solutions provide reference answers and complete code implementations for Level 1 and Level 2 revision exercises, along with research guidance for Level 3.

---

## Level 1: Recall & Conceptual

### 1. Supervised vs. Self-Supervised Learning
- **Supervised Learning**: Models learn from explicitly human-annotated input-target pairs $(X, y)$ (e.g., patient vital signs paired with a medical diagnosis, or images paired with integer class labels).
- **Self-Supervised Learning**: The supervisory training signal is derived directly from the unlabelled structure of the input data itself without manual human labelling. In language models, this takes the form of *Next-Token Prediction* (causal language modeling) or *Masked Language Modeling* (BERT), where previous tokens serve as inputs and subsequent tokens serve as self-generated labels.

### 2. High Precision (99%) vs. Low Recall (20%) in Airport Facial Scanner
- **Precision = 99%**: When the facial scanner rings an alarm identifying an individual on a high-risk watchlist, it is correct 99% of the time (only 1% false alarms, meaning innocent travelers are rarely detained).
- **Recall = 20%**: The scanner only detects 20% of all actual flagged individuals passing through security. 80% of wanted individuals pass by undetected (False Negatives). In high-stakes security settings, low recall presents an unacceptable safety vulnerability.

### 3. Data Leakage from Dataset-Wide Imputation
**Data leakage** occurs when information from outside the training dataset is mistakenly used to create the model.
If you compute the global mean or median over the entire dataset before performing a train/test split, the imputed training values incorporate distributional statistics from the held-out test split. The model will appear to achieve artificially high evaluation performance during development, but will suffer significant degradation when deployed on genuine out-of-distribution real-world data. **Rule**: Always perform train/test splitting first, fit transformers/imputers solely on `X_train`, and then use `.transform(X_test)`.

---

## Level 2: Hands-On Implementation

### 1. `LogisticRegressionScratch` with Gradient Descent

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class LogisticRegressionScratch:
    def __init__(self, lr: float = 0.05, epochs: int = 1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        # Numerically stable clipping to avoid overflow
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Gradient of Binary Cross-Entropy
            dw = (1.0 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1.0 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

# Verification
if __name__ == "__main__":
    X, y = make_classification(n_samples=500, n_features=2, n_informative=2,
                               n_redundant=0, random_state=42)
    split = 400
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scratch model
    model_scratch = LogisticRegressionScratch(lr=0.1, epochs=1500)
    model_scratch.fit(X_train, y_train)
    scratch_preds = model_scratch.predict(X_test)
    scratch_acc = accuracy_score(y_test, scratch_preds)

    # Scikit-learn reference
    model_sk = LogisticRegression(penalty=None, solver="lbfgs")
    model_sk.fit(X_train, y_train)
    sk_preds = model_sk.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_preds)

    print(f"Scratch Model Accuracy:    {scratch_acc * 100:.2f}%")
    print(f"Scikit-Learn Accuracy:     {sk_acc * 100:.2f}%")
    assert abs(scratch_acc - sk_acc) < 0.05, "Scratch implementation diverges significantly!"
```

### 2. Confusion Matrix from Scratch

```python
def confusion_matrix_scratch(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Returns (TP, FP, TN, FN) and formatted confusion matrix.
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    matrix = np.array([[tn, fp],
                       [fn, tp]])
    return matrix, {"TP": tp, "FP": fp, "TN": tn, "FN": fn}

# Example validation
cm, metrics = confusion_matrix_scratch(y_test, scratch_preds)
print("Confusion Matrix [[TN, FP], [FN, TP]]:")
print(cm)
print("Counts:", metrics)
```

---

## Level 3: Research Deep Dive

### 1. The Double Descent Phenomenon & Overparameterization
- **Classical U-Curve (Underparameterized Regime)**: Classical statistics considers models up to the *interpolation threshold* (where model parameter count $P$ equals the number of training samples $N$). Approaching $P \approx N$, models fit noise in the training data, leading to divergent parameter norms and peaking test error.
- **Deep Double Descent (Overparameterized Regime)**: When $P \gg N$, the system is heavily underdetermined: there exist infinitely many parameter configurations that achieve zero training error ($L_{train} \approx 0$).
- **Implicit Regularization of Gradient Descent**: Gradient descent algorithms (and optimizers like AdamW) implicitly favor minimum-norm interpolators—the smoothest possible decision surfaces that interpolate the points. As parameter capacity grows further, the minimum-norm interpolator becomes smoother and less sensitive to individual noisy perturbations, driving test error *down* in a second descent curve.
