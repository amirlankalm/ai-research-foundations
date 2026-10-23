"""
Generates high-resolution educational diagrams for all 20 modules of AI Research Foundations.
Saves to /Users/amirlankalmukhan/ai-research-foundations/assets/
"""

import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("assets", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E0"
plt.rcParams["axes.linewidth"] = 0.8

# -------------------------------------------------------------
# 01: Dev Ecosystem & Research Tooling
# -------------------------------------------------------------
def make_01():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    boxes = [
        ("1. Terminal & CLI", "Ghostty / zsh\nCommand history, btop/htop", "#3182CE", (0.1, 0.6)),
        ("2. Isolated Env", "uv / venv (.venv)\nZero dependency collision", "#2B6CB0", (0.35, 0.6)),
        ("3. Experiment VCS", "Git Branching\n.gitignore large .pt weights", "#234E52", (0.6, 0.6)),
        ("4. Compute Target", "Apple MPS / NVIDIA CUDA\nPyTorch device dispatch", "#44337A", (0.85, 0.6)),
    ]
    
    for title, desc, color, (x, y) in boxes:
        rect = plt.Rectangle((x-0.11, y-0.2), 0.22, 0.4, facecolor="white", edgecolor=color, lw=2.5, transform=ax.transAxes, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y+0.12, title, ha="center", va="center", fontsize=11, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x, y-0.03, desc, ha="center", va="center", fontsize=9, color="#2D3748", transform=ax.transAxes)

    # Arrows
    for x in [0.22, 0.47, 0.72]:
        ax.annotate("", xy=(x+0.02, 0.6), xytext=(x-0.01, 0.6),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#718096"), xycoords="axes fraction")

    ax.set_title("Module 01: The Modern AI Research Development Stack", fontsize=13, fontweight="bold", pad=20, color="#1A202C")
    plt.tight_layout()
    plt.savefig("assets/01_dev_ecosystem.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 02: Tensor Shapes & NumPy Broadcasting
# -------------------------------------------------------------
def make_02():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
    fig.patch.set_facecolor("#F7FAFC")
    
    # Left: Tensor dimensions
    ax1.axis("off")
    dims = [
        ("0D Scalar", "Loss = 0.42", "()", "#4A5568"),
        ("1D Vector", "Token Embedding", "(768,)", "#2B6CB0"),
        ("2D Matrix", "Linear Weights W", "(768, 3072)", "#2C7A7B"),
        ("3D Tensor", "Batch Sequences", "(32, 128, 768)", "#744210"),
        ("4D Tensor", "Batch Vision Images", "(64, 3, 224, 224)", "#44337A")
    ]
    ax1.text(0.5, 0.95, "Tensor Dimensionality in Deep Learning", ha="center", fontsize=11, fontweight="bold", transform=ax1.transAxes)
    for i, (name, role, shape, col) in enumerate(dims):
        y = 0.78 - i * 0.18
        rect = plt.Rectangle((0.05, y-0.06), 0.9, 0.12, facecolor="white", edgecolor=col, lw=2, transform=ax1.transAxes)
        ax1.add_patch(rect)
        ax1.text(0.1, y, f"{name}: {role}", va="center", fontsize=9.5, fontweight="bold", color=col, transform=ax1.transAxes)
        ax1.text(0.88, y, shape, va="center", ha="right", fontsize=9, family="monospace", color="#4A5568", transform=ax1.transAxes)

    # Right: Broadcasting visual
    ax2.axis("off")
    ax2.text(0.5, 0.95, "Broadcasting Alignment Rule", ha="center", fontsize=11, fontweight="bold", transform=ax2.transAxes)
    
    # Array A: (4, 3)
    ax2.text(0.25, 0.75, "Token Activations A\nShape: (4, 3)", ha="center", fontsize=9.5, fontweight="bold", color="#2B6CB0", transform=ax2.transAxes)
    grid_A = np.ones((4, 3))
    # Array B: (1, 3)
    ax2.text(0.75, 0.75, "Bias Vector B\nShape: (1, 3)", ha="center", fontsize=9.5, fontweight="bold", color="#C53030", transform=ax2.transAxes)
    
    ax2.text(0.5, 0.45, "+\n(Stretched along dimension of size 1)", ha="center", va="center", fontsize=10, fontweight="bold", color="#718096", transform=ax2.transAxes)
    ax2.text(0.5, 0.20, "Output Shape: (4, 3)\nZero extra memory copies!", ha="center", va="center", fontsize=10, fontweight="bold", color="#22543D", transform=ax2.transAxes)
    
    fig.suptitle("Module 02: Tensor Dimensionality & Broadcasting Mechanics", fontsize=13, fontweight="bold", color="#1A202C")
    plt.tight_layout()
    plt.savefig("assets/02_tensor_broadcasting.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 03: Linear Algebra & LoRA Low-Rank
# -------------------------------------------------------------
def make_03():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
    fig.patch.set_facecolor("#F7FAFC")
    
    # Left: Dot Product Alignment
    angles = np.linspace(0, np.pi, 200)
    cos_vals = np.cos(angles)
    ax1.plot(angles * 180 / np.pi, cos_vals, color="#3182CE", lw=2.5)
    ax1.axhline(0, color="#A0AEC0", linestyle="--")
    ax1.scatter([0, 90, 180], [1, 0, -1], color=["#276749", "#D69E2E", "#C53030"], s=70, zorder=4)
    ax1.annotate("Aligned (Similarity = +1.0)", xy=(0, 1), xytext=(20, 0.8), arrowprops=dict(arrowstyle="->", color="#276749"))
    ax1.annotate("Orthogonal (No Correlation = 0.0)", xy=(90, 0), xytext=(95, 0.3), arrowprops=dict(arrowstyle="->", color="#D69E2E"))
    ax1.annotate("Opposite (Conflict = -1.0)", xy=(180, -1), xytext=(110, -0.7), arrowprops=dict(arrowstyle="->", color="#C53030"))
    ax1.set_title("Dot Product as Semantic Alignment", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Angle θ Between Vectors (degrees)")
    ax1.set_ylabel("Cosine Similarity: a · b / (||a|| ||b||)")

    # Right: LoRA Low-Rank Decomposition
    ax2.axis("off")
    ax2.text(0.5, 0.95, "Low-Rank Adaptation (LoRA: ΔW = B · A)", ha="center", fontsize=11, fontweight="bold", transform=ax2.transAxes)
    
    # Full matrix W
    rect_full = plt.Rectangle((0.05, 0.25), 0.35, 0.55, facecolor="#EBF8FF", edgecolor="#3182CE", lw=2, transform=ax2.transAxes)
    ax2.add_patch(rect_full)
    ax2.text(0.225, 0.52, "Full Weight Matrix W\n(4096 × 4096)\n16.7M Parameters\n[FROZEN]", ha="center", va="center", fontsize=8.5, color="#2B6CB0", transform=ax2.transAxes)
    
    ax2.text(0.45, 0.52, "+", ha="center", va="center", fontsize=16, fontweight="bold", color="#4A5568", transform=ax2.transAxes)

    # Matrix B (4096 x 8)
    rect_B = plt.Rectangle((0.52, 0.25), 0.1, 0.55, facecolor="#FEFCBF", edgecolor="#D69E2E", lw=2, transform=ax2.transAxes)
    ax2.add_patch(rect_B)
    ax2.text(0.57, 0.52, "B\n(4096×8)\nInit 0", ha="center", va="center", fontsize=8, color="#744210", transform=ax2.transAxes)

    ax2.text(0.66, 0.52, "·", ha="center", va="center", fontsize=16, fontweight="bold", color="#4A5568", transform=ax2.transAxes)

    # Matrix A (8 x 4096)
    rect_A = plt.Rectangle((0.71, 0.45), 0.25, 0.15, facecolor="#C6F6D5", edgecolor="#38A169", lw=2, transform=ax2.transAxes)
    ax2.add_patch(rect_A)
    ax2.text(0.835, 0.525, "A (8×4096) [Init N(0, σ²)]", ha="center", va="center", fontsize=8, color="#22543D", transform=ax2.transAxes)
    
    ax2.text(0.72, 0.15, "Parameters: ~65,000 (< 0.5%)\nRank r = 8 bottleneck", ha="center", fontsize=9, fontweight="bold", color="#276749", transform=ax2.transAxes)

    fig.suptitle("Module 03: Intuitive Linear Algebra & LoRA Geometry", fontsize=13, fontweight="bold", color="#1A202C")
    plt.tight_layout()
    plt.savefig("assets/03_linear_algebra_visuals.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 04: Loss Landscapes & Optimizers
# -------------------------------------------------------------
def make_04():
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor("#F7FAFC")
    
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = (1 - X)**2 + 10 * (Y - X**2)**2  # Elliptical ravine
    
    contours = ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 15), cmap="Blues_r", alpha=0.6)
    ax.scatter([1], [1], marker="*", color="#E53E3E", s=200, label="Global Minimum (1, 1)", zorder=5)
    
    # Synthetic paths
    sgd_x = [-1.5, -1.2, -1.4, -0.9, -1.1, -0.5, -0.7, 0.2, 0.0, 0.6, 0.8, 1.0]
    sgd_y = [2.2, 1.1, 1.9, 0.5, 1.2, 0.2, 0.6, 0.1, 0.2, 0.4, 0.7, 1.0]
    ax.plot(sgd_x, sgd_y, color="#E53E3E", marker="o", markersize=3, lw=1.5, linestyle=":", label="SGD (Bounces between walls)")

    adam_x = np.linspace(-1.5, 1.0, 15)
    adam_y = 2.2 * np.exp(-1.5 * np.linspace(0, 1, 15)) + np.linspace(0, 1, 15)**2
    ax.plot(adam_x, adam_y, color="#276749", marker="s", markersize=4, lw=2.5, label="Adam (Momentum + Adaptive Step)")

    ax.set_title("Module 04: Optimization Dynamics in Loss Ravines", fontsize=13, fontweight="bold")
    ax.set_xlabel("Parameter w1")
    ax.set_ylabel("Parameter w2")
    ax.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    plt.savefig("assets/04_optimization_landscapes.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 05: Probability, Entropy & KL Divergence
# -------------------------------------------------------------
def make_05():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
    fig.patch.set_facecolor("#F7FAFC")
    
    # Left: Shannon Entropy of binary distribution
    p = np.linspace(0.001, 0.999, 100)
    entropy = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
    ax1.plot(p, entropy, color="#3182CE", lw=2.5)
    ax1.scatter([0.5], [1.0], color="#C53030", s=60, zorder=4)
    ax1.annotate("Max Uncertainty\n(H = 1.0 bit)", xy=(0.5, 1.0), xytext=(0.4, 0.7),
                 arrowprops=dict(arrowstyle="->", color="#C53030"))
    ax1.set_title("Shannon Entropy H(P) vs Probability", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Probability of Positive Outcome p")
    ax1.set_ylabel("Entropy (bits)")

    # Right: KL Divergence Forward vs Reverse
    x = np.linspace(-4, 6, 200)
    p_true = 0.5 * (1/np.sqrt(2*np.pi)) * (np.exp(-0.5*(x+1)**2) + np.exp(-0.5*(x-3)**2))
    q_forward = (1/np.sqrt(2*np.pi*2.5)) * np.exp(-0.5*(x-1)**2 / 2.5)
    q_reverse = (1/np.sqrt(2*np.pi*0.6)) * np.exp(-0.5*(x+1)**2 / 0.6)

    ax2.plot(x, p_true, color="#2D3748", lw=2.5, label="True Target P (Bimodal)")
    ax2.plot(x, q_forward, color="#3182CE", lw=2, linestyle="--", label="Forward KL: D_KL(P||Q) (Mean-Seeking)")
    ax2.plot(x, q_reverse, color="#38A169", lw=2, linestyle="-.", label="Reverse KL: D_KL(Q||P) (Mode-Seeking)")
    ax2.set_title("KL Divergence: Forward vs Reverse Mode", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Latent State x")
    ax2.set_ylabel("Probability Density")
    ax2.legend(frameon=True, fontsize=8.5)

    fig.suptitle("Module 05: Information Theory & Divergence Metrics", fontsize=13, fontweight="bold", color="#1A202C")
    plt.tight_layout()
    plt.savefig("assets/05_information_theory.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 06: Classical ML & Bias-Variance
# -------------------------------------------------------------
def make_06():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
    fig.patch.set_facecolor("#F7FAFC")
    
    # Left: Bias-Variance tradeoff
    complexity = np.linspace(1, 10, 100)
    bias = 5.0 / complexity
    variance = 0.05 * complexity**2
    total_error = bias + variance + 0.5

    ax1.plot(complexity, bias, label="Bias² (Underfitting)", color="#3182CE", lw=2, linestyle="--")
    ax1.plot(complexity, variance, label="Variance (Overfitting)", color="#E53E3E", lw=2, linestyle="--")
    ax1.plot(complexity, total_error, label="Test Error (Total)", color="#2D3748", lw=3)
    ax1.axvline(3.7, color="#38A169", linestyle=":", lw=2, label="Optimal Model Capacity")
    ax1.set_title("The Bias-Variance Tradeoff", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Model Capacity / Complexity")
    ax1.set_ylabel("Error")
    ax1.legend(frameon=True, fontsize=8.5)

    # Right: Confusion Matrix visual
    ax2.axis("off")
    ax2.text(0.5, 0.95, "Confusion Matrix & Metrics", ha="center", fontsize=11, fontweight="bold", transform=ax2.transAxes)
    
    grid = [
        ("True Positive (TP)\nTarget: 1 | Model: 1", "#C6F6D5", (0.1, 0.5)),
        ("False Positive (FP)\nTarget: 0 | Model: 1\n(Type I Error)", "#FED7D7", (0.55, 0.5)),
        ("False Negative (FN)\nTarget: 1 | Model: 0\n(Type II Error)", "#FED7D7", (0.1, 0.15)),
        ("True Negative (TN)\nTarget: 0 | Model: 0", "#C6F6D5", (0.55, 0.15)),
    ]
    for text, color, (x, y) in grid:
        rect = plt.Rectangle((x, y), 0.38, 0.3, facecolor=color, edgecolor="#718096", lw=1.5, transform=ax2.transAxes)
        ax2.add_patch(rect)
        ax2.text(x+0.19, y+0.15, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1A202C", transform=ax2.transAxes)

    fig.suptitle("Module 06: Generalization, Bias-Variance & Evaluation", fontsize=13, fontweight="bold", color="#1A202C")
    plt.tight_layout()
    plt.savefig("assets/06_ml_bias_variance.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 07: Micrograd Computational Graph
# -------------------------------------------------------------
def make_07():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    nodes = [
        ("x = 2.0\ndL/dx = +18.0", (0.1, 0.7), "#2B6CB0"),
        ("y = -3.0\ndL/dy = -12.0", (0.1, 0.3), "#2B6CB0"),
        ("u = x * y = -6.0\ndL/du = -6.0", (0.35, 0.5), "#2C7A7B"),
        ("v = u + 3.0 = -3.0\ndL/dv = -6.0", (0.6, 0.5), "#D69E2E"),
        ("Loss = v² = 9.0\ndL/dLoss = 1.0", (0.85, 0.5), "#C53030"),
    ]
    
    for text, (x, y), color in nodes:
        rect = plt.Rectangle((x-0.08, y-0.12), 0.16, 0.24, facecolor="white", edgecolor=color, lw=2.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, fontweight="bold", color=color, transform=ax.transAxes)

    # Forward arrows
    arrows = [
        ((0.18, 0.7), (0.27, 0.55)),
        ((0.18, 0.3), (0.27, 0.45)),
        ((0.43, 0.5), (0.52, 0.5)),
        ((0.68, 0.5), (0.77, 0.5)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2, color="#3182CE"), xycoords="axes fraction")

    ax.text(0.5, 0.88, "FORWARD PASS: Compute Values (Left to Right)", ha="center", fontsize=10, fontweight="bold", color="#3182CE", transform=ax.transAxes)
    ax.text(0.5, 0.12, "BACKWARD PASS: Propagate Gradients via Chain Rule (Right to Left)", ha="center", fontsize=10, fontweight="bold", color="#C53030", transform=ax.transAxes)

    ax.set_title("Module 07: Reverse-Mode Automatic Differentiation (Micrograd Engine)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/07_autograd_computational_graph.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 08: PyTorch Architecture & Device Loop
# -------------------------------------------------------------
def make_08():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    steps = [
        ("1. Data Pipeline", "Dataset & DataLoader\nBatching, shuffle, multi-worker", "#2B6CB0", (0.1, 0.5)),
        ("2. Model Forward", "nn.Module\nGPU / MPS dispatch", "#2C7A7B", (0.35, 0.5)),
        ("3. Loss Function", "nn.CrossEntropyLoss\nLog-sum-exp stabilization", "#D69E2E", (0.6, 0.5)),
        ("4. Backward & Step", "loss.backward()\noptimizer.step() (AdamW)", "#C53030", (0.85, 0.5)),
    ]
    for title, desc, color, (x, y) in steps:
        rect = plt.Rectangle((x-0.1, y-0.2), 0.2, 0.4, facecolor="white", edgecolor=color, lw=2.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, y+0.12, title, ha="center", va="center", fontsize=10, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x, y-0.03, desc, ha="center", va="center", fontsize=8.5, color="#2D3748", transform=ax.transAxes)

    for x in [0.2, 0.45, 0.7]:
        ax.annotate("", xy=(x+0.05, 0.5), xytext=(x, 0.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#4A5568"), xycoords="axes fraction")

    # Feedback loop arrow
    ax.annotate("", xy=(0.1, 0.25), xytext=(0.85, 0.25),
                arrowprops=dict(arrowstyle="->", lw=2, color="#718096", linestyle="--"), xycoords="axes fraction")
    ax.text(0.475, 0.28, "Repeat for N Epochs (model.train() vs model.eval())", ha="center", fontsize=9, color="#718096", transform=ax.transAxes)

    ax.set_title("Module 08: The Canonical PyTorch Research Execution Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/08_pytorch_ecosystem.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 09: Activations & Normalization
# -------------------------------------------------------------
def make_09():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
    fig.patch.set_facecolor("#F7FAFC")
    
    # Left: Activations
    x = np.linspace(-3, 3, 200)
    relu = np.maximum(0, x)
    gelu = x * 0.5 * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
    swish = x * (1 / (1 + np.exp(-x)))

    ax1.plot(x, relu, label="ReLU: max(0, x)", color="#718096", lw=2, linestyle=":")
    ax1.plot(x, gelu, label="GELU (GPT-2, BERT)", color="#3182CE", lw=2.5)
    ax1.plot(x, swish, label="Swish (SwiGLU component)", color="#38A169", lw=2.5)
    ax1.set_title("Modern Activation Functions", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Input Activation x")
    ax1.set_ylabel("f(x)")
    ax1.legend(frameon=True, fontsize=9)

    # Right: LayerNorm vs RMSNorm
    ax2.axis("off")
    ax2.text(0.5, 0.95, "LayerNorm vs. RMSNorm (Modern Frontier)", ha="center", fontsize=11, fontweight="bold", transform=ax2.transAxes)
    
    rect_ln = plt.Rectangle((0.05, 0.48), 0.9, 0.38, facecolor="white", edgecolor="#3182CE", lw=2, transform=ax2.transAxes)
    ax2.add_patch(rect_ln)
    ax2.text(0.1, 0.76, "Standard LayerNorm (Vaswani 2017)", fontsize=9.5, fontweight="bold", color="#2B6CB0", transform=ax2.transAxes)
    ax2.text(0.1, 0.60, "Formula: y = (x - μ) / √(σ² + ε) * γ + β\nComputes both mean AND variance. Has offset bias β.", fontsize=8.5, color="#2D3748", transform=ax2.transAxes)

    rect_rms = plt.Rectangle((0.05, 0.05), 0.9, 0.38, facecolor="white", edgecolor="#38A169", lw=2, transform=ax2.transAxes)
    ax2.add_patch(rect_rms)
    ax2.text(0.1, 0.33, "RMSNorm (LLaMA, DeepSeek-V3, Mistral)", fontsize=9.5, fontweight="bold", color="#22543D", transform=ax2.transAxes)
    ax2.text(0.1, 0.17, "Formula: y = x / RMS(x) * γ\nDrops mean calculation! Saves ~15% runtime with zero accuracy loss.", fontsize=8.5, color="#2D3748", transform=ax2.transAxes)

    fig.suptitle("Module 09: Neural Dynamics, Activations & Normalization", fontsize=13, fontweight="bold", color="#1A202C")
    plt.tight_layout()
    plt.savefig("assets/09_activations_and_norms.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 10: Byte-Pair Encoding
# -------------------------------------------------------------
def make_10():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    flow = [
        ("Raw Text", "'learning'", "#4A5568", 0.1),
        ("UTF-8 Bytes", "[108, 101, 97, ...]\n(256 Base Tokens)", "#2B6CB0", 0.32),
        ("BPE Merge Rules", "('e', 'a') -> 'ea'\n('ea', 'r') -> 'ear'", "#D69E2E", 0.58),
        ("Embedding Table", "nn.Embedding(V, D)\n[Tokens -> Dense Vectors]", "#276749", 0.85),
    ]
    for title, desc, color, x in flow:
        rect = plt.Rectangle((x-0.09, 0.3), 0.18, 0.4, facecolor="white", edgecolor=color, lw=2.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, 0.62, title, ha="center", va="center", fontsize=10, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x, 0.45, desc, ha="center", va="center", fontsize=8.5, color="#2D3748", transform=ax.transAxes)

    for x in [0.19, 0.41, 0.67]:
        ax.annotate("", xy=(x+0.04, 0.5), xytext=(x, 0.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#718096"), xycoords="axes fraction")

    ax.set_title("Module 10: Byte-Pair Encoding (BPE) & Vocabulary Construction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/10_bpe_tokenization.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 11: Attention & The Transformer
# -------------------------------------------------------------
def make_11():
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    # Schematic of Attention
    ax.text(0.5, 0.95, "Scaled Dot-Product Attention: Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V", ha="center", fontsize=11, fontweight="bold", transform=ax.transAxes)

    rect_q = plt.Rectangle((0.1, 0.65), 0.15, 0.15, facecolor="#EBF8FF", edgecolor="#3182CE", lw=2, transform=ax.transAxes)
    rect_k = plt.Rectangle((0.3, 0.65), 0.15, 0.15, facecolor="#FEFCBF", edgecolor="#D69E2E", lw=2, transform=ax.transAxes)
    rect_v = plt.Rectangle((0.75, 0.35), 0.15, 0.15, facecolor="#C6F6D5", edgecolor="#38A169", lw=2, transform=ax.transAxes)
    
    ax.add_patch(rect_q)
    ax.add_patch(rect_k)
    ax.add_patch(rect_v)

    ax.text(0.175, 0.725, "Query Q\n(Search)", ha="center", va="center", fontsize=9, fontweight="bold", color="#2B6CB0", transform=ax.transAxes)
    ax.text(0.375, 0.725, "Key K\n(Index)", ha="center", va="center", fontsize=9, fontweight="bold", color="#744210", transform=ax.transAxes)
    ax.text(0.825, 0.425, "Value V\n(Content)", ha="center", va="center", fontsize=9, fontweight="bold", color="#22543D", transform=ax.transAxes)

    rect_mid = plt.Rectangle((0.15, 0.35), 0.45, 0.18, facecolor="white", edgecolor="#4A5568", lw=2, transform=ax.transAxes)
    ax.add_patch(rect_mid)
    ax.text(0.375, 0.44, "Matmul Q @ K.T\nScale by 1/√d_k\n+ Causal Lower-Triangular Mask\nSoftmax along last axis", ha="center", va="center", fontsize=8.5, color="#1A202C", transform=ax.transAxes)

    rect_out = plt.Rectangle((0.35, 0.08), 0.3, 0.15, facecolor="#EDF2F7", edgecolor="#2D3748", lw=2, transform=ax.transAxes)
    ax.add_patch(rect_out)
    ax.text(0.5, 0.155, "Contextualized Output\nTokens Attended to Past", ha="center", va="center", fontsize=9, fontweight="bold", color="#1A202C", transform=ax.transAxes)

    ax.annotate("", xy=(0.25, 0.53), xytext=(0.2, 0.65), arrowprops=dict(arrowstyle="->", lw=2, color="#3182CE"), xycoords="axes fraction")
    ax.annotate("", xy=(0.35, 0.53), xytext=(0.35, 0.65), arrowprops=dict(arrowstyle="->", lw=2, color="#D69E2E"), xycoords="axes fraction")
    ax.annotate("", xy=(0.55, 0.23), xytext=(0.4, 0.35), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"), xycoords="axes fraction")
    ax.annotate("", xy=(0.6, 0.23), xytext=(0.75, 0.35), arrowprops=dict(arrowstyle="->", lw=2, color="#38A169"), xycoords="axes fraction")

    ax.set_title("Module 11: The Self-Attention Communication Mechanism", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/11_transformer_attention.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 12: GPT Decoder Architecture
# -------------------------------------------------------------
def make_12():
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    layers = [
        ("Final LM Head (tied to wte) -> Vocab Logits", "#C53030", 0.88),
        ("Final LayerNorm / RMSNorm", "#4A5568", 0.76),
        ("Transformer Block N (Pre-LN, Causal Attention, FFN)", "#2B6CB0", 0.60),
        ("... Repeated N Layers (e.g. 12 to 32 blocks) ...", "#718096", 0.46),
        ("Transformer Block 1 (Pre-LN, Causal Attention, FFN)", "#2B6CB0", 0.32),
        ("Token Embeddings + Positional Embeddings (wte + wpe)", "#2C7A7B", 0.18),
        ("Input Prompt Tokens [B, T]", "#1A202C", 0.06),
    ]
    for text, color, y in layers:
        rect = plt.Rectangle((0.1, y-0.04), 0.8, 0.08, facecolor="white", edgecolor=color, lw=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.5, y, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color=color, transform=ax.transAxes)

    ax.set_title("Module 12: Decoder-Only GPT Foundation Architecture (nanoGPT)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/12_gpt_decoder_stack.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 13: Reinforcement Learning & MDP
# -------------------------------------------------------------
def make_13():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    rect_agent = plt.Rectangle((0.15, 0.4), 0.25, 0.35, facecolor="white", edgecolor="#2B6CB0", lw=2.5, transform=ax.transAxes)
    rect_env = plt.Rectangle((0.6, 0.4), 0.25, 0.35, facecolor="white", edgecolor="#22543D", lw=2.5, transform=ax.transAxes)
    ax.add_patch(rect_agent)
    ax.add_patch(rect_env)

    ax.text(0.275, 0.62, "AGENT\n(Policy π)", ha="center", va="center", fontsize=11, fontweight="bold", color="#2B6CB0", transform=ax.transAxes)
    ax.text(0.275, 0.48, "Selects action a_t\nLearns from rewards", ha="center", va="center", fontsize=8.5, color="#4A5568", transform=ax.transAxes)

    ax.text(0.725, 0.62, "ENVIRONMENT\n(Dynamics P)", ha="center", va="center", fontsize=11, fontweight="bold", color="#22543D", transform=ax.transAxes)
    ax.text(0.725, 0.48, "Emits state s_{t+1}\nDelivers reward r_{t+1}", ha="center", va="center", fontsize=8.5, color="#4A5568", transform=ax.transAxes)

    # Top Arrow: Action
    ax.annotate("", xy=(0.6, 0.65), xytext=(0.4, 0.65), arrowprops=dict(arrowstyle="->", lw=2.5, color="#3182CE"), xycoords="axes fraction")
    ax.text(0.5, 0.70, "Action a_t", ha="center", fontsize=10, fontweight="bold", color="#3182CE", transform=ax.transAxes)

    # Bottom Arrow: State & Reward
    ax.annotate("", xy=(0.4, 0.45), xytext=(0.6, 0.45), arrowprops=dict(arrowstyle="->", lw=2.5, color="#38A169"), xycoords="axes fraction")
    ax.text(0.5, 0.38, "Reward r_{t+1} & Next State s_{t+1}", ha="center", fontsize=10, fontweight="bold", color="#38A169", transform=ax.transAxes)

    ax.text(0.5, 0.15, "Bellman Equation: V(s) = R(s) + γ Σ P(s'|s) V(s')\nReturn: G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...", ha="center", fontsize=9.5, fontweight="bold", color="#2D3748", transform=ax.transAxes)

    ax.set_title("Module 13: The Reinforcement Learning Interaction Loop (MDP)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/13_rl_mdp_bellman.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 14: PPO Clipped Objective
# -------------------------------------------------------------
def make_14():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
    fig.patch.set_facecolor("#F7FAFC")
    
    r = np.linspace(0.5, 1.5, 100)
    eps = 0.2

    # Positive advantage A > 0
    A_pos = 1.0
    surr1_pos = r * A_pos
    surr2_pos = np.clip(r, 1 - eps, 1 + eps) * A_pos
    obj_pos = np.minimum(surr1_pos, surr2_pos)

    ax1.plot(r, surr1_pos, label="Unclipped: r · A", color="#A0AEC0", linestyle="--")
    ax1.plot(r, obj_pos, label="Clipped Objective L_CLIP", color="#276749", lw=3)
    ax1.axvline(1 - eps, color="#E53E3E", linestyle=":", label="1 - ε (0.8)")
    ax1.axvline(1 + eps, color="#E53E3E", linestyle=":", label="1 + ε (1.2)")
    ax1.set_title("Positive Advantage (A > 0): Good Action", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Probability Ratio r(θ) = π_θ / π_old")
    ax1.set_ylabel("Surrogate Objective")
    ax1.legend(frameon=True, fontsize=8.5)

    # Negative advantage A < 0
    A_neg = -1.0
    surr1_neg = r * A_neg
    surr2_neg = np.clip(r, 1 - eps, 1 + eps) * A_neg
    obj_neg = np.minimum(surr1_neg, surr2_neg)

    ax2.plot(r, surr1_neg, label="Unclipped: r · A", color="#A0AEC0", linestyle="--")
    ax2.plot(r, obj_neg, label="Clipped Objective L_CLIP", color="#C53030", lw=3)
    ax2.axvline(1 - eps, color="#E53E3E", linestyle=":")
    ax2.axvline(1 + eps, color="#E53E3E", linestyle=":")
    ax2.set_title("Negative Advantage (A < 0): Bad Action", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Probability Ratio r(θ) = π_θ / π_old")
    ax2.set_ylabel("Surrogate Objective")
    ax2.legend(frameon=True, fontsize=8.5)

    fig.suptitle("Module 14: Proximal Policy Optimization (PPO) Clipped Objective", fontsize=13, fontweight="bold", color="#1A202C")
    plt.tight_layout()
    plt.savefig("assets/14_ppo_clipping.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 15: Post-Training SFT & DPO
# -------------------------------------------------------------
def make_15():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    ax.text(0.5, 0.95, "Direct Preference Optimization (DPO): Bypassing the Reward Model", ha="center", fontsize=11, fontweight="bold", transform=ax.transAxes)

    rect_pref = plt.Rectangle((0.05, 0.35), 0.25, 0.45, facecolor="white", edgecolor="#2B6CB0", lw=2, transform=ax.transAxes)
    ax.add_patch(rect_pref)
    ax.text(0.175, 0.70, "Preference Pair", ha="center", fontsize=10, fontweight="bold", color="#2B6CB0", transform=ax.transAxes)
    ax.text(0.175, 0.50, "Prompt x\nWinner y_w (Chosen)\nLoser y_l (Rejected)", ha="center", fontsize=8.5, color="#2D3748", transform=ax.transAxes)

    rect_models = plt.Rectangle((0.38, 0.35), 0.26, 0.45, facecolor="white", edgecolor="#D69E2E", lw=2, transform=ax.transAxes)
    ax.add_patch(rect_models)
    ax.text(0.51, 0.70, "Policy & Reference", ha="center", fontsize=10, fontweight="bold", color="#744210", transform=ax.transAxes)
    ax.text(0.51, 0.50, "Active Policy π_θ\nFrozen Reference π_ref\nImplicit Reward:\nr(x,y) = β log(π_θ/π_ref)", ha="center", fontsize=8.5, color="#2D3748", transform=ax.transAxes)

    rect_loss = plt.Rectangle((0.72, 0.35), 0.23, 0.45, facecolor="#C6F6D5", edgecolor="#22543D", lw=2, transform=ax.transAxes)
    ax.add_patch(rect_loss)
    ax.text(0.835, 0.70, "DPO Loss", ha="center", fontsize=10, fontweight="bold", color="#22543D", transform=ax.transAxes)
    ax.text(0.835, 0.50, "-log σ(β · [Δr_w - Δr_l])\nDirect gradients\nNo RL Actor-Critic!", ha="center", fontsize=8.5, color="#22543D", transform=ax.transAxes)

    ax.annotate("", xy=(0.38, 0.58), xytext=(0.3, 0.58), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"), xycoords="axes fraction")
    ax.annotate("", xy=(0.72, 0.58), xytext=(0.64, 0.58), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"), xycoords="axes fraction")

    ax.set_title("Module 15: Post-Training Preference Alignment with DPO", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/15_dpo_alignment.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 16: DeepSeek GRPO
# -------------------------------------------------------------
def make_16():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    ax.text(0.5, 0.95, "DeepSeek Group Relative Policy Optimization (GRPO: arXiv:2402.03300)", ha="center", fontsize=11, fontweight="bold", transform=ax.transAxes)

    rect_q = plt.Rectangle((0.05, 0.4), 0.2, 0.35, facecolor="white", edgecolor="#2B6CB0", lw=2, transform=ax.transAxes)
    ax.add_patch(rect_q)
    ax.text(0.15, 0.65, "Prompt q", ha="center", fontsize=10, fontweight="bold", color="#2B6CB0", transform=ax.transAxes)
    ax.text(0.15, 0.50, "Sample G outputs\n{o1, o2, ..., oG}\nfrom Policy π_θ", ha="center", fontsize=8.5, color="#2D3748", transform=ax.transAxes)

    rect_ver = plt.Rectangle((0.35, 0.4), 0.25, 0.35, facecolor="white", edgecolor="#38A169", lw=2, transform=ax.transAxes)
    ax.add_patch(rect_ver)
    ax.text(0.475, 0.65, "Rule-Based Verifier", ha="center", fontsize=10, fontweight="bold", color="#22543D", transform=ax.transAxes)
    ax.text(0.475, 0.50, "Score each output: r_i\nTrue/False math check\n+ format bonuses", ha="center", fontsize=8.5, color="#2D3748", transform=ax.transAxes)

    rect_norm = plt.Rectangle((0.68, 0.4), 0.27, 0.35, facecolor="#FEFCBF", edgecolor="#D69E2E", lw=2, transform=ax.transAxes)
    ax.add_patch(rect_norm)
    ax.text(0.815, 0.65, "Group Advantage", ha="center", fontsize=10, fontweight="bold", color="#744210", transform=ax.transAxes)
    ax.text(0.815, 0.50, "A_i = (r_i - mean) / std\nNo Critic Network!\nSaves 50% GPU memory!", ha="center", fontsize=8.5, color="#744210", transform=ax.transAxes)

    ax.annotate("", xy=(0.35, 0.58), xytext=(0.25, 0.58), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"), xycoords="axes fraction")
    ax.annotate("", xy=(0.68, 0.58), xytext=(0.60, 0.58), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"), xycoords="axes fraction")

    ax.set_title("Module 16: Group Relative Policy Optimization (GRPO Architecture)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/16_deepseek_grpo.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 17: Reasoning Models & Test-Time Compute
# -------------------------------------------------------------
def make_17():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=200)
    fig.patch.set_facecolor("#F7FAFC")
    
    # Left: Test-time compute curve
    tokens = np.array([100, 250, 500, 1000, 2000, 4000, 8000])
    acc = 45 + 40 * (1 - np.exp(-tokens / 1200))
    ax1.plot(tokens, acc, color="#3182CE", marker="o", lw=2.5)
    ax1.set_title("Inference-Time Compute Scaling", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Reasoning Tokens Generated per Problem")
    ax1.set_ylabel("Competition Math Accuracy (%)")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Right: System 1 vs System 2 schematic
    ax2.axis("off")
    ax2.text(0.5, 0.95, "System 1 vs. System 2 Reasoning", ha="center", fontsize=11, fontweight="bold", transform=ax2.transAxes)
    
    rect_s1 = plt.Rectangle((0.05, 0.52), 0.9, 0.35, facecolor="#FED7D7", edgecolor="#E53E3E", lw=2, transform=ax2.transAxes)
    ax2.add_patch(rect_s1)
    ax2.text(0.1, 0.76, "System 1: Fast / Direct Predict", fontsize=9.5, fontweight="bold", color="#9B2C2C", transform=ax2.transAxes)
    ax2.text(0.1, 0.62, "Generates instant answer in 1 forward pass.\nProne to arithmetic slips and hallucinations.", fontsize=8.5, color="#2D3748", transform=ax2.transAxes)

    rect_s2 = plt.Rectangle((0.05, 0.08), 0.9, 0.35, facecolor="#C6F6D5", edgecolor="#38A169", lw=2, transform=ax2.transAxes)
    ax2.add_patch(rect_s2)
    ax2.text(0.1, 0.32, "System 2: DeepSeek-R1 / Deliberate Search", fontsize=9.5, fontweight="bold", color="#22543D", transform=ax2.transAxes)
    ax2.text(0.1, 0.18, "Generates <think> traces, backtracks from errors, self-verifies.\nDynamic computational depth unlocked via RL!", fontsize=8.5, color="#2D3748", transform=ax2.transAxes)

    fig.suptitle("Module 17: Reasoning Models & Test-Time Compute Dynamics", fontsize=13, fontweight="bold", color="#1A202C")
    plt.tight_layout()
    plt.savefig("assets/17_reasoning_models.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 18: Interpretability & Sparse Autoencoders
# -------------------------------------------------------------
def make_18():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    ax.text(0.5, 0.95, "Sparse Autoencoder (SAE): Untangling Superposition", ha="center", fontsize=11, fontweight="bold", transform=ax.transAxes)

    rect_in = plt.Rectangle((0.08, 0.35), 0.2, 0.4, facecolor="#E2E8F0", edgecolor="#4A5568", lw=2, transform=ax.transAxes)
    ax.add_patch(rect_in)
    ax.text(0.18, 0.60, "Model Layer\nActivation x\n(d_model = 4096)", ha="center", fontsize=9, fontweight="bold", color="#2D3748", transform=ax.transAxes)
    ax.text(0.18, 0.43, "Polysemantic\n(Superposition)", ha="center", fontsize=8, color="#718096", transform=ax.transAxes)

    rect_sae = plt.Rectangle((0.38, 0.25), 0.24, 0.6, facecolor="#FEFCBF", edgecolor="#D69E2E", lw=2.5, transform=ax.transAxes)
    ax.add_patch(rect_sae)
    ax.text(0.5, 0.74, "Sparse Latents f\n(dict_size = 65,536)", ha="center", fontsize=9.5, fontweight="bold", color="#744210", transform=ax.transAxes)
    ax.text(0.5, 0.50, "Only ~30 fire per token!\nMonosemantic Features:\n• Golden Gate Bridge\n• Code SQL Injection\n• French Grammar", ha="center", fontsize=8, color="#744210", transform=ax.transAxes)

    rect_out = plt.Rectangle((0.72, 0.35), 0.2, 0.4, facecolor="#E2E8F0", edgecolor="#4A5568", lw=2, transform=ax.transAxes)
    ax.add_patch(rect_out)
    ax.text(0.82, 0.58, "Reconstructed x_hat\n||x - x_hat||²\n+ λ ||f||_1", ha="center", fontsize=9, fontweight="bold", color="#2D3748", transform=ax.transAxes)

    ax.annotate("", xy=(0.38, 0.55), xytext=(0.28, 0.55), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"), xycoords="axes fraction")
    ax.annotate("", xy=(0.72, 0.55), xytext=(0.62, 0.55), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"), xycoords="axes fraction")

    ax.set_title("Module 18: Mechanistic Interpretability & Feature Extraction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/18_interpretability_sae.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 19: Keshav Three-Pass & Paperleet
# -------------------------------------------------------------
def make_19():
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    passes = [
        ("Pass 1: Scan (5-10m)", "Title, abstract, headings\nAssess 5 Cs (Category, Context,\nCorrectness, Contributions, Clarity)", "#2B6CB0", 0.15),
        ("Pass 2: Grasp (1 hour)", "Deep figures & tables inspection\nAnnotate math equations\nCheck ablation studies", "#2C7A7B", 0.5),
        ("Pass 3: Rebuild (2-5h)", "Mentally re-implement paper\nIdentify hidden assumptions\nReproduce in PyTorch", "#44337A", 0.85),
    ]
    for title, desc, color, x in passes:
        rect = plt.Rectangle((x-0.14, 0.25), 0.28, 0.5, facecolor="white", edgecolor=color, lw=2.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, 0.65, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x, 0.45, desc, ha="center", va="center", fontsize=8.5, color="#2D3748", transform=ax.transAxes)

    for x in [0.29, 0.64]:
        ax.annotate("", xy=(x+0.07, 0.5), xytext=(x, 0.5), arrowprops=dict(arrowstyle="->", lw=2.5, color="#718096"), xycoords="axes fraction")

    ax.text(0.5, 0.12, "Community Hub: Publish reproduction logs & critique on Paperleet (paperleet.com)", ha="center", fontsize=9.5, fontweight="bold", color="#276749", transform=ax.transAxes)

    ax.set_title("Module 19: The S. Keshav Three-Pass Paper Reading Method", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/19_three_pass_reading.png", bbox_inches="tight")
    plt.close()

# -------------------------------------------------------------
# 20: Capstone Paper Reproduction
# -------------------------------------------------------------
def make_20():
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    ax.axis("off")
    fig.patch.set_facecolor("#F7FAFC")
    
    steps = [
        ("1. Paperleet Setup", "paperleet.com\narXiv:2402.03300", "#3182CE", (0.1, 0.6)),
        ("2. GSM8K Verifier", "Regex rule check\n+ format bonuses", "#38A169", (0.35, 0.6)),
        ("3. Group Sampling", "Policy samples G=4\nanswers per prompt", "#D69E2E", (0.6, 0.6)),
        ("4. GRPO Engine", "Advantage standardizer\n+ Clipped surrogate loss", "#C53030", (0.85, 0.6)),
    ]
    for title, desc, color, (x, y) in steps:
        rect = plt.Rectangle((x-0.1, y-0.2), 0.2, 0.4, facecolor="white", edgecolor=color, lw=2.5, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x, y+0.12, title, ha="center", va="center", fontsize=10, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(x, y-0.03, desc, ha="center", va="center", fontsize=8.5, color="#2D3748", transform=ax.transAxes)

    for x in [0.2, 0.45, 0.7]:
        ax.annotate("", xy=(x+0.05, 0.6), xytext=(x, 0.6), arrowprops=dict(arrowstyle="->", lw=2, color="#4A5568"), xycoords="axes fraction")

    rect_report = plt.Rectangle((0.15, 0.08), 0.7, 0.2, facecolor="#EDF2F7", edgecolor="#2D3748", lw=2, transform=ax.transAxes)
    ax.add_patch(rect_report)
    ax.text(0.5, 0.18, "Final Deliverable: Runnable reproduce_deepseek_grpo.py & Paperleet Reproduction Report", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1A202C", transform=ax.transAxes)

    ax.set_title("Module 20: Capstone Paper Reproduction Workflow (DeepSeekMath GRPO)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/20_capstone_grpo_paper.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    print("Generating all 20 module educational diagrams...")
    make_01()
    make_02()
    make_03()
    make_04()
    make_05()
    make_06()
    make_07()
    make_08()
    make_09()
    make_10()
    make_11()
    make_12()
    make_13()
    make_14()
    make_15()
    make_16()
    make_17()
    make_18()
    make_19()
    make_20()
    print("✅ All 20 educational diagrams generated successfully in assets/!")
