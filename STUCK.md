# 🆘 Troubleshooting & Common Errors (`STUCK.md`)

When beginning in AI research and PyTorch development, you will inevitably hit cryptic terminal errors. Below are the **Top 15 most common errors** along with step-by-step diagnostic checks and exact fixes.

---

## 1. `ModuleNotFoundError: No module named 'torch'` (or `numpy`)
- **Root Cause**: You installed dependencies in a virtual environment, but your current terminal session has not activated it, or your IDE is pointing to global Python.
- **Fix**:
  ```bash
  source .venv/bin/activate    # macOS / Linux
  # or .venv\Scripts\activate  # Windows
  which python                # Verify it points to .../ai-research-foundations/.venv/bin/python
  ```

---

## 2. `RuntimeError: The size of tensor a (N) must match the size of tensor b (M) at non-singleton dimension 1`
- **Root Cause**: A matrix multiplication or broadcasting mismatch between layers.
- **Fix**:
  Add shape print statements before the line:
  ```python
  print("Tensor a shape:", a.shape)
  print("Tensor b shape:", b.shape)
  ```
  Ensure inner matrix dimensions match: `(B, T, D) @ (B, D, H) -> (B, T, H)`. If one is transposed, use `.transpose(-2, -1)` or `.T`.

---

## 3. `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (or mps:0)`
- **Root Cause**: One tensor or model parameter was moved to the GPU (`.to(device)`), but another tensor remained on CPU.
- **Fix**:
  Explicitly dispatch all newly created tensors to the active device:
  ```python
  device = next(model.parameters()).device
  x = x.to(device)
  target = target.to(device)
  ```

---

## 4. `RuntimeError: CUDA out of memory` (or `MPS allocated memory exceeds limit`)
- **Root Cause**: The combination of batch size, sequence length, and model weights exceeds your graphics card's physical VRAM.
- **Fix**:
  1. Reduce your batch size: e.g. from `batch_size = 64` to `batch_size = 16`.
  2. Shorten the sequence context window: e.g. `block_size = 256` down to `128`.
  3. Ensure old gradients are freed: `optimizer.zero_grad(set_to_none=True)`.
  4. Detach values when computing running statistics: `total_loss += loss.item()` (never `total_loss += loss`).

---

## 5. `RuntimeError: Trying to backward through the graph a second time`
- **Root Cause**: You called `loss.backward()` twice without resetting the computation graph or retained a live computational graph across loop iterations.
- **Fix**:
  Make sure you extract scalar numbers with `.item()` when logging loss:
  ```python
  # BAD:
  running_loss += loss  # Keeps the entire computation graph in memory!
  # GOOD:
  running_loss += loss.item()  # Extracts raw float, allowing graph to be freed
  ```

---

## 6. Model Loss Explodes into `nan` (Not a Number)
- **Root Cause**: Learning rate is set too high, gradient explosion occurred, or an unconstrained mathematical operation (like `torch.log(0.0)` or `sqrt(-1)`) was computed.
- **Fix**:
  1. Add gradient clipping in your training loop:
     ```python
     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
     ```
  2. Add an epsilon buffer inside logarithms: `torch.log(p.clamp(min=1e-12))`.
  3. Lower the learning rate: e.g. from `1e-3` down to `3e-4` or `1e-4`.

---

## 7. Model Loss Never Changes (Stuck at Constant Value)
- **Root Cause**: You forgot to call `optimizer.step()`, forgot `optimizer.zero_grad()`, or all gradients are zero due to a dying ReLU or detached tensor.
- **Fix**:
  Verify the standard 5-step cadence is intact:
  ```python
  optimizer.zero_grad(set_to_none=True)
  outputs = model(inputs)
  loss = criterion(outputs, targets)
  loss.backward()
  optimizer.step()
  ```
  Check that model weights have gradients: `print(model.linear.weight.grad)`.

---

## 8. `TypeError: Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64')`
- **Root Cause**: NumPy array type mismatch when trying to write floating point results into an integer array in-place.
- **Fix**:
  Ensure the array is initialized with float dtype: `arr = np.zeros(shape, dtype=np.float32)`.

---

## 9. PyTorch `CrossEntropyLoss` Producing Bizarrely High Losses
- **Root Cause**: Applying `torch.softmax()` to model outputs before passing them into `nn.CrossEntropyLoss()`.
- **Fix**:
  `nn.CrossEntropyLoss` expects **raw unnormalized logits**. Remove any final `nn.Softmax()` layer from your model definition.

---

## 10. `RuntimeError: view size is not compatible with input tensor's size and stride`
- **Root Cause**: The tensor has been transposed or sliced, meaning its elements are no longer stored in contiguous memory.
- **Fix**:
  Call `.contiguous()` immediately prior to `.view()`:
  ```python
  y = x.transpose(1, 2).contiguous().view(B, T, C)
  ```
  Alternatively, use `.reshape()` which calls `.contiguous()` automatically if necessary.

---

## 11. Git Accidental Large File Commit: `File is 150 MB; this exceeds GitHub's limit`
- **Root Cause**: You committed a `.pt` checkpoint or large `.safetensors` weight file to git.
- **Fix**:
  ```bash
  # Remove the staged file from index without deleting it from disk:
  git rm --cached path/to/large_model.pt
  echo "*.pt" >> .gitignore
  git commit --amend -CHEAD
  ```

---

## 12. Python Generator Yields Nothing on Second Iteration
- **Root Cause**: Generators in Python stream data once and then exhaust themselves.
- **Fix**:
  Wrap the generator creation in a function or class so you can instantiate a fresh iterator for each training epoch.

---

## 13. `IndexError: Target N is out of bounds`
- **Root Cause**: Your target label integer is greater than or equal to `num_classes` in classification.
- **Fix**:
  Remember that classes must be indexed from `0` to `num_classes - 1`. If your labels are 1-indexed (1, 2, 3), subtract 1: `y = y - 1`.

---

## 14. Validation Accuracy High, but Real-World Generations Hallucinate
- **Root Cause**: Data leakage between training and validation sets, or target sequence shifted incorrectly during tokenization.
- **Fix**:
  Verify your context $(x)$ and target $(y)$ alignment:
  ```python
  x = tokens[i : i + block_size]
  y = tokens[i + 1 : i + block_size + 1]
  ```
  Each target token must be exactly one position into the future.

---

## 15. `UserWarning: The operator 'aten::std.correction' is not currently supported on the MPS backend`
- **Root Cause**: Certain rare PyTorch operations lack dedicated Metal Performance Shader kernels on Apple Silicon.
- **Fix**:
  Add an environment fallback to CPU for unsupported MPS operators:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
  Or run that specific operation on CPU: `x_cpu = x.cpu(); res = op(x_cpu).to(device)`.
