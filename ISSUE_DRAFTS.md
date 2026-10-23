# Draft GitHub Issues for Community Contribution (`ISSUE_DRAFTS.md`)

*Note: These are local drafts for the author's review. None have been published to GitHub.*

---

## Issue 1: [Good First Issue] Add interactive Jupyter Notebooks for Modules 01 to 05
- **Labels:** `good first issue`, `documentation`
- **Description:**  
  Currently, Modules 01 through 05 contain standalone Python scripts (`tooling_demo.py`, `numpy_foundations.py`, etc.). Beginners often prefer running code cells interactively with immediate visual output.
- **Expected Deliverable:**  
  Add `notebook.ipynb` in each of the first 5 module folders containing the markdown text, formulas, and runnable code blocks with inline Matplotlib plots.

---

## Issue 2: [Enhancement] Add a visual interactive comparison of BPE vs WordPiece tokenization in Module 10
- **Labels:** `enhancement`, `educational`
- **Description:**  
  Module 10 implements Byte-Pair Encoding (BPE). Adding a small comparison section or script contrasting BPE with WordPiece (used in BERT) and Unigram (SentencePiece) would deepen the student's understanding of subword algorithms.
- **Expected Deliverable:**  
  A clean paragraph and small Python function demonstrating the score difference when selecting candidates to merge.

---

## Issue 3: [Feature] Implement AdamW weight decay comparison test in Module 04
- **Labels:** `good first issue`, `testing`
- **Description:**  
  Module 04 covers SGD, Momentum, and Adam. Level 3 Exercise 1 asks about the difference between standard L2 regularization and AdamW's decoupled weight decay.
- **Expected Deliverable:**  
  Add an automated test in `tests/test_module_04.py` verifying that decoupled weight decay $w \leftarrow w(1 - \eta \lambda)$ shrinks weights directly compared to adding $\lambda w$ to the gradient.

---

## Issue 4: [Documentation] Expand Module 19 with a practical arXiv API script
- **Labels:** `enhancement`, `tooling`
- **Description:**  
  Module 19 covers reading papers and the 3-pass method, but currently lacks an executable python utility.
- **Expected Deliverable:**  
  Add `arxiv_search.py` using Python's standard `urllib` to query the public arXiv API for recent papers by query keyword and print title, authors, and summary without external API keys.

---

## Issue 5: [Testing] Add parameterized tests for RoPE relative distance invariance in Module 11
- **Labels:** `testing`, `math`
- **Description:**  
  In Module 11, Rotary Position Embeddings (RoPE) are discussed. We need an explicit unit test showing that for any two position indices $m$ and $n$, the inner product of rotated vectors $\langle R_m q, R_n k \rangle$ is strictly a function of $m - n$.
- **Expected Deliverable:**  
  Add a parameterized test in `tests/test_module_11.py` validating rotation invariance across random vectors.
