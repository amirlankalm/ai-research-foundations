# Release Checklist (`RELEASE_CHECKLIST.md`)

Use this checklist before merging changes into `main` or drafting a release tag:

- [ ] **Honesty & Claims**: Verify that no title, badge, or section header claims capabilities not backed by runnable code in the repository.
- [ ] **Clean Virtual Environment**: Run a test installation from both `requirements-core.txt` and `requirements-torch.txt`.
- [ ] **CPU Execution**: Confirm all demo scripts run cleanly on CPU without requiring GPU hardware:
  ```bash
  CUDA_VISIBLE_DEVICES="" pytest tests/
  ```
- [ ] **Links & Assets**: Verify all internal links between modules, images, and documents resolve properly.
- [ ] **Third-Party Art Disclaimers**: Verify `assets/banner.png` exclusion is documented in `ASSETS_NOTICE.md` and `README.md`.
- [ ] **No Secrets**: Ensure `.env`, private tokens, or large weight binaries (`*.pt`) are excluded via `.gitignore`.
- [ ] **CI Workflow**: Confirm GitHub Actions pipeline is passing.
