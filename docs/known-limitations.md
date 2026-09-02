# Known workflow limits

This branch shares the implementation without changing its training math.
Publication checks cover source disclosure, 22 CPU unit/integration tests with
existing upstream dependencies, and synthetic batch CLI dry-runs. They are not
a new full-data GPU run, a clean-environment installation test, or a guarantee
of identical reconstructed images.

- **Installation:** `pyproject.toml` retains Torch 2.6.x/torchvision 0.21.x and a
  CUDA 12.4 `uv` source mapping. The README's primary setup targets that stack.
  Do not substitute CUDA 12.1 wheels into the same dependency constraints.
  CPU/macOS users need a compatible separately managed environment and must
  bypass the CUDA source mapping as described in the README. The old conda
  environment file is not the fast branch's supported installation recipe.
- **Ordinary batch resume:** `run_batch_training.py` and
  `run_batch_training_brats.py` select existing phase checkpoints independently
  by filename and modification time. Do not use their `--resume` option when
  configs or Phase 1 parents may differ. Use the dedicated fast-exact runners
  for validated completed-phase reuse, or pass verified checkpoint pairs
  explicitly to the individual phase commands.
- **Preprocessing repeatability:** OASIS/BraTS transforms use Python's built-in
  string hash. BraTS `--seed` controls subject selection, not that hash. Preserve
  generated inputs; for repeat preprocessing, hold Python/runtime settings and
  `PYTHONHASHSEED` fixed before process startup. The default invocation is not
  a cross-process transform-reproducibility guarantee.
- **Fast-backend context bookkeeping:** successful lookups can leave dead weak
  references in the forward-context dictionaries. This does not retain the
  expired tensors themselves, but may add Python bookkeeping overhead in long
  runs. A small synthetic check confirmed stale-entry retention; its runtime
  impact on full MRI training has not been measured here.
- **Auxiliary methods:** ECLARE/SMORE batch adapters require separate method
  implementations not included in this source tree.

These inherited limitations remain open; the passing numerical tests are not a
claim that all research workflows or deployment environments are validated.
