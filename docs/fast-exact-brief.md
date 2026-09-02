# Fast-exact branch: differences from main

- Same three-phase SIMS-MRI method, with an opt-in faster registration stage;
  original phase commands remain available.
- Faster Phase 2 through analytic SIREN bending derivatives and avoiding unused
  Phase 1 parameter gradients, while retaining registration gradient flow.
- Adds full-pipeline and registration-only batch runners, with configuration and
  checkpoint-parent validation when reusing completed phases.
- Adds sagittal-view pair configurations, BraTS data workflows, and support for
  nested or flat subject folders.
- Includes numerical loss/gradient and resume regression tests. Exact refers to
  the loss calculation, not identical final reconstructions; runtime and image
  quality still depend on the data and environment.

Use `scripts/run_batch_training_fast_exact.py` for the complete fast workflow,
or `sims-mri-train-phase2-fast-exact` for registration alone. See the README for
setup. Dataset images, checkpoints, and experiment logs are not included.
