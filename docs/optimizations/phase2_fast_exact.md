# Phase 2 Fast Exact Launcher

This is an opt-in launcher for Phase 2 registration:

```bash
PYTHONPATH=src:. python -m sims_mri.training.phase2_registration_fast_exact <same args as phase2_registration>
```

The launcher delegates to the existing `sims_mri.training.phase2_registration`
CLI after installing two process-local exact optimizations:

1. Freeze Phase-1 modules that Phase 2 already puts in `eval()` mode:
   `HashEncoderWrapper`, `MLPv1`, and generation-shaped IDIR `Siren` modules.
   The wrapper only sets parameter `requires_grad=False`; it does not detach
   tensors, use `torch.no_grad()`, or enter inference mode, so coordinate and
   registration gradients still flow through the frozen forward operations.
2. Replace `IDIR.objectives.regularizers.compute_bending_energy` with a strict
   exact SIREN backend in the current process. CPU/CUDA use the six-term
   analytic SIREN backend by default; MPS uses the full analytic backend.

The backend policy can be overridden for validation:

```bash
SIMS_MRI_PHASE2_BENDING_BACKEND=auto|terms6|analytic_siren
```

The source optimization pass measured the accepted exact launcher against
upstream bending with the freeze optimization enabled:

| Workload | Upstream + freeze | Fast exact + freeze | Speedup | Correctness |
|---|---:|---:|---:|---|
| CPU compact, `N=512` | 15.058 ms | 5.206 ms | 2.89x | exact total/reg loss parity |
| CPU reduced production-like, `N=256` | 13.740 ms | 5.296 ms | 2.59x | exact total/reg loss parity |
| MPS compact, `N=256` | 132.746 ms | 25.208 ms | 5.27x | float-noise total/reg loss parity |

Cadenced/periodic bending updates were tested in the source pass but are not
included here because K=2 and K=4 both failed the held-out exact-bending
smoothness gate by a wide margin. This PR keeps only the exact opt-in path.

## Batch execution and resume

`scripts/run_batch_training_fast_exact.py` runs all three phases;
`scripts/run_batch_training_oasis_phase2_fast_exact.py` runs registration from
an existing Phase 1 checkpoint. Both require a valid strict-backend and freeze
audit from a freshly completed Phase 2 subprocess.

With `--resume`, Phase 1 selection requires the model, its hash-encoder weights,
and a matching config snapshot. Registration is reused only when a successful
fast-exact batch result names the same Phase 1 checkpoint, the effective config
snapshot agrees, and the recorded audit matches the requested backend. The
selected command, source record and audit are carried into the new result JSON,
so a validated skip remains resumable even when replacing the prior results file.
File age alone
does not prove that checkpoints belong together.

Keep the original successful batch result JSON in `runs/` (or the Phase-2-only
runner's `--checkpoint_search_root`), alongside the run directories and their
config snapshots. Both full-pipeline and Phase-2-only result formats are
supported. If you use `--results_json`, save it in that search root so it can be
discovered. The Phase-2-only runner defaults its result JSON to its configured
checkpoint search root. Parent checkpoint paths remain authoritative when the
upstream optional parent-ID label is absent for a custom folder.
A registration without retained successful evidence, with a missing
checkpoint, or from a different config/parent/backend is recomputed. The full
runner also recomputes Phase 1 if its model/hash pair cannot be validated; the
Phase-2-only runner reports that a compatible Phase 1 is required.

These checks validate recorded artifacts and configuration, not unchanged input
data or identical hardware. Keep the dataset and upstream environment associated
with an experiment when resuming it.

## Regression checks

After installing project dependencies and cloning upstreams with
`bash scripts/setup_upstreams.sh`, run:

```bash
PYTHONPATH=src:. python -m unittest discover -s tests -v
```

The suite covers checkpoint lineage/config/audit rejection and valid resume,
real IDIR SIREN scalar and parameter/coordinate gradients in float32/float64,
gradient flow through frozen generation modules, and restoration of the installed
hooks after success or failure. Missing upstream clones are explicitly reported
as skipped integration coverage; they are required for a complete test run.
