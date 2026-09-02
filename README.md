# Single-Subject Multi-View MRI Super-Resolution via Implicit Neural Representations

This repository contains code for the SIMS-MRI pipeline described in the paper
[Single-Subject Multi-View MRI Super-Resolution via Implicit Neural Representations](https://arxiv.org/abs/2603.22627).

## Fast-exact branch

This branch adds opt-in Phase 2 acceleration while retaining the original
three-phase method and default training entrypoints. It also includes batch
runners, validated completed-phase resume, sagittal-view pair configs, BraTS
data workflows, and nested/flat input layout support.

After setup, run the complete fast pipeline on your own preprocessed subject:

```bash
python scripts/run_batch_training_fast_exact.py \
  --config configs/oasis_ngp_rigid.yaml --subjects YOUR_SUBJECT_ID --logging
```

For an existing Phase 1 checkpoint, use `sims-mri-train-phase2-fast-exact`
with the same arguments as `sims-mri-train-phase2`. Exact refers to the bending
loss and preserved gradients, not bitwise-identical final reconstructions.
See [optimization and validation notes](docs/optimizations/phase2_fast_exact.md)
and the [short differences brief](docs/fast-exact-brief.md).

This is a research-code snapshot. Read the [known workflow limits](docs/known-limitations.md)
before installing or resuming experiments; the fast runners have stronger resume
checks than the ordinary batch runners.

Only source, example configuration, tests, and documentation are included. Obtain
datasets separately under their access terms; no scans, checkpoints, subject
lists, or run logs are distributed here. Replace example subject IDs with your
own selected scan. Auxiliary ECLARE/SMORE scripts require their separate method
implementations; those implementations are not bundled in this repository.

## Method Overview

![Overview of the three-phase SIMS-MRI framework](assets/overview.png)

We propose a three-stage single-subject pipeline:

1. Phase 1 fits an implicit image model to one anisotropic input view.
2. Phase 2 learns inter-view alignment.
3. Phase 3 refines the final reconstruction using both image and registration
   checkpoints.

Phase 1 and Phase 3 use FreeNeRF-style progressive hash unlock by default.

![Qualitative comparison of unseen-view brain MRI reconstruction across methods](assets/result-brain.png)

*Qualitative comparison of unseen-view brain MRI reconstruction across methods.*

## Implementation Overview

Training is split into three phases:

- `sims-mri-train-phase1` for image generation
- `sims-mri-train-phase2` for registration
- `sims-mri-train-phase3` for mixed training

The checked-in example configs are under `configs/` (e.g. `configs/oasis_ngp_rigid.yaml`, `configs/oasis_ngp_rigid_axial_sagittal.yaml`, and `configs/oasis_ngp_rigid_coronal_sagittal.yaml`). The default configs expect data under `./oasis_random`, cache processed tensors under `preprocessed_data/`, and write experiment outputs under `runs/` plus optional MLflow runs under `mlruns/`.

## Setup

### Prerequisites

- Python 3.10
- NVIDIA GPU with a driver compatible with the PyTorch CUDA wheel you install
- `uv` or `conda`

The training scripts prefer `cuda`, then `mps`, then `cpu`.

### With uv

This branch requires PyTorch 2.6.x and torchvision 0.21.x. Its `uv` project
configuration selects the CUDA 12.4 wheel index. The following setup therefore
targets Linux/Windows with a compatible NVIDIA driver; it is not a universal
CPU/macOS installation recipe. The versions and wheel index follow the
[official PyTorch 2.6 installation options](https://pytorch.org/get-started/previous-versions/#v260).

```bash
uv venv --python 3.10
source .venv/bin/activate

# Install the matching CUDA 12.4 pair.
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0

# Install the package in editable mode (also installs remaining dependencies).
uv pip install -e .
```

For a separately managed CPU or macOS/MPS PyTorch 2.6 environment, use
`uv pip install --no-sources -e .` to bypass this project's CUDA-only source
mapping. See [uv dependency-source controls](https://docs.astral.sh/uv/concepts/projects/dependencies/#disabling-sources).
Use the activated environment's `python` commands below; `uv run`/`uv sync`
without overrides can reapply the CUDA-only mapping. Fresh CPU/MPS environment
installation has not been validated by this branch's publication checks.

After installing PyTorch, verify that CUDA is actually usable in the same
environment you will train in:

```bash
python -c "import torch; print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'device_count': torch.cuda.device_count()})"
```

If `cuda_available` is `False` and PyTorch warns that the NVIDIA driver is too
old, reinstall a PyTorch wheel that matches the driver on that machine.

### Existing conda environments

The inherited `environment.yml` targets the original CUDA 12.1 stack and is kept
for reference; it is not a validated installer for this branch's PyTorch 2.6
requirements. Prefer the wheel-based flow above, or provide a separately managed
compatible environment.

### Clone upstream dependencies

```bash
bash scripts/setup_upstreams.sh
```

This clones [IDIR](https://github.com/MIAGroupUT/IDIR) and [multi_contrast_inr](https://github.com/jqmcginnis/multi_contrast_inr) into the project root.

## OASIS-3 Dataset Setup

The brain experiments use the [OASIS-3](https://www.oasis-brains.org/) dataset. It can be accessed here: https://www.oasis-brains.org/.

### 1. Download and organize the raw data

After obtaining access, download the T1-weighted MRI scans and arrange them in a BIDS-style directory:

```text
OASIS-3-BIDS-selected/
  selected_participants_brain.tsv
  sub-OAS30001/
    ses-M006/
      anat/
        sub-OAS30001_ses-M006_run-01_T1w.nii.gz
  sub-OAS30433/
    ses-M066/
      anat/
        sub-OAS30433_ses-M066_run-02_T1w.nii.gz
  ...
```

The TSV file drives preprocessing. It must have two columns:

| participant_id | selected_scan |
|---|---|
| sub-OAS30001_ses-M006_run-01_T1w | sub-OAS30001/ses-M006/anat/sub-OAS30001_ses-M006_run-01_T1w.nii.gz |
| sub-OAS30433_ses-M066_run-02_T1w | sub-OAS30433/ses-M066/anat/sub-OAS30433_ses-M066_run-02_T1w.nii.gz |

- `participant_id` — identifier kept from the TSV for selection/logging
- `selected_scan` — path to the NIfTI file relative to the dataset root

The preprocessing output is keyed by the selected scan stem, so a scan such as
`sub-OAS30433_ses-M066_run-02_T1w.nii.gz` will be written under
`oasis_random/sub-OAS30433_ses-M066_run-02_T1w/`.

### 2. Run the preprocessing script

```bash
python scripts/preprocessing_brain.py \
  --dataset-root OASIS-3-BIDS-selected \
  --output-root oasis_random
```

For each subject this produces axial/coronal/sagittal low-resolution views, a zero-padded 256^3 volume, and translated/rotated/rigid transform variants under the output root. The preprocessing step also writes transformed coronal and sagittal low-resolution moving-view files such as `*_rigid_coronal_LR.nii.gz` and `*_rigid_sagittal_LR.nii.gz`.

### 3. Point the config at your preprocessed data

The `oasis_ngp_rigid.yaml` config expects the preprocessing output:

```yaml
SETTINGS:
  DIRECTORY: "./oasis_random"
DATASET:
  SUBJECT_ID: "sub-OAS30433_ses-M066_run-02_T1w"
```

Update `SUBJECT_ID` to match the selected scan stem you want to train on. The
loader combines `DIRECTORY`, `SUBJECT_ID`, and the `LR_CONTRAST` suffixes to
locate input files under `DIRECTORY/SUBJECT_ID/`.

For brain experiments, the checked-in rigid pairwise ablation configs are:

- `configs/oasis_ngp_rigid.yaml` for `axial_LR + rigid_coronal_LR`
- `configs/oasis_ngp_rigid_axial_sagittal.yaml` for `axial_LR + rigid_sagittal_LR`
- `configs/oasis_ngp_rigid_coronal_sagittal.yaml` for `coronal_LR + rigid_sagittal_LR`

The current brain pipeline is pairwise and order-sensitive: `LR_CONTRAST1` is the Phase 1 anchor / fixed view, and `LR_CONTRAST2` is the moving view passed through registration. The sagittal configs above therefore use a single ordering only.

## BraTS 2023 GLI Dataset Setup

The BraTS workflow targets the [BraTS 2023 GLI](https://www.synapse.org/Synapse:syn51156910/wiki/622351)
dataset and uses the **T1 contrast-enhanced (T1ce)** sequence. The scripts are
research utilities; their inclusion does not establish tumor-region accuracy or
clinical validity. Set `DATASET.SUBJECT_ID` in a BraTS config to your own
preprocessed subject before running the individual phase commands.

### 1. Preprocess

```bash
python scripts/preprocessing_brats.py \
  --dataset-root brats-data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
  --output-root brats_random \
  --num-subjects 25 \
  --seed 42
```

### 2. Train

The BraTS configs follow the same three-phase workflow as OASIS:

- `configs/brats_ngp_rigid.yaml` for `axial_LR + rigid_coronal_LR`
- `configs/brats_ngp_rigid_axial_sagittal.yaml` for `axial_LR + rigid_sagittal_LR`

## Typical Workflow

Run the three stages in order, passing the previous stage checkpoints forward:

```bash
# Phase 1
sims-mri-train-phase1 --config configs/oasis_ngp_rigid.yaml

# Phase 2
sims-mri-train-phase2 \
  --config configs/oasis_ngp_rigid.yaml \
  --phase1_checkpoint runs/<phase1_id>/<weight_subdir>/<checkpoint>_model.pt

# Phase 3
sims-mri-train-phase3 \
  --config configs/oasis_ngp_rigid.yaml \
  --phase1_checkpoint runs/<phase1_id>/<weight_subdir>/<checkpoint>_model.pt \
  --phase2_checkpoint runs/<phase2_id>/<weight_subdir>/<checkpoint>_model_rotation.pt
```

The exact checkpoint paths depend on your run settings and early-stopping
behavior. Each phase writes its artifacts under `runs/<unique_id>/` with
a descriptive subdirectory (e.g. `hash_L16_F32_cat_<project>_w_es/`).

The default config enables FreeNeRF-style progressive hash unlock for Phase 1 and Phase 3. To disable it, set `PROGRESSIVE_HASH_UNLOCK: False` in your config under `GENERATION_MODEL`.

You can also run the modules directly:

```bash
python -m sims_mri.training.phase1_generation --help
```

For the opt-in exact Phase 2 optimization wrapper, use the same Phase 2
arguments with:

```bash
python -m sims_mri.training.phase2_registration_fast_exact --help
```

Details and validation notes are in
`docs/optimizations/phase2_fast_exact.md`.

## Citation

If you use SIMS-MRI in your work, please cite the following paper:

```bibtex
@misc{kim2026singlesubjectmultiviewmrisuperresolution,
      title={Single-Subject Multi-View MRI Super-Resolution via Implicit Neural Representations},
      author={Heejong Kim and Abhishek Thanki and Roel van Herten and Daniel Margolis and Mert R Sabuncu},
      year={2026},
      eprint={2603.22627},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2603.22627},
}
```
