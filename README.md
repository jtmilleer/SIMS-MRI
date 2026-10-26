# Single-Subject Multi-View MRI Super-Resolution via Implicit Neural Representations

This repository contains code for the SIMS-MRI pipeline described in the paper
[_Single-Subject Multi-View MRI Super-Resolution via Implicit Neural
Representations_](https://arxiv.org/abs/2603.22627).

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

The checked-in example configs are under `configs/` (e.g. `configs/oasis_ngp_rigid.yaml`). The default configs expect data under `./oasis_random`, cache processed tensors under `preprocessed_data/`, and write experiment outputs under `runs/` plus optional MLflow runs under `mlruns/`.

## Setup

### Prerequisites

- Python 3.10
- NVIDIA GPU with a CUDA 12.1-compatible driver recommended
- `uv`, `conda`, or the built-in `venv` module

The training scripts prefer `cuda`, then `mps`, then `cpu`.

### With uv

We intentionally exclude the PyTorch stack from `requirements.txt` so you can install the CUDA build that matches your setup first.

```bash
uv venv --python 3.10
source .venv/bin/activate

# Install the CUDA-enabled PyTorch build first.
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# Install the package in editable mode (also installs remaining dependencies).
uv pip install -e .
```

### With conda

```bash
conda env create -f environment.yml
conda activate sims-mri
python -c "import torch; print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available()})"
```

### With venv

If you don't have `uv` or `conda`, the standard library `venv` works the same way.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the CUDA-enabled PyTorch build first.
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# Install the package in editable mode (also installs remaining dependencies).
pip install -e .
```

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

- `participant_id` — subject identifier used for output directory naming
- `selected_scan` — path to the NIfTI file relative to the dataset root

### 2. Run the preprocessing script

```bash
uv run python scripts/preprocessing_brain.py \
  --dataset-root OASIS-3-BIDS-selected \
  --output-root oasis_random
```

For each subject this produces axial/coronal low-resolution views, a zero-padded 256^3 volume, and translated/rotated/rigid transform variants under the output root.

### 3. Point the config at your preprocessed data

The `oasis_ngp_rigid.yaml` config expects the preprocessing output:

```yaml
SETTINGS:
  DIRECTORY: "./oasis_random"
DATASET:
  SUBJECT_ID: "sub-OAS30433_ses-M066_run-02_T1w"
```

Update `SUBJECT_ID` to match the subject you want to train on. The loader combines `DIRECTORY`, `SUBJECT_ID`, and the `LR_CONTRAST` suffixes to locate input files.

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
uv run python -m sims_mri.training.phase1_generation --help
```

## Batch Runs on BraTS

`scripts/run_brats_t1.py` runs preprocessing and Phases 1-3 over the BraTS
2020 T1 scans, one subject at a time, passing checkpoints between phases
automatically.

### 1. Get the data

Download the BraTS 2020 training data and keep the stock layout:

```text
MICCAI_BraTS2020_TrainingData/
  BraTS20_Training_001/
    BraTS20_Training_001_t1.nii
    ...
  BraTS20_Training_002/
  ...
```

Subjects without a `<subject>_t1.nii` (or `.nii.gz`) are skipped. Each scan is
zero-padded to a multiple of 4 per axis and run through the same preprocessing
as OASIS (`scripts/preprocessing_brain.py`).

### 2. Run the script

Run from the repo root with the environment from [Setup](#setup) active:

```bash
python scripts/run_brats_t1.py   --dataset-root /path/to/MICCAI_BraTS2020_TrainingData   --output-root brats_t1
```

To try it on a couple of subjects first, add `--limit 2` or
`--subjects BraTS20_Training_001 BraTS20_Training_002`.

| Option | Default | Meaning |
|---|---|---|
| `--dataset-root` | (required) | Folder containing the `BraTS20_Training_XXX/` folders |
| `--output-root` | `brats_t1` | Where preprocessed data, configs and logs go |
| `--config` | `configs/oasis_ngp_rigid.yaml` | Config copied per subject (`DIRECTORY`, `SUBJECT_ID`, `PROJECT_NAME` are filled in) |
| `--project-name` | `brats_t1_rigid` | `PROJECT_NAME` written to each subject config |
| `--modality` | `t1` | BraTS scan suffix to use |
| `--subjects` | all | Only these BraTS IDs |
| `--limit` | all | Only the first N subjects |
| `--preprocess-only` | off | Stop after preprocessing |
| `--skip-preprocess` | off | Reuse data that is already preprocessed |
| `--logging` | off | Pass `--logging` (MLflow) to each phase |

Subjects that are already preprocessed are not preprocessed again, but the
phases always rerun. If a subject fails, the error is recorded and the script
moves on to the next one.

### 3. Find the outputs

```text
brats_t1/
  BraTS20_Training_001_t1/
    BraTS20_Training_001_t1_*_LR.nii.gz   # preprocessed views
    config.yaml                           # per-subject config
  logs/
    BraTS20_Training_001_t1_phase{1,2,3}.log
  pipeline_summary.csv   # one row per subject: status, checkpoints, outputs, error
runs/                    # training checkpoints and reconstructions
```

`pipeline_summary.csv` lists each subject's status, Phase 1 checkpoint,
Phase 2 checkpoint, Phase 3 output directory under `runs/`, and the error
message if it failed. Rows are appended, so rerunning adds new rows rather than
replacing old ones.

### Running with Snakemake

`workflow/Snakefile` runs the same steps as Snakemake jobs. Preprocessing runs
in parallel on the CPU, and the phases are pipelined on the GPU: one Phase 1,
one Phase 2 and one Phase 3 job can run at the same time, each on a different
subject. Finished steps are not rerun, so an interrupted run picks up where it
left off.

Snakemake is not installed with the package, so install it into the same
environment first. Snakemake 8+ needs Python 3.11, so on the Python 3.10
environment use the last 7.x release (newer `pulp` versions break it):

```bash
pip install "snakemake==7.32.4" "pulp<2.8"
```

Then run from the repo root:

```bash
snakemake --config dataset_root=/path/to/MICCAI_BraTS2020_TrainingData
```

Snakemake finds `workflow/Snakefile` on its own and loads the default options
from `workflow/profiles/default/config.yaml`: 8 cores, `--keep-going`, and one
slot per phase (`phase1_slot`, `phase2_slot`, `phase3_slot`). Edit that file, or
override on the command line (e.g. `--cores 16`), to change them. A
`--resources` flag replaces the profile's whole list, so pass all three slots,
e.g. `--resources phase1_slot=2 phase2_slot=1 phase3_slot=1` to allow two
Phase 1 jobs at once (if the GPU has the memory for it).

Add `--dry-run` (`-n`) to list the jobs without running them. Settings live in
`workflow/config.yaml`, and you can override any of them with `--config`, e.g.
`--config dataset_root=... limit=2 timing=true`:

| Key | Default | Meaning |
|---|---|---|
| `dataset_root` | (required) | Folder containing the `BraTS20_Training_XXX/` folders |
| `output_root` | `brats_t1` | Where preprocessed data, configs, logs and step markers go |
| `base_config` | `configs/oasis_ngp_rigid.yaml` | Config copied per subject |
| `project_name` | `brats_t1_rigid` | `PROJECT_NAME` written to each subject config |
| `modality` | `t1` | BraTS scan suffix to use |
| `subjects` | all | Only these BraTS IDs (edit as a list in `config.yaml`) |
| `limit` | all | Only the first N subjects |
| `logging` | `false` | Pass `--logging` (MLflow) to each phase |
| `timing` | `false` | Report total run time and subject count at the end |

Each phase writes a JSON marker (`phase{1,2,3}.json`) to
`<output_root>/<subject>/` that records its checkpoint or output paths, and the
next phase reads it. Logs go to `<output_root>/logs/`. When every subject
finishes, `<output_root>/summary.csv` lists the checkpoints and Phase 3 output
directory for each subject. With `--keep-going`, a failed subject doesn't stop
the others, but `summary.csv` is only written once every subject succeeds.

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
