"""Preprocess BraTS 2020 T1 scans and run all three SIMS-MRI phases per subject.

Expects the stock BraTS layout:

    MICCAI_BraTS2020_TrainingData/
      BraTS20_Training_001/
        BraTS20_Training_001_t1.nii
        ...

Each subject's T1 is converted to .nii.gz and passed through the same
preprocessing as the OASIS pipeline (scripts/preprocessing_brain.py), giving

    <output-root>/BraTS20_Training_001_t1/BraTS20_Training_001_t1_axial_LR.nii.gz
    <output-root>/BraTS20_Training_001_t1/BraTS20_Training_001_t1_rigid_coronal_LR.nii.gz
    ...

A per-subject config is written next to the data, then Phase 1 -> 2 -> 3 run
in sequence with checkpoints chained automatically. Results are appended to
<output-root>/pipeline_summary.csv.
"""

import argparse
import copy
import csv
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from preprocessing_brain import preprocess_oasis_subject

REPO_ROOT = Path(__file__).resolve().parent.parent


def find_subjects(dataset_root, modality):
    subjects = []
    for subject_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        candidates = [
            subject_dir / f"{subject_dir.name}_{modality}.nii",
            subject_dir / f"{subject_dir.name}_{modality}.nii.gz",
        ]
        scan = next((c for c in candidates if c.exists()), None)
        if scan is None:
            print(f"[!] Skipping {subject_dir.name}: no {modality} scan found")
            continue
        subjects.append((f"{subject_dir.name}_{modality}", scan))
    return subjects


def preprocess_subject(subject_id, scan, output_root):
    subject_dir = output_root / subject_id
    # rigid_coronal_LR is the last LR view the Phase 1-3 config needs
    if (subject_dir / f"{subject_id}_rigid_coronal_LR.nii.gz").exists():
        print(f"[*] {subject_id} already preprocessed, skipping")
        return

    # preprocess_oasis_subject derives output names via .replace(".nii.gz", ...),
    # so stage the raw .nii as .nii.gz first
    staging_dir = output_root / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = staging_dir / f"{subject_id}.nii.gz"
    img = nib.load(str(scan))
    # BraTS is 240x240x155; zero-pad each axis up to a multiple of 4 so the
    # 4x LR views upsample back to matching shapes (155 -> 156)
    data = np.asanyarray(img.dataobj)
    pad = [(0, -dim % 4) for dim in data.shape]
    nib.save(nib.Nifti1Image(np.pad(data, pad), img.affine, img.header), str(staged))
    try:
        preprocess_oasis_subject(str(staged), subject_id=subject_id, output_root=output_root)
    finally:
        staged.unlink(missing_ok=True)


def write_subject_config(base_config, subject_id, output_root, project_name):
    config = copy.deepcopy(base_config)
    config["SETTINGS"]["DIRECTORY"] = str((output_root / subject_id).resolve())
    config["SETTINGS"]["PROJECT_NAME"] = project_name
    config["DATASET"]["SUBJECT_ID"] = subject_id
    config_path = output_root / subject_id / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return config_path


def run_phase(module, args, log_path):
    cmd = [sys.executable, "-m", f"sims_mri.training.{module}", *args]
    print(f"[>] {' '.join(cmd)}")
    lines = []
    with open(log_path, "w") as log, subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            lines.append(line.rstrip("\n"))
    if proc.returncode != 0:
        raise RuntimeError(f"{module} exited with code {proc.returncode} (log: {log_path})")
    return lines


def parse_field(lines, label):
    pattern = re.compile(rf"^{re.escape(label)}:\s*(.+)$")
    for line in reversed(lines):
        match = pattern.match(line.strip())
        if match:
            return match.group(1).strip()
    raise RuntimeError(f"Could not find '{label}' in phase output")


def run_pipeline(subject_id, config_path, log_dir, extra_args):
    log_dir.mkdir(parents=True, exist_ok=True)
    config_arg = ["--config", str(config_path.resolve())]

    # Phase 1
    lines = run_phase("phase1_generation", config_arg + extra_args, log_dir / f"{subject_id}_phase1.log")
    phase1_dir = parse_field(lines, "Output directory")
    phase1_ckpt = os.path.join(parse_field(lines, "Weights directory"), parse_field(lines, "Saved checkpoint"))

    # Phase 2, co-located with Phase 1 so its checkpoint is easy to find
    run_phase(
        "phase2_registration",
        config_arg + ["--phase1_checkpoint", phase1_ckpt, "--output_dir", phase1_dir] + extra_args,
        log_dir / f"{subject_id}_phase2.log",
    )
    rotation_ckpts = sorted(glob.glob(str(REPO_ROOT / phase1_dir / "*_phase2_*" / "*_model_rotation.pt")))
    if not rotation_ckpts:
        raise RuntimeError(f"No Phase 2 rotation checkpoint found under {phase1_dir}")
    phase2_ckpt = rotation_ckpts[-1]

    # Phase 3
    lines = run_phase(
        "phase3_mixed_training",
        config_arg
        + ["--phase1_checkpoint", phase1_ckpt, "--phase2_checkpoint", phase2_ckpt, "--output_dir", phase1_dir]
        + extra_args,
        log_dir / f"{subject_id}_phase3.log",
    )
    phase3_dir = parse_field(lines, "Output directory")

    return {
        "subject_id": subject_id,
        "phase1_checkpoint": phase1_ckpt,
        "phase2_checkpoint": phase2_ckpt,
        "phase3_output": phase3_dir,
    }


def append_summary(summary_path, row):
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["subject_id", "status", "phase1_checkpoint", "phase2_checkpoint", "phase3_output", "error"]
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Directory containing BraTS20_Training_XXX/ subject folders")
    parser.add_argument("--output-root", type=Path, default=Path("brats_t1"),
                        help="Output directory for preprocessed subjects (default: brats_t1)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "oasis_ngp_rigid.yaml",
                        help="Base config; DIRECTORY/SUBJECT_ID/PROJECT_NAME are overridden per subject")
    parser.add_argument("--project-name", default="brats_t1_rigid")
    parser.add_argument("--modality", default="t1", help="BraTS modality suffix (default: t1)")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Only process these BraTS IDs, e.g. BraTS20_Training_001")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N subjects")
    parser.add_argument("--preprocess-only", action="store_true", help="Stop after preprocessing")
    parser.add_argument("--skip-preprocess", action="store_true", help="Assume data is already preprocessed")
    parser.add_argument("--logging", action="store_true", help="Enable MLflow logging in each phase")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects(args.dataset_root, args.modality)
    if args.subjects:
        wanted = set(args.subjects)
        subjects = [s for s in subjects if s[0].rsplit("_", 1)[0] in wanted]
    if args.limit is not None:
        subjects = subjects[: args.limit]
    if not subjects:
        raise RuntimeError("No subjects to process")
    print(f"[*] {len(subjects)} subject(s) selected")

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    extra_args = ["--logging"] if args.logging else []
    summary_path = output_root / "pipeline_summary.csv"
    log_dir = output_root / "logs"

    for subject_id, scan in subjects:
        print("=" * 60)
        print(f"[*] Subject {subject_id}")
        print("=" * 60)
        try:
            if not args.skip_preprocess:
                preprocess_subject(subject_id, scan, output_root)
            if args.preprocess_only:
                continue
            config_path = write_subject_config(base_config, subject_id, output_root, args.project_name)
            result = run_pipeline(subject_id, config_path, log_dir, extra_args)
            append_summary(summary_path, {**result, "status": "ok", "error": ""})
        except Exception as exc:
            print(f"[!] {subject_id} failed: {exc}")
            append_summary(summary_path, {"subject_id": subject_id, "status": "failed", "error": str(exc)})

    print(f"[*] Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
