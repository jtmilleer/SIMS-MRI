#!/usr/bin/env python
"""Batch training orchestrator for SIMS-MRI (BraTS 2023 GLI).

Runs Phase 1 (generation) -> Phase 2 (registration) -> Phase 3 (mixed training)
for every subject found in BraTS TSV split files, using config file defaults for
all hyperparameters.

Checkpoint paths are automatically threaded between phases by parsing stdout
from each phase's CLI entry point.

Examples:
  # Dry run all subjects from all splits
  python scripts/run_batch_training_brats.py --config configs/brats_ngp_rigid.yaml --dry-run

  # Run specific splits with logging
  python scripts/run_batch_training_brats.py --config configs/brats_ngp_rigid.yaml --parts 01 02 --logging

  # Run a single explicit subject
  python scripts/run_batch_training_brats.py --config configs/brats_ngp_rigid.yaml \
      --subjects YOUR_BRATS_SUBJECT_ID --logging

  # Resume a previous batch (skips phases with existing checkpoints)
  python scripts/run_batch_training_brats.py --config configs/brats_ngp_rigid.yaml --resume --logging
"""

from __future__ import annotations

import argparse
import glob as glob_module
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matches TSV filenames: brats_part01.tsv, brats_part02.tsv, etc.
TSV_PART_RE = re.compile(
    r"brats_part(\d+[a-z]?)\.tsv$", re.IGNORECASE
)

# Phase 1 prints: "      --phase1_checkpoint runs/<id>/<subdir>/<name>_model.pt \"
PHASE1_CKPT_RE = re.compile(r"--phase1_checkpoint\s+(runs/\S+_model\.pt)")

# Phase 2 prints: "Weights directory: runs/<id>/<subdir>"
PHASE2_WEIGHTS_DIR_RE = re.compile(r"Weights directory:\s+(runs/\S+)")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    phase: str  # "phase1", "phase2", "phase3"
    subject_id: str
    status: str = "pending"  # "success", "failed", "skipped", "dry-run"
    duration_seconds: float = 0.0
    command: list[str] = field(default_factory=list)
    checkpoint_path: str | None = None
    error_message: str | None = None
    log_file: str | None = None


# ---------------------------------------------------------------------------
# TSV handling
# ---------------------------------------------------------------------------


def discover_tsv_files(
    tsv_dir: str, parts: list[str] | None = None
) -> list[Path]:
    """Find and optionally filter TSV split files in *tsv_dir*."""
    tsv_dir_path = Path(tsv_dir)
    if not tsv_dir_path.is_dir():
        raise FileNotFoundError(f"TSV directory not found: {tsv_dir}")

    all_tsvs = sorted(
        tsv_dir_path.glob("brats_part*.tsv")
    )

    if parts is None:
        return all_tsvs

    # Normalize requested parts: "1" -> "01", "4a" -> "04a"
    normalized = set()
    for p in parts:
        m = re.match(r"^0*(\d+)([a-z]?)$", p)
        if m:
            normalized.add(f"{int(m.group(1)):02d}{m.group(2)}")
        else:
            normalized.add(p)

    filtered = []
    for tsv_path in all_tsvs:
        m = TSV_PART_RE.search(tsv_path.name)
        if m:
            # Normalize the discovered label the same way
            raw_label = m.group(1)
            lm = re.match(r"^0*(\d+)([a-z]?)$", raw_label)
            if lm:
                norm_label = f"{int(lm.group(1)):02d}{lm.group(2)}"
            else:
                norm_label = raw_label
            if norm_label in normalized:
                filtered.append(tsv_path)

    return filtered


def extract_subjects_from_tsvs(tsv_files: list[Path]) -> list[str]:
    """Read subject IDs from BraTS TSV split files (deduplicated, order-preserving).

    Each TSV has a single ``subject_id`` column with BraTS subject IDs.
    """
    seen: set[str] = set()
    subjects: list[str] = []

    for tsv_path in tsv_files:
        df = pd.read_csv(tsv_path, sep="\t")
        if "subject_id" not in df.columns:
            raise ValueError(
                f"TSV file {tsv_path} missing required 'subject_id' column"
            )
        for subject_id in df["subject_id"]:
            subject_id = str(subject_id).strip()
            if subject_id not in seen:
                seen.add(subject_id)
                subjects.append(subject_id)

    return subjects


# ---------------------------------------------------------------------------
# Checkpoint discovery (from phase stdout)
# ---------------------------------------------------------------------------


def extract_phase1_checkpoint(stdout: str) -> str | None:
    """Parse Phase 1 stdout for the saved model checkpoint path."""
    match = PHASE1_CKPT_RE.search(stdout)
    return match.group(1) if match else None


def extract_phase2_checkpoint(stdout: str) -> str | None:
    """Parse Phase 2 stdout for the rotation checkpoint.

    Extracts the weights directory, then globs for ``*_model_rotation.pt``.
    """
    match = PHASE2_WEIGHTS_DIR_RE.search(stdout)
    if not match:
        return None

    weights_dir = match.group(1)
    rotation_files = sorted(
        glob_module.glob(os.path.join(weights_dir, "*_model_rotation.pt")),
        key=os.path.getmtime,
    )
    return rotation_files[-1] if rotation_files else None


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


def find_existing_phase1_checkpoint(subject_id: str) -> str | None:
    """Search ``runs/`` for an existing Phase 1 model checkpoint."""
    pattern = f"runs/**/{subject_id}_*_model.pt"
    candidates = [
        c
        for c in glob_module.glob(pattern, recursive=True)
        if not c.endswith("_model_rotation.pt")
    ]
    return max(candidates, key=os.path.getmtime) if candidates else None


def find_existing_phase2_checkpoint(subject_id: str) -> str | None:
    """Search ``runs/`` for an existing Phase 2 rotation checkpoint."""
    pattern = f"runs/**/{subject_id}_*_model_rotation.pt"
    candidates = glob_module.glob(pattern, recursive=True)
    return max(candidates, key=os.path.getmtime) if candidates else None


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------


def run_phase(
    cmd: list[str],
    phase_name: str,
    subject_id: str,
    log_dir: str,
    dry_run: bool = False,
) -> tuple[PhaseResult, str]:
    """Execute a training phase as a subprocess.

    Returns ``(PhaseResult, stdout_text)``.
    """
    result = PhaseResult(
        phase=phase_name,
        subject_id=subject_id,
        command=cmd,
    )

    if dry_run:
        result.status = "dry-run"
        print(f"  [DRY-RUN] {' '.join(cmd)}")
        return result, ""

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{subject_id}_{phase_name}.log")
    result.log_file = log_file

    print(f"  Running {phase_name} for {subject_id} ...")
    start = time.time()

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )
        result.duration_seconds = time.time() - start

        with open(log_file, "w") as f:
            f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
            f.write(f"=== RETURN CODE === {proc.returncode}\n\n")
            f.write(f"=== STDOUT ===\n{proc.stdout}\n\n")
            f.write(f"=== STDERR ===\n{proc.stderr}\n")

        if proc.returncode == 0:
            result.status = "success"
            hrs = result.duration_seconds / 3600
            print(
                f"  Completed {phase_name} for {subject_id} "
                f"({result.duration_seconds:.1f}s / {hrs:.2f}h)"
            )
        else:
            result.status = "failed"
            stderr_lines = proc.stderr.strip().splitlines()
            result.error_message = "\n".join(stderr_lines[-5:])
            print(
                f"  FAILED {phase_name} for {subject_id} "
                f"(exit code {proc.returncode})"
            )
            if result.log_file:
                print(f"    Log: {result.log_file}")

        return result, proc.stdout

    except Exception as exc:
        result.duration_seconds = time.time() - start
        result.status = "failed"
        result.error_message = str(exc)
        return result, ""


# ---------------------------------------------------------------------------
# Per-subject orchestration
# ---------------------------------------------------------------------------


def train_subject(
    subject_id: str,
    config_path: str,
    logging: bool,
    dry_run: bool,
    resume: bool,
    log_dir: str,
) -> list[PhaseResult]:
    """Run Phase 1 -> Phase 2 -> Phase 3 for one subject."""
    results: list[PhaseResult] = []

    # ---- Phase 1 ----
    phase1_ckpt: str | None = None

    if resume:
        phase1_ckpt = find_existing_phase1_checkpoint(subject_id)
        if phase1_ckpt:
            print(f"  [RESUME] Phase 1 checkpoint: {phase1_ckpt}")
            results.append(
                PhaseResult(
                    phase="phase1",
                    subject_id=subject_id,
                    status="skipped",
                    checkpoint_path=phase1_ckpt,
                )
            )

    if phase1_ckpt is None:
        cmd = [
            sys.executable,
            "-m",
            "sims_mri.training.phase1_generation",
            "--config",
            config_path,
            "--subject_id",
            subject_id,
        ]
        if logging:
            cmd.append("--logging")

        p1_result, p1_stdout = run_phase(
            cmd, "phase1", subject_id, log_dir, dry_run
        )

        if not dry_run and p1_result.status == "success":
            phase1_ckpt = extract_phase1_checkpoint(p1_stdout)
            p1_result.checkpoint_path = phase1_ckpt
            if phase1_ckpt is None:
                p1_result.status = "failed"
                p1_result.error_message = (
                    "Phase 1 succeeded but checkpoint path could not be "
                    "parsed from stdout"
                )

        results.append(p1_result)
        if p1_result.status not in ("success", "dry-run"):
            return results

    # ---- Phase 2 ----
    phase2_ckpt: str | None = None

    if resume:
        phase2_ckpt = find_existing_phase2_checkpoint(subject_id)
        if phase2_ckpt:
            print(f"  [RESUME] Phase 2 checkpoint: {phase2_ckpt}")
            results.append(
                PhaseResult(
                    phase="phase2",
                    subject_id=subject_id,
                    status="skipped",
                    checkpoint_path=phase2_ckpt,
                )
            )

    if phase2_ckpt is None:
        cmd = [
            sys.executable,
            "-m",
            "sims_mri.training.phase2_registration",
            "--config",
            config_path,
            "--subject_id",
            subject_id,
            "--phase1_checkpoint",
            phase1_ckpt or "UNKNOWN",
        ]
        if logging:
            cmd.append("--logging")

        p2_result, p2_stdout = run_phase(
            cmd, "phase2", subject_id, log_dir, dry_run
        )

        if not dry_run and p2_result.status == "success":
            phase2_ckpt = extract_phase2_checkpoint(p2_stdout)
            p2_result.checkpoint_path = phase2_ckpt
            if phase2_ckpt is None:
                p2_result.status = "failed"
                p2_result.error_message = (
                    "Phase 2 succeeded but rotation checkpoint could not "
                    "be discovered"
                )

        results.append(p2_result)
        if p2_result.status not in ("success", "dry-run"):
            return results

    # ---- Phase 3 ----
    cmd = [
        sys.executable,
        "-m",
        "sims_mri.training.phase3_mixed_training",
        "--config",
        config_path,
        "--subject_id",
        subject_id,
        "--phase1_checkpoint",
        phase1_ckpt or "UNKNOWN",
        "--phase2_checkpoint",
        phase2_ckpt or "UNKNOWN",
    ]
    if logging:
        cmd.append("--logging")

    p3_result, _ = run_phase(cmd, "phase3", subject_id, log_dir, dry_run)
    results.append(p3_result)

    return results


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------


def write_results_json(
    path: str,
    all_results: dict[str, list[PhaseResult]],
    config_path: str,
    batch_duration: float,
    total_subjects: int,
) -> None:
    """Write batch results summary as JSON."""
    subjects_data = {}
    for subject_id, phase_results in all_results.items():
        subjects_data[subject_id] = {
            "phases": [
                {
                    "phase": r.phase,
                    "status": r.status,
                    "duration_seconds": round(r.duration_seconds, 2),
                    "checkpoint_path": r.checkpoint_path,
                    "error_message": r.error_message,
                    "log_file": r.log_file,
                    "command": r.command,
                }
                for r in phase_results
            ],
            "all_succeeded": all(
                r.status in ("success", "skipped", "dry-run")
                for r in phase_results
            ),
        }

    output = {
        "batch_metadata": {
            "config": config_path,
            "timestamp": datetime.now().isoformat(),
            "total_subjects": total_subjects,
            "subjects_attempted": len(all_results),
            "subjects_succeeded": sum(
                1 for v in subjects_data.values() if v["all_succeeded"]
            ),
            "subjects_failed": sum(
                1 for v in subjects_data.values() if not v["all_succeeded"]
            ),
            "total_duration_seconds": round(batch_duration, 2),
        },
        "subjects": subjects_data,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def verify_working_directory() -> None:
    """Ensure we are running from the repo root."""
    if not Path("configs").is_dir() or not Path("src/sims_mri").is_dir():
        print(
            "ERROR: This script must be run from the SIMS-MRI repository root.\n"
            "  Expected to find: configs/ and src/sims_mri/\n"
            f"  Current directory: {Path.cwd()}"
        )
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch training orchestrator for SIMS-MRI: "
        "runs Phase 1 -> Phase 2 -> Phase 3 per subject.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file (e.g., configs/oasis_ngp_rigid.yaml).",
    )
    parser.add_argument(
        "--tsv_dir",
        type=str,
        default="brats_random/splits",
        help="Directory containing brats_partXX.tsv files.",
    )
    parser.add_argument(
        "--parts",
        type=str,
        nargs="+",
        default=None,
        help="TSV part identifiers to include (e.g., --parts 01 02 04a). "
        "Default: all detected parts.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help="Explicit subject IDs (overrides --tsv_dir / --parts).",
    )

    parser.add_argument(
        "--logging",
        action="store_true",
        help="Pass --logging to each phase entry point.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the entire batch on the first subject failure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip phases whose checkpoints already exist in runs/.",
    )

    parser.add_argument(
        "--results_json",
        type=str,
        default=None,
        help="Path for results JSON. Default: runs/batch_results_<timestamp>.json",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help="Directory for per-subject per-phase logs. Default: runs/batch_logs_<timestamp>",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    verify_working_directory()

    if not Path(args.config).is_file():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)

    # ---- Resolve subjects ----
    if args.subjects:
        subjects = args.subjects
        print(f"Using {len(subjects)} explicitly specified subject(s)")
    else:
        tsv_files = discover_tsv_files(args.tsv_dir, args.parts)
        if not tsv_files:
            available = sorted(
                Path(args.tsv_dir).glob("brats_part*.tsv")
            )
            print(
                f"ERROR: No TSV files found in {args.tsv_dir}"
                + (f" for parts {args.parts}" if args.parts else "")
            )
            if available:
                labels = [
                    TSV_PART_RE.search(p.name).group(1)
                    for p in available
                    if TSV_PART_RE.search(p.name)
                ]
                print(f"  Available parts: {labels}")
            sys.exit(1)

        print(f"Found {len(tsv_files)} TSV file(s):")
        for f in tsv_files:
            print(f"  {f}")

        subjects = extract_subjects_from_tsvs(tsv_files)
        if not subjects:
            print("ERROR: No subjects found in TSV files.")
            sys.exit(1)

    # ---- Setup output paths ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path("runs").mkdir(exist_ok=True)

    results_json = args.results_json or f"runs/batch_results_{timestamp}.json"
    log_dir = args.log_dir or f"runs/batch_logs_{timestamp}"

    # ---- Banner ----
    print("\n" + "=" * 60)
    print("SIMS-MRI BATCH TRAINING (BraTS)")
    print("=" * 60)
    print(f"Subjects:     {len(subjects)}")
    print(f"Config:       {args.config}")
    print(f"Results JSON: {results_json}")
    print(f"Log dir:      {log_dir}")
    print(f"Dry run:      {args.dry_run}")
    print(f"Resume:       {args.resume}")
    print(f"Fail fast:    {args.fail_fast}")
    print(f"Logging:      {args.logging}")
    print("=" * 60)

    # ---- Main loop ----
    all_results: dict[str, list[PhaseResult]] = {}
    batch_start = time.time()
    failed_subjects: list[str] = []

    for i, subject_id in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] Subject: {subject_id}")
        print("-" * 40)

        subject_results = train_subject(
            subject_id=subject_id,
            config_path=args.config,
            logging=args.logging,
            dry_run=args.dry_run,
            resume=args.resume,
            log_dir=log_dir,
        )
        all_results[subject_id] = subject_results

        if any(r.status == "failed" for r in subject_results):
            failed_subjects.append(subject_id)
            if args.fail_fast:
                print(
                    f"\n--fail-fast: aborting batch after failure "
                    f"on {subject_id}"
                )
                break

    batch_duration = time.time() - batch_start

    # ---- Write results ----
    if not args.dry_run:
        write_results_json(
            results_json,
            all_results,
            config_path=args.config,
            batch_duration=batch_duration,
            total_subjects=len(subjects),
        )

    # ---- Summary ----
    n_ok = sum(
        1
        for subj_results in all_results.values()
        if all(
            r.status in ("success", "skipped", "dry-run")
            for r in subj_results
        )
    )

    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(
        f"Total time:        {batch_duration:.1f}s "
        f"({batch_duration / 3600:.2f}h)"
    )
    print(f"Subjects OK:       {n_ok}/{len(all_results)}")
    if failed_subjects:
        print(f"Subjects failed:   {', '.join(failed_subjects)}")
    if not args.dry_run:
        print(f"Results JSON:      {results_json}")
    print("=" * 60)

    if failed_subjects:
        sys.exit(1)


if __name__ == "__main__":
    main()
