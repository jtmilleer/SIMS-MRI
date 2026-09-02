#!/usr/bin/env python
"""Batch training orchestrator with fast/exact Phase 2.

Runs Phase 1 (generation) -> fast/exact Phase 2 (registration) -> Phase 3
(mixed training) for every subject found in TSV split files, using config file
defaults for all hyperparameters.

Checkpoint paths are automatically threaded between phases by parsing stdout
from each phase's CLI entry point. Phase 2 uses
``sims_mri.training.phase2_registration_fast_exact`` and requires the wrapper's
``PHASE2_FAST_EXACT`` audit line before Phase 3 is launched.

Examples:
  # Dry run all subjects from all splits
  python scripts/run_batch_training_fast_exact.py \
      --config configs/oasis_ngp_rigid.yaml --dry-run

  # Run specific splits with logging
  python scripts/run_batch_training_fast_exact.py \
      --config configs/oasis_ngp_rigid.yaml --parts 01 02 --logging

  # Run a single explicit subject
  python scripts/run_batch_training_fast_exact.py \
      --config configs/oasis_ngp_rigid.yaml \
      --subjects sub-OAS30433_ses-M066_run-02_T1w --logging

  # Resume compatible checkpoints with completed fast/exact result evidence
  python scripts/run_batch_training_fast_exact.py \
      --config configs/oasis_ngp_rigid.yaml --resume --logging
"""

from __future__ import annotations

import argparse
import csv
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
from typing import Any

from _fast_exact_resume import (
    find_phase1_checkpoint,
    find_registration_resume,
    valid_fast_exact_audit,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matches TSV filenames: selected_participants_brain_part01.tsv, part04a.tsv, etc.
TSV_PART_RE = re.compile(
    r"selected_participants_brain_part(\d+[a-z]?)\.tsv$", re.IGNORECASE
)

# Phase 1 prints: "      --phase1_checkpoint runs/<id>/<subdir>/<name>_model.pt \"
PHASE1_CKPT_RE = re.compile(r"--phase1_checkpoint\s+(runs/\S+_model\.pt)")

# Phase 2 prints: "Weights directory: runs/<id>/<subdir>"
PHASE2_WEIGHTS_DIR_RE = re.compile(r"Weights directory:\s+(runs/\S+)")

# The fast/exact Phase 2 wrapper emits one audit line before delegating to
# the production Phase 2 CLI.
FAST_EXACT_AUDIT_PREFIX = "PHASE2_FAST_EXACT "
FAST_EXACT_BACKEND_ENV = "SIMS_MRI_PHASE2_BENDING_BACKEND"
VALID_BENDING_BACKENDS = ("auto", "terms6", "analytic_siren")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PhaseResult:
    phase: str
    subject_id: str
    status: str = "pending"  # "success", "failed", "skipped", "dry-run"
    duration_seconds: float = 0.0
    command: list[str] = field(default_factory=list)
    checkpoint_path: str | None = None
    error_message: str | None = None
    log_file: str | None = None
    fast_exact_audit: dict[str, Any] | None = None
    bending_backend_requested: str | None = None
    resume_result_file: str | None = None


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
        tsv_dir_path.glob("selected_participants_brain_part*.tsv")
    )

    if parts is None:
        return all_tsvs

    normalized = set()
    for part in parts:
        match = re.match(r"^0*(\d+)([a-z]?)$", part)
        if match:
            normalized.add(f"{int(match.group(1)):02d}{match.group(2)}")
        else:
            normalized.add(part)

    filtered = []
    for tsv_path in all_tsvs:
        match = TSV_PART_RE.search(tsv_path.name)
        if not match:
            continue
        raw_label = match.group(1)
        label_match = re.match(r"^0*(\d+)([a-z]?)$", raw_label)
        if label_match:
            norm_label = f"{int(label_match.group(1)):02d}{label_match.group(2)}"
        else:
            norm_label = raw_label
        if norm_label in normalized:
            filtered.append(tsv_path)

    return filtered


def extract_subjects_from_tsvs(tsv_files: list[Path]) -> list[str]:
    """Read subject IDs from TSV split files, deduplicated in first-seen order."""
    seen: set[str] = set()
    subjects: list[str] = []

    for tsv_path in tsv_files:
        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None or "selected_scan" not in reader.fieldnames:
                raise ValueError(
                    f"TSV file {tsv_path} missing required 'selected_scan' column"
                )
            scan_paths = [row["selected_scan"] for row in reader]

        for scan_path in scan_paths:
            fname = Path(scan_path).name
            if fname.endswith(".nii.gz"):
                subject_id = fname[: -len(".nii.gz")]
            else:
                subject_id = Path(fname).stem
            if subject_id not in seen:
                seen.add(subject_id)
                subjects.append(subject_id)

    return subjects


# ---------------------------------------------------------------------------
# Checkpoint and audit helpers
# ---------------------------------------------------------------------------


def extract_phase1_checkpoint(stdout: str) -> str | None:
    """Parse Phase 1 stdout for the saved model checkpoint path."""
    match = PHASE1_CKPT_RE.search(stdout)
    return match.group(1) if match else None


def extract_phase2_checkpoint(stdout: str) -> str | None:
    """Parse Phase 2 stdout and discover the rotation checkpoint."""
    match = PHASE2_WEIGHTS_DIR_RE.search(stdout)
    if not match:
        return None

    weights_dir = match.group(1)
    rotation_files = sorted(
        glob_module.glob(os.path.join(weights_dir, "*_model_rotation.pt")),
        key=os.path.getmtime,
    )
    return rotation_files[-1] if rotation_files else None


def extract_fast_exact_audit(stdout: str) -> dict[str, Any] | None:
    """Parse the fast/exact Phase 2 wrapper audit payload from stdout."""
    for line in stdout.splitlines():
        if not line.startswith(FAST_EXACT_AUDIT_PREFIX):
            continue
        payload = line[len(FAST_EXACT_AUDIT_PREFIX) :]
        return json.loads(payload)
    return None


def _fallback_read_regularization(config_path: str) -> str:
    """Read REGISTRATION_MODEL.REGULARIZATION when PyYAML is unavailable."""
    in_registration_model = False
    with open(config_path) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw_line.startswith((" ", "\t")) and stripped.endswith(":"):
                in_registration_model = stripped[:-1] == "REGISTRATION_MODEL"
                continue
            if not in_registration_model or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            if key.strip() == "REGULARIZATION":
                return value.split("#", 1)[0].strip().strip("'\"") or "none"
    return "none"


def validate_fast_exact_config(config_path: str) -> None:
    """Fail early unless the config requests bending regularization."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        regularization = _fallback_read_regularization(config_path)
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        reg_model = (config or {}).get("REGISTRATION_MODEL", {})
        regularization = reg_model.get("REGULARIZATION", "none")

    if regularization != "bending":
        raise ValueError(
            "Fast-exact Phase 2 batch runner expects "
            "REGISTRATION_MODEL.REGULARIZATION: bending; "
            f"{config_path} has {regularization!r}."
        )


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------


def run_phase(
    cmd: list[str],
    phase_name: str,
    subject_id: str,
    log_dir: str,
    dry_run: bool = False,
    env_overrides: dict[str, str] | None = None,
) -> tuple[PhaseResult, str]:
    """Execute a training phase as a subprocess."""
    result = PhaseResult(
        phase=phase_name,
        subject_id=subject_id,
        command=cmd,
    )

    if dry_run:
        result.status = "dry-run"
        env_prefix = ""
        if env_overrides:
            env_prefix = " ".join(
                f"{key}={value}" for key, value in sorted(env_overrides.items())
            )
            env_prefix += " "
        print(f"  [DRY-RUN] {env_prefix}{' '.join(cmd)}")
        return result, ""

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{subject_id}_{phase_name}.log")
    result.log_file = log_file

    print(f"  Running {phase_name} for {subject_id} ...")
    start = time.time()
    env = None
    if env_overrides:
        env = os.environ.copy()
        env.update(env_overrides)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )
        result.duration_seconds = time.time() - start

        with open(log_file, "w") as f:
            f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
            if env_overrides:
                f.write("=== ENVIRONMENT OVERRIDES ===\n")
                for key, value in sorted(env_overrides.items()):
                    f.write(f"{key}={value}\n")
                f.write("\n")
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
    bending_backend: str,
) -> list[PhaseResult]:
    """Run Phase 1 -> fast/exact Phase 2 -> Phase 3 for one subject."""
    results: list[PhaseResult] = []

    # ---- Phase 1 ----
    phase1_ckpt: str | None = None

    if resume:
        phase1_ckpt = find_phase1_checkpoint(subject_id, config_path)
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

    # ---- Phase 2 fast/exact ----
    phase2_ckpt: str | None = None

    if resume and phase1_ckpt is not None:
        saved = find_registration_resume(
            subject_id, config_path, phase1_ckpt, bending_backend
        )
        if saved is not None:
            phase2_ckpt = saved.checkpoint_path
            print(f"  [RESUME] Phase 2 checkpoint: {phase2_ckpt}")
            print(f"    Validated result: {saved.result_file}")
            results.append(
                PhaseResult(
                    phase="phase2_fast_exact",
                    subject_id=subject_id,
                    status="skipped",
                    checkpoint_path=phase2_ckpt,
                    bending_backend_requested=bending_backend,
                    fast_exact_audit=saved.audit,
                    log_file=saved.log_file,
                    resume_result_file=saved.result_file,
                    command=saved.command,
                )
            )

    if phase2_ckpt is None:
        cmd = [
            sys.executable,
            "-m",
            "sims_mri.training.phase2_registration_fast_exact",
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
            cmd,
            "phase2_fast_exact",
            subject_id,
            log_dir,
            dry_run,
            env_overrides={FAST_EXACT_BACKEND_ENV: bending_backend},
        )
        p2_result.bending_backend_requested = bending_backend

        if not dry_run and p2_result.status == "success":
            try:
                p2_result.fast_exact_audit = extract_fast_exact_audit(p2_stdout)
            except json.JSONDecodeError as exc:
                p2_result.status = "failed"
                p2_result.error_message = (
                    f"Malformed {FAST_EXACT_AUDIT_PREFIX.strip()} "
                    f"audit JSON: {exc}"
                )

            if p2_result.status == "success" and not valid_fast_exact_audit(
                p2_result.fast_exact_audit, bending_backend
            ):
                p2_result.status = "failed"
                p2_result.error_message = (
                    "Phase 2 exited successfully but no "
                    f"valid {FAST_EXACT_AUDIT_PREFIX.strip()} audit matched the requested backend"
                )

            phase2_ckpt = extract_phase2_checkpoint(p2_stdout)
            p2_result.checkpoint_path = phase2_ckpt
            if p2_result.status == "success" and phase2_ckpt is None:
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
    bending_backend: str,
) -> None:
    """Write batch results summary as JSON."""
    subjects_data = {}
    for subject_id, phase_results in all_results.items():
        subjects_data[subject_id] = {
            "phases": [
                {
                    "phase": result.phase,
                    "status": result.status,
                    "duration_seconds": round(result.duration_seconds, 2),
                    "checkpoint_path": result.checkpoint_path,
                    "error_message": result.error_message,
                    "log_file": result.log_file,
                    "command": result.command,
                    "fast_exact_audit": result.fast_exact_audit,
                    "bending_backend_requested": result.bending_backend_requested,
                    "resume_result_file": result.resume_result_file,
                }
                for result in phase_results
            ],
            "all_succeeded": all(
                result.status in ("success", "skipped", "dry-run")
                for result in phase_results
            ),
        }

    output = {
        "batch_metadata": {
            "runner": "run_batch_training_fast_exact.py",
            "config": config_path,
            "phase2_fast_exact": True,
            "bending_backend": bending_backend,
            "timestamp": datetime.now().isoformat(),
            "total_subjects": total_subjects,
            "subjects_attempted": len(all_results),
            "subjects_succeeded": sum(
                1 for value in subjects_data.values() if value["all_succeeded"]
            ),
            "subjects_failed": sum(
                1 for value in subjects_data.values() if not value["all_succeeded"]
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
        description=(
            "Batch training orchestrator for SIMS-MRI: runs Phase 1 -> "
            "fast/exact Phase 2 -> Phase 3 per subject."
        ),
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
        default="oasis_random/selected_participants_brain_splits",
        help="Directory containing selected_participants_brain_partXX.tsv files.",
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
        "--bending_backend",
        type=str,
        choices=VALID_BENDING_BACKENDS,
        default="auto",
        help="Value for SIMS_MRI_PHASE2_BENDING_BACKEND in Phase 2.",
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
        help="Reuse compatible Phase 1 and successful fast-exact Phase 2 records in runs/.",
    )

    parser.add_argument(
        "--results_json",
        type=str,
        default=None,
        help=(
            "Path for results JSON. Default: "
            "runs/batch_fast_exact_results_<timestamp>.json"
        ),
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help=(
            "Directory for per-subject per-phase logs. Default: "
            "runs/batch_fast_exact_logs_<timestamp>"
        ),
    )

    return parser


def resolve_subjects(args: argparse.Namespace) -> list[str]:
    """Resolve subject IDs from explicit args or split TSV files."""
    if args.subjects:
        print(f"Using {len(args.subjects)} explicitly specified subject(s)")
        return args.subjects

    tsv_files = discover_tsv_files(args.tsv_dir, args.parts)
    if not tsv_files:
        available = sorted(
            Path(args.tsv_dir).glob("selected_participants_brain_part*.tsv")
        )
        print(
            f"ERROR: No TSV files found in {args.tsv_dir}"
            + (f" for parts {args.parts}" if args.parts else "")
        )
        if available:
            labels = [
                TSV_PART_RE.search(path.name).group(1)
                for path in available
                if TSV_PART_RE.search(path.name)
            ]
            print(f"  Available parts: {labels}")
        sys.exit(1)

    print(f"Found {len(tsv_files)} TSV file(s):")
    for tsv_path in tsv_files:
        print(f"  {tsv_path}")

    subjects = extract_subjects_from_tsvs(tsv_files)
    if not subjects:
        print("ERROR: No subjects found in TSV files.")
        sys.exit(1)
    return subjects


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    verify_working_directory()

    if not Path(args.config).is_file():
        print(f"ERROR: Config file not found: {args.config}")
        sys.exit(1)
    try:
        validate_fast_exact_config(args.config)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    subjects = resolve_subjects(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.dry_run:
        Path("runs").mkdir(exist_ok=True)

    results_json = (
        args.results_json or f"runs/batch_fast_exact_results_{timestamp}.json"
    )
    log_dir = args.log_dir or f"runs/batch_fast_exact_logs_{timestamp}"

    print("\n" + "=" * 68)
    print("SIMS-MRI BATCH TRAINING FAST-EXACT")
    print("=" * 68)
    print(f"Subjects:        {len(subjects)}")
    print(f"Config:          {args.config}")
    print(f"Results JSON:    {results_json}")
    print(f"Log dir:         {log_dir}")
    print(f"Bending backend: {args.bending_backend}")
    print(f"Dry run:         {args.dry_run}")
    print(f"Resume:          {args.resume}")
    print(f"Fail fast:       {args.fail_fast}")
    print(f"Logging:         {args.logging}")
    print("=" * 68)

    all_results: dict[str, list[PhaseResult]] = {}
    batch_start = time.time()
    failed_subjects: list[str] = []

    for index, subject_id in enumerate(subjects, 1):
        print(f"\n[{index}/{len(subjects)}] Subject: {subject_id}")
        print("-" * 40)

        subject_results = train_subject(
            subject_id=subject_id,
            config_path=args.config,
            logging=args.logging,
            dry_run=args.dry_run,
            resume=args.resume,
            log_dir=log_dir,
            bending_backend=args.bending_backend,
        )
        all_results[subject_id] = subject_results

        if any(result.status == "failed" for result in subject_results):
            failed_subjects.append(subject_id)
            if args.fail_fast:
                print(
                    f"\n--fail-fast: aborting batch after failure "
                    f"on {subject_id}"
                )
                break

    batch_duration = time.time() - batch_start

    if not args.dry_run:
        write_results_json(
            results_json,
            all_results,
            config_path=args.config,
            batch_duration=batch_duration,
            total_subjects=len(subjects),
            bending_backend=args.bending_backend,
        )

    n_ok = sum(
        1
        for subject_results in all_results.values()
        if all(
            result.status in ("success", "skipped", "dry-run")
            for result in subject_results
        )
    )

    print("\n" + "=" * 68)
    print("BATCH SUMMARY")
    print("=" * 68)
    print(
        f"Total time:        {batch_duration:.1f}s "
        f"({batch_duration / 3600:.2f}h)"
    )
    print(f"Subjects OK:       {n_ok}/{len(all_results)}")
    if failed_subjects:
        print(f"Subjects failed:   {', '.join(failed_subjects)}")
    if not args.dry_run:
        print(f"Results JSON:      {results_json}")
    print("=" * 68)

    if failed_subjects:
        sys.exit(1)


if __name__ == "__main__":
    main()
