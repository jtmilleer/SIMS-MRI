#!/usr/bin/env python
"""Batch OASIS Phase 2 registration with the fast/exact launcher.

Runs Phase 2 only for subjects discovered from OASIS split TSVs, using the
latest compatible Phase 1 model/hash checkpoint pair and config snapshot found
under ``runs/`` by default. The training subprocess uses
``sims_mri.training.phase2_registration_fast_exact``
and requires the wrapper's ``PHASE2_FAST_EXACT`` audit line before marking a
subject successful.

Examples:
  # Dry run one split
  uv run python scripts/run_batch_training_oasis_phase2_fast_exact.py \
      --config configs/oasis_ngp_rigid.yaml --parts 01 --dry-run

  # Run one split with logging and resume existing Phase 2 checkpoints
  uv run python scripts/run_batch_training_oasis_phase2_fast_exact.py \
      --config configs/oasis_ngp_rigid.yaml --parts 01 --logging --resume \
      --bending_backend auto

  # Run one explicit subject
  uv run python scripts/run_batch_training_oasis_phase2_fast_exact.py \
      --config configs/oasis_ngp_rigid.yaml \
      --subjects sub-OAS30433_ses-M066_run-02_T1w --logging
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

# Phase 2 prints: "Weights directory: runs/<id>/<subdir>"
PHASE2_WEIGHTS_DIR_RE = re.compile(r"Weights directory:\s+(runs/\S+)")

# The fast/exact wrapper emits one audit line before delegating to Phase 2.
FAST_EXACT_AUDIT_PREFIX = "PHASE2_FAST_EXACT "
FAST_EXACT_BACKEND_ENV = "SIMS_MRI_PHASE2_BENDING_BACKEND"
VALID_BENDING_BACKENDS = ("auto", "terms6", "analytic_siren")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Phase2FastExactResult:
    subject_id: str
    status: str = "pending"  # "success", "failed", "skipped", "dry-run"
    duration_seconds: float = 0.0
    command: list[str] = field(default_factory=list)
    phase1_checkpoint: str | None = None
    phase2_checkpoint: str | None = None
    error_message: str | None = None
    log_file: str | None = None
    fast_exact_audit: dict[str, Any] | None = None
    bending_backend_requested: str = "auto"
    resume_result_file: str | None = None


# ---------------------------------------------------------------------------
# TSV handling
# ---------------------------------------------------------------------------


def discover_tsv_files(
    tsv_dir: str, parts: list[str] | None = None
) -> list[Path]:
    """Find and optionally filter OASIS split TSV files in *tsv_dir*."""
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
    """Read scan stems from split TSVs, deduplicated in first-seen order."""
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
# Config and checkpoint helpers
# ---------------------------------------------------------------------------


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


def validate_bending_config(config_path: str) -> None:
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


def extract_phase2_checkpoint(stdout: str) -> str | None:
    """Parse Phase 2 stdout and discover the emitted rotation checkpoint."""
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
    """Return the parsed ``PHASE2_FAST_EXACT`` payload from stdout."""
    for line in stdout.splitlines():
        if not line.startswith(FAST_EXACT_AUDIT_PREFIX):
            continue
        payload = line[len(FAST_EXACT_AUDIT_PREFIX) :]
        return json.loads(payload)
    return None


# ---------------------------------------------------------------------------
# Phase 2 execution
# ---------------------------------------------------------------------------


def build_phase2_command(
    config_path: str, subject_id: str, phase1_checkpoint: str, logging: bool
) -> list[str]:
    """Build the fast/exact Phase 2 command for one subject."""
    cmd = [
        sys.executable,
        "-m",
        "sims_mri.training.phase2_registration_fast_exact",
        "--config",
        config_path,
        "--subject_id",
        subject_id,
        "--phase1_checkpoint",
        phase1_checkpoint,
    ]
    if logging:
        cmd.append("--logging")
    return cmd


def run_phase2_fast_exact(
    cmd: list[str],
    subject_id: str,
    phase1_checkpoint: str,
    log_dir: str,
    bending_backend: str,
    dry_run: bool = False,
) -> Phase2FastExactResult:
    """Execute one fast/exact Phase 2 subprocess and validate its provenance."""
    result = Phase2FastExactResult(
        subject_id=subject_id,
        command=cmd,
        phase1_checkpoint=phase1_checkpoint,
        bending_backend_requested=bending_backend,
    )

    if dry_run:
        result.status = "dry-run"
        print(f"  [DRY-RUN] {FAST_EXACT_BACKEND_ENV}={bending_backend} {' '.join(cmd)}")
        return result

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, f"{subject_id}_phase2_fast_exact.log")
    result.log_file = log_file

    print(f"  Running phase2_fast_exact for {subject_id} ...")
    start = time.time()
    env = os.environ.copy()
    env[FAST_EXACT_BACKEND_ENV] = bending_backend

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
            f.write(f"=== ENVIRONMENT OVERRIDES ===\n")
            f.write(f"{FAST_EXACT_BACKEND_ENV}={bending_backend}\n\n")
            f.write(f"=== RETURN CODE === {proc.returncode}\n\n")
            f.write(f"=== STDOUT ===\n{proc.stdout}\n\n")
            f.write(f"=== STDERR ===\n{proc.stderr}\n")

        if proc.returncode != 0:
            result.status = "failed"
            stderr_lines = proc.stderr.strip().splitlines()
            result.error_message = "\n".join(stderr_lines[-5:])
            print(
                f"  FAILED phase2_fast_exact for {subject_id} "
                f"(exit code {proc.returncode})"
            )
            print(f"    Log: {log_file}")
            return result

        try:
            result.fast_exact_audit = extract_fast_exact_audit(proc.stdout)
        except json.JSONDecodeError as exc:
            result.status = "failed"
            result.error_message = (
                f"Malformed {FAST_EXACT_AUDIT_PREFIX.strip()} audit JSON: {exc}"
            )
            print(f"  FAILED phase2_fast_exact for {subject_id}: malformed audit line")
            print(f"    Log: {log_file}")
            return result

        if not valid_fast_exact_audit(result.fast_exact_audit, bending_backend):
            result.status = "failed"
            result.error_message = (
                f"Phase 2 exited successfully but no "
                f"valid {FAST_EXACT_AUDIT_PREFIX.strip()} audit matched the requested backend"
            )
            print(f"  FAILED phase2_fast_exact for {subject_id}: missing or incompatible audit")
            print(f"    Log: {log_file}")
            return result

        result.phase2_checkpoint = extract_phase2_checkpoint(proc.stdout)
        if result.phase2_checkpoint is None:
            result.status = "failed"
            result.error_message = (
                "Phase 2 exited successfully but rotation checkpoint could "
                "not be discovered"
            )
            print(
                f"  FAILED phase2_fast_exact for {subject_id}: "
                "rotation checkpoint not found"
            )
            print(f"    Log: {log_file}")
            return result

        result.status = "success"
        hrs = result.duration_seconds / 3600
        print(
            f"  Completed phase2_fast_exact for {subject_id} "
            f"({result.duration_seconds:.1f}s / {hrs:.2f}h)"
        )
        print(f"    Phase 2 checkpoint: {result.phase2_checkpoint}")
        return result

    except Exception as exc:
        result.duration_seconds = time.time() - start
        result.status = "failed"
        result.error_message = str(exc)
        return result


def train_subject_phase2_fast_exact(
    subject_id: str,
    config_path: str,
    logging: bool,
    dry_run: bool,
    resume: bool,
    log_dir: str,
    bending_backend: str,
    checkpoint_search_root: str,
) -> Phase2FastExactResult:
    """Run or resume fast/exact Phase 2 for one subject."""
    phase1_ckpt = find_phase1_checkpoint(subject_id, config_path, checkpoint_search_root)
    if phase1_ckpt is None:
        message = (
            f"No compatible Phase 1 model, hash encoder and config snapshot found for {subject_id} under "
            f"{checkpoint_search_root!r}"
        )
        print(f"  FAILED phase2_fast_exact for {subject_id}: {message}")
        return Phase2FastExactResult(
            subject_id=subject_id,
            status="failed",
            error_message=message,
            bending_backend_requested=bending_backend,
        )

    if resume:
        saved = find_registration_resume(
            subject_id, config_path, phase1_ckpt, bending_backend, checkpoint_search_root
        )
        if saved is not None:
            print(f"  [RESUME] Phase 2 checkpoint: {saved.checkpoint_path}")
            print(f"    Validated result: {saved.result_file}")
            return Phase2FastExactResult(
                subject_id=subject_id,
                status="skipped",
                phase1_checkpoint=phase1_ckpt,
                phase2_checkpoint=saved.checkpoint_path,
                bending_backend_requested=bending_backend,
                fast_exact_audit=saved.audit,
                log_file=saved.log_file,
                resume_result_file=saved.result_file,
                command=saved.command,
            )

    print(f"  Phase 1 checkpoint: {phase1_ckpt}")
    cmd = build_phase2_command(config_path, subject_id, phase1_ckpt, logging)
    return run_phase2_fast_exact(
        cmd=cmd,
        subject_id=subject_id,
        phase1_checkpoint=phase1_ckpt,
        log_dir=log_dir,
        bending_backend=bending_backend,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# Results output
# ---------------------------------------------------------------------------


def write_results_json(
    path: str,
    all_results: dict[str, Phase2FastExactResult],
    config_path: str,
    checkpoint_search_root: str,
    batch_duration: float,
    total_subjects: int,
) -> None:
    """Write batch results summary as JSON."""
    subjects_data = {}
    for subject_id, result in all_results.items():
        subjects_data[subject_id] = {
            "phase": "phase2_fast_exact",
            "status": result.status,
            "duration_seconds": round(result.duration_seconds, 2),
            "phase1_checkpoint": result.phase1_checkpoint,
            "phase2_checkpoint": result.phase2_checkpoint,
            "error_message": result.error_message,
            "log_file": result.log_file,
            "command": result.command,
            "bending_backend_requested": result.bending_backend_requested,
            "fast_exact_audit": result.fast_exact_audit,
            "resume_result_file": result.resume_result_file,
            "all_succeeded": result.status in ("success", "skipped", "dry-run"),
        }

    output = {
        "batch_metadata": {
            "runner": "run_batch_training_oasis_phase2_fast_exact.py",
            "config": config_path,
            "checkpoint_search_root": checkpoint_search_root,
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
    """Ensure this script runs from the SIMS-MRI repository root."""
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
            "Batch OASIS Phase 2 fast/exact registration runner. "
            "Discovers subjects, finds Phase 1 checkpoints, and runs only "
            "sims_mri.training.phase2_registration_fast_exact."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to OASIS YAML config file (e.g., configs/oasis_ngp_rigid.yaml).",
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
        "--checkpoint_search_root",
        type=str,
        default="runs",
        help="Root to search for Phase 1 and resume Phase 2 checkpoints.",
    )
    parser.add_argument(
        "--bending_backend",
        type=str,
        choices=VALID_BENDING_BACKENDS,
        default="auto",
        help=(
            "Value for SIMS_MRI_PHASE2_BENDING_BACKEND in the Phase 2 subprocess."
        ),
    )

    parser.add_argument(
        "--logging",
        action="store_true",
        help="Pass --logging to the Phase 2 entry point.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing Phase 2.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort the batch on the first subject failure.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful fast-exact records matching the Phase 1 checkpoint and config.",
    )

    parser.add_argument(
        "--results_json",
        type=str,
        default=None,
        help=(
            "Path for results JSON. Default: "
            "<checkpoint_search_root>/batch_phase2_fast_exact_results_<timestamp>.json"
        ),
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=None,
        help=(
            "Directory for per-subject Phase 2 logs. Default: "
            "runs/batch_phase2_fast_exact_logs_<timestamp>"
        ),
    )

    return parser


def resolve_subjects(args: argparse.Namespace) -> list[str]:
    """Resolve subject IDs from explicit args or OASIS split TSVs."""
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
                TSV_PART_RE.search(p.name).group(1)
                for p in available
                if TSV_PART_RE.search(p.name)
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
    if not Path(args.checkpoint_search_root).is_dir():
        print(
            "ERROR: Checkpoint search root not found: "
            f"{args.checkpoint_search_root}"
        )
        sys.exit(1)
    try:
        validate_bending_config(args.config)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    subjects = resolve_subjects(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_json = (
        args.results_json
        or str(Path(args.checkpoint_search_root) / f"batch_phase2_fast_exact_results_{timestamp}.json")
    )
    log_dir = args.log_dir or f"runs/batch_phase2_fast_exact_logs_{timestamp}"

    print("\n" + "=" * 68)
    print("SIMS-MRI OASIS PHASE 2 FAST-EXACT BATCH")
    print("=" * 68)
    print(f"Subjects:               {len(subjects)}")
    print(f"Config:                 {args.config}")
    print(f"Checkpoint search root: {args.checkpoint_search_root}")
    print(f"Bending backend:        {args.bending_backend}")
    print(f"Results JSON:           {results_json}")
    print(f"Log dir:                {log_dir}")
    print(f"Dry run:                {args.dry_run}")
    print(f"Resume:                 {args.resume}")
    print(f"Fail fast:              {args.fail_fast}")
    print(f"Logging:                {args.logging}")
    print("=" * 68)

    all_results: dict[str, Phase2FastExactResult] = {}
    failed_subjects: list[str] = []
    batch_start = time.time()

    for index, subject_id in enumerate(subjects, 1):
        print(f"\n[{index}/{len(subjects)}] Subject: {subject_id}")
        print("-" * 40)

        result = train_subject_phase2_fast_exact(
            subject_id=subject_id,
            config_path=args.config,
            logging=args.logging,
            dry_run=args.dry_run,
            resume=args.resume,
            log_dir=log_dir,
            bending_backend=args.bending_backend,
            checkpoint_search_root=args.checkpoint_search_root,
        )
        all_results[subject_id] = result

        if result.status == "failed":
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
            checkpoint_search_root=args.checkpoint_search_root,
            batch_duration=batch_duration,
            total_subjects=len(subjects),
        )

    n_ok = sum(
        1
        for result in all_results.values()
        if result.status in ("success", "skipped", "dry-run")
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
