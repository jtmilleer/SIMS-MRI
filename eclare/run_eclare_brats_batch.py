#!/usr/bin/env python3
"""
Batch runner for ECLARE over the BraTS 2023 GLI dataset.

Adapts the OASIS batch script for BraTS-specific layout:
- Flat file structure: <dataset-root>/<sid>_<scan>_LR.nii.gz (no sub-directories)
- TSV splits: brats_part01.tsv .. brats_part05.tsv in <dataset-root>/splits/
- Nine scan types: axial, coronal, sagittal, plus rigid/rotated/translated
  variants of coronal and sagittal
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .run_eclare_batch import RunItem, read_subject_ids, run_one, write_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "brats_random"
DEFAULT_TSV_DIR = DEFAULT_DATASET_ROOT / "splits"
SCAN_DEFAULTS = ("axial", "coronal", "sagittal", "rigid-coronal", "rigid-sagittal")
SCAN_ALIASES = {
    "ax": "axial",
    "axial": "axial",
    "axial_lr": "axial",
    "cor": "coronal",
    "coronal": "coronal",
    "coronal_lr": "coronal",
    "sag": "sagittal",
    "sagittal": "sagittal",
    "sagittal_lr": "sagittal",
    "rigid-cor": "rigid-coronal",
    "rigid_cor": "rigid-coronal",
    "rigid-coronal": "rigid-coronal",
    "rigid_coronal": "rigid-coronal",
    "rigid-sag": "rigid-sagittal",
    "rigid_sag": "rigid-sagittal",
    "rigid-sagittal": "rigid-sagittal",
    "rigid_sagittal": "rigid-sagittal",
    "rot-cor": "rotated-coronal",
    "rot_cor": "rotated-coronal",
    "rotated-coronal": "rotated-coronal",
    "rotated_coronal": "rotated-coronal",
    "rot-sag": "rotated-sagittal",
    "rot_sag": "rotated-sagittal",
    "rotated-sagittal": "rotated-sagittal",
    "rotated_sagittal": "rotated-sagittal",
    "trans-cor": "translated-coronal",
    "trans_cor": "translated-coronal",
    "translated-coronal": "translated-coronal",
    "translated_coronal": "translated-coronal",
    "trans-sag": "translated-sagittal",
    "trans_sag": "translated-sagittal",
    "translated-sagittal": "translated-sagittal",
    "translated_sagittal": "translated-sagittal",
}


@dataclass(frozen=True)
class ScanSpec:
    key: str
    suffixes: Tuple[str, ...]
    out_subdir: str
    description: str


SCAN_SPECS = {
    "axial": ScanSpec(
        key="axial",
        suffixes=("_axial_LR.nii.gz", "_axial_LR.nii"),
        out_subdir="axial",
        description="Axial LR",
    ),
    "coronal": ScanSpec(
        key="coronal",
        suffixes=("_coronal_LR.nii.gz", "_coronal_LR.nii"),
        out_subdir="coronal",
        description="Coronal LR",
    ),
    "sagittal": ScanSpec(
        key="sagittal",
        suffixes=("_sagittal_LR.nii.gz", "_sagittal_LR.nii"),
        out_subdir="sagittal",
        description="Sagittal LR",
    ),
    "rigid-coronal": ScanSpec(
        key="rigid-coronal",
        suffixes=("_rigid_coronal_LR.nii.gz", "_rigid_coronal_LR.nii"),
        out_subdir="rigid_coronal",
        description="Rigid coronal LR",
    ),
    "rigid-sagittal": ScanSpec(
        key="rigid-sagittal",
        suffixes=("_rigid_sagittal_LR.nii.gz", "_rigid_sagittal_LR.nii"),
        out_subdir="rigid_sagittal",
        description="Rigid sagittal LR",
    ),
    "rotated-coronal": ScanSpec(
        key="rotated-coronal",
        suffixes=("_rotated_coronal_LR.nii.gz", "_rotated_coronal_LR.nii"),
        out_subdir="rotated_coronal",
        description="Rotated coronal LR",
    ),
    "rotated-sagittal": ScanSpec(
        key="rotated-sagittal",
        suffixes=("_rotated_sagittal_LR.nii.gz", "_rotated_sagittal_LR.nii"),
        out_subdir="rotated_sagittal",
        description="Rotated sagittal LR",
    ),
    "translated-coronal": ScanSpec(
        key="translated-coronal",
        suffixes=("_translated_coronal_LR.nii.gz", "_translated_coronal_LR.nii"),
        out_subdir="translated_coronal",
        description="Translated coronal LR",
    ),
    "translated-sagittal": ScanSpec(
        key="translated-sagittal",
        suffixes=("_translated_sagittal_LR.nii.gz", "_translated_sagittal_LR.nii"),
        out_subdir="translated_sagittal",
        description="Translated sagittal LR",
    ),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch ECLARE over BraTS scans")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing BraTS NIfTI files (default: brats_random/)",
    )
    p.add_argument(
        "--tsv-dir",
        type=Path,
        default=DEFAULT_TSV_DIR,
        help="Directory with brats_part*.tsv split files (default: <dataset-root>/splits)",
    )
    p.add_argument(
        "--tsv",
        type=Path,
        default=None,
        help="Single TSV file override (ignores --tsv-dir and --parts)",
    )
    p.add_argument(
        "--parts",
        nargs="+",
        default=None,
        help="Filter split files by part number, e.g. --parts 01 03",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Directory to store outputs. Defaults to <dataset-root>/<model-name>",
    )
    p.add_argument(
        "--model-name",
        type=str,
        default="eclare",
        help="Model directory name when deriving --out-root automatically",
    )
    p.add_argument(
        "--scans",
        type=str,
        default=",".join(SCAN_DEFAULTS),
        help="Comma-separated scans to process (default: axial,coronal,sagittal,rigid-coronal,rigid-sagittal)",
    )
    p.add_argument("--gpu-id", type=int, default=0, help="GPU id for ECLARE (use -1 for CPU)")
    p.add_argument("--dry-run", action="store_true", help="List planned runs without executing")
    p.add_argument("--force", action="store_true", help="Re-run even if output exists (overrides resume)")
    p.add_argument("--resume", action="store_true", help="Skip completed outputs")
    p.add_argument("--suffix", type=str, default="_eclare", help="Suffix for ECLARE outputs")
    p.add_argument("--debug", action="store_true", help="Print discovery diagnostics for missing scans")
    p.add_argument("--start", type=int, default=0, help="Zero-based index of the first subject to process")
    p.add_argument("--limit", type=int, default=None, help="Only process N subjects (starting from --start)")

    # Pass-through options for eclare.main
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-patches", type=int, default=None)
    p.add_argument("--relative-slice-thickness", type=float, default=None)
    p.add_argument("--relative-slice-profile-type", type=str, default=None)
    p.add_argument("--relative-slice-profile-fpath", type=Path, default=None)
    p.add_argument("--inplane-acq-res", type=float, nargs=2, default=None)
    p.add_argument("--verbose", action="store_true")

    return p.parse_args(argv)


def parse_scans(scans_arg: str) -> Tuple[str, ...]:
    if not scans_arg:
        return SCAN_DEFAULTS
    result: List[str] = []
    for raw in scans_arg.split(","):
        key = raw.strip().lower()
        if not key:
            continue
        canonical = SCAN_ALIASES.get(key, key)
        if canonical not in SCAN_SPECS:
            raise ValueError(
                f"Unsupported scan '{raw}'. Supported scans: {','.join(sorted(SCAN_SPECS.keys()))}"
            )
        if canonical not in result:
            result.append(canonical)
    return tuple(result) if result else SCAN_DEFAULTS


def collect_subject_ids(args: argparse.Namespace) -> List[str]:
    """Read subject IDs from TSV splits, with optional part filtering.

    If --tsv is given, reads that single file. Otherwise reads all
    brats_part*.tsv files from --tsv-dir, optionally filtered by --parts.
    Deduplicates while preserving order.
    """
    if args.tsv is not None:
        return read_subject_ids(args.tsv.resolve())

    tsv_dir = args.tsv_dir.resolve()
    if not tsv_dir.exists():
        print(f"TSV directory not found: {tsv_dir}")
        return []

    tsv_files = sorted(tsv_dir.glob("brats_part*.tsv"))
    if not tsv_files:
        print(f"No brats_part*.tsv files found in {tsv_dir}")
        return []

    if args.parts is not None:
        filtered: List[Path] = []
        for part in args.parts:
            pat = re.compile(rf"brats_part0*{re.escape(part)}\.tsv$")
            matched = [f for f in tsv_files if pat.search(f.name)]
            filtered.extend(matched)
        tsv_files = sorted(set(filtered), key=lambda p: p.name)
        if not tsv_files:
            print(f"No TSV files matched --parts {args.parts} in {tsv_dir}")
            return []

    seen: set[str] = set()
    all_sids: List[str] = []
    for tsv_path in tsv_files:
        for sid in read_subject_ids(tsv_path):
            if sid not in seen:
                seen.add(sid)
                all_sids.append(sid)
    return all_sids


def find_scan_input(dataset_root: Path, sid: str, spec: ScanSpec, debug: bool = False) -> Optional[Path]:
    for suffix in spec.suffixes:
        candidate = dataset_root / f"{sid}{suffix}"
        if candidate.exists():
            return candidate

    if debug:
        print(
            f"[discovery] no {spec.description} input for {sid}. "
            f"Tried suffixes {spec.suffixes} under {dataset_root}"
        )
    return None


def derive_output_paths(out_root: Path, sid: str, spec: ScanSpec, in_fpath: Path, suffix: str) -> Tuple[Path, Path]:
    subject_dir = out_root / sid
    out_dir = subject_dir / spec.out_subdir
    out_fname = in_fpath.name.replace(".nii", f"{suffix}.nii")
    return out_dir, out_dir / out_fname


def build_runlist(
    dataset_root: Path,
    out_root: Path,
    sids: Iterable[str],
    scan_keys: Sequence[str],
    suffix: str,
    debug: bool = False,
) -> Tuple[List[RunItem], List[Tuple[str, str]]]:
    run_items: List[RunItem] = []
    missing: List[Tuple[str, str]] = []
    for sid in sids:
        for scan_key in scan_keys:
            spec = SCAN_SPECS[scan_key]
            in_fpath = find_scan_input(dataset_root, sid, spec, debug=debug)
            if in_fpath is None:
                missing.append((sid, spec.key))
                continue
            out_dir, out_fpath = derive_output_paths(out_root, sid, spec, in_fpath, suffix)
            run_items.append(
                RunItem(
                    sid=sid,
                    view=spec.key,
                    in_fpath=in_fpath,
                    out_dir=out_dir,
                    out_fpath=out_fpath,
                )
            )
    return run_items, missing


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}")
        return 1

    out_root = args.out_root.resolve() if args.out_root else (dataset_root / args.model_name).resolve()

    sids = collect_subject_ids(args)
    if args.start > 0:
        sids = sids[args.start:]
    if args.limit is not None:
        sids = sids[: max(0, args.limit)]
    if not sids:
        print("No subject IDs collected. Check --tsv-dir / --tsv / --parts.")
        return 1

    scan_keys = parse_scans(args.scans)

    runlist, missing = build_runlist(
        dataset_root=dataset_root,
        out_root=out_root,
        sids=sids,
        scan_keys=scan_keys,
        suffix=args.suffix,
        debug=args.debug,
    )

    if not runlist:
        print("No matching inputs found for requested subjects/scans. Use --debug for diagnostics.")
        return 1

    if missing:
        print(
            f"Warning: missing {len(missing)} subject/scan combinations. "
            "Enable --debug to see detailed discovery output."
        )

    print(f"Planned items: {len(runlist)} (dry-run={args.dry_run})")

    rows: List[List[str]] = []
    for item in runlist:
        start_ts = time.strftime("%Y-%m-%d %H:%M:%S")
        st = time.time()
        if not item.in_fpath.exists():
            status, err = ("missing_input", None)
        else:
            status, err = ("planned", None) if args.dry_run else run_one(item, args)
        en = time.time()
        end_ts = time.strftime("%Y-%m-%d %H:%M:%S")

        rows.append(
            [
                item.sid,
                item.view,
                str(item.in_fpath),
                str(item.out_fpath),
                status,
                start_ts,
                end_ts,
                err or "",
            ]
        )

        print(
            f"{item.sid} {item.view}: {status}"
            + (" (exists)" if status == "skipped_exists" else "")
            + (f" err={err}" if err else "")
        )

    if args.dry_run:
        print("Dry-run: manifest not written (no filesystem changes).")
        return 0

    manifest = write_manifest(out_root, rows)
    print(f"Manifest written: {manifest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
