import argparse
import csv
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCAN_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "axial": {"label": "Axial", "pattern": "_axial_LR.nii.gz"},
    "coronal": {"label": "Coronal", "pattern": "_coronal_LR.nii.gz"},
    "sagittal": {"label": "Sagittal", "pattern": "_sagittal_LR.nii.gz"},
    "rigid_coronal": {"label": "Rigid Coronal", "pattern": "_rigid_coronal_LR.nii.gz"},
    "rigid_sagittal": {"label": "Rigid Sagittal", "pattern": "_rigid_sagittal_LR.nii.gz"},
    "rotated_coronal": {"label": "Rotated Coronal", "pattern": "_rotated_coronal_LR.nii.gz"},
    "rotated_sagittal": {"label": "Rotated Sagittal", "pattern": "_rotated_sagittal_LR.nii.gz"},
    "translated_coronal": {"label": "Translated Coronal", "pattern": "_translated_coronal_LR.nii.gz"},
    "translated_sagittal": {"label": "Translated Sagittal", "pattern": "_translated_sagittal_LR.nii.gz"},
}
SCAN_DEFAULT_ORDER = tuple(SCAN_DEFINITIONS.keys())


def parse_scans(scans_arg: Optional[str]) -> List[str]:
    if not scans_arg:
        return list(SCAN_DEFAULT_ORDER)
    scans: List[str] = []
    for token in scans_arg.split(","):
        name = token.strip()
        if not name:
            continue
        normalized = name.lower().replace("-", "_")
        if normalized not in SCAN_DEFINITIONS:
            valid = ", ".join(SCAN_DEFINITIONS.keys())
            raise ValueError(f"Unsupported scan '{name}'. Choose from {valid}.")
        if normalized not in scans:
            scans.append(normalized)
    return scans if scans else list(SCAN_DEFAULT_ORDER)


def read_participant_ids(tsv_path: Path, column: str = "subject_id") -> List[str]:
    participants: List[str] = []
    with tsv_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        return participants

    header = rows[0]
    start_idx = 0
    column_idx = 0
    if header:
        lowered = [cell.strip().lower() for cell in header]
        if column.lower() in lowered:
            column_idx = lowered.index(column.lower())
            start_idx = 1

    for row in rows[start_idx:]:
        if not row:
            continue
        cell = (row[column_idx] or "").strip()
        if not cell or cell.startswith("#"):
            continue
        participants.append(cell)
    return participants


def collect_participant_ids(args: argparse.Namespace) -> List[str]:
    if args.tsv:
        return read_participant_ids(Path(args.tsv))

    tsv_dir = Path(args.tsv_dir)
    if not tsv_dir.is_dir():
        raise FileNotFoundError(f"TSV directory not found: {tsv_dir}")

    all_tsvs = sorted(tsv_dir.glob("brats_part*.tsv"))
    if not all_tsvs:
        raise FileNotFoundError(f"No brats_part*.tsv files found in {tsv_dir}")

    if args.parts:
        selected: List[Path] = []
        for part in args.parts:
            pat = re.compile(rf"brats_part0*{re.escape(part)}\.tsv$")
            matched = [t for t in all_tsvs if pat.search(t.name)]
            if not matched:
                raise FileNotFoundError(
                    f"No TSV matching part '{part}' in {tsv_dir}"
                )
            selected.extend(matched)
        all_tsvs = selected

    seen: set = set()
    participants: List[str] = []
    for tsv_path in all_tsvs:
        for pid in read_participant_ids(tsv_path):
            if pid not in seen:
                seen.add(pid)
                participants.append(pid)
    return participants


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_status_row(status_csv: Path, row: List[str]) -> None:
    ensure_parent(status_csv)
    exists = status_csv.exists()
    with status_csv.open("a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "participant_id",
                    "scan",
                    "input",
                    "out_dir",
                    "status",
                    "message",
                    "elapsed_s",
                    "timestamp",
                ]
            )
        writer.writerow(row)


def build_run_smore_cmd(
    in_fpath: Path,
    out_dir: Path,
    gpu_id: int,
    suffix: str,
    slice_thickness: Optional[float],
    blur_kernel_fpath: Optional[Path],
    patch_sampling: Optional[str],
    extra_args: Optional[str],
    prefer_module: bool = False,
) -> List[str]:
    run_smore = None if prefer_module else shutil.which("run-smore")
    if run_smore:
        cmd: List[str] = [
            run_smore,
            "--in-fpath",
            str(in_fpath),
            "--out-dir",
            str(out_dir),
            "--gpu-id",
            str(gpu_id),
            "--suffix",
            suffix,
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "smore.main",
            "--in-fpath",
            str(in_fpath),
            "--out-dir",
            str(out_dir),
            "--gpu-id",
            str(gpu_id),
            "--suffix",
            suffix,
        ]
    if slice_thickness is not None:
        cmd += ["--slice-thickness", str(slice_thickness)]
    if blur_kernel_fpath is not None:
        cmd += ["--blur-kernel-fpath", str(blur_kernel_fpath)]
    if patch_sampling is not None:
        cmd += ["--patch-sampling", patch_sampling]
    if extra_args:
        cmd += shlex.split(extra_args)
    return cmd


def infer_output_file(in_fpath: Path, out_dir: Path, participant_id: str, suffix: str) -> Path:
    name = in_fpath.name
    ext = name.split(".", maxsplit=1)[1] if "." in name else "nii.gz"
    return out_dir / participant_id / f"{participant_id}{suffix}.{ext}"


def build_module_cmds(
    in_fpath: Path,
    out_dir_base: Path,
    participant_id: str,
    suffix: str,
    gpu_id: int,
    slice_thickness: Optional[float],
    blur_kernel_fpath: Optional[Path],
    patch_sampling: Optional[str],
    extra_args: Optional[str],
) -> Tuple[List[str], List[str]]:
    out_dir = out_dir_base / participant_id
    weight_dir = out_dir / "weights"
    out_fpath = infer_output_file(in_fpath, out_dir_base, participant_id, suffix)

    train_cmd: List[str] = [
        sys.executable,
        "-m",
        "smore.train",
        "--in-fpath",
        str(in_fpath),
        "--weight-dir",
        str(weight_dir),
        "--gpu-id",
        str(gpu_id),
        "--patch-sampling",
        patch_sampling or "gradient",
    ]
    if slice_thickness is not None:
        train_cmd += ["--slice-thickness", str(slice_thickness)]
    if blur_kernel_fpath is not None:
        train_cmd += ["--blur-kernel-fpath", str(blur_kernel_fpath)]
    if extra_args:
        train_cmd += shlex.split(extra_args)

    test_cmd: List[str] = [
        sys.executable,
        "-m",
        "smore.test",
        "--in-fpath",
        str(in_fpath),
        "--out-fpath",
        str(out_fpath),
        "--weight-dir",
        str(weight_dir),
        "--gpu-id",
        str(gpu_id),
    ]
    if extra_args:
        test_cmd += shlex.split(extra_args)
    return train_cmd, test_cmd


def find_scan_input(data_root: Path, participant_id: str, scan_key: str) -> Optional[Path]:
    pattern = SCAN_DEFINITIONS[scan_key]["pattern"]
    candidate = data_root / f"{participant_id}{pattern}"
    if candidate.is_file():
        return candidate
    return None


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch runner for SMORE on the BraTS brain dataset."
    )
    parser.add_argument(
        "--tsv",
        type=str,
        help="Single TSV file containing subject_id column (overrides --tsv-dir).",
    )
    parser.add_argument(
        "--tsv-dir",
        type=str,
        default="brats_random/splits",
        help="Directory with brats_part*.tsv split files (default: brats_random/splits).",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        type=str,
        help="Which parts to process (e.g. 1 2 3). Default: all parts in --tsv-dir.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="brats_random",
        help="Root directory containing BraTS NIfTI files (default: brats_random).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        help="Override output root (default: data-root).",
    )
    parser.add_argument(
        "--model-dir-name",
        type=str,
        default="smore",
        help="Name of the model directory created under each scan (default: smore).",
    )
    parser.add_argument(
        "--scans",
        type=str,
        help="Comma-separated scan keys to process (axial,rotated_coronal). Default: all.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--slice-thickness", type=float)
    parser.add_argument("--blur-kernel-fpath", type=str)
    parser.add_argument(
        "--patch-sampling",
        type=str,
        default="gradient",
        choices=["gradient", "uniform"],
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_smore4",
        help="Output filename suffix before extension.",
    )
    parser.add_argument(
        "--smore-args",
        type=str,
        help="Extra args to pass to run-smore (quoted).",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--prefer-module",
        action="store_true",
        default=False,
        help="Force python -m smore.* instead of run-smore CLI.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Zero-based index of the first participant to process.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only process N participants (starting from --start).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip subject if expected output file exists.",
    )

    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    data_root = Path(args.data_root)
    out_root = Path(args.out_root) if args.out_root else data_root
    try:
        scans = parse_scans(args.scans)
    except ValueError as exc:
        parser.error(str(exc))

    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    participants = collect_participant_ids(args)
    if args.start > 0:
        participants = participants[args.start:]
    if args.limit is not None:
        participants = participants[: max(0, args.limit)]
    if not participants:
        print("No participants found.")
        return 1

    scan_labels = ", ".join(SCAN_DEFINITIONS[s]["label"] for s in scans)
    print(f"Found {len(participants)} participants; scans: {scan_labels}")

    total_ok = 0
    total_fail = 0
    blur_kernel = Path(args.blur_kernel_fpath) if args.blur_kernel_fpath else None

    for scan in scans:
        scan_label = SCAN_DEFINITIONS[scan]["label"]
        pattern = SCAN_DEFINITIONS[scan]["pattern"]
        out_dir_base = out_root / scan / args.model_dir_name
        status_csv = out_dir_base / "smore_batch_status.csv"

        for participant_id in participants:
            input_fpath = find_scan_input(data_root, participant_id, scan)
            if input_fpath is None:
                msg = f"input matching '{pattern}' not found"
                print(f"[{scan_label}][{participant_id}] {msg}")
                write_status_row(
                    status_csv,
                    [
                        participant_id,
                        scan,
                        "",
                        str(out_dir_base),
                        "fail",
                        msg,
                        "",
                        datetime.utcnow().isoformat(),
                    ],
                )
                total_fail += 1
                continue

            expected_out = infer_output_file(
                input_fpath, out_dir_base, participant_id, args.suffix
            )
            if args.skip_existing and expected_out.exists():
                msg = "exists; skipped"
                print(f"[{scan_label}][{participant_id}] {msg}")
                write_status_row(
                    status_csv,
                    [
                        participant_id,
                        scan,
                        str(input_fpath),
                        str(out_dir_base),
                        "skip",
                        msg,
                        "",
                        datetime.utcnow().isoformat(),
                    ],
                )
                continue

            prefer_module = args.prefer_module
            if not prefer_module:
                cmd = build_run_smore_cmd(
                    in_fpath=input_fpath,
                    out_dir=out_dir_base,
                    gpu_id=args.gpu_id,
                    suffix=args.suffix,
                    slice_thickness=args.slice_thickness,
                    blur_kernel_fpath=blur_kernel,
                    patch_sampling=args.patch_sampling,
                    extra_args=args.smore_args,
                    prefer_module=False,
                )
            else:
                train_cmd, test_cmd = build_module_cmds(
                    in_fpath=input_fpath,
                    out_dir_base=out_dir_base,
                    participant_id=participant_id,
                    suffix=args.suffix,
                    gpu_id=args.gpu_id,
                    slice_thickness=args.slice_thickness,
                    blur_kernel_fpath=blur_kernel,
                    patch_sampling=args.patch_sampling,
                    extra_args=args.smore_args,
                )

            if args.dry_run:
                if not prefer_module:
                    cmd_display = " ".join(shlex.quote(c) for c in cmd)
                    print(f"[DRY-RUN][{scan_label}][{participant_id}] => {cmd_display}")
                else:
                    train_display = " ".join(shlex.quote(c) for c in train_cmd)
                    test_display = " ".join(shlex.quote(c) for c in test_cmd)
                    print(f"[DRY-RUN][{scan_label}][{participant_id}] TRAIN => {train_display}")
                    print(f"[DRY-RUN][{scan_label}][{participant_id}] TEST  => {test_display}")
                write_status_row(
                    status_csv,
                    [
                        participant_id,
                        scan,
                        str(input_fpath),
                        str(out_dir_base),
                        "dry-run",
                        "",
                        "",
                        datetime.utcnow().isoformat(),
                    ],
                )
                continue

            start = time.time()
            try:
                ensure_parent(status_csv)
                if not prefer_module:
                    subprocess.run(cmd, check=True)
                else:
                    env = os.environ.copy()
                    repo_root = Path(__file__).resolve().parents[1]
                    env["PYTHONPATH"] = (
                        str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
                    )
                    subprocess.run(train_cmd, check=True, env=env)
                    subprocess.run(test_cmd, check=True, env=env)
                elapsed = f"{time.time() - start:.2f}"
                print(f"[{scan_label}][{participant_id}] ok ({elapsed}s)")
                write_status_row(
                    status_csv,
                    [
                        participant_id,
                        scan,
                        str(input_fpath),
                        str(out_dir_base),
                        "ok",
                        "",
                        elapsed,
                        datetime.utcnow().isoformat(),
                    ],
                )
                total_ok += 1
            except subprocess.CalledProcessError as exc:
                elapsed = f"{time.time() - start:.2f}"
                msg = f"error code {exc.returncode}"
                print(f"[{scan_label}][{participant_id}] fail: {msg}")
                write_status_row(
                    status_csv,
                    [
                        participant_id,
                        scan,
                        str(input_fpath),
                        str(out_dir_base),
                        "fail",
                        msg,
                        elapsed,
                        datetime.utcnow().isoformat(),
                    ],
                )
                total_fail += 1

    print(f"Done. ok={total_ok} fail={total_fail}")
    return 0 if total_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
