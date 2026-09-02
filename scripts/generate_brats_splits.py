"""Generate split files for BraTS batch training.

Discovers preprocessed BraTS subjects in the output directory and writes
them into equal-sized TSV split files.

Examples:
  python scripts/generate_brats_splits.py
  python scripts/generate_brats_splits.py --data-dir brats_random --subjects-per-split 5
"""

import argparse
import math
from pathlib import Path

import pandas as pd


def discover_subjects(data_dir: Path) -> list[str]:
    """Find preprocessed BraTS subject IDs by looking for GT files.

    GT files match ``{subject_id}.nii.gz`` (no underscore suffix).
    """
    subjects = []
    for f in sorted(data_dir.glob("BraTS-*.nii.gz")):
        name = f.name.replace(".nii.gz", "")
        # GT files have no suffix (no underscore after the subject ID pattern)
        if "_" not in name.split("BraTS-GLI-")[-1]:
            subjects.append(name)
    return subjects


def main():
    parser = argparse.ArgumentParser(
        description="Generate BraTS split files for batch training."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("brats_random"),
        help="Directory with preprocessed BraTS subjects (default: brats_random)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for split files (default: <data-dir>/splits)",
    )
    parser.add_argument(
        "--subjects-per-split",
        type=int,
        default=5,
        help="Number of subjects per split file (default: 5)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = discover_subjects(args.data_dir)
    if not subjects:
        print(f"No BraTS subjects found in {args.data_dir}")
        return

    print(f"Found {len(subjects)} subjects in {args.data_dir}")

    n_splits = math.ceil(len(subjects) / args.subjects_per_split)
    for i in range(n_splits):
        start = i * args.subjects_per_split
        end = min(start + args.subjects_per_split, len(subjects))
        chunk = subjects[start:end]

        part_num = f"{i + 1:02d}"
        tsv_path = output_dir / f"brats_part{part_num}.tsv"
        df = pd.DataFrame({"subject_id": chunk})
        df.to_csv(tsv_path, sep="\t", index=False)
        print(f"  {tsv_path.name}: {len(chunk)} subjects")

    print(f"Wrote {n_splits} split files to {output_dir}")


if __name__ == "__main__":
    main()
