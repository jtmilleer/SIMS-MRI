"""
Classical interpolation baseline for BraTS (skull-stripped) data WITHOUT registration.

Supports arbitrary view pairs (e.g., axial+coronal, axial+sagittal) and applies
brain masking to zero out background predictions in skull-stripped outputs.
Expects unrotated LR scans that share the same coordinate system.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import string
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import SimpleITK as sitk
import yaml

SUPPORTED_INTERPOLATORS = {
    "linear": sitk.sitkLinear,
    "bspline3": sitk.sitkBSpline,
}

INTERPOLATOR_ALIASES = {
    "linear": "linear",
    "lin": "linear",
    "bspline": "bspline3",
    "bspline3": "bspline3",
    "bicubic": "bspline3",
}


def generate_unique_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{timestamp}_{random_str}"


def nested_get(config: dict, *keys: str, default=None):
    current = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_interpolator_list(spec: str) -> List[str]:
    tokens = [tok.strip().lower() for tok in spec.split(",") if tok.strip()]
    if not tokens:
        raise ValueError("No interpolators provided in --interp_variants.")

    out: List[str] = []
    for token in tokens:
        canonical = INTERPOLATOR_ALIASES.get(token)
        if canonical is None:
            supported = ", ".join(sorted(SUPPORTED_INTERPOLATORS.keys()))
            raise ValueError(
                f"Unsupported interpolator '{token}'. Supported: {supported} "
                "(aliases: bspline, bicubic)."
            )
        if canonical not in out:
            out.append(canonical)
    return out


def parse_view_pairs(spec: str) -> List[Tuple[str, str]]:
    """Parse view pair spec like 'axial_LR:coronal_LR,axial_LR:sagittal_LR'."""
    pairs = []
    for pair_str in spec.split(","):
        pair_str = pair_str.strip()
        if not pair_str:
            continue
        parts = pair_str.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid view pair '{pair_str}'. Expected format: 'view1:view2'."
            )
        pairs.append((parts[0].strip(), parts[1].strip()))
    if not pairs:
        raise ValueError("No view pairs provided.")
    return pairs


def load_subjects_from_tsv(tsv_path: str) -> List[str]:
    """Load subject IDs from a TSV file (column: subject_id)."""
    subjects = []
    with open(tsv_path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row.get("subject_id", "").strip()
            if sid:
                subjects.append(sid)
    return subjects


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classical baseline for BraTS isotropic MRI reconstruction using "
            "interpolation fusion only (no registration). Supports brain masking."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--subject_id",
        type=str,
        default=None,
        help="Single subject ID (default: DATASET.SUBJECT_ID from config).",
    )
    parser.add_argument(
        "--subjects_tsv",
        type=str,
        default=None,
        help="TSV file with subject_id column for batch execution.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Default: runs/<unique_id>/baseline_brats",
    )
    parser.add_argument(
        "--view_pairs",
        type=str,
        default="axial_LR:coronal_LR",
        help=(
            "Comma-separated view pairs as view1:view2. "
            "Default: 'axial_LR:coronal_LR'. "
            "Example: 'axial_LR:coronal_LR,axial_LR:sagittal_LR'"
        ),
    )
    parser.add_argument(
        "--fusion_weight_view1",
        type=float,
        default=0.5,
        help="Fusion weight for view1 isotropic volume (default: 0.5).",
    )
    parser.add_argument(
        "--fusion_weight_view2",
        type=float,
        default=0.5,
        help="Fusion weight for view2 isotropic volume (default: 0.5).",
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        default=1.0,
        help="Isotropic output spacing in mm (default: 1.0).",
    )
    parser.add_argument(
        "--interp_variants",
        type=str,
        default="linear,bspline3",
        help="Comma-separated interpolator variants (default: linear,bspline3).",
    )
    parser.add_argument(
        "--no_skull_stripped",
        action="store_true",
        help="Disable brain masking (default: masking enabled for BraTS).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: config TRAINING.SEED or 12).",
    )
    return parser.parse_args()


def resolve_subject_volume_path(image_dir: Path, subject_id: str, suffix: str) -> Path:
    suffix = suffix.replace(".nii.gz", "").replace(".nii", "")
    candidate_nii_gz = image_dir / f"{subject_id}_{suffix}.nii.gz"
    candidate_nii = image_dir / f"{subject_id}_{suffix}.nii"
    if candidate_nii_gz.exists():
        return candidate_nii_gz
    if candidate_nii.exists():
        return candidate_nii
    return candidate_nii_gz


def resolve_gt_path(image_dir: Path, subject_id: str) -> Path:
    """Resolve ground truth NIfTI path for brain mask derivation."""
    candidate = image_dir / f"{subject_id}.nii.gz"
    if candidate.exists():
        return candidate
    candidate_nii = image_dir / f"{subject_id}.nii"
    if candidate_nii.exists():
        return candidate_nii
    return candidate


def ensure_float_image(image: sitk.Image, name: str) -> sitk.Image:
    if image.GetDimension() != 3:
        raise ValueError(f"{name} must be 3D, got dimension={image.GetDimension()}")
    return sitk.Cast(image, sitk.sitkFloat32)


def make_isotropic_reference(
    fixed_image: sitk.Image, target_spacing: float
) -> sitk.Image:
    if target_spacing <= 0:
        raise ValueError(f"target_spacing must be > 0, got {target_spacing}")

    in_spacing = fixed_image.GetSpacing()
    in_size = fixed_image.GetSize()
    out_size = [
        max(1, int(round(in_size[i] * (in_spacing[i] / target_spacing))))
        for i in range(3)
    ]

    reference = sitk.Image(out_size, sitk.sitkFloat32)
    reference.SetOrigin(fixed_image.GetOrigin())
    reference.SetDirection(fixed_image.GetDirection())
    reference.SetSpacing((target_spacing, target_spacing, target_spacing))
    return reference


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    transform: sitk.Transform,
    interpolator: int,
    default_value: float = 0.0,
    output_pixel_type: int = sitk.sitkFloat32,
) -> sitk.Image:
    return sitk.Resample(
        image, reference, transform, interpolator, default_value, output_pixel_type,
    )


def make_unit_mask_like(image: sitk.Image) -> sitk.Image:
    arr = np.ones(tuple(reversed(image.GetSize())), dtype=np.uint8)
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(image)
    return mask


def normalize_weights(weight1: float, weight2: float) -> Tuple[float, float]:
    if weight1 < 0 or weight2 < 0:
        raise ValueError("Fusion weights must be non-negative.")
    total = weight1 + weight2
    if total <= 0:
        raise ValueError("At least one fusion weight must be > 0.")
    return weight1 / total, weight2 / total


def fuse_arrays(
    view1_np: np.ndarray,
    view2_np: np.ndarray,
    view1_mask_np: np.ndarray,
    view2_mask_np: np.ndarray,
    weight1: float,
    weight2: float,
) -> Tuple[np.ndarray, Dict[str, int]]:
    valid1 = view1_mask_np > 0
    valid2 = view2_mask_np > 0
    both = valid1 & valid2
    only1 = valid1 & ~valid2
    only2 = valid2 & ~valid1
    neither = ~(valid1 | valid2)

    fused = np.zeros_like(view1_np, dtype=np.float32)
    fused[both] = weight1 * view1_np[both] + weight2 * view2_np[both]
    fused[only1] = view1_np[only1]
    fused[only2] = view2_np[only2]
    fused[neither] = 0.0

    counts = {
        "total_voxels": int(fused.size),
        "view1_valid_voxels": int(valid1.sum()),
        "view2_valid_voxels": int(valid2.sum()),
        "both_valid_voxels": int(both.sum()),
        "only_view1_voxels": int(only1.sum()),
        "only_view2_voxels": int(only2.sum()),
        "neither_valid_voxels": int(neither.sum()),
    }
    return fused, counts


def create_brain_mask(
    gt_image: sitk.Image, reference: sitk.Image
) -> Tuple[np.ndarray, int, int]:
    """Create brain mask from GT NIfTI, resampled to isotropic reference grid.

    Returns (mask_np, foreground_count, total_count).
    """
    # Threshold GT > 0 to get binary brain mask
    brain_mask = sitk.BinaryThreshold(gt_image, lowerThreshold=1e-6, upperThreshold=1e10)
    # Resample mask to isotropic reference with nearest-neighbor
    mask_iso = resample_to_reference(
        image=brain_mask,
        reference=reference,
        transform=sitk.Transform(3, sitk.sitkIdentity),
        interpolator=sitk.sitkNearestNeighbor,
        default_value=0,
        output_pixel_type=sitk.sitkUInt8,
    )
    mask_np = sitk.GetArrayFromImage(mask_iso).astype(np.float32)
    fg_count = int((mask_np > 0).sum())
    total_count = int(mask_np.size)
    return mask_np, fg_count, total_count


def save_image(image: sitk.Image, path: Path) -> None:
    sitk.WriteImage(image, str(path))


def image_to_np(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image).astype(np.float32)


def np_to_image(array_zyx: np.ndarray, reference: sitk.Image) -> sitk.Image:
    out = sitk.GetImageFromArray(array_zyx.astype(np.float32))
    out.CopyInformation(reference)
    return out


def run_single_subject(
    subject_id: str,
    image_dir_path: Path,
    output_dir: Path,
    view_pairs: List[Tuple[str, str]],
    interp_variants: List[str],
    weight1: float,
    weight2: float,
    target_spacing: float,
    skull_stripped: bool,
) -> Dict:
    """Run baseline interpolation for a single subject across all view pairs."""
    print(f"\n{'=' * 60}")
    print(f"Subject: {subject_id}")
    print(f"{'=' * 60}")

    subject_output_dir = output_dir / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve GT path for brain masking and isotropic reference grid
    gt_path = resolve_gt_path(image_dir_path, subject_id)
    if not gt_path.exists():
        print(f"  WARNING: GT not found at {gt_path}, skipping subject.")
        return {"subject_id": subject_id, "brain_mask_stats": {}, "view_pairs": {}}
    gt_image = ensure_float_image(sitk.ReadImage(str(gt_path)), "gt")

    brain_mask_np = None
    brain_mask_ref_size = None
    mask_stats = {}
    all_view_pair_results = {}

    for view1_suffix, view2_suffix in view_pairs:
        pair_label = f"{view1_suffix}+{view2_suffix}"
        print(f"\n--- View pair: {pair_label} ---")

        view1_path = resolve_subject_volume_path(image_dir_path, subject_id, view1_suffix)
        view2_path = resolve_subject_volume_path(image_dir_path, subject_id, view2_suffix)

        if not view1_path.exists():
            print(f"  SKIP: {view1_suffix} not found at {view1_path}")
            continue
        if not view2_path.exists():
            print(f"  SKIP: {view2_suffix} not found at {view2_path}")
            continue

        view1_image = ensure_float_image(sitk.ReadImage(str(view1_path)), view1_suffix)
        view2_image = ensure_float_image(sitk.ReadImage(str(view2_path)), view2_suffix)

        print(
            f"  {view1_suffix} size/spacing: {view1_image.GetSize()} / "
            f"{tuple(round(v, 4) for v in view1_image.GetSpacing())}"
        )
        print(
            f"  {view2_suffix} size/spacing: {view2_image.GetSize()} / "
            f"{tuple(round(v, 4) for v in view2_image.GetSpacing())}"
        )

        # Build isotropic reference from GT (not view1) to match GT origin and size
        isotropic_ref = make_isotropic_reference(gt_image, target_spacing)
        identity = sitk.Transform(3, sitk.sitkIdentity)

        # Create or recreate brain mask when isotropic reference size changes
        ref_size = isotropic_ref.GetSize()
        if skull_stripped and (brain_mask_np is None or brain_mask_ref_size != ref_size):
            brain_mask_np, fg_count, total_count = create_brain_mask(gt_image, isotropic_ref)
            brain_mask_ref_size = ref_size
            mask_stats = {
                "foreground_voxels": fg_count,
                "total_voxels": total_count,
                "foreground_fraction": round(fg_count / total_count, 4),
            }
            print(f"  Brain mask: {fg_count}/{total_count} foreground voxels "
                  f"({mask_stats['foreground_fraction']:.1%})")

        # Validity masks for fusion
        view1_mask = make_unit_mask_like(view1_image)
        view2_mask = make_unit_mask_like(view2_image)
        view1_mask_iso = resample_to_reference(
            view1_mask, isotropic_ref, identity,
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )
        view2_mask_iso = resample_to_reference(
            view2_mask, isotropic_ref, identity,
            sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8,
        )
        view1_mask_np = image_to_np(view1_mask_iso)
        view2_mask_np = image_to_np(view2_mask_iso)

        pair_results = {}

        for variant in interp_variants:
            interpolator = SUPPORTED_INTERPOLATORS[variant]
            print(f"  Interpolator: {variant}")

            view1_iso = resample_to_reference(
                view1_image, isotropic_ref, identity, interpolator,
            )
            view2_iso = resample_to_reference(
                view2_image, isotropic_ref, identity, interpolator,
            )

            view1_iso_np = image_to_np(view1_iso)
            view2_iso_np = image_to_np(view2_iso)

            fused_np, counts = fuse_arrays(
                view1_iso_np, view2_iso_np,
                view1_mask_np, view2_mask_np,
                weight1, weight2,
            )

            # Apply brain mask
            if skull_stripped and brain_mask_np is not None:
                bg_mask = brain_mask_np <= 0
                view1_iso_np[bg_mask] = 0.0
                view2_iso_np[bg_mask] = 0.0
                fused_np[bg_mask] = 0.0

            # Short labels for filenames
            v1_short = view1_suffix.replace("_LR", "")
            v2_short = view2_suffix.replace("_LR", "")

            view1_fname = f"{subject_id}_baseline_{v1_short}_iso_{variant}.nii.gz"
            view2_fname = f"{subject_id}_baseline_{v2_short}_iso_{variant}.nii.gz"
            fused_fname = f"{subject_id}_baseline_{v1_short}_{v2_short}_fused_{variant}.nii.gz"

            save_image(np_to_image(view1_iso_np, isotropic_ref),
                       subject_output_dir / view1_fname)
            save_image(np_to_image(view2_iso_np, isotropic_ref),
                       subject_output_dir / view2_fname)
            save_image(np_to_image(fused_np, isotropic_ref),
                       subject_output_dir / fused_fname)

            pair_results[variant] = {
                "fusion_stats": counts,
                "outputs": {
                    "view1_iso": str(subject_output_dir / view1_fname),
                    "view2_iso": str(subject_output_dir / view2_fname),
                    "fused": str(subject_output_dir / fused_fname),
                },
            }
            print(
                f"    Saved: {fused_fname} "
                f"(both_valid={counts['both_valid_voxels']}, "
                f"neither={counts['neither_valid_voxels']})"
            )

        all_view_pair_results[pair_label] = pair_results

    return {
        "subject_id": subject_id,
        "brain_mask_stats": mask_stats,
        "view_pairs": all_view_pair_results,
    }


def main() -> None:
    args = parse_args()
    run_start = time.time()
    started_at = datetime.now().isoformat()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must parse to a dict. Got: {type(config)}")

    seed = (
        args.seed
        if args.seed is not None
        else int(nested_get(config, "TRAINING", "SEED", default=12))
    )
    random.seed(seed)
    np.random.seed(seed)

    # Resolve subjects
    if args.subjects_tsv:
        subjects = load_subjects_from_tsv(args.subjects_tsv)
        if not subjects:
            raise ValueError(f"No subjects found in {args.subjects_tsv}")
        print(f"Loaded {len(subjects)} subjects from {args.subjects_tsv}")
    else:
        subject_id = args.subject_id or nested_get(config, "DATASET", "SUBJECT_ID")
        if not subject_id:
            raise ValueError(
                "Subject ID required. Provide --subject_id, --subjects_tsv, "
                "or DATASET.SUBJECT_ID in config."
            )
        subjects = [subject_id]

    image_dir = nested_get(config, "SETTINGS", "DIRECTORY")
    if not image_dir:
        raise ValueError("SETTINGS.DIRECTORY is required in config.")
    image_dir_path = Path(image_dir).expanduser().resolve()
    if not image_dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir_path}")

    view_pairs = parse_view_pairs(args.view_pairs)
    interp_variants = parse_interpolator_list(args.interp_variants)
    weight1, weight2 = normalize_weights(
        args.fusion_weight_view1, args.fusion_weight_view2,
    )
    skull_stripped = not args.no_skull_stripped

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (Path("runs") / generate_unique_id() / "baseline_brats").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== BraTS Baseline: interpolation only (no registration) ===")
    print(f"Config: {Path(args.config).resolve()}")
    print(f"Subjects: {len(subjects)}")
    print(f"Image directory: {image_dir_path}")
    print(f"View pairs: {[f'{v1}+{v2}' for v1, v2 in view_pairs]}")
    print(f"Target spacing: {args.target_spacing} mm")
    print(f"Interpolators: {interp_variants}")
    print(f"Fusion weights (normalized): view1={weight1:.3f}, view2={weight2:.3f}")
    print(f"Skull-stripped masking: {skull_stripped}")
    print(f"Output directory: {output_dir}")

    all_results = []
    for subject_id in subjects:
        result = run_single_subject(
            subject_id=subject_id,
            image_dir_path=image_dir_path,
            output_dir=output_dir,
            view_pairs=view_pairs,
            interp_variants=interp_variants,
            weight1=weight1,
            weight2=weight2,
            target_spacing=args.target_spacing,
            skull_stripped=skull_stripped,
        )
        all_results.append(result)

    finished_at = datetime.now().isoformat()
    total_runtime_sec = time.time() - run_start

    metadata = {
        "script_name": "baseline_interpolation_no_registration_brats.py",
        "variant": "no_registration_brats",
        "started_at": started_at,
        "finished_at": finished_at,
        "runtime_sec": total_runtime_sec,
        "config_path": str(Path(args.config).resolve()),
        "seed": seed,
        "settings": {
            "view_pairs": [f"{v1}:{v2}" for v1, v2 in view_pairs],
            "target_spacing_mm": args.target_spacing,
            "interpolators": interp_variants,
            "fusion_weights_normalized": {"view1": weight1, "view2": weight2},
            "skull_stripped": skull_stripped,
        },
        "subjects": all_results,
    }
    metadata_path = output_dir / "baseline_brats_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Processed {len(subjects)} subject(s) in {total_runtime_sec:.2f}s")


if __name__ == "__main__":
    main()
