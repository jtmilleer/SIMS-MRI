"""Preprocess BraTS 2023 brain MRI scans for SIMS-MRI training."""

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import nibabel.processing as nip
import numpy as np

from sims_mri.utils import get_rotation_matrix_numpy

RESAMPLING_FACTORS = {
    "axial": (1, 1, 4),
    "coronal": (1, 4, 1),
    "sagittal": (4, 1, 1),
}


def resample_nib(img, voxel_spacing=(1, 1, 1), order=3):
    """Resample a NIfTI image to the given voxel spacing."""
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    new_shp = tuple(
        np.rint(
            [
                shp[0] * zms[0] / voxel_spacing[0],
                shp[1] * zms[1] / voxel_spacing[1],
                shp[2] * zms[2] / voxel_spacing[2],
            ]
        ).astype(int)
    )
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=0)
    print("[*] Image resampled to voxel size:", voxel_spacing)
    return new_img


def resample_and_save(input_file, output_file, view):
    """Load a NIfTI, resample to the given view's factor, and save."""
    img = nib.load(input_file)
    voxel_spacing = RESAMPLING_FACTORS.get(view.lower())
    if voxel_spacing is None:
        print(f"Invalid view: {view}. Use axial, sagittal, or coronal.")
        return
    new_img = resample_nib(img, voxel_spacing=voxel_spacing)
    nib.save(new_img, output_file)
    print(f"Resampled image saved as {output_file}")


def preprocess_brats_subject(subject_folder, subject_id, output_root=Path("brats_random"), skip_existing=False):
    output_root.mkdir(parents=True, exist_ok=True)
    fname = f"{subject_id}.nii.gz"
    output_base = output_root / fname
    if skip_existing and output_base.exists():
        print(f"[*] Skipping {subject_id} (already exists)")
        return
    print(f"[*] Preprocessing {subject_id}")

    # Copy T1 contrast-enhanced as ground truth (tumor visible in this sequence).
    t1c_path = subject_folder / f"{subject_id}-t1c.nii.gz"
    shutil.copyfile(t1c_path, output_base)

    # Copy segmentation mask for evaluation.
    seg_path = subject_folder / f"{subject_id}-seg.nii.gz"
    seg_output = output_root / f"{subject_id}_seg.nii.gz"
    shutil.copyfile(seg_path, seg_output)

    # Axial, coronal, and sagittal LR views from the original scan.
    for view in ["axial", "coronal", "sagittal"]:
        output_fname = str(output_base).replace(".nii.gz", f"_{view}_LR.nii.gz")
        resample_and_save(input_file=str(output_base), output_file=output_fname, view=view)

    # Padded baseline (for CNN based baseline)
    padded_target_shape = (256, 256, 256)
    img = nib.load(output_base)
    data = np.asanyarray(img.dataobj)
    affine = img.affine
    padded = np.zeros(padded_target_shape, dtype=data.dtype)
    src_shape = data.shape

    slices = []
    for axis in range(3):
        target_dim = padded_target_shape[axis]
        src_dim = src_shape[axis]
        if src_dim > target_dim:
            raise ValueError(
                f"Source dimension {src_dim} exceeds target {target_dim} along axis {axis} "
                f"for volume {fname}"
            )
        pad_before = (target_dim - src_dim) // 2
        pad_after = target_dim - src_dim - pad_before
        slices.append(slice(pad_before, target_dim - pad_after))

    padded[tuple(slices)] = data
    padded_fname = str(output_base).replace(".nii.gz", "_padded.nii.gz")
    nib.save(nib.Nifti1Image(padded, affine=affine), padded_fname)

    for view in ["axial", "sagittal", "coronal"]:
        output_fname = padded_fname.replace(".nii.gz", f"_{view}_LR.nii.gz")
        resample_and_save(input_file=padded_fname, output_file=output_fname, view=view)

    # Per-subject RNG for reproducible random transforms
    img = nib.load(output_base)
    rng = np.random.RandomState(hash(subject_id) % (2**31))
    base = str(output_base)

    # Translation (affine shift)
    translate_vec = np.append(rng.uniform(-8, 8, size=3), 0)
    print(f"[*] Random translation for {subject_id}: {translate_vec[:3]} mm")
    new_affine = img.affine.copy()
    new_affine[:, -1] += translate_vec
    nib.save(
        nib.Nifti1Image(np.asanyarray(img.dataobj), affine=new_affine),
        base.replace(".nii.gz", "_translated.nii.gz"),
    )

    # Translation (voxel shift, affine unchanged)
    orig_data = np.array(img.dataobj)
    shifted = np.zeros(orig_data.shape)
    shifted[8:, 8:, 8:] = orig_data[:-8, :-8, :-8]
    nib.save(
        nib.Nifti1Image(shifted, affine=img.affine),
        base.replace(".nii.gz", "_translated2.nii.gz"),
    )

    # Rotation only
    theta_x = rng.uniform(-0.1, 0.1)
    print(f"[*] Random x-axis rotation for {subject_id}: {theta_x:.6f} rad")
    rotation_mat = get_rotation_matrix_numpy(theta_x, 0.0, 0.0)
    rotated_affine = img.affine.copy().dot(rotation_mat)
    nib.save(
        nib.Nifti1Image(np.asanyarray(img.dataobj), affine=rotated_affine),
        base.replace(".nii.gz", "_rotated.nii.gz"),
    )

    # Rigid (rotation + translation)
    rigid_affine = img.affine.copy().dot(rotation_mat)
    rigid_affine[:, -1] += translate_vec
    nib.save(
        nib.Nifti1Image(np.asanyarray(img.dataobj), affine=rigid_affine),
        base.replace(".nii.gz", "_rigid.nii.gz"),
    )

    # Coronal and sagittal LR views for each transform variant.
    for suffix in ["_translated", "_rotated", "_translated2", "_rigid"]:
        variant_fname = base.replace(".nii.gz", f"{suffix}.nii.gz")
        for view in ["coronal", "sagittal"]:
            output_fname = variant_fname.replace(".nii.gz", f"_{view}_LR.nii.gz")
            resample_and_save(input_file=variant_fname, output_file=output_fname, view=view)

    # Mean baseline from axial + coronal LR resampled
    axial_fname = base.replace(".nii.gz", "_axial_LR.nii.gz")
    coronal_fname = base.replace(".nii.gz", "_coronal_LR.nii.gz")
    axial_img = resample_nib(nib.load(axial_fname), voxel_spacing=(1, 1, 1))
    coronal_img = resample_nib(nib.load(coronal_fname), voxel_spacing=(1, 1, 1))

    nib.save(axial_img, axial_fname.replace("_LR.nii.gz", "_LR_resampled.nii.gz"))
    nib.save(coronal_img, coronal_fname.replace("_LR.nii.gz", "_LR_resampled.nii.gz"))

    # BraTS Z-dim (155) isn't divisible by 4, so the axial LR resampled back
    # to 1mm has 156 slices (ceil(155/4)*4) while coronal keeps 155. Crop to
    # the min overlapping shape to discard the extra boundary voxel.
    # Alternative: resample both onto the original GT grid via
    # nip.resample_from_to(img, gt_img) for exact alignment.
    axial_data = np.asanyarray(axial_img.dataobj)
    coronal_data = np.asanyarray(coronal_img.dataobj)
    min_shape = tuple(min(a, c) for a, c in zip(axial_data.shape, coronal_data.shape))
    mean_data = (
        axial_data[:min_shape[0], :min_shape[1], :min_shape[2]]
        + coronal_data[:min_shape[0], :min_shape[1], :min_shape[2]]
    ) / 2
    mean_img = nib.Nifti1Image(mean_data, axial_img.affine, axial_img.header)
    nib.save(mean_img, base.replace(".nii.gz", "_axial_LR_coronal_LR_mean.nii.gz"))


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess BraTS 2023 brain MRI scans for SIMS-MRI training."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("brats-data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"),
        help="Root directory of the BraTS training dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("brats_random"),
        help="Output directory for preprocessed subjects (default: brats_random)",
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=25,
        help="Number of subjects to randomly select (default: 25)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subject selection (default: 42)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip subjects whose GT already exists in output-root",
    )
    args = parser.parse_args()

    # Enumerate subject folders.
    subject_folders = sorted(
        [d for d in args.dataset_root.iterdir() if d.is_dir()]
    )
    print(f"[*] Found {len(subject_folders)} subjects in {args.dataset_root}")

    # Select N subjects randomly.
    rng = np.random.RandomState(args.seed)
    selected_indices = rng.choice(len(subject_folders), size=args.num_subjects, replace=False)
    selected_folders = [subject_folders[i] for i in sorted(selected_indices)]

    print(f"[*] Selected {len(selected_folders)} subjects (seed={args.seed}):")
    for folder in selected_folders:
        print(f"    {folder.name}")

    for folder in selected_folders:
        try:
            preprocess_brats_subject(
                subject_folder=folder,
                subject_id=folder.name,
                output_root=args.output_root,
                skip_existing=args.skip_existing,
            )
        except Exception as exc:
            print(f"[!] Failed to process {folder.name}: {exc}")


if __name__ == "__main__":
    main()
