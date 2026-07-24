"""Reorient a raw MRI to RAS+ and resample to 1mm isotropic voxels -- the
input specification mr-to-pct expects (third_party/mr-to-pct/README.md,
"Input to network"). Raw IXI scans are not RAS+ or 1mm isotropic, so this
has to happen before mr-to-pct can be run on them.
"""

from pathlib import Path

import nibabel as nib
import nibabel.processing


def reorient_and_resample(input_path: Path, output_path: Path, voxel_size_mm: float = 1.0) -> nib.Nifti1Image:
    img = nib.load(input_path)
    canonical = nib.as_closest_canonical(img)
    resampled = nibabel.processing.resample_to_output(canonical, voxel_sizes=voxel_size_mm, order=3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(resampled, output_path)
    return resampled
