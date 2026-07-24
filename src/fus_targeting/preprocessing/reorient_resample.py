"""Reorient a raw MRI to RAS+ and resample to 1mm isotropic voxels -- the
input specification mr-to-pct expects (third_party/mr-to-pct/README.md,
"Input to network"). Raw IXI scans are not RAS+ or 1mm isotropic, so this
has to happen before mr-to-pct can be run on them.

Uses trilinear (order=1) resampling, not cubic spline. Cubic spline
over-smooths the thin cortical bone rim (~1-2mm), which collapsed the
pseudo-CT's peak Hounsfield value from ~2500 to ~800 in testing on subject
IXI002 -- confirmed by comparing order=1 vs order=3 output through
mr-to-pct directly. This matters because acoustic properties (Phase C) are
scaled from that peak, so an artificially low peak would understate skull
density/sound-speed/attenuation across the whole simulation.
"""

from pathlib import Path

import nibabel as nib
import nibabel.processing


def reorient_and_resample(input_path: Path, output_path: Path, voxel_size_mm: float = 1.0) -> nib.Nifti1Image:
    img = nib.load(input_path)
    canonical = nib.as_closest_canonical(img)
    resampled = nibabel.processing.resample_to_output(canonical, voxel_sizes=voxel_size_mm, order=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(resampled, output_path)
    return resampled
