"""Phase A of the one-subject pipeline: reorient + resample one real IXI dev
subject to RAS+ 1mm isotropic (mr-to-pct's required input format), and save
a before/after diagnostic figure.

Usage: python3 scripts/prepare_ixi_subject.py [subject_prefix]
Defaults to the first subject found in data/raw/ixi_t1_dev/.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fus_targeting.preprocessing.reorient_resample import reorient_and_resample  # noqa: E402
from fus_targeting.viz.style import ANATOMY_CMAP, CHROME, save_figure  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "ixi_t1_dev"
INTERIM_DIR = ROOT / "data" / "interim"


def mid_axial(volume: np.ndarray) -> np.ndarray:
    return np.rot90(volume[:, :, volume.shape[2] // 2])


def main() -> None:
    subject_prefix = sys.argv[1] if len(sys.argv) > 1 else None
    candidates = sorted(RAW_DIR.glob("*.nii"))
    if not candidates:
        raise SystemExit(f"No subjects found in {RAW_DIR} -- run scripts/fetch_ixi_dev_subjects.sh first")

    if subject_prefix:
        matches = [f for f in candidates if f.name.startswith(subject_prefix)]
        if not matches:
            raise SystemExit(f"No file starting with {subject_prefix!r} in {RAW_DIR}")
        input_path = matches[0]
    else:
        input_path = candidates[0]

    subject_id = input_path.name.split("-")[0]
    output_path = INTERIM_DIR / subject_id / "t1_ras_1mm.nii"

    print(f"Subject: {subject_id}")
    print(f"Input:   {input_path.name}")

    orig_img = nib.load(input_path)
    print(f"  original orientation: {nib.aff2axcodes(orig_img.affine)}, "
          f"voxel size: {tuple(round(z, 3) for z in orig_img.header.get_zooms()[:3])}, "
          f"shape: {orig_img.shape}")

    resampled_img = reorient_and_resample(input_path, output_path)
    print(f"  resampled orientation: {nib.aff2axcodes(resampled_img.affine)}, "
          f"voxel size: {tuple(round(z, 3) for z in resampled_img.header.get_zooms()[:3])}, "
          f"shape: {resampled_img.shape}")
    print(f"Output:  {output_path}")

    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(9, 5))
    ax_before.imshow(mid_axial(orig_img.get_fdata()), cmap=ANATOMY_CMAP)
    ax_before.set_title("Before (raw IXI)", fontsize=11)
    ax_before.set_xlabel(
        f"{nib.aff2axcodes(orig_img.affine)} | "
        f"{'x'.join(f'{z:.2f}' for z in orig_img.header.get_zooms()[:3])} mm",
        fontsize=9, color=CHROME["ink_secondary"],
    )
    ax_before.set_xticks([]); ax_before.set_yticks([])

    ax_after.imshow(mid_axial(resampled_img.get_fdata()), cmap=ANATOMY_CMAP)
    ax_after.set_title("After (RAS+, 1mm isotropic)", fontsize=11)
    ax_after.set_xlabel(
        f"{nib.aff2axcodes(resampled_img.affine)} | "
        f"{'x'.join(f'{z:.2f}' for z in resampled_img.header.get_zooms()[:3])} mm",
        fontsize=9, color=CHROME["ink_secondary"],
    )
    ax_after.set_xticks([]); ax_after.set_yticks([])

    fig.suptitle(f"Phase A: reorient + resample ({subject_id})", fontsize=13)
    fig.tight_layout()

    saved_paths = save_figure(fig, f"pipeline_one_subject/{subject_id}_phase_a_reorient_resample")
    print(f"Figure:  {saved_paths[0]}")


if __name__ == "__main__":
    main()
