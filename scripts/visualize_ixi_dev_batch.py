"""Paper-quality figure showing the 5-subject IXI dev batch (Immediate Next
Action #3): confirms every scan loaded correctly and shows intact skull
anatomy (not skull-stripped) before they're fed into the pipeline.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fus_targeting.viz.style import ANATOMY_CMAP, CHROME, save_figure  # noqa: E402

DEV_BATCH_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "ixi_t1_dev"


def main() -> None:
    files = sorted(DEV_BATCH_DIR.glob("*.nii"))
    if not files:
        raise SystemExit(f"No .nii files found in {DEV_BATCH_DIR} -- run scripts/fetch_ixi_dev_subjects.sh first")

    fig, axes = plt.subplots(1, len(files), figsize=(3.2 * len(files), 3.6))
    if len(files) == 1:
        axes = [axes]

    for ax, f in zip(axes, files):
        img = nib.load(f)
        data = img.get_fdata()
        mid = data.shape[2] // 2
        ax.imshow(np.rot90(data[:, :, mid]), cmap=ANATOMY_CMAP)
        subject_id = f.name.split("-")[0]
        zooms = img.header.get_zooms()[:3]
        ax.set_title(subject_id, fontsize=11, color=CHROME["ink_primary"])
        ax.set_xlabel(f"{zooms[0]:.2f}x{zooms[1]:.2f}x{zooms[2]:.2f} mm", fontsize=8, color=CHROME["ink_muted"])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"IXI dev batch: {len(files)} raw T1 MRI subjects (mid-axial slice, skull intact)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    saved_paths = save_figure(fig, "ixi_download/dev_subjects_grid")
    print(f"figure saved: {saved_paths[0]}")


if __name__ == "__main__":
    main()
