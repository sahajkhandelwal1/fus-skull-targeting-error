"""Paper-quality figure comparing the mr-to-pct tool's input T1 MRI against
its output pseudo-CT, across all three anatomical planes, for the example
subject used to verify the environment (Immediate Next Action #2).

Run this after third_party/mr-to-pct's own example conversion has produced
example_data_prep.nii and example_pct_output.nii in that directory.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fus_targeting.viz.style import ANATOMY_CMAP, CHROME, save_figure  # noqa: E402

MR_TO_PCT_DIR = Path(__file__).resolve().parents[1] / "third_party" / "mr-to-pct"


def mid_slices(volume: np.ndarray) -> list[tuple[str, np.ndarray]]:
    ax, cor, sag = (s // 2 for s in volume.shape)
    return [
        ("Axial", np.rot90(volume[:, :, ax])),
        ("Coronal", np.rot90(volume[:, cor, :])),
        ("Sagittal", np.rot90(volume[sag, :, :])),
    ]


def main() -> None:
    mr = nib.load(MR_TO_PCT_DIR / "example_data_prep.nii").get_fdata()
    pct = nib.load(MR_TO_PCT_DIR / "example_pct_output.nii").get_fdata()

    mr_slices = mid_slices(mr)
    pct_slices = mid_slices(pct)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))

    for col, ((label, mr_slice), (_, pct_slice)) in enumerate(zip(mr_slices, pct_slices)):
        axes[0, col].imshow(mr_slice, cmap=ANATOMY_CMAP)
        axes[0, col].set_title(label, fontsize=11)
        axes[0, col].axis("off")

        im = axes[1, col].imshow(pct_slice, cmap=ANATOMY_CMAP, vmin=-1000, vmax=2000)
        axes[1, col].axis("off")

    axes[0, 0].text(
        -0.08, 0.5, "Input T1 MRI\n(bias-corrected, masked)",
        transform=axes[0, 0].transAxes, rotation=90,
        va="center", ha="center", fontsize=10, color=CHROME["ink_secondary"],
    )
    axes[1, 0].text(
        -0.08, 0.5, "Output pseudo-CT\n(mr-to-pct)",
        transform=axes[1, 0].transAxes, rotation=90,
        va="center", ha="center", fontsize=10, color=CHROME["ink_secondary"],
    )

    cbar = fig.colorbar(im, ax=axes[1, :], location="bottom", fraction=0.05, pad=0.06, aspect=40)
    cbar.set_label("Pseudo-CT intensity (Hounsfield-like units)", color=CHROME["ink_secondary"])

    fig.suptitle("MRI -> Pseudo-CT Conversion (mr-to-pct environment check)", fontsize=13, y=0.96)

    saved_paths = save_figure(fig, "mr_to_pct/example_subject_comparison")
    print(f"figure saved: {saved_paths[0]}")


if __name__ == "__main__":
    main()
