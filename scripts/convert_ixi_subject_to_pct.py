"""Phase B of the one-subject pipeline: run mr-to-pct on the real,
reoriented/resampled IXI subject from Phase A (not the tool's own bundled
example), and save a before/after diagnostic figure.

Run with the mr-to-pct environment (.venv-mrtopct, Python 3.10):
  source .venv-mrtopct/bin/activate
  python3 scripts/convert_ixi_subject_to_pct.py [subject_id]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MR_TO_PCT_DIR = ROOT / "third_party" / "mr-to-pct"
sys.path.insert(0, str(MR_TO_PCT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from utils.infer_funcs import do_mr_to_pct  # noqa: E402
from fus_targeting.viz.style import ANATOMY_CMAP, CHROME, save_figure  # noqa: E402

INTERIM_DIR = ROOT / "data" / "interim"


def mid_slices(volume: np.ndarray) -> list[tuple[str, np.ndarray]]:
    ax, cor, sag = (s // 2 for s in volume.shape)
    return [
        ("Axial", np.rot90(volume[:, :, ax])),
        ("Coronal", np.rot90(volume[:, cor, :])),
        ("Sagittal", np.rot90(volume[sag, :, :])),
    ]


def main() -> None:
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "IXI002"
    input_path = INTERIM_DIR / subject_id / "t1_ras_1mm.nii"
    output_path = INTERIM_DIR / subject_id / "pct.nii"

    if not input_path.exists():
        raise SystemExit(f"{input_path} not found -- run scripts/prepare_ixi_subject.py {subject_id} first")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    saved_model = torch.load(MR_TO_PCT_DIR / "pretrained_net_final_20220825.pth", map_location=device)

    print(f"Subject: {subject_id}")
    print(f"Device:  {device}")
    do_mr_to_pct(str(input_path), str(output_path), saved_model, device, prep_t1=True, plot_mrct=False)

    mr = nib.load(str(input_path).replace(".nii", "_prep.nii")).get_fdata()
    pct = nib.load(output_path).get_fdata()

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
        -0.08, 0.5, "Input T1 MRI\n(prepped)",
        transform=axes[0, 0].transAxes, rotation=90,
        va="center", ha="center", fontsize=10, color=CHROME["ink_secondary"],
    )
    axes[1, 0].text(
        -0.08, 0.5, "Output pseudo-CT",
        transform=axes[1, 0].transAxes, rotation=90,
        va="center", ha="center", fontsize=10, color=CHROME["ink_secondary"],
    )

    cbar = fig.colorbar(im, ax=axes[1, :], location="bottom", fraction=0.05, pad=0.06, aspect=40)
    cbar.set_label("Pseudo-CT intensity (Hounsfield-like units)", color=CHROME["ink_secondary"])

    fig.suptitle(f"Phase B: MRI -> Pseudo-CT ({subject_id})", fontsize=13, y=0.96)

    saved_paths = save_figure(fig, f"pipeline_one_subject/{subject_id}_phase_b_mr_to_pct")
    print(f"Figure:  {saved_paths[0]}")


if __name__ == "__main__":
    main()
