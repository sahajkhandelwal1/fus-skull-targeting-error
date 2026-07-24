"""Phase F of the one-subject pipeline: extract the simple, fast skull
descriptors at the beam entry point (density, thickness, curvature, entry
angle, SDR) -- the input features the eventual predictive model uses.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fus_targeting.features.skull_descriptors import compute_skull_descriptors  # noqa: E402
from fus_targeting.simulation.skull_simulation import (  # noqa: E402
    KNOWN_TARGET_VOXELS,
    find_lateral_transducer_position,
)
from fus_targeting.viz.style import CHROME, legend_on_image, save_figure, sequential_cmap  # noqa: E402

INTERIM_DIR = ROOT / "data" / "interim"
CONFIG_PATH = ROOT / "configs" / "simulation_matrix.yaml"


def display_xy(voxel_idx: np.ndarray, shape: tuple) -> tuple:
    r, a, _ = voxel_idx
    return r, shape[1] - 1 - a


def main() -> None:
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "IXI002"

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    aperture_diameter_mm = config["baseline"]["transducer"]["aperture_diameter_mm"]

    pct_hu = nib.load(INTERIM_DIR / subject_id / "pct.nii").get_fdata()
    acoustic = np.load(INTERIM_DIR / subject_id / "acoustic_maps.npz")
    skull_mask = acoustic["skull_mask"]

    target_voxel = KNOWN_TARGET_VOXELS[subject_id]
    roc_mm = config["baseline"]["transducer"]["radius_of_curvature_mm"]
    transducer_voxel = find_lateral_transducer_position(skull_mask, target_voxel, roc_mm)

    descriptors = compute_skull_descriptors(
        pct_hu, skull_mask, target_voxel, transducer_voxel, aperture_diameter_mm,
    )

    result = {
        "subject_id": subject_id,
        "entry_point_voxel": descriptors.entry_point_voxel.tolist(),
        "thickness_mm": descriptors.thickness_mm,
        "density_hu_mean": descriptors.density_hu_mean,
        "density_hu_max": descriptors.density_hu_max,
        "sdr": descriptors.sdr,
        "entry_angle_deg": descriptors.entry_angle_deg,
        "radius_of_curvature_mm": descriptors.radius_of_curvature_mm,
    }
    print(json.dumps(result, indent=2))

    out_path = INTERIM_DIR / subject_id / "skull_descriptors.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    mid = int(round(target_voxel[2]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].imshow(np.rot90(skull_mask[:, :, mid]), cmap="gray")
    axes[0].scatter(*display_xy(target_voxel, skull_mask.shape), c=CHROME["good"], marker="+", s=200, label="Target")
    axes[0].scatter(*display_xy(transducer_voxel, skull_mask.shape), c="#e34948", marker="o", s=60, label="Transducer")
    axes[0].scatter(*display_xy(descriptors.entry_point_voxel, skull_mask.shape), c="#eda100", marker="*", s=180, label="Entry point")
    axes[0].set_title("Beam geometry + entry point", fontsize=11)
    legend_on_image(axes[0])
    axes[0].axis("off")

    axes[1].imshow(np.rot90(pct_hu[:, :, mid]), cmap="gray", vmin=-1000, vmax=2000)
    axes[1].scatter(*display_xy(descriptors.entry_point_voxel, pct_hu.shape), c="#eda100", marker="*", s=180)
    zoom = 40
    ep = descriptors.entry_point_voxel
    axes[1].set_xlim(max(0, ep[0] - zoom), min(pct_hu.shape[0], ep[0] + zoom))
    axes[1].set_ylim(min(pct_hu.shape[1], pct_hu.shape[1] - ep[1] + zoom), max(0, pct_hu.shape[1] - ep[1] - zoom))
    axes[1].set_title("Entry point close-up (pseudo-CT)", fontsize=11)
    axes[1].axis("off")

    summary = (
        f"thickness={descriptors.thickness_mm:.1f}mm  "
        f"density(mean/max)={descriptors.density_hu_mean:.0f}/{descriptors.density_hu_max:.0f}HU\n"
        f"SDR={descriptors.sdr:.2f}  entry angle={descriptors.entry_angle_deg:.1f}deg  "
        f"ROC={descriptors.radius_of_curvature_mm:.0f}mm"
    )
    fig.suptitle(f"Phase F: skull descriptors ({subject_id})\n{summary}", fontsize=12)
    fig.tight_layout()

    saved_paths = save_figure(fig, f"pipeline_one_subject/{subject_id}_phase_f_skull_descriptors")
    print(f"Figure:  {saved_paths[0]}")
    print(f"Labels:  {out_path}")


if __name__ == "__main__":
    main()
