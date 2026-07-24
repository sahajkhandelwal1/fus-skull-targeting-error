"""Phase D of the one-subject pipeline: run a 3D k-Wave simulation of the
baseline transducer/target condition (single-element 650kHz bowl -> VIM
thalamus) through the subject's skull, and save a diagnostic figure of the
resulting pressure field.

NOTE on targeting: the target voxel (fus_targeting.simulation.skull_simulation
.KNOWN_TARGET_VOXELS) was picked by visual inspection of subject IXI002's
anatomy, not atlas registration -- adequate for proving the pipeline
mechanics work end-to-end on one subject, but real atlas-based targeting is
needed before scaling to the full cohort (flagged in
configs/simulation_matrix.yaml).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fus_targeting.simulation.skull_simulation import prepare_subject_domain, run_through_skull_simulation  # noqa: E402
from fus_targeting.viz.style import CHROME, legend_on_image, save_figure, sequential_cmap  # noqa: E402

INTERIM_DIR = ROOT / "data" / "interim"
CONFIG_PATH = ROOT / "configs" / "simulation_matrix.yaml"


def display_xy(voxel_idx: np.ndarray, shape: tuple) -> tuple:
    """Convert (R,A,S) voxel index to the (x,y) coords used by imshow(rot90(slice))."""
    r, a, _ = voxel_idx
    return r, shape[1] - 1 - a


def main() -> None:
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "IXI002"

    domain, tx = prepare_subject_domain(subject_id, INTERIM_DIR, CONFIG_PATH, half_size=100)
    print(f"Subject: {subject_id}")
    print(f"  target voxel (in crop):     {domain.target_voxel}")
    print(f"  transducer voxel (in crop): {domain.transducer_voxel}")
    print(f"  cropped domain shape: {domain.density.shape}")
    print(f"  skull voxel fraction in crop: {domain.skull_mask.mean():.3%}")

    p_max, target_pos, transducer_pos = run_through_skull_simulation(
        domain,
        dx_m=1e-3,
        freq_hz=tx["freq_hz"],
        roc_m=tx["roc_mm"] * 1e-3,
        diameter_m=tx["diameter_mm"] * 1e-3,
    )

    np.savez(
        INTERIM_DIR / subject_id / "skull_simulation_result.npz",
        p_max=p_max, target_voxel_in_crop=domain.target_voxel,
        transducer_voxel_in_crop=domain.transducer_voxel, crop_origin=domain.crop_origin,
        skull_mask=domain.skull_mask, dx_m=1e-3, freq_hz=tx["freq_hz"],
    )

    achieved_focus_idx = np.unravel_index(np.argmax(p_max), p_max.shape)
    print(f"  achieved focus voxel (in crop): {achieved_focus_idx}")
    print(f"  intended target voxel (in crop): {domain.target_voxel}")
    print(f"  max pressure: {p_max.max():.3g} Pa")

    mid = domain.target_voxel[2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].imshow(np.rot90(domain.skull_mask[:, :, mid]), cmap="gray")
    axes[0].scatter(*display_xy(domain.target_voxel, domain.skull_mask.shape), c=CHROME["good"], marker="+", s=200, label="Intended target")
    axes[0].scatter(*display_xy(domain.transducer_voxel, domain.skull_mask.shape), c="#e34948", marker="o", s=60, label="Transducer")
    axes[0].set_title("Skull mask + geometry", fontsize=11)
    legend_on_image(axes[0])
    axes[0].axis("off")

    im = axes[1].imshow(np.rot90(p_max[:, :, mid]), cmap=sequential_cmap)
    axes[1].scatter(*display_xy(domain.target_voxel, p_max.shape), c=CHROME["good"], marker="+", s=200)
    axes[1].scatter(*display_xy(np.array(achieved_focus_idx), p_max.shape), c="#e34948", marker="x", s=120, label="Achieved focus")
    axes[1].set_title("Max pressure field", fontsize=11)
    legend_on_image(axes[1])
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Pressure (Pa)", color=CHROME["ink_secondary"])

    fig.suptitle(f"Phase D: through-skull k-Wave simulation ({subject_id})", fontsize=13)
    fig.tight_layout()

    saved_paths = save_figure(fig, f"pipeline_one_subject/{subject_id}_phase_d_skull_simulation")
    print(f"Figure:  {saved_paths[0]}")


if __name__ == "__main__":
    main()
