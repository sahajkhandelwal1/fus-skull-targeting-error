"""Phase D of the one-subject pipeline: run a 3D k-Wave simulation of the
baseline transducer/target condition (single-element 650kHz bowl -> VIM
thalamus) through the subject's skull, and save a diagnostic figure of the
resulting pressure field.

NOTE on targeting: the target voxel below was picked by visual inspection
of subject IXI002's anatomy (see the axial-slice scan used to locate it),
not atlas registration -- adequate for proving the pipeline mechanics work
end-to-end on one subject, but real atlas-based targeting is needed before
scaling to the full cohort (flagged in configs/simulation_matrix.yaml).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fus_targeting.simulation.skull_simulation import crop_around, run_through_skull_simulation  # noqa: E402
from fus_targeting.viz.style import CHROME, save_figure, sequential_cmap  # noqa: E402

INTERIM_DIR = ROOT / "data" / "interim"

# Approximate right-thalamus target for IXI002, picked by visual inspection
# (voxel indices in the subject's RAS+ 1mm acoustic-map grid).
TARGET_VOXEL = np.array([103, 117, 140])


def find_lateral_transducer_position(skull_mask: np.ndarray, target_voxel: np.ndarray, roc_mm: float, standoff_mm: float = 3.0) -> np.ndarray:
    """Cast a ray laterally (+R direction) from the target through the skull to
    find the temporal window, then place the transducer ROC millimeters from
    the target along that ray (so the bowl's natural geometric focus lands at
    the target), nudged outward past the skull surface if ROC alone would land
    inside tissue."""
    r0, a0, s0 = target_voxel
    max_r = skull_mask.shape[0]

    outer_skull_r = None
    for r in range(r0, max_r):
        if skull_mask[r, a0, s0]:
            outer_skull_r = r
    if outer_skull_r is None:
        raise RuntimeError("No skull crossing found along +R ray from target -- pick a different target/direction.")

    roc_based_r = r0 + roc_mm
    min_allowed_r = outer_skull_r + standoff_mm
    transducer_r = max(roc_based_r, min_allowed_r)

    return np.array([transducer_r, a0, s0])


def main() -> None:
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "IXI002"

    with open(ROOT / "configs" / "simulation_matrix.yaml") as f:
        config = yaml.safe_load(f)
    tx_cfg = config["baseline"]["transducer"]
    freq_hz = config["baseline"]["frequency_hz"]
    roc_mm = tx_cfg["radius_of_curvature_mm"]
    diameter_mm = tx_cfg["aperture_diameter_mm"]

    npz_path = INTERIM_DIR / subject_id / "acoustic_maps.npz"
    if not npz_path.exists():
        raise SystemExit(f"{npz_path} not found -- run scripts/compute_acoustic_maps.py {subject_id} first")

    data = np.load(npz_path)
    maps = {k: data[k] for k in ["density", "sound_speed", "alpha_coeff", "skull_mask"]}
    maps["alpha_power"] = float(data["alpha_power"])

    transducer_voxel = find_lateral_transducer_position(maps["skull_mask"], TARGET_VOXEL, roc_mm)
    print(f"Subject: {subject_id}")
    print(f"  target voxel:     {TARGET_VOXEL}")
    print(f"  transducer voxel: {transducer_voxel}")
    print(f"  distance (voxels/mm, 1mm iso): {np.linalg.norm(transducer_voxel - TARGET_VOXEL):.1f}")

    domain = crop_around(maps, TARGET_VOXEL, transducer_voxel, half_size=100)
    print(f"  cropped domain shape: {domain.density.shape}")
    print(f"  skull voxel fraction in crop: {domain.skull_mask.mean():.3%}")

    p_max, target_pos, transducer_pos = run_through_skull_simulation(
        domain,
        dx_m=1e-3,
        freq_hz=freq_hz,
        roc_m=roc_mm * 1e-3,
        diameter_m=diameter_mm * 1e-3,
    )

    np.savez(
        INTERIM_DIR / subject_id / "skull_simulation_result.npz",
        p_max=p_max, target_voxel_in_crop=domain.target_voxel,
        transducer_voxel_in_crop=domain.transducer_voxel, crop_origin=domain.crop_origin,
        skull_mask=domain.skull_mask, dx_m=1e-3, freq_hz=freq_hz,
    )

    achieved_focus_idx = np.unravel_index(np.argmax(p_max), p_max.shape)
    print(f"  achieved focus voxel (in crop): {achieved_focus_idx}")
    print(f"  intended target voxel (in crop): {domain.target_voxel}")
    print(f"  max pressure: {p_max.max():.3g} Pa")

    mid = domain.target_voxel[2]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    axes[0].imshow(np.rot90(domain.skull_mask[:, :, mid]), cmap="gray")
    axes[0].scatter(*_display_xy(domain.target_voxel, domain.skull_mask.shape), c=CHROME["good"], marker="+", s=200, label="Intended target")
    axes[0].scatter(*_display_xy(domain.transducer_voxel, domain.skull_mask.shape), c="#e34948", marker="o", s=60, label="Transducer")
    axes[0].set_title("Skull mask + geometry", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].axis("off")

    im = axes[1].imshow(np.rot90(p_max[:, :, mid]), cmap=sequential_cmap)
    axes[1].scatter(*_display_xy(domain.target_voxel, p_max.shape), c=CHROME["good"], marker="+", s=200)
    axes[1].scatter(*_display_xy(np.array(achieved_focus_idx), p_max.shape), c="#e34948", marker="x", s=120, label="Achieved focus")
    axes[1].set_title("Max pressure field", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Pressure (Pa)", color=CHROME["ink_secondary"])

    fig.suptitle(f"Phase D: through-skull k-Wave simulation ({subject_id})", fontsize=13)
    fig.tight_layout()

    saved_paths = save_figure(fig, f"pipeline_one_subject/{subject_id}_phase_d_skull_simulation")
    print(f"Figure:  {saved_paths[0]}")


def _display_xy(voxel_idx: np.ndarray, shape: tuple) -> tuple:
    """Convert (R,A,S) voxel index to the (x,y) coords used by imshow(rot90(slice))."""
    r, a, _ = voxel_idx
    return r, shape[1] - 1 - a


if __name__ == "__main__":
    main()
