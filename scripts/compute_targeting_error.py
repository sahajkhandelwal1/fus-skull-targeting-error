"""Phase E of the one-subject pipeline: run a free-field (homogeneous water,
no skull) reference simulation with the identical transducer/target
geometry as Phase D, then compute the two labels this whole project is
about -- targeting error (how far the achieved focus is from the intended
target) and energy loss (how much peak pressure the skull cost vs. water).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fus_targeting.simulation.skull_simulation import (  # noqa: E402
    make_free_field_domain,
    prepare_subject_domain,
    run_through_skull_simulation,
)
from fus_targeting.viz.style import CATEGORICAL, CHROME, save_figure, sequential_cmap  # noqa: E402

INTERIM_DIR = ROOT / "data" / "interim"
CONFIG_PATH = ROOT / "configs" / "simulation_matrix.yaml"
DX_M = 1e-3


def display_xy(voxel_idx: np.ndarray, shape: tuple) -> tuple:
    r, a, _ = voxel_idx
    return r, shape[1] - 1 - a


def main() -> None:
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "IXI002"

    skull_result_path = INTERIM_DIR / subject_id / "skull_simulation_result.npz"
    if not skull_result_path.exists():
        raise SystemExit(f"{skull_result_path} not found -- run scripts/run_skull_simulation.py {subject_id} first")
    skull_result = np.load(skull_result_path)
    p_max_skull = skull_result["p_max"]
    target_voxel_in_crop = skull_result["target_voxel_in_crop"]
    freq_hz = float(skull_result["freq_hz"])

    # Rebuild the identical geometry Phase D used, then swap in a
    # homogeneous water medium for the free-field reference run.
    domain, tx = prepare_subject_domain(subject_id, INTERIM_DIR, CONFIG_PATH, half_size=100)
    assert np.array_equal(domain.target_voxel, target_voxel_in_crop), "geometry mismatch vs. Phase D result"
    free_field_domain = make_free_field_domain(domain)

    print(f"Subject: {subject_id}")
    print("Running free-field (water-only) reference simulation...")
    p_max_free, _, _ = run_through_skull_simulation(
        free_field_domain, dx_m=DX_M, freq_hz=freq_hz,
        roc_m=tx["roc_mm"] * 1e-3, diameter_m=tx["diameter_mm"] * 1e-3,
    )

    intended_target = domain.target_voxel
    achieved_focus_skull = np.array(np.unravel_index(np.argmax(p_max_skull), p_max_skull.shape))
    achieved_focus_free = np.array(np.unravel_index(np.argmax(p_max_free), p_max_free.shape))

    targeting_error_mm = float(np.linalg.norm((achieved_focus_skull - intended_target) * DX_M * 1000))
    free_field_offset_mm = float(np.linalg.norm((achieved_focus_free - intended_target) * DX_M * 1000))

    peak_pressure_skull = float(p_max_skull.max())
    peak_pressure_free = float(p_max_free.max())
    energy_loss_fraction = 1 - (peak_pressure_skull / peak_pressure_free)
    insertion_loss_db = 20 * np.log10(peak_pressure_free / peak_pressure_skull)

    labels = {
        "subject_id": subject_id,
        "targeting_error_mm": targeting_error_mm,
        "free_field_geometric_offset_mm": free_field_offset_mm,
        "peak_pressure_skull_pa": peak_pressure_skull,
        "peak_pressure_free_field_pa": peak_pressure_free,
        "energy_loss_fraction": energy_loss_fraction,
        "insertion_loss_db": insertion_loss_db,
    }
    print(json.dumps(labels, indent=2))

    out_dir = INTERIM_DIR / subject_id
    with open(out_dir / "targeting_error_labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    np.savez(out_dir / "free_field_simulation_result.npz", p_max=p_max_free)

    mid = intended_target[2]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    vmax = max(peak_pressure_skull, peak_pressure_free)
    for ax, p_max, title, achieved in [
        (axes[0], p_max_free, "Free field (no skull)", achieved_focus_free),
        (axes[1], p_max_skull, "Through skull", achieved_focus_skull),
    ]:
        im = ax.imshow(np.rot90(p_max[:, :, mid]), cmap=sequential_cmap, vmin=0, vmax=vmax)
        ax.scatter(*display_xy(intended_target, p_max.shape), c=CHROME["good"], marker="+", s=200, label="Intended target")
        ax.scatter(*display_xy(achieved, p_max.shape), c="#e34948", marker="x", s=120, label="Achieved focus")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label("Pressure (Pa)", color=CHROME["ink_secondary"])

    ax_bar = axes[2]
    bars = ax_bar.bar(
        ["Free field", "Through skull"],
        [peak_pressure_free / 1e6, peak_pressure_skull / 1e6],
        color=[CATEGORICAL[0], CATEGORICAL[1]],
    )
    ax_bar.set_ylabel("Peak pressure (MPa)")
    ax_bar.set_title(
        f"Energy loss: {energy_loss_fraction:.1%}\nTargeting error: {targeting_error_mm:.2f} mm",
        fontsize=11,
    )
    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(f"{height:.2f}", (bar.get_x() + bar.get_width() / 2, height),
                         ha="center", va="bottom", fontsize=9, color=CHROME["ink_primary"])

    fig.suptitle(f"Phase E: targeting error / energy loss labels ({subject_id})", fontsize=13)
    fig.tight_layout()

    saved_paths = save_figure(fig, f"pipeline_one_subject/{subject_id}_phase_e_targeting_error")
    print(f"Figure:  {saved_paths[0]}")
    print(f"Labels:  {out_dir / 'targeting_error_labels.json'}")


if __name__ == "__main__":
    main()
