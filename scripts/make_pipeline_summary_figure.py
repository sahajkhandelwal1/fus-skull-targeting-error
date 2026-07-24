"""Phase G (part 2): composite every phase's figure plus the final labeled
record into one summary image -- the single picture that tells the whole
one-subject pipeline story end to end.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fus_targeting.viz.style import CHROME, save_figure  # noqa: E402

FIGURES_DIR = ROOT / "results" / "figures" / "pipeline_one_subject"
PROCESSED_DIR = ROOT / "data" / "processed"

PHASES = [
    ("a", "Reorient + resample"),
    ("b", "MRI -> pseudo-CT"),
    ("c", "Acoustic maps"),
    ("d", "Through-skull simulation"),
    ("e", "Targeting error / energy loss"),
    ("f", "Skull descriptors"),
]


def main() -> None:
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "IXI002"

    record_path = PROCESSED_DIR / f"{subject_id}_record.json"
    if not record_path.exists():
        raise SystemExit(f"{record_path} not found -- run scripts/assemble_subject_record.py {subject_id} first")
    with open(record_path) as f:
        record = json.load(f)

    fig = plt.figure(figsize=(14, 16))
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.7], hspace=0.25, wspace=0.05)

    for i, (letter, title) in enumerate(PHASES):
        img_path = FIGURES_DIR / f"{subject_id}_phase_{letter}_" \
            f"{'reorient_resample' if letter == 'a' else 'mr_to_pct' if letter == 'b' else 'acoustic_maps' if letter == 'c' else 'skull_simulation' if letter == 'd' else 'targeting_error' if letter == 'e' else 'skull_descriptors'}.png"
        ax = fig.add_subplot(grid[i // 2, i % 2])
        if img_path.exists():
            ax.imshow(plt.imread(img_path))
        else:
            ax.text(0.5, 0.5, f"[missing: {img_path.name}]", ha="center", va="center")
        ax.set_title(f"Phase {letter.upper()}: {title}", fontsize=11, color=CHROME["ink_primary"])
        ax.axis("off")

    ax_summary = fig.add_subplot(grid[3, :])
    ax_summary.axis("off")
    summary_text = (
        f"Subject {record['subject_id']} -- final labeled record\n\n"
        f"Inputs (fast, from pseudo-CT):  "
        f"thickness={record['skull_thickness_mm']:.1f}mm   "
        f"density(mean/max)={record['skull_density_hu_mean']:.0f}/{record['skull_density_hu_max']:.0f}HU   "
        f"SDR={record['skull_density_ratio_sdr']:.2f}   "
        f"entry angle={record['beam_entry_angle_deg']:.1f}deg   "
        f"ROC={record['skull_radius_of_curvature_mm']:.0f}mm\n\n"
        f"Outputs (expensive, from full simulation):  "
        f"targeting error={record['targeting_error_mm']:.2f}mm   "
        f"energy loss={record['energy_loss_fraction']:.1%}   "
        f"insertion loss={record['insertion_loss_db']:.2f}dB"
    )
    ax_summary.text(
        0.02, 0.6, summary_text, transform=ax_summary.transAxes,
        fontsize=12, va="center", color=CHROME["ink_primary"],
        bbox=dict(boxstyle="round,pad=0.6", facecolor=CHROME["surface"], edgecolor=CHROME["baseline"]),
    )

    fig.suptitle(f"One-Subject Pipeline: {subject_id}", fontsize=16, y=0.995)

    saved_paths = save_figure(fig, f"pipeline_one_subject/{subject_id}_phase_g_summary")
    print(f"Figure: {saved_paths[0]}")


if __name__ == "__main__":
    main()
