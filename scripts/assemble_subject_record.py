"""Phase G (part 1) of the one-subject pipeline: combine Phase E's labels
(targeting error, energy loss) and Phase F's skull descriptors (density,
thickness, curvature, entry angle, SDR) into the one record this whole
project is trying to predict -- what training row #1 of the eventual model
looks like.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "IXI002"
    subject_dir = INTERIM_DIR / subject_id

    labels_path = subject_dir / "targeting_error_labels.json"
    descriptors_path = subject_dir / "skull_descriptors.json"
    for p in (labels_path, descriptors_path):
        if not p.exists():
            raise SystemExit(f"{p} not found -- run the earlier phase scripts for {subject_id} first")

    with open(labels_path) as f:
        labels = json.load(f)
    with open(descriptors_path) as f:
        descriptors = json.load(f)

    record = {
        "subject_id": subject_id,
        # inputs (Phase F): what a fast screening model would have access to
        "skull_thickness_mm": descriptors["thickness_mm"],
        "skull_density_hu_mean": descriptors["density_hu_mean"],
        "skull_density_hu_max": descriptors["density_hu_max"],
        "skull_density_ratio_sdr": descriptors["sdr"],
        "beam_entry_angle_deg": descriptors["entry_angle_deg"],
        "skull_radius_of_curvature_mm": descriptors["radius_of_curvature_mm"],
        # outputs (Phase E): the expensive simulation's ground truth
        "targeting_error_mm": labels["targeting_error_mm"],
        "energy_loss_fraction": labels["energy_loss_fraction"],
        "insertion_loss_db": labels["insertion_loss_db"],
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{subject_id}_record.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    print(json.dumps(record, indent=2))
    print(f"\nRecord: {out_path}")


if __name__ == "__main__":
    main()
