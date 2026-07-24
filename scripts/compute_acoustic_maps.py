"""Phase C of the one-subject pipeline: convert the pseudo-CT into k-Wave
acoustic medium properties (density, sound speed, attenuation) and save a
diagnostic figure of all three maps.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fus_targeting.preprocessing.pct_to_acoustic import pct_to_acoustic_maps  # noqa: E402
from fus_targeting.viz.style import CHROME, save_figure, sequential_cmap  # noqa: E402

INTERIM_DIR = ROOT / "data" / "interim"


def main() -> None:
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "IXI002"
    pct_path = INTERIM_DIR / subject_id / "pct.nii"
    if not pct_path.exists():
        raise SystemExit(f"{pct_path} not found -- run scripts/convert_ixi_subject_to_pct.py {subject_id} first")

    img = nib.load(pct_path)
    pct = img.get_fdata()

    maps = pct_to_acoustic_maps(pct)

    print(f"Subject: {subject_id}")
    print(f"  hu_min={maps.hu_min}, hu_max={maps.hu_max}")
    print(f"  skull voxel fraction: {maps.skull_mask.mean():.3%}")
    print(f"  density range (skull): {maps.density[maps.skull_mask].min():.0f}-{maps.density[maps.skull_mask].max():.0f} kg/m^3")
    print(f"  sound speed range (skull): {maps.sound_speed[maps.skull_mask].min():.0f}-{maps.sound_speed[maps.skull_mask].max():.0f} m/s")
    print(f"  alpha_coeff range (skull): {maps.alpha_coeff[maps.skull_mask].min():.2f}-{maps.alpha_coeff[maps.skull_mask].max():.2f} dB/(MHz^y cm)")

    np.savez(
        INTERIM_DIR / subject_id / "acoustic_maps.npz",
        density=maps.density, sound_speed=maps.sound_speed,
        alpha_coeff=maps.alpha_coeff, skull_mask=maps.skull_mask,
        alpha_power=maps.alpha_power, affine=img.affine,
    )

    mid = pct.shape[2] // 2
    panels = [
        ("Density", maps.density, "kg/m$^3$"),
        ("Sound speed", maps.sound_speed, "m/s"),
        ("Attenuation coeff.", maps.alpha_coeff, "dB/(MHz$^y$ cm)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (title, arr, units) in zip(axes, panels):
        im = ax.imshow(np.rot90(arr[:, :, mid]), cmap=sequential_cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(units, color=CHROME["ink_secondary"], fontsize=9)

    fig.suptitle(f"Phase C: pseudo-CT -> k-Wave acoustic maps ({subject_id})", fontsize=13)
    fig.tight_layout()

    saved_paths = save_figure(fig, f"pipeline_one_subject/{subject_id}_phase_c_acoustic_maps")
    print(f"Figure:  {saved_paths[0]}")


if __name__ == "__main__":
    main()
