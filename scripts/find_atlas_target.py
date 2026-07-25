"""Automated replacement for manually eyeballing the target voxel: register
the subject's T1 to MNI152 and bring the standard indirect VIM target
coordinate into the subject's native voxel space.

Run in the mr-to-pct venv (needs ANTsPy + nilearn):
  source .venv-mrtopct/bin/activate
  python3 scripts/find_atlas_target.py IXI002
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fus_targeting.preprocessing.atlas_targeting import (  # noqa: E402
    MNI_VIM_TARGET_RIGHT,
    get_mni_template_path,
    mni_point_to_subject_voxel,
    register_subject_to_mni,
)
from fus_targeting.viz.style import ANATOMY_CMAP, CHROME, legend_on_image, save_figure  # noqa: E402

INTERIM_DIR = ROOT / "data" / "interim"
CACHE_DIR = ROOT / "data" / "interim" / "_mni_template"


def main() -> None:
    subject_id = sys.argv[1] if len(sys.argv) > 1 else "IXI002"
    subject_t1_path = INTERIM_DIR / subject_id / "t1_ras_1mm.nii"
    if not subject_t1_path.exists():
        raise SystemExit(f"{subject_t1_path} not found -- run scripts/prepare_ixi_subject.py {subject_id} first")

    mni_template_path = get_mni_template_path(CACHE_DIR)

    print(f"Subject: {subject_id}")
    print("Registering to MNI152 (affine)...")
    registration = register_subject_to_mni(subject_t1_path, mni_template_path)

    subject_img = nib.load(subject_t1_path)
    target_voxel = mni_point_to_subject_voxel(MNI_VIM_TARGET_RIGHT, registration, subject_img.affine)
    print(f"  MNI target (right VIM): {MNI_VIM_TARGET_RIGHT}")
    print(f"  -> subject voxel: {target_voxel}")

    out_path = INTERIM_DIR / subject_id / "atlas_target_voxel.npy"
    np.save(out_path, target_voxel)
    print(f"Saved: {out_path}")

    subject_data = subject_img.get_fdata()
    mni_img = nib.load(mni_template_path)
    mni_data = mni_img.get_fdata()
    mni_target_voxel = (np.linalg.inv(mni_img.affine) @ np.append(MNI_VIM_TARGET_RIGHT, 1.0))[:3]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    s_mni = int(round(mni_target_voxel[2]))
    axes[0].imshow(np.rot90(mni_data[:, :, s_mni]), cmap=ANATOMY_CMAP)
    axes[0].scatter(mni_target_voxel[0], mni_data.shape[1] - 1 - mni_target_voxel[1],
                     c=CHROME["good"], marker="+", s=200, label="MNI VIM target")
    axes[0].set_title("MNI152 template + fixed target", fontsize=11)
    legend_on_image(axes[0])
    axes[0].axis("off")

    s_sub = int(round(target_voxel[2]))
    axes[1].imshow(np.rot90(subject_data[:, :, s_sub]), cmap=ANATOMY_CMAP)
    axes[1].scatter(target_voxel[0], subject_data.shape[1] - 1 - target_voxel[1],
                     c=CHROME["good"], marker="+", s=200, label="Atlas-derived target")
    axes[1].set_title(f"{subject_id} + registered target", fontsize=11)
    legend_on_image(axes[1])
    axes[1].axis("off")

    fig.suptitle(f"Atlas-based VIM targeting ({subject_id})", fontsize=13)
    fig.tight_layout()

    saved_paths = save_figure(fig, f"pipeline_one_subject/{subject_id}_atlas_target_registration")
    print(f"Figure: {saved_paths[0]}")


if __name__ == "__main__":
    main()
