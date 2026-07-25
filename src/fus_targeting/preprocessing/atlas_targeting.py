"""Automated VIM thalamus target coordinates via atlas registration.

Replaces per-subject manual eyeballing (used for the IXI002 proof of
concept) with the standard clinical "indirect" AC-PC-relative targeting
formula, applied by registering each subject to the MNI152 template:

    X = 14mm lateral from midline
    Y = (AC-PC length / 3) - 2mm anterior to PC
    Z = 0 (AC-PC plane)

This is the classical indirect VIM targeting formula used across DBS/FUS
centers (see e.g. Papavassiliou et al. 2004; corroborated against
connectivity-derived VIM coordinates -- Middlebrooks et al. 2018 reports a
connectivity-derived VIM centroid at MNI (-16,-19,6), close to this
formula's Y). In MNI152 space, AC sits at ~(0,0,0) and PC at ~y=-25mm for
the "average" adult brain (AC-PC length ~25mm), giving:

    MNI target (right VIM):  (14, -19, 0)
    MNI target (left VIM):   (-14, -19, 0)

Known limitation: indirect (atlas-based) targeting has a reported mean
error of ~2mm (up to 5mm) against true clinical VIM targets even before
any skull effects -- acceptable for this project's proof-of-concept
cohort-scale targeting, not a substitute for clinical direct/tractography
targeting.
"""

from pathlib import Path

import ants
import nibabel as nib
import numpy as np
from nilearn import datasets

MNI_VIM_TARGET_RIGHT = np.array([14.0, -19.0, 0.0])   # MNI world (R,A,S) mm
MNI_VIM_TARGET_LEFT = np.array([-14.0, -19.0, 0.0])


def get_mni_template_path(cache_dir: Path) -> Path:
    """Save nilearn's bundled MNI152 1mm template to disk (as a plain NIfTI
    ANTs can read) and return the path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    template_path = cache_dir / "MNI152_T1_1mm.nii.gz"
    if not template_path.exists():
        template_img = datasets.load_mni152_template(resolution=1)
        nib.save(template_img, template_path)
    return template_path


def register_subject_to_mni(subject_t1_path: Path, mni_template_path: Path):
    """Affine-register the subject's T1 to MNI152. Returns the ANTs
    transform list (subject -> MNI direction) needed to bring points from
    MNI space into the subject's native space."""
    fixed = ants.image_read(str(mni_template_path))
    moving = ants.image_read(str(subject_t1_path))
    registration = ants.registration(fixed=fixed, moving=moving, type_of_transform="Affine")
    return registration


def mni_point_to_subject_voxel(mni_point_mm: np.ndarray, registration, subject_affine: np.ndarray) -> np.ndarray:
    """Transform a point from MNI world space into the subject's voxel grid.

    ants.registration(fixed=MNI, moving=subject) computed fwdtransforms
    that warp the *image* subject->MNI; per ANTs convention, *point*
    mapping runs the opposite direction, so applying fwdtransforms to a
    point maps MNI-space -> subject-space (verified against ants' own
    apply_transforms_to_points example)."""
    import pandas as pd

    # ants.apply_transforms_to_points expects LPS-convention points and a
    # dataframe with x,y,z columns; our coordinates are RAS (nibabel/MNI
    # convention), so flip the first two axes going in and coming out.
    # NOTE: a single-row points DataFrame trips a bug in ants' CSV
    # round-trip (raises "number of columns fewer than 3" / a pandas
    # length-mismatch) -- duplicating the row is a harmless workaround
    # (verified both rows come back identical).
    x, y, z = -mni_point_mm[0], -mni_point_mm[1], mni_point_mm[2]
    pt_lps = pd.DataFrame({"x": [x, x], "y": [y, y], "z": [z, z]})
    transformed = ants.apply_transforms_to_points(
        dim=3, points=pt_lps, transformlist=registration["fwdtransforms"],
    )
    subject_world_ras = np.array([-transformed["x"][0], -transformed["y"][0], transformed["z"][0]])

    inv_affine = np.linalg.inv(subject_affine)
    voxel = inv_affine @ np.append(subject_world_ras, 1.0)
    return voxel[:3]
