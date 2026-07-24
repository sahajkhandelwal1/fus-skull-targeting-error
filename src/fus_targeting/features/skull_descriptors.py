"""Extract simple, fast skull descriptors at the beam entry point -- the
input features the whole project is trying to predict targeting error and
energy loss from, without running the full simulation.
"""

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class SkullDescriptors:
    entry_point_voxel: np.ndarray
    thickness_mm: float
    density_hu_mean: float
    density_hu_max: float
    sdr: float                     # skull density ratio (established clinical metric)
    entry_angle_deg: float         # angle between beam and skull surface normal (0 = perpendicular)
    radius_of_curvature_mm: float  # local outer-surface radius of curvature at entry point


def _sample_ray(volume: np.ndarray, start: np.ndarray, end: np.ndarray, n_samples: int) -> np.ndarray:
    """Nearest-neighbor sample `volume` along the straight line from start to
    end (voxel coordinates); fine at 1mm grid resolution for this purpose."""
    t = np.linspace(0, 1, n_samples)
    points = start[None, :] + t[:, None] * (end - start)[None, :]
    idx = np.round(points).astype(int)
    idx = np.clip(idx, 0, np.array(volume.shape) - 1)
    return volume[idx[:, 0], idx[:, 1], idx[:, 2]]


def _find_outer_skull_surface_points(skull_mask: np.ndarray, center_voxel: np.ndarray, radius_vox: int) -> np.ndarray:
    """Surface voxels (skull with at least one non-skull neighbor) within a
    cube of the given radius around center_voxel."""
    center_voxel = np.round(np.asarray(center_voxel)).astype(int)
    lo = np.clip(center_voxel - radius_vox, 0, None)
    hi = np.clip(center_voxel + radius_vox + 1, None, np.array(skull_mask.shape))
    sub = skull_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    eroded = np.ones_like(sub, dtype=bool)
    eroded[1:-1, 1:-1, 1:-1] = sub[1:-1, 1:-1, 1:-1] & sub[:-2, 1:-1, 1:-1] & sub[2:, 1:-1, 1:-1] \
        & sub[1:-1, :-2, 1:-1] & sub[1:-1, 2:, 1:-1] & sub[1:-1, 1:-1, :-2] & sub[1:-1, 1:-1, 2:]
    surface = sub & ~eroded

    local_idx = np.argwhere(surface)
    return local_idx + lo


def _estimate_curvature(skull_mask: np.ndarray, entry_point: np.ndarray, direction: np.ndarray, patch_radius_mm: float = 15.0) -> float:
    """Fit a quadratic surface to local outer-skull points near the entry
    point (in a frame aligned with the beam direction) and estimate the mean
    radius of curvature from the fit's second-order coefficients."""
    surface_points = _find_outer_skull_surface_points(skull_mask, entry_point, int(patch_radius_mm))
    if len(surface_points) < 10:
        return np.nan

    centered = surface_points - entry_point
    dist = np.linalg.norm(centered, axis=1)
    nearby = centered[dist <= patch_radius_mm]
    if len(nearby) < 10:
        return np.nan

    # local frame: u,v span the plane perpendicular to `direction`; w along `direction`
    w = direction / np.linalg.norm(direction)
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(w, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(w, u)

    # nearby @ u (etc.) triggers spurious divide-by-zero/overflow/invalid
    # RuntimeWarnings from macOS's Accelerate BLAS backend whenever u/v/w
    # has an exact-zero component -- verified against random synthetic data
    # that the matmul result is bit-identical to a manual per-row dot
    # product (no NaN/Inf), so this is a known false positive, not a bug.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        uu = nearby @ u
        vv = nearby @ v
        ww = nearby @ w

    # fit w = a*u^2 + b*v^2 + c*u*v + d*u + e*v + f
    A = np.column_stack([uu**2, vv**2, uu * vv, uu, vv, np.ones_like(uu)])
    coeffs, *_ = np.linalg.lstsq(A, ww, rcond=None)
    a, b = coeffs[0], coeffs[1]

    mean_curvature = (a + b)  # for a near-flat patch, w ~ a*u^2+b*v^2 -> curvature ~ 2a,2b; a+b is proportional to mean curvature
    if abs(mean_curvature) < 1e-8:
        return np.inf
    return float(1.0 / (2 * abs(mean_curvature)))


def compute_skull_descriptors(
    pct_hu: np.ndarray,
    skull_mask: np.ndarray,
    target_voxel: np.ndarray,
    transducer_voxel: np.ndarray,
    aperture_diameter_mm: float,
    dx_mm: float = 1.0,
    n_beam_rays: int = 9,
) -> SkullDescriptors:
    target_voxel = np.asarray(target_voxel, dtype=float)
    transducer_voxel = np.asarray(transducer_voxel, dtype=float)
    direction = target_voxel - transducer_voxel
    axis_length = np.linalg.norm(direction)
    direction_unit = direction / axis_length

    n_samples = int(axis_length) * 2
    profile_hu = _sample_ray(pct_hu, transducer_voxel, target_voxel, n_samples)
    profile_mask = _sample_ray(skull_mask, transducer_voxel, target_voxel, n_samples) > 0.5

    if not profile_mask.any():
        raise RuntimeError("No skull crossing found along the central beam ray.")

    skull_hu = profile_hu[profile_mask]
    thickness_mm = float(profile_mask.sum()) * (axis_length * dx_mm / n_samples)

    entry_idx = np.argmax(profile_mask)  # first sample (from transducer side) that's skull
    entry_point = transducer_voxel + (entry_idx / n_samples) * direction

    # SDR (skull density ratio): min/max HU ratio along parallel rays across
    # the aperture, averaged -- approximates the clinical SDR definition
    # (mean of per-element-path min/max ratios) with a small ray fan instead
    # of full per-element modeling.
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(direction_unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(direction_unit, arbitrary); u /= np.linalg.norm(u)
    v = np.cross(direction_unit, u)

    sdr_values = []
    aperture_radius_vox = (aperture_diameter_mm / 2) / dx_mm
    offsets = np.linspace(-aperture_radius_vox, aperture_radius_vox, int(np.sqrt(n_beam_rays)))
    for du in offsets:
        for dv in offsets:
            if du**2 + dv**2 > aperture_radius_vox**2:
                continue
            offset = du * u + dv * v
            ray_start = transducer_voxel + offset
            ray_end = target_voxel + offset
            ray_hu = _sample_ray(pct_hu, ray_start, ray_end, n_samples)
            ray_mask = _sample_ray(skull_mask, ray_start, ray_end, n_samples) > 0.5
            if ray_mask.sum() < 2:
                continue
            ray_skull_hu = ray_hu[ray_mask]
            if ray_skull_hu.max() > 0:
                sdr_values.append(ray_skull_hu.min() / ray_skull_hu.max())
    sdr = float(np.mean(sdr_values)) if sdr_values else float(skull_hu.min() / skull_hu.max())

    surface_points = _find_outer_skull_surface_points(skull_mask, np.round(entry_point).astype(int), radius_vox=5)
    if len(surface_points) > 0:
        dists = np.linalg.norm(surface_points - entry_point, axis=1)
        nearest_surface_point = surface_points[np.argmin(dists)].astype(float)
    else:
        nearest_surface_point = entry_point

    normal = _estimate_surface_normal(skull_mask, nearest_surface_point)
    cos_angle = np.clip(abs(np.dot(normal, direction_unit)), -1, 1)
    entry_angle_deg = float(np.degrees(np.arccos(cos_angle)))

    radius_of_curvature_mm = _estimate_curvature(skull_mask, nearest_surface_point, direction_unit)

    return SkullDescriptors(
        entry_point_voxel=entry_point,
        thickness_mm=thickness_mm,
        density_hu_mean=float(skull_hu.mean()),
        density_hu_max=float(skull_hu.max()),
        sdr=sdr,
        entry_angle_deg=entry_angle_deg,
        radius_of_curvature_mm=radius_of_curvature_mm,
    )


def _estimate_surface_normal(skull_mask: np.ndarray, point: np.ndarray, radius_vox: int = 4) -> np.ndarray:
    """Outward normal estimated as the direction from the local skull-voxel
    centroid to `point` (points from bone mass toward air/outside)."""
    center = np.round(point).astype(int)
    lo = np.clip(center - radius_vox, 0, None)
    hi = np.clip(center + radius_vox + 1, None, np.array(skull_mask.shape))
    sub = skull_mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    local_idx = np.argwhere(sub) + lo
    if len(local_idx) == 0:
        return np.array([1.0, 0.0, 0.0])
    centroid = local_idx.mean(axis=0)
    normal = point - centroid
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
