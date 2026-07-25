"""Run a 3D k-Wave transcranial FUS simulation through one subject's
acoustic maps, for a single-element focused bowl transducer.

Geometry/driving approach follows tussim_skull_3D.m from the same lab's
companion tool (sitiny/BRIC_TUS_Simulation_Tools): crop a manageable
sub-volume around the transducer/target pair, build the bowl source with
kWaveArray, and run kspaceFirstOrder3D.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.kwave_array import kWaveArray
from kwave.utils.signals import tone_burst


# Fallback per-subject target voxel (R,A,S index in that subject's RAS+ 1mm
# acoustic-map grid), picked by visual inspection of anatomy -- only used if
# scripts/find_atlas_target.py hasn't been run for that subject yet. The
# atlas-registered target (data/interim/<subject>/atlas_target_voxel.npy,
# see fus_targeting.preprocessing.atlas_targeting) is preferred and is what
# scales to the full cohort; this dict is legacy from the initial
# proof-of-concept before atlas-based targeting existed.
KNOWN_TARGET_VOXELS = {
    "IXI002": np.array([103, 117, 140]),  # approximate right VIM thalamus (eyeballed)
}


def get_target_voxel(subject_id: str, interim_dir: Path) -> np.ndarray:
    atlas_path = interim_dir / subject_id / "atlas_target_voxel.npy"
    if atlas_path.exists():
        return np.round(np.load(atlas_path)).astype(int)
    if subject_id in KNOWN_TARGET_VOXELS:
        return KNOWN_TARGET_VOXELS[subject_id]
    raise KeyError(
        f"No target voxel for {subject_id} -- run scripts/find_atlas_target.py {subject_id} first"
    )


@dataclass
class CroppedDomain:
    density: np.ndarray
    sound_speed: np.ndarray
    alpha_coeff: np.ndarray
    alpha_power: float
    skull_mask: np.ndarray
    target_voxel: np.ndarray       # target position, index within cropped array
    transducer_voxel: np.ndarray   # transducer position, index within cropped array
    crop_origin: np.ndarray        # crop's origin in the original (uncropped) voxel grid


def crop_around(maps: dict, target_voxel: np.ndarray, transducer_voxel: np.ndarray, half_size: int) -> CroppedDomain:
    """Crop the acoustic maps to a cube of side 2*half_size centered on the
    midpoint between transducer and target."""
    midpoint = np.round((np.asarray(target_voxel) + np.asarray(transducer_voxel)) / 2).astype(int)
    full_shape = np.array(maps["density"].shape)

    lo = np.clip(midpoint - half_size, 0, None)
    hi = np.clip(midpoint + half_size, None, full_shape)
    slices = tuple(slice(lo[i], hi[i]) for i in range(3))

    return CroppedDomain(
        density=maps["density"][slices],
        sound_speed=maps["sound_speed"][slices],
        alpha_coeff=maps["alpha_coeff"][slices],
        alpha_power=float(maps["alpha_power"]),
        skull_mask=maps["skull_mask"][slices],
        target_voxel=np.asarray(target_voxel) - lo,
        transducer_voxel=np.asarray(transducer_voxel) - lo,
        crop_origin=lo,
    )


def make_free_field_domain(domain: CroppedDomain, water_density: float = 1000.0, water_sound_speed: float = 1500.0) -> CroppedDomain:
    """Same geometry (shape, target/transducer voxels) as `domain`, but
    homogeneous water everywhere -- i.e. what the beam would do with no
    skull in the way. Used as the reference for targeting error / energy
    loss (Phase E)."""
    shape = domain.density.shape
    return CroppedDomain(
        density=np.full(shape, water_density, dtype=np.float32),
        sound_speed=np.full(shape, water_sound_speed, dtype=np.float32),
        alpha_coeff=np.zeros(shape, dtype=np.float32),
        alpha_power=domain.alpha_power,
        skull_mask=np.zeros(shape, dtype=bool),
        target_voxel=domain.target_voxel,
        transducer_voxel=domain.transducer_voxel,
        crop_origin=domain.crop_origin,
    )


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


def prepare_subject_domain(subject_id: str, interim_dir: Path, config_path: Path, half_size: int = 100):
    """Load a subject's acoustic maps + baseline config, find the transducer
    position, and return the cropped simulation domain plus the transducer
    spec (roc_mm, diameter_mm, freq_hz). Shared by Phase D (through-skull)
    and Phase E (free-field reference) so both use identical geometry."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    tx_cfg = config["baseline"]["transducer"]
    freq_hz = config["baseline"]["frequency_hz"]
    roc_mm = tx_cfg["radius_of_curvature_mm"]
    diameter_mm = tx_cfg["aperture_diameter_mm"]

    target_voxel = get_target_voxel(subject_id, interim_dir)

    npz_path = interim_dir / subject_id / "acoustic_maps.npz"
    if not npz_path.exists():
        raise SystemExit(f"{npz_path} not found -- run scripts/compute_acoustic_maps.py {subject_id} first")
    data = np.load(npz_path)
    maps = {k: data[k] for k in ["density", "sound_speed", "alpha_coeff", "skull_mask"]}
    maps["alpha_power"] = float(data["alpha_power"])

    transducer_voxel = find_lateral_transducer_position(maps["skull_mask"], target_voxel, roc_mm)
    domain = crop_around(maps, target_voxel, transducer_voxel, half_size=half_size)

    return domain, dict(roc_mm=roc_mm, diameter_mm=diameter_mm, freq_hz=freq_hz)


def voxel_to_kgrid_coords(voxel_idx: np.ndarray, shape: tuple, dx: float) -> np.ndarray:
    """k-Wave grids are centered at 0; voxel i on an axis of length N maps to
    physical coordinate (i - N//2) * dx (matches kWaveGrid's own x_vec/y_vec/z_vec)."""
    shape = np.asarray(shape)
    return (np.asarray(voxel_idx, dtype=float) - shape // 2) * dx


def run_through_skull_simulation(
    domain: CroppedDomain,
    dx_m: float,
    freq_hz: float,
    roc_m: float,
    diameter_m: float,
    source_amp_pa: float = 500e3,
    cfl: float = 0.3,
    record_periods: int = 3,
    ppp: int = 12,
):
    shape = domain.density.shape
    grid_size = Vector(list(shape))
    grid_spacing = Vector([dx_m, dx_m, dx_m])
    kgrid = kWaveGrid(grid_size, grid_spacing)

    medium = kWaveMedium(
        sound_speed=domain.sound_speed,
        density=domain.density,
        alpha_coeff=domain.alpha_coeff,
        alpha_power=domain.alpha_power,
    )

    dt = 1.0 / (ppp * freq_hz)
    t_end = record_periods / freq_hz + np.max(domain.sound_speed.shape) * dx_m / np.min(domain.sound_speed)
    kgrid.setTime(int(np.ceil(t_end / dt)), dt)

    target_pos = voxel_to_kgrid_coords(domain.target_voxel, shape, dx_m)
    transducer_pos = voxel_to_kgrid_coords(domain.transducer_voxel, shape, dx_m)

    # NOTE: when transducer->target is exactly axis-aligned (as it is for our
    # lateral-ray placement), kwave-python's compute_rotation_between_vectors
    # hits a parallel/anti-parallel edge case and emits divide-by-zero /
    # overflow / invalid-value RuntimeWarnings from an unused intermediate
    # branch. Verified harmless: output bowl mask, source signal, and p_max
    # all come back fully finite with no NaN/Inf (checked directly on
    # IXI002's result) and the focus lands where geometry predicts.
    karray = kWaveArray(bli_tolerance=0.05, upsampling_rate=10)
    karray.add_bowl_element(transducer_pos.tolist(), roc_m, diameter_m, target_pos.tolist())

    num_cycles = int(kgrid.Nt / ppp)  # fill (most of) the simulation duration -- quasi-CW
    source_sig = source_amp_pa * tone_burst(
        1 / dt, freq_hz, num_cycles, envelope=[4, 4], signal_length=kgrid.Nt,
    ).squeeze()

    source = kSource()
    source.p_mask = karray.get_array_binary_mask(kgrid)
    source.p = karray.get_distributed_source_signal(kgrid, source_sig[np.newaxis, :])

    sensor = kSensor()
    sensor.mask = np.ones(shape, dtype=bool)
    sensor.record = ["p_max"]

    simulation_options = SimulationOptions(pml_inside=False, save_to_disk=True, data_cast="single")
    execution_options = SimulationExecutionOptions(is_gpu_simulation=False)

    result = kspaceFirstOrder3D(
        kgrid=kgrid, source=source, sensor=sensor, medium=medium,
        simulation_options=simulation_options, execution_options=execution_options,
    )

    p_max = result["p_max"].reshape(shape, order="F")
    return p_max, target_pos, transducer_pos
