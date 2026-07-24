"""Run a 3D k-Wave transcranial FUS simulation through one subject's
acoustic maps, for a single-element focused bowl transducer.

Geometry/driving approach follows tussim_skull_3D.m from the same lab's
companion tool (sitiny/BRIC_TUS_Simulation_Tools): crop a manageable
sub-volume around the transducer/target pair, build the bowl source with
kWaveArray, and run kspaceFirstOrder3D.
"""

from dataclasses import dataclass

import numpy as np
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
