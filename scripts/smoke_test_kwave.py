"""Minimal k-Wave smoke test: confirms the local CPU simulation binary
runs end-to-end and produces a plausible (non-NaN, non-zero) pressure field.

Not part of the FUS pipeline itself -- just a fast sanity check for the
dev environment, per Phase 1 / Immediate Next Action #1.
"""

import numpy as np

from kwave.data import Vector
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.utils.kwave_array import kWaveArray


def main() -> None:
    grid_size = Vector([128, 128])
    grid_spacing = Vector([0.1e-3, 0.1e-3])
    kgrid = kWaveGrid(grid_size, grid_spacing)

    medium = kWaveMedium(sound_speed=1500.0, density=1000.0)
    kgrid.makeTime(medium.sound_speed)

    source = kSource()
    source_mask = np.zeros(grid_size)
    source_mask[grid_size[0] // 2, grid_size[1] // 2] = 1
    source.p_mask = source_mask
    source.p = np.atleast_2d(np.sin(2 * np.pi * 1e6 * kgrid.t_array))

    sensor = kSensor()
    sensor_mask = np.zeros(grid_size)
    sensor_mask[grid_size[0] // 4, :] = 1
    sensor.mask = sensor_mask
    sensor.record = ["p"]

    simulation_options = SimulationOptions(
        pml_inside=False,
        save_to_disk=True,
        data_cast="single",
    )
    execution_options = SimulationExecutionOptions(is_gpu_simulation=False)

    result = kspaceFirstOrder2D(
        kgrid=kgrid,
        source=source,
        sensor=sensor,
        medium=medium,
        simulation_options=simulation_options,
        execution_options=execution_options,
    )

    p = result["p"]
    assert np.isfinite(p).all(), "Simulation produced non-finite values"
    assert np.any(p != 0), "Simulation produced an all-zero field"

    print("k-Wave smoke test passed.")
    print(f"  recorded pressure shape: {p.shape}")
    print(f"  max |p|: {np.max(np.abs(p)):.6g}")


if __name__ == "__main__":
    main()
