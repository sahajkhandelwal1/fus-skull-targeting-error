"""Minimal k-Wave smoke test: confirms the local CPU simulation binary
runs end-to-end and produces a plausible (non-NaN, non-zero) pressure field.

Not part of the FUS pipeline itself -- just a fast sanity check for the
dev environment, per Phase 1 / Immediate Next Action #1. Also saves a
paper-quality figure of the recorded wavefield so it's clear what the
simulation actually did, not just that it exited 0.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fus_targeting.viz.style import CATEGORICAL, CHROME, save_figure  # noqa: E402


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

    time_us = kgrid.t_array.squeeze() * 1e6
    sensor_positions_mm = np.arange(p.shape[1]) * grid_spacing[0] * 1e3
    center_trace_idx = p.shape[1] // 2

    fig, (ax_field, ax_trace) = plt.subplots(1, 2, figsize=(11, 4.2))

    im = ax_field.imshow(
        p.T,
        aspect="auto",
        origin="lower",
        extent=[time_us[0], time_us[-1], sensor_positions_mm[0], sensor_positions_mm[-1]],
        cmap="RdBu_r",
        vmin=-np.max(np.abs(p)),
        vmax=np.max(np.abs(p)),
    )
    ax_field.set_xlabel("Time (µs)")
    ax_field.set_ylabel("Position along sensor line (mm)")
    ax_field.set_title("Recorded wavefield (point source → line sensor)")
    cbar = fig.colorbar(im, ax=ax_field, pad=0.02)
    cbar.set_label("Pressure (Pa)", color=CHROME["ink_secondary"])

    ax_trace.plot(time_us, p[:, center_trace_idx], color=CATEGORICAL[0], linewidth=1.4)
    ax_trace.set_xlabel("Time (µs)")
    ax_trace.set_ylabel("Pressure (Pa)")
    ax_trace.set_title("Trace at sensor line midpoint")

    fig.suptitle(
        "k-Wave CPU environment smoke test (kspaceFirstOrder-OMP, 2D homogeneous medium)",
        fontsize=12,
    )
    fig.tight_layout()

    saved_paths = save_figure(fig, "kwave_smoke_test/pressure_field")
    print(f"  figure saved: {saved_paths[0]}")


if __name__ == "__main__":
    main()
