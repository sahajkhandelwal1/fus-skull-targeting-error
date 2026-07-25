# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A research pipeline that predicts transcranial focused ultrasound (FUS) targeting error and energy loss from simple, fast skull measurements (density, thickness, curvature, entry angle, SDR) instead of running a full physics simulation per patient. The "ground truth" for training this predictive model comes from running the real, expensive simulation (k-Wave) on MRI-derived pseudo-CT skull models for a cohort of subjects from the public IXI dataset. See the README's "In Plain English" and "Executive Summary" sections for the full research framing, and `configs/simulation_matrix.yaml` for the locked baseline condition (VIM thalamus target, 650kHz single-element bowl transducer).

## Environment setup — two separate venvs

The two toolchains have incompatible dependency constraints (MONAI 1.1 / ANTsPy 0.4.2 predate Python 3.13/NumPy 2.0 support), so this project uses two Python environments. Always activate the correct one before running a script — check the script's own docstring/header if unsure.

**`.venv/` — Python 3.13 — k-Wave simulation, everything except MRI->pseudo-CT conversion:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export SSL_CERT_FILE=$(python3 -m certifi)   # needed once per shell if kwave's binary download hits an SSL error
```
On Apple Silicon, k-Wave's binary needs native `fftw`/`hdf5`/`zlib`/`libomp` from an **arm64** Homebrew at `/opt/homebrew` — a Rosetta/x86_64 shell installs these to `/usr/local` instead, which the arm64 k-Wave binary can't link against (`dyld: Library not loaded` if this is wrong; check with `uname -m`).

**`.venv-mrtopct/` — Python 3.10 — MRI->pseudo-CT conversion and atlas-based targeting (needs ANTsPy/MONAI/nilearn):**
```bash
arch -arm64 /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 -m venv .venv-mrtopct
source .venv-mrtopct/bin/activate
pip install -r third_party/requirements-mrtopct.txt
```
`third_party/mr-to-pct` is a git submodule ([sitiny/mr-to-pct](https://github.com/sitiny/mr-to-pct)) — run `git submodule update --init` if empty. Its pretrained weights and example dataset are fetched separately from OSF (not committed).

## Running the pipeline

`scripts/run_one_subject.sh <subject_id>` runs the entire one-subject pipeline (Phases A-G) end to end, switching venvs automatically. Individual phase scripts can also be run standalone (each states its required venv in its docstring) — useful for iterating on one stage without re-running the expensive simulation phases.

Verify the k-Wave environment alone with: `python3 scripts/smoke_test_kwave.py` (in `.venv`).

Kaggle API auth (`~/.kaggle/access_token` or `kaggle auth login`) is required for `scripts/fetch_ixi_dev_subjects.sh`, since the official IXI download host has been intermittently returning 403s (see README "Data" section) — that script pulls from a Kaggle mirror instead.

## Pipeline architecture (one subject, Phases A-G)

Each phase is a standalone script in `scripts/`, backed by reusable logic in `src/fus_targeting/`, and each phase **always** produces a figure via the shared visualization module — this is a hard project convention, not optional (see below). Data flows through `data/{raw,interim,processed}/<subject_id>/` (gitignored except `data/processed/`, which holds the small final per-subject records).

| Phase | Purpose | Script | Key module |
|---|---|---|---|
| A | Reorient raw MRI to RAS+, resample to 1mm iso (mr-to-pct's required input format) | `prepare_ixi_subject.py` | `preprocessing/reorient_resample.py` |
| B | MRI -> pseudo-CT via `mr-to-pct` | `convert_ixi_subject_to_pct.py` | (calls into `third_party/mr-to-pct`) |
| — | Atlas-based VIM target auto-registration (MNI152 affine registration + standard indirect targeting formula) | `find_atlas_target.py` | `preprocessing/atlas_targeting.py` |
| C | Pseudo-CT Hounsfield values -> k-Wave acoustic properties (density, sound speed, attenuation) | `compute_acoustic_maps.py` | `preprocessing/pct_to_acoustic.py` |
| D | 3D k-Wave simulation of the baseline condition through the subject's skull | `run_skull_simulation.py` | `simulation/skull_simulation.py` |
| E | Free-field (no-skull) reference simulation + targeting error / energy loss labels | `compute_targeting_error.py` | `simulation/skull_simulation.py` (`make_free_field_domain`) |
| F | Skull descriptors at the beam entry point (thickness, density, SDR, entry angle, curvature) | `extract_skull_descriptors.py` | `features/skull_descriptors.py` |
| G | Assemble one labeled record + composite summary figure | `assemble_subject_record.py`, `make_pipeline_summary_figure.py` | — |

Important implementation details future work will run into:
- **Resampling interpolation order matters**: use linear (order=1), not cubic spline, in Phase A — cubic spline over-smooths the thin (~1-2mm) cortical bone rim and silently collapses pseudo-CT peak bone density (this was a real bug found via the visualization requirement).
- **Acoustic property formulas** (Phase C) follow `tussim_skull_3D.m` from the same lab's companion tool ([sitiny/BRIC_TUS_Simulation_Tools](https://github.com/sitiny/BRIC_TUS_Simulation_Tools)), citing Marsac et al. 2017 / Bancel et al. 2021 / F. A. Duck 2013 / Robertson et al. 2017 — don't invent different constants without checking that reference.
- **Transducer geometry** (Phase D) is built with k-Wave's `kWaveArray.add_bowl_element`; the transducer is placed by ray-casting laterally from the target through the skull mask to the temporal window, at the bowl's ROC distance so its natural geometric focus lands on target.
- **Two categories of RuntimeWarning have been investigated and are benign, not bugs** (both documented in code comments where they occur): (1) divide-by-zero/overflow/invalid warnings from `kwave-python`'s bowl rotation code when transducer->target is axis-aligned, and (2) the same category of warning from macOS's Accelerate BLAS backend in the curvature fit whenever a basis vector has an exact-zero component — both verified to produce fully finite, bit-identical-to-manual-computation results. Don't "fix" these by changing the geometry; if new warnings appear elsewhere, verify with the same approach (check for NaN/Inf, compare against a manual/unvectorized computation) before assuming a bug.
- **Curvature estimation is noise-prone**: the outer-skull surface is a binary mask boundary (blocky at 1mm resolution), so the local quadratic-fit curvature estimate is capped at a realistic 300mm radius rather than reporting fit noise as signal.
- **Atlas targeting**: `ants.apply_transforms_to_points` has a bug with single-row point DataFrames (worked around by duplicating the row) — see `atlas_targeting.py`. Also note ANTs point-mapping direction is the *opposite* of image-mapping direction (documented in that module).
- Per-subject target voxels: `skull_simulation.get_target_voxel()` prefers the atlas-registered target (`data/interim/<subject>/atlas_target_voxel.npy`) over the legacy `KNOWN_TARGET_VOXELS` dict (manually-eyeballed, proof-of-concept only, IXI002 only).

## Visualization convention (applies project-wide, not just Phases A-G)

Every pipeline stage — existing or new — must save a figure as a side effect of running, using `src/fus_targeting/viz/style.py` (`apply_style()`, `save_figure()`, `legend_on_image()` for legends drawn over anatomical-slice images since the default frameon=False legend is invisible on a black background). The categorical color palette there is CVD-safety-validated; don't reorder it or introduce ad-hoc colors elsewhere. Figures are saved to `results/figures/<stage>/` and **are tracked in git** (not gitignored) — they're a deliverable in their own right (articles/videos), not just regenerable scratch output. When adding a new pipeline stage, follow the existing phase scripts' pattern: compute, save data artifact, save a styled figure, print a one-line status.

## Known constraints for future work

- The official IXI MRI download host has been intermittently down (403 on the specific `IXI/` subfolder) — `scripts/fetch_ixi_dev_subjects.sh` uses a Kaggle mirror as a fallback; retry the official source before assuming the Kaggle mirror is the long-term path for the full cohort.
- k-Wave 3D simulations (Phases D/E) take roughly 3-25 minutes each depending on thermal state on the dev Mac mini M4 — expect to background these and poll rather than block, and budget accordingly when scaling to the full 150-300 subject cohort (two simulations per subject: through-skull + free-field reference).
- Atlas-based targeting uses the standard *indirect* AC-PC-relative VIM formula (not tractography/direct targeting), with a reported literature accuracy of ~2-5mm even before skull effects — acceptable for this project's cohort-scale screening use case, not a claim of clinical-grade precision.
