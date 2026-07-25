# Predicting Where Focused Ultrasound Actually Lands in the Brain

A skull-heterogeneity-based targeting error model for transcranial focused ultrasound (FUS), aimed at predicting simulation-derived targeting error and energy loss from simple, quickly-measurable skull characteristics — without running a full physics simulation per patient.

## In Plain English

Doctors can now treat certain brain conditions (essential tremor, some cases of depression and OCD) without opening the skull at all — by aiming ultrasound waves through the head from outside, so they converge on one tiny spot deep in the brain, kind of like focusing sunlight through a magnifying glass to a single burning point. This is called **focused ultrasound (FUS)**.

The catch: **skull bone gets in the way and bends the beam.** Everyone's skull is a little different — thicker in some spots, thinner in others, denser here, more porous there. Sound waves passing through bone slow down unevenly and lose energy, so the "focus point" often doesn't land exactly where the doctor aimed it, and it arrives weaker than intended. Being off by even a few millimeters matters a lot when you're targeting a structure deep in the brain.

Right now, the only way to know in advance how much a *specific* patient's skull will throw off the beam is to run a full, expensive, slow computer simulation of the sound waves traveling through that person's skull (built from a brain scan). That works, but it's too slow/expensive to do for every single patient just as a first-pass check.

**What this project is trying to do:** build a much faster shortcut. Instead of running the whole expensive simulation, can we predict *how far off-target the beam will land* and *how much strength it will lose* just from a handful of simple, quick measurements of someone's skull (like its thickness and density at the entry point)? If yes, that becomes a fast screening tool — "this skull looks straightforward" vs. "this one needs the careful full simulation."

**How we're getting the data to test that idea:** since we can't ethically run real ultrasound treatments on 150-300 people just to gather training data, we instead:
1. Take real, publicly available brain MRI scans from a research dataset (IXI).
2. Use an existing AI tool to turn each MRI into a "pseudo-CT" — basically a computer-generated estimate of what that person's skull bone looks like, since MRI doesn't naturally show bone well (CT scans do, but those use radiation and MRI doesn't).
3. Run the real, expensive physics simulation (a well-established tool called k-Wave, used throughout this field) on each of those simulated skulls, aiming at the same target every time — that gives us "ground truth" answers for how much each person's skull would throw off the beam.
4. Measure simple properties of each skull (density, thickness, curvature, etc.).
5. Train a simple, explainable statistical model to predict the ground-truth answers from just those simple skull measurements.
6. Check how well that shortcut actually works.

Every step above produces its own explanatory picture (see `results/figures/`) — the goal is that anyone, technical or not, can look at the figures folder and see exactly what happened at each stage without reading code.

## Executive Summary

**Problem.** Transcranial focused ultrasound (FUS) aims ultrasound waves through the skull to a precise point in the brain, for use in treating conditions like depression, essential tremor, and OCD. Skull bone varies in thickness, density, and structure across the head and between people, which bends and weakens the beam unpredictably — so it often lands off-target ("targeting error") with reduced energy. Today, catching this requires a full physics-based acoustic simulation per patient, built from a CT/MRI scan — accurate, but too slow and expensive to run on everyone.

**Research question.** Can targeting error be predicted from a handful of simple, fast skull measurements (density, thickness, curvature, entry angle, skull density ratio, etc.), without running the full simulation — as a quick triage/screening tool to flag which patients need careful full simulation versus which are likely straightforward?

**Why this is new.** Existing work either compares simulation platforms/methods (not skipping simulation) or predicts the entire pressure field with deep learning (heavier than needed). This project instead predicts just the key numbers — targeting offset and energy loss — directly from simple skull features, aiming for a lightweight, interpretable screening tool rather than a simulation replacement.

**Method.**
1. Pull MRI brain scans from the public IXI dataset.
2. Convert each MRI to a skull acoustic model using the open-source `mr-to-pct` tool (no CT/radiation needed).
3. Run full physics simulations (k-Wave) per subject as ground truth for targeting error and energy loss, across a baseline condition plus generalization subsets varying target, frequency, and transducer geometry.
4. Extract simple skull descriptors per subject/condition (density, thickness, curvature, entry angle, SDR, and related metrics).
5. Train interpretable models (not deep nets, given modest sample size) mapping skull descriptors → targeting error and energy loss.
6. Validate on held-out subjects and test generalization across targets/frequencies/transducers; sanity-check against published literature.

**Success looks like** a working, interpretable model plus a clear answer to which skull characteristics matter most — an honest proof-of-concept given a moderate sample size (150-300 subjects), not a clinic-ready product.

**Known limitations.** Sample size bounds precision; MRI-derived pseudo-CT is an estimate of true bone structure, not a direct measurement; the chosen skull descriptors may not capture every relevant feature.

**Target venues.** International Society for Therapeutic Ultrasound (ISTU), IEEE UFFC, *Brain Stimulation* (as a possible extension), or an arXiv/bioRxiv preprint — no venue is fixed yet; the priority is a complete, submittable writeup.

## Repo layout

```
configs/            Simulation design matrix and other run configs
data/
  raw/               Downloaded IXI MRI scans (gitignored)
  interim/           Pseudo-CT / intermediate conversions (gitignored)
  processed/         Extracted skull features + simulation labels (gitignored)
src/fus_targeting/
  preprocessing/      MRI -> pseudo-CT conversion, reorient/resample, pseudo-CT -> acoustic maps
  simulation/          k-Wave simulation setup and batch runners
  features/            Skull descriptor extraction (SDR, thickness, curvature, etc.)
  modeling/            Predictive model training and evaluation [not yet built -- needs the full cohort]
  viz/                 Shared plotting style used by every figure in the project
notebooks/           Exploratory analysis
results/
  figures/             Generated plots (tracked in git -- see Visualization standard below)
  tables/              Generated result tables (gitignored)
paper/               Writeup drafts
scripts/             CLI entry points / pipeline runners
```

## Environment setup

Two separate Python environments are used, because the two toolchains have incompatible dependency constraints:

**k-Wave (CPU simulation), Python 3.13, `.venv/`:**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
The k-Wave binary auto-downloads on first `import kwave` and needs the system CA bundle (`pip install certifi` + `export SSL_CERT_FILE=$(python3 -m certifi)` if you hit an SSL error). On Apple Silicon it also needs native `fftw`/`hdf5`/`zlib`/`libomp` from an **arm64** Homebrew at `/opt/homebrew` (a Rosetta/x86_64 shell will install these to `/usr/local` instead, which the arm64 binary can't link against — check with `uname -m` if `dyld: Library not loaded` errors appear).

Verify with: `python3 scripts/smoke_test_kwave.py`

**mr-to-pct (MRI -> pseudo-CT), Python 3.10, `.venv-mrtopct/`:**
```
arch -arm64 /Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 -m venv .venv-mrtopct
source .venv-mrtopct/bin/activate
pip install -r third_party/requirements-mrtopct.txt
```
`third_party/mr-to-pct` is a git submodule ([sitiny/mr-to-pct](https://github.com/sitiny/mr-to-pct)) — run `git submodule update --init` if it's empty after cloning. Pretrained weights and the example dataset are downloaded separately from OSF (not committed, see the tool's own README) into `third_party/mr-to-pct/`.

## Data

The official IXI download source (`biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/`) is currently returning `403 Forbidden` for that specific subfolder (confirmed broken from multiple independent networks, not just an IP block) — the dataset itself is still fully open (CC BY-SA 3.0, no registration required), just temporarily unreachable at the canonical host. `scripts/fetch_ixi_dev_subjects.sh` fetches a small 5-subject dev batch from a Kaggle mirror ([kbacon/ixi-t1](https://www.kaggle.com/datasets/kbacon/ixi-t1)) instead, which hosts the same raw per-subject NIfTI files. Requires `pip install kaggle` and Kaggle API auth (`~/.kaggle/access_token` or `kaggle auth login`). Retry the official source once it's back up before scaling to the full 150-300 subject cohort.

## Visualization standard

Every pipeline stage saves a paper-quality figure as a side effect of running it — not an optional extra step. This is deliberate: the figures are both a correctness check (see the resampling bug below, caught because a figure looked wrong) and a growing archive of visual material for articles/videos later, independent of what ends up in the eventual paper. All figures use one consistent style (`src/fus_targeting/viz/style.py`, palette validated for colorblind-safety) and are tracked in git under `results/figures/<stage>/`, browsable directly on GitHub.

## One-subject pipeline progress

Building the pipeline one phase at a time on a single real subject (IXI002) before scaling to the full cohort. Each phase's figure is linked below.

| Phase | What it does | Script | Figure |
|---|---|---|---|
| A | Reorient raw IXI scan to RAS+ and resample to 1mm isotropic (mr-to-pct's required input format) | `scripts/prepare_ixi_subject.py` | [phase A](results/figures/pipeline_one_subject/IXI002_phase_a_reorient_resample.png) |
| B | Convert the prepped MRI to a pseudo-CT skull model via `mr-to-pct` | `scripts/convert_ixi_subject_to_pct.py` | [phase B](results/figures/pipeline_one_subject/IXI002_phase_b_mr_to_pct.png) |
| C | Convert pseudo-CT Hounsfield values into k-Wave acoustic properties (density, sound speed, attenuation) using the same lab's published formulas | `scripts/compute_acoustic_maps.py` | [phase C](results/figures/pipeline_one_subject/IXI002_phase_c_acoustic_maps.png) |
| D | Run a 3D k-Wave simulation of the baseline condition (VIM thalamus target, single-element 650kHz bowl transducer) through the subject's skull | `scripts/run_skull_simulation.py` | [phase D](results/figures/pipeline_one_subject/IXI002_phase_d_skull_simulation.png) |
| E | Compute targeting error (focus offset) and energy loss labels vs. a free-field reference | `scripts/compute_targeting_error.py` | [phase E](results/figures/pipeline_one_subject/IXI002_phase_e_targeting_error.png) |
| F | Extract skull descriptors (density, thickness, curvature, entry angle, SDR) at the beam entry point | `scripts/extract_skull_descriptors.py` | [phase F](results/figures/pipeline_one_subject/IXI002_phase_f_skull_descriptors.png) |
| G | Tie A-F into one pipeline, producing one labeled record + a summary figure | `scripts/run_one_subject.sh` | [phase G summary](results/figures/pipeline_one_subject/IXI002_phase_g_summary.png) |

**IXI002's completed record** (`data/processed/IXI002_record.json`) — the shape of one training-data row for the eventual model:

| Inputs (fast, from pseudo-CT) | Outputs (expensive, from full simulation) |
|---|---|
| thickness: 6.0mm | targeting error: 4.12mm |
| density (mean/max): 1151/1655 HU | energy loss: 57.2% |
| SDR: 0.33 | insertion loss: 7.37dB |
| entry angle: 4.8° | |
| radius of curvature: 89mm | |

Cross-checked against a free-field (no-skull) reference simulation: the achieved focus there landed exactly on the intended target (0.0mm offset), confirming the through-skull offset is a real skull effect and not a geometry bug.

**Baseline condition** (locked in `configs/simulation_matrix.yaml`): VIM thalamus target (most common clinical tcMRgFUS target, e.g. Insightec Exablate Neuro), single-element focused bowl transducer at 650kHz (ROC 63.2mm, 64mm aperture, matching Sonic Concepts CTX-500 bowl geometry driven as one uniform-phase element). The per-subject target voxel is now found automatically (`scripts/find_atlas_target.py`) by affine-registering the subject to the MNI152 template (via nilearn's bundled template, no download needed) and applying the standard clinical indirect VIM targeting formula — replacing the manually-eyeballed target used for the initial proof of concept. Validated against the original manual target: landed within ~5.5mm, consistent with the ~2-5mm accuracy indirect targeting is reported to have in the literature itself.

**A real bug caught by the visualization requirement:** Phase A originally resampled using cubic-spline interpolation, which over-smoothed the thin (~1-2mm) cortical bone rim and collapsed IXI002's pseudo-CT peak bone value from the expected ~2500 Hounsfield-like units down to ~795 — silently understating skull density/sound-speed/attenuation everywhere downstream. Caught by comparing figures against the tool's own bundled example output, fixed by switching to linear (order=1) resampling. See the Phase A/B commit history for the full comparison.

## Status

**The one-subject pipeline (Phases A-G) is complete and verified end to end on real subject IXI002** — raw MRI → pseudo-CT → acoustic maps → through-skull k-Wave simulation → targeting error/energy loss labels → skull descriptors → one final labeled record, runnable as a single command (`scripts/run_one_subject.sh IXI002`). See the [pipeline summary figure](results/figures/pipeline_one_subject/IXI002_phase_g_summary.png) for the whole story in one image.

Environment set up and verified for both k-Wave (CPU, Apple M4) and mr-to-pct (MPS). 5 raw IXI T1 scans downloaded for pipeline dev (`data/raw/ixi_t1_dev/`, gitignored).

**Next up:** atlas-based auto-targeting is done (see above) -- the pipeline no longer needs a human to look at each subject's anatomy. Remaining before the full run: batch-download the full cohort (retry the official IXI source, which was down as of this writing), run the baseline + generalization-subset simulation matrix, build the feature-extraction pipeline at scale, and train the predictive model.
