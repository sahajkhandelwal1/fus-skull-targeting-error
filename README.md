# Predicting Where Focused Ultrasound Actually Lands in the Brain

A skull-heterogeneity-based targeting error model for transcranial focused ultrasound (FUS), aimed at predicting simulation-derived targeting error and energy loss from simple, quickly-measurable skull characteristics — without running a full physics simulation per patient.

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
  preprocessing/      MRI -> pseudo-CT conversion (mr-to-pct wrapper)
  simulation/          k-Wave simulation setup and batch runners
  features/            Skull descriptor extraction (SDR, thickness, curvature, etc.)
  modeling/            Predictive model training and evaluation
notebooks/           Exploratory analysis
results/
  figures/             Generated plots (gitignored)
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

## Status

Environment verified for both k-Wave (CPU) and mr-to-pct — a single example subject runs end-to-end through mr-to-pct producing a plausible pseudo-CT. Full pipeline (Phases 1-2 of the plan) not yet implemented.
