# Predicting Where Focused Ultrasound Actually Lands in the Brain

A skull-heterogeneity-based targeting error model for transcranial focused ultrasound (FUS), aimed at predicting simulation-derived targeting error and energy loss from simple, quickly-measurable skull characteristics — without running a full physics simulation per patient.

See [`FUS_Project_Full_Plan_and_Executive_Summary.md`](./FUS_Project_Full_Plan_and_Executive_Summary.md) for the full background, research question, method, and publication targets.

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

## Status

Project scaffold only — pipeline implementation not yet started. See the plan document for phased next steps.
