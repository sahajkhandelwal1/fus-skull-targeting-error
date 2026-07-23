# Predicting Where Focused Ultrasound Actually Lands in the Brain: A Skull-Heterogeneity-Based Targeting Error Model

## Full Project Plan & Executive Summary

---

## 1. Plain-Language Background: What Problem Are We Solving?

**Transcranial focused ultrasound (FUS)** is a non-invasive brain treatment technique. Instead of surgery or implanted electrodes, doctors aim ultrasound waves from outside the skull so that they converge — like sunlight through a magnifying glass — on a small, specific point deep inside the brain. This can be used to stimulate or suppress activity in a targeted brain region, and is being investigated for conditions like depression, essential tremor, and obsessive-compulsive disorder.

The problem: **the skull gets in the way.** Bone is not a simple, uniform material — it varies in thickness, density, and internal structure across the head and from person to person. When an ultrasound wave passes through bone, it bends, slows unevenly, and loses energy in ways that depend heavily on that specific person's skull. The practical consequence is well documented in the scientific literature: the ultrasound beam frequently ends up landing somewhere slightly different from where it was aimed, and with less energy than intended, a phenomenon called **targeting error**. In a brain treatment context, being off by even a few millimeters can mean stimulating the wrong tissue entirely.

Currently, the way clinicians and researchers deal with this is to run a full physics-based computer simulation of the ultrasound wave passing through that specific patient's skull (built from a CT or MRI scan) before treatment. This works, but it's computationally expensive and slow — not something you can do instantly or at scale. There's no fast way to get a quick, rough estimate of "will this patient's skull cause a big problem or not" without running the whole expensive simulation.

## 2. The Research Question

**Can we predict how much targeting error a patient's skull will cause — without running the full expensive simulation — using only a handful of simple, quickly-measurable skull characteristics?**

If this works, it would function as a fast screening tool: a way to flag "this patient's skull geometry suggests significant targeting error, so run the full simulation carefully" versus "this skull looks straightforward, standard treatment planning is probably fine" — without needing to run the expensive full physics simulation on every single patient just to find that out.

## 3. Why This Hasn't Already Been Done (The Gap)

This is an active research area, so it's important to be precise about what's new here versus what already exists:

- A **January 2026 preprint** compared different simulation software platforms and different ways of estimating skull structure from scans — a valuable but different question (which tool/method is more accurate), not "can we skip the simulation entirely for a quick estimate."
- A dataset called **TFUScapes** already includes a deep learning model that predicts the *entire* pressure field (the full 2D/3D map of ultrasound intensity everywhere in the brain) from skull scans — an impressive but heavier tool than what's proposed here.

**What's actually new in this project:** instead of predicting the entire complex pressure field, this project predicts just a few key numbers — how far off-target the beam lands, and how much energy is lost — directly from simple skull measurements. It's a lighter-weight, faster, more interpretable tool aimed specifically at triage/screening use, not at replacing detailed simulation.

## 4. Step-by-Step Method (What Actually Gets Built)

**Step 1 — Gather skull anatomy data.**
Use MRI brain scans from a public research dataset (the IXI dataset, freely downloadable, no application required) covering dozens of different people, so the model sees real anatomical variation.

**Step 2 — Convert MRI to a skull acoustic model.**
Real bone-mapping normally requires a CT scan (which uses radiation), but an open-source tool exists that estimates the same acoustic properties directly from a standard MRI, avoiding the need for actual CT scans.

**Step 3 — Run the "ground truth" simulation.**
For each person's skull, run a full physics-based acoustic wave simulation (using the open-source k-Wave simulation engine) for a fixed treatment target and a fixed ultrasound transducer setup. This tells us, for that person, exactly how much the beam actually shifted and weakened — the "correct answer" we're trying to predict without doing this expensive step every time.

**Step 4 — Measure simple skull characteristics.**
For each person, extract a handful of quick numbers describing their skull at the entry point: how dense the bone is, how thick it is, how curved it is, and the angle the beam enters at.

**Step 5 — Train a predictive model.**
Using the simple skull numbers as input and the simulation's "correct answer" (targeting error and energy loss) as the target, train a model — starting with simple, interpretable methods (not a complex neural network, since the number of people in the study is modest) — to see how well those simple numbers alone can predict the error.

**Step 6 — Test and validate.**
Check the model's predictions against people's data it wasn't trained on, and test whether it still works if we change the treatment target or the ultrasound frequency, to see how generally useful the model is.

## 5. What "Success" Looks Like

- A working, interpretable model that connects specific skull characteristics to predictable amounts of targeting error.
- A clear answer to "which skull characteristics matter most" — itself a useful scientific finding, independent of how accurate the predictions turn out to be.
- Honest reporting of accuracy and limitations — with a modest sample size, this is a proof-of-concept demonstrating the relationship exists and is quantifiable, not a clinic-ready deployable tool.

## 6. Tools and Data (All Free, All Immediately Accessible)

| Component | Tool/Source | Access |
|---|---|---|
| Brain MRI scans | IXI dataset | Free, instant download |
| MRI → skull model conversion | `mr-to-pct` (open-source) | Free, GitHub |
| Acoustic simulation engine | k-Wave / k-Wave-python | Free, open-source, scriptable |
| Alternative precomputed data | TFUScapes | Free, instant download |
| Compute | Personal computer (Mac mini M4) | Already owned |

No institutional approval, data use agreement, or waiting period is required for any part of this pipeline — everything can start immediately.

## 7. Full Timeline (2-Week Sprint)

| Days | What Happens |
|---|---|
| **1–2** | Set up simulation software locally; download MRI scans; convert a first test subject to a skull model to confirm the pipeline works end-to-end on one case |
| **3–5** | Scale up: run the full simulation across the entire set of subjects (~50–100 people) for one fixed target and transducer setup; extract the skull characteristic measurements for each person |
| **6–9** | Build and train the predictive model; evaluate how well it predicts held-out subjects |
| **10–12** | Robustness testing: check whether the model still works for a different treatment target or ultrasound frequency; sanity-check the error magnitudes against published values from the literature |
| **13–14** | Write up methods, results, and figures for submission |

## 8. Known Limitations (Stated Upfront, Honestly)

- With roughly 50–100 subjects, the model's precision is inherently limited — this is appropriately sized for demonstrating and characterizing the relationship, not for building a clinically deployable product.
- The MRI-to-skull conversion is an *estimate* of true bone structure, not a direct measurement — some error is baked in from that step alone, independent of the modeling.
- Only a small number of skull descriptors are used; there may be relevant skull features this approach doesn't capture.

## 9. Where This Could Go (Publication Targets)

- **International Society for Therapeutic Ultrasound (ISTU)** — active, current venue for exactly this kind of work
- **IEEE UFFC** (Ultrasonics, Ferroelectrics, and Frequency Control) — relevant engineering venue
- **Brain Stimulation** (journal) — possible extension after conference presentation
- **ISEF** (Biomedical Engineering / Translational Medical Science category) — if the project is extended and deepened beyond the initial two-week scope

## 10. Immediate Next Actions

1. Install k-Wave-python and confirm the acoustic simulation binary runs correctly on the local machine
2. Install `mr-to-pct` and confirm it can convert a single MRI to a usable skull model
3. Download an initial batch of MRI scans from IXI
4. Choose and fix the treatment target coordinate and transducer specifications to be used across all subjects, for consistency
5. Run the full pipeline (MRI → skull model → simulation → skull metrics → error labels) on one subject start-to-finish before scaling to the full cohort
