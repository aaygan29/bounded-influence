# Gütlin et al. (2026) reproduction (model-side)

A controlled, runnable reproduction of the *mechanism* in Gütlin, Kittelmann &
Auksztulewicz, "Predictive coding networks capture human neural representations
missing in supervised DNNs" (bioRxiv 2026, doi 10.1101/2026.09.18.752626), built
to support the learning-as-control extension in `../../theory/learning_as_control.md`.

## What this reproduces, and what it does not

The paper's load-bearing design is a **factorial isolation**: hold architecture,
task, and capacity fixed; vary only the learning objective (predictive /
contrastive / supervised / supervised-shuffled / untrained) and the mechanism
(local vs global credit assignment); then ask which objective's representations
best match human EEG, early vs late in statistical learning.

This code reproduces the **model side** faithfully:
- the statistical-learning paradigm (leading/trailing categories, transition
  matrix with valid 0.75 / invalid 0.25 / control 0.50), `paradigm.py`;
- all seven conditions in one parameter-matched two-layer RNN, `models.py` + `train.py`;
- the RSA pipeline with **cross-indexing** (category target vs trailing-grouped
  reps; predictive target vs leading-grouped reps) and a bootstrap BMS, `rsa.py`.

It does **not** reproduce brain alignment. We do not have the McDermott et al.
(2026) EEG. The "neural" targets are **synthetic ground truth** that instantiate
the paper's two learning stages (a category/identity RDM and a predictive/predictor
RDM), so we can test whether the pipeline recovers the reported dissociation
against a controlled target instead of claiming it against unavailable data.

Two further honest reductions:
- **Stimuli are category embeddings, not images.** Fixed random unit vectors per
  category, shared across conditions. The mechanism the paper isolates does not
  depend on pixels; the paper's own controls hold the stimulus set fixed.
- **"Parameter-matched" here means matched _across conditions_** (the load-bearing
  control), not matched to the paper's absolute 1.34M count. Every condition uses
  the identical architecture (~124k params); only loss and gradient scope change.
- The **local** mechanism is approximated by detaching the cross-layer activation
  so no gradient crosses the layer boundary. This is a coarse stand-in for the
  paper's layer-restricted target and does **not** reliably separate local from
  global predictive models here (see results).

## Result (5 seeds, 40 epochs, 5000 trials)

| claim | outcome |
|---|---|
| predictive objective uniquely captures predictive structure (predictor-indexed RDM); supervised/contrastive do not | **5/5 seeds** (pred rho 0.46–0.66; supervised rho ≈ 0 or negative) |
| full double dissociation (also: supervised best-on-category among trained models) | **3/5 seeds** (category half is fragile) |
| local vs global predictive distinguished | **no** (crude local approximation) |

Per-seed detail:

```
seed 0: dd=True   pred_late=0.49 sup_late=0.02  sup_early=0.44
seed 1: dd=True   pred_late=0.62 sup_late=-0.17 sup_early=0.64
seed 2: dd=True   pred_late=0.60 sup_late=0.01  sup_early=0.46
seed 7: dd=False  pred_late=0.66 sup_late=-0.05 sup_early=0.36
seed 13:dd=False  pred_late=0.46 sup_late=-0.08 sup_early=0.28
```

**Honest read.** The paper's *headline* claim — that a predictive objective
captures predictive representational structure that a supervised objective misses
— reproduces robustly (5/5). The *category-stage* half (supervised best matches
category structure) is seed-fragile in this reduction, because the untrained
architecture is the ceiling on raw category geometry (the category target is just
input geometry, which an untrained net already carries — exactly what the paper's
untrained baseline exists to expose). The finding is therefore reported as a
**partial, mechanism-level reproduction**, not a brain result.

## Run

```
python run_repro.py --epochs 40 --n-trials 5000 --seed 0 --out results.json
```

`results.json` (committed) is seed 0. Runtime ~25 s/seed on CPU.

## Two-channel manipulation detector (`detect.py`)

The extension this reproduction exists for (see the detector section of
`../../theory/learning_as_control.md`). A control arm learns the true predictive
rule; a manipulated arm has its predictive *target* bent to an attacker-chosen
partner for a fraction of trials (matched experience, only the learning target
moves). Two difference-in-differences vs a clean-arm null:

- **behavioral** channel (next-step prediction fidelity): separates attack from
  null cleanly at every scale.
- **representational** channel (RSA predictive-axis drift): underpowered at 8
  categories (one seed flips sign), **reliable at 20** (attack DiD -0.13±0.02 vs
  null +0.04±0.05; both channels move under attack, both separated from null, 5/5).

```
python detect.py --epochs 40 --atk 0.4 --seeds 5 --ncat 20 --out detect_results_n20.json
```

`detect_results.json` (ncat 8, representational channel underpowered) and
`detect_results_n20.json` (ncat 20, rescued) are both committed so the scale
dependence is inspectable.

**Open, not closed.** This attack changes behavior, so the behavioral channel
alone already flags it; the detector does not yet prove the representational
channel adds information in the matched-behavior / different-rule case that the
identifiability crux is about. The decisive next experiment is a
**behavior-matched manipulation** with the same two-channel readout.

## Why this lives in bounded-influence

`../../theory/learning_as_control.md` argues that manipulating the *learning rule*
(objective + local/global mechanism) reshapes an agent's representational
landscape, and that this reshaping has a measurable neural signature. This repro
is the working demonstration of that premise on a controlled system, and it
supplies the RSA-axis machinery for the two-channel manipulation detector
(behavioral difference-in-differences + representational-axis drift). See
`detect.py` and the detector section of `learning_as_control.md`.
