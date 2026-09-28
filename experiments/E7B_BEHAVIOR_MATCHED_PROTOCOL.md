# E7B. Real-data test of the behavior-matched two-channel break

`gutlin_repro/behavior_matched.py` showed, on synthetic controlled data, that a
learning-rule manipulation which leaves behavior identical is still detectable on a
representational channel, beyond a clean-student noise floor. E7B is the honest
real-data version of exactly that claim. It is a companion to
[E7_PROTOCOL.md](E7_PROTOCOL.md): E7 asks "does `u` enter the inferred rule"; E7B
asks the sharper question the identifiability crux turns on, "when the behavioral
channel is held matched, does a representational channel still separate manipulated
from control, beyond matched-control noise?"

## What carries over from the synthetic result, and what does not

The synthetic experiment had three ingredients. Each needs a real-data analog, and
one of them cannot be reproduced faithfully, which is stated up front.

| synthetic ingredient | real-data analog | faithful? |
|---|---|---|
| behavior pinned by distilling two students to one teacher's outputs | behavior matched by covariate matching / matched sampling across a real manipulation | approximately: matched, not identical |
| attacker aux head bends the latent | a real external signal `u_t` (block bias, advice, framing, LLM advice) that may reshape the rule | the manipulation is real but not chosen by us |
| clean-student noise floor (two inits) | matched-control representational variability (split-half, matched control pairs, between-session) | yes |

The load-bearing caveat: on synthetic data behavior was *identical* by construction
(output agreement 1.000). On real data behavior can only be *matched* to a
tolerance. So E7B cannot claim the behavioral channel is exactly blind; it must
report the residual behavioral separation and show the representational separation
exceeds what that residual could explain.

## Name the failure modes first (/litadapt discipline)

Stated before any data is touched:

1. **Matching leakage.** "Behavior-matched" strata still differ behaviorally in some
   unmatched statistic, so the representational separation is just downstream of a
   behavioral difference we failed to match on. Guard: report the residual
   behavioral separation on every matched stratum, and a behavioral-only classifier
   as a control (below). If behavior alone separates the strata, the matching failed.
2. **Representational confound.** The representational readout differs across
   conditions for a reason unrelated to the rule (arousal, motion, time on task,
   signal quality). Guard: the placebo channel and the matched-control noise floor.
3. **Noise-floor underestimation.** The clean-vs-clean variability is larger on real
   data than any manipulation effect, so nothing clears it. This is the honest null
   and the kill criterion; it is not tuned away.
4. **Circularity.** The representational readout is fit using labels that encode the
   manipulation, so it "separates" by construction. Guard: the readout is defined on
   held-out data and, where it is a probed direction, validated on an independent
   axis first (the `whisper_audit` tuned-lens / rich-lazy control).

## Estimand

Among behavior-matched strata (manipulated vs control matched on the behavioral
readout), the separation of a representational channel between manipulated and
control, expressed as its distance beyond the matched-control noise floor. Reported
as a call / no-call / abstain at level alpha, alongside (a) the residual behavioral
separation on the same strata and (b) the incremental separation the representational
channel gives *over* a behavioral-only classifier (the quantity that would be zero if
the second channel adds nothing).

The single headline statistic mirrors the synthetic one:

    repr_gap = D(rep_manipulated, rep_control) - D(rep_control, rep_control')

where `D` is a representational distance (1 - RSA correlation, or a decoder's
cross-condition error), `rep_control'` is a matched-control split (the noise floor),
and a positive `repr_gap` that survives its surrogate distribution is the break. On
synthetic data this was 0.25 +/- 0.03 at lambda 0.5, growing to 0.52 at lambda 5.0.

## Two channels, defined per data type

The behavioral channel is always the trial-by-trial choice/policy readout (the
dynamic-GLM output, PsyTrack [1] / GLM-HMM [2]). The representational channel depends
on what internal signal the dataset carries:

- **Neural datasets (Tier 1-2):** the representational channel is a neural RDM (or a
  population decoder's geometry) over the learning task, the direct analog of the
  synthetic RDM. This is the cleanest instantiation.
- **Behavior-only datasets (Tier 3):** there is no neural readout, so the
  "representation" is the *inferred learning-rule functional* `f_phi` itself (Liu,
  Geadah & Pillow [5]). The two channels become choice-output vs update-rule-shape;
  the break is whether two subjects with matched choices have separable inferred
  rules. This is the hardest case and the purest test of the theory's claim that the
  rule carries information the output does not.
- **AI-in-the-loop / model subjects (Tier 3+):** the representational channel is a
  probed internal direction (J-Lens / tuned-lens), i.e. the `whisper_audit` machinery
  pointed at the learning trajectory rather than at loyalty. Here behavior can again
  be pinned tightly (a model's outputs can be matched by construction, as in the
  synthetic teacher-student), so this is the real setting where E7B most closely
  reproduces the synthetic design.

## Data requirements

A dataset qualifies only if it has all three:
- trial-by-trial de novo learning, so the behavioral readout is a real trajectory;
- a known external signal `u_t` (the candidate manipulation) that varies across
  conditions or blocks;
- an internal/representational readout: neural (EEG/fMRI/Neuropixels) for Tiers 1-2,
  or enough trials per subject to infer `f_phi` for Tier 3.

## Candidate datasets (all public or already in-program; runs on the local machine)

Tiered by how cleanly behavior can be matched while a real manipulation differs.

- **Tier 1, IBL brainwide + behavior.** International Brain Laboratory public
  Neuropixels + biased-block behavior. Behavior matched across blocks on the
  psychometric; representational channel = population RDM in a value/decision region.
  Strong positive control because the block manipulation is known and the neural
  readout is rich. Downloadable via the ONE API; fits on the local machine per region.
- **Tier 2, human RL with fMRI/EEG and a framing/advice manipulation.** Public
  OpenNeuro learning studies where advice, social information, or framing is
  manipulated. Behavior matched across advice conditions; representational channel =
  neural RDM. Closest human analog of external influence on learning.
- **Tier 3, behavior-only human learning with a manipulation.** Any of the above
  behavioral streams without the neural arm, or the program's own SPAR agent-steering
  / steering-in-the-wild logs. Representational channel = inferred `f_phi` geometry.
  Hardest power; run last.
- **Tier 3+, model-subject (the design that most faithfully reproduces the synthetic
  test).** A small RL agent or an LLM-as-subject given advice over trials; behavior
  pinned by distillation exactly as in `behavior_matched.py`; representational channel
  = a probed internal direction. This bridges the synthetic result to real influence
  data and is where the two-channel break can be shown before the neural arms mature.

## Controls (each reported alongside the number)

- **Matched-control noise floor.** The denominator of `repr_gap`: representational
  distance between two matched-control splits (split-half within control, or paired
  matched controls). The manipulation must drift beyond it. This is the direct analog
  of the synthetic clean-student noise floor.
- **Behavioral-only classifier.** Train a classifier to separate manipulated vs
  control from the behavioral readout alone on the matched strata. It must perform at
  chance if the matching worked. The reported result is the representational
  separation *minus* whatever this classifier achieves: the incremental identifiability
  the second channel buys. A zero increment is a null for the whole claim.
- **Placebo influence channel.** A logged covariate that is not the manipulation
  (time-of-day, trial-index) with matched autocorrelation. The representational
  separation on the placebo must be within the noise floor.
- **Surrogate calibration.** Circular-shift the condition labels / `u` preserving
  autocorrelation to build the null distribution of `repr_gap`.
- **Held-out readout.** The representational readout (RDM cells, decoder, or probed
  direction) is estimated on held-out sessions/subjects to defeat circularity.

## Kill criterion

If, on the Tier-1 positive control (or the Tier-3+ model-subject, whichever proves
out first), the representational `repr_gap` on behavior-matched strata does not exceed
the matched-control noise floor and its surrogate distribution, *and* the
behavioral-only classifier already separates the strata, then the behavior-matched
two-channel break **does not hold on real data** and E7B reports that. The synthetic
existence result stands as an in-silico proof of concept, and Layers 1 and 3 of the
learning-as-control defense do not depend on it, so an E7B null is not fatal to the
extension, exactly as in `model.md`.

## Procedure

1. Fit the dynamic-GLM behavioral trajectory per subject (PsyTrack), holding out
   sessions for the readouts.
2. Build behavior-matched strata: match manipulated vs control trials/subjects on the
   behavioral readout (propensity or coarsened-exact matching on choice rate,
   accuracy, psychometric slope). Report match quality and residual behavioral
   separation.
3. Compute the representational channel on held-out data per condition (neural RDM,
   decoder geometry, inferred `f_phi`, or probed direction).
4. Compute `repr_gap` = manipulated-vs-control representational distance minus the
   matched-control noise floor; calibrate against the surrogate label-shift null.
5. Compute the behavioral-only classifier baseline on the same strata; report the
   incremental separation of the representational channel over it.
6. Run the placebo channel through steps 3-5; it must return no-call.
7. Where a call survives, report persistence (does the representational separation
   outlast the manipulation within a session) and the effect sizes from
   `learning_as_control.md`.

## Reporting rules

- Council-review this protocol before running, and again after the first positive
  control.
- No threshold tuning until the positive control passes, the placebo returns no-call,
  and the behavioral-only classifier is at chance on the matched strata.
- Every number ships with its noise floor, its behavioral-only baseline, its placebo,
  and a one-line statement of what result would have falsified it.
- The synthetic numbers (`behavior_matched_results.json`) are cited as the in-silico
  reference the real-data effect is compared against, not as evidence about brains.

## References

[1] Roy, Bak, Akrami, Brody, Pillow. *Extracting the dynamics of behavior in sensory
decision-making experiments.* Neuron, 2021 (PsyTrack).

[2] Ashwood, Roy, Stone, et al., Pillow. *Mice alternate between discrete strategies
during perceptual decision-making.* Nature Neuroscience, 2022 (GLM-HMM).

[3] Gütlin, Kittelmann, Auksztulewicz. *Predictive coding networks capture human
neural representations missing in supervised DNNs.* bioRxiv, 2026.
doi:10.1101/2026.09.18.752626.

[4] International Brain Laboratory et al. *A brain-wide map of neural activity during
complex behaviour.* (IBL public data; ONE API.)

[5] Liu, Geadah, Pillow. *Flexible inference for animal learning rules using neural
networks.* NeurIPS, 2025.
