# E7B-R1. Pre-registration: identifiability-constrained cloning rescue

E7B (`e7b_model_subject/`) returned a null: on real IBL model-subjects, a
behavior-matched learning-rule manipulation is not representationally detectable. The
diagnostic located the cause precisely: two *clean* behavior-cloned subjects already
have uncorrelated raw hidden states (cosine drift ~1.0), so the representation is
underdetermined by behavior and any coordinate-level attack signature is swamped by the
clean-clean noise floor. In the rotation-invariant condition-RDM the floor is tight
(rho ~0.98) but the attack does not move the RDM (it changes coordinates, not relational
geometry), so that readout is blind by a different route.

This document pre-registers the rescue the null motivates: make the representation
identifiable across subjects so a coordinate-level manipulation has a tight floor to be
measured against. It is written and frozen BEFORE running, because the rescue has a
built-in way to fool itself, and the whole point is the control that catches that.

## The hypothesis, and the trap it must avoid

- **H (rescue).** If clean subjects that share behavior are constrained to also share a
  representational frame, the clean-clean noise floor shrinks, and the behavior-matched
  attack then separates above it. Detectability was floor-limited, not absent.
- **The trap (why this is not automatically a positive).** Tightening the floor can
  *manufacture* separation. In the limit of identical clean subjects (shared init, same
  data, same loss) the floor goes to zero and ANY perturbation, including a meaningless
  one, separates. That is the Gate-3 circularity of the synthetic result in a new form:
  the constraint, not the attack, produces the signal. A result is only real if, at a
  floor tight enough to see the attack, a manipulation known to be inert stays below the
  floor while the attack exceeds it.

The scientific content of E7B-R1 is therefore not "does the attack separate" but "does
the attack separate MORE than an inert sham at the same floor tightness." Everything
below is built around that contrast.

## Name the failure modes first (/litadapt discipline)

1. **Manufactured separation.** The identifiability constraint drives the clean-clean
   floor so low that the sham separates too. Guard: the sham-attack control (below) is
   the primary readout; the attack-minus-sham gap is the estimand, not the raw attack
   separation.
2. **Behavioral-residual leakage.** Real-data cloning matches behavior only to ~0.945,
   so the (control, attack) pair differs behaviorally more than a (control, control)
   pair, and the representational gap is just that residual. Guard: behavior-residual
   matching (below), and the behavioral-only classifier must stay at chance.
3. **Alignment overfSitting.** A post-hoc alignment fit on the same subjects it scores
   can absorb or invent differences. Guard: alignment is fit on held-out trials / a
   held-out clean subject and applied frozen.
4. **Constraint changes the subject.** A cloning constraint strong enough to tighten the
   floor may stop the model-subject from actually fitting the real behavior (it is no
   longer a faithful subject). Guard: report clone accuracy vs real choices at every
   constraint strength; a subject that has stopped matching behavior is disqualified.
5. **Regime confound.** The constraint might itself push all subjects toward one
   rich/lazy regime, and the "detection" is really a regime read. Guard: report the
   effective-rank order parameter for every arm and constraint strength.

## The intervention: identifiability-constrained cloning (two knobs, pre-specified)

Both knobs keep the model-subject cloned to the REAL IBL choices; they only constrain
where in representation space it lands.

- **Knob A, shared-frame cloning (training-side).** Control-A, control-B, and attack
  share the same initialisation. Optionally add an anchor penalty
  `beta * || h_t - h_t^ref ||^2` pulling each subject's per-trial state toward a fixed
  reference trajectory (the mean clean state), with `beta` swept on a pre-declared grid
  `{0, 0.01, 0.03, 0.1, 0.3}`. `beta = 0` with shared init is the weakest constraint;
  higher `beta` tightens the frame.
- **Knob B, alignment-constrained readout (readout-side, no training change).** Leave
  cloning unconstrained; instead fit an orthogonal Procrustes (and, as a sensitivity
  arm, a ridge-CCA) map between two clean subjects' hidden states on HELD-OUT trials,
  then compare all arms in that aligned frame. This tests the same hypothesis without
  altering the subject, so it is the cleaner of the two; Knob A is the stronger but
  riskier.

Primary analysis = Knob B (alignment-constrained readout). Knob A is the confirmatory
sensitivity analysis. Both are run at every declared setting; nothing is selected post
hoc.

## Estimand

At each constraint setting, in the identifiable frame:

    rescue_gap = D(control_A, attack) - D(control_A, sham)

where `D` is the raw-space representational distance (1 - mean per-trial cosine, in the
aligned/shared frame), and `sham` is a control-C subject processed identically to the
attack arm but with the attack objective REPLACED by a label-shuffled version of the
same objective (same optimiser, same capacity, same aux head, zero real manipulation).
`rescue_gap > 0`, surviving its surrogate null across the 6 real sessions, is the
positive. The raw attack separation `D(control_A, attack) - D(control_A, control_B)` is
reported too, but it is NOT the estimand, because the trap inflates it.

## Controls (each reported alongside the number)

- **Sham attack (primary control).** Control-C with a label-shuffled attack objective.
  Defines what separation the constraint manufactures with no real manipulation. The
  estimand subtracts it.
- **Behavior-residual matching.** Down-weight or resample trials so the choice
  disagreement in the (control, attack) pair equals that in the (control, control) pair,
  so `rescue_gap` cannot be the behavioral residual. Report the matched residuals.
- **Behavioral-only classifier.** Choices -> arm label; must stay at chance at every
  constraint strength. If the constraint makes behavior separable, the subject changed
  (failure mode 4) and that setting is disqualified.
- **Clone-accuracy floor.** Clone accuracy vs real choices per arm per `beta`; a subject
  below the unconstrained accuracy minus a pre-set tolerance is disqualified.
- **Surrogate null.** Circular-shift the attack labels / re-pair arms to build the null
  distribution of `rescue_gap`.
- **Effective-rank order parameter** per arm and setting (regime confound guard).

## Success and kill criteria

- **Positive (moves the verdict to a real break):** `rescue_gap > 0` surviving the
  surrogate null on a majority of the 6 sessions, at a constraint setting where (a) the
  behavioral-only classifier is at chance, (b) clone accuracy is within tolerance of
  unconstrained, and (c) the sham stays within the clean-clean floor. Only then is the
  behavior-matched manipulation genuinely detectable once representation is identifiable.
- **Kill criterion:** if across the entire declared constraint grid there is no setting
  where the attack separates while the sham does not (either the attack never exceeds the
  sham, or it only does so where the subject has stopped matching behavior), then
  identifiability-constrained cloning does NOT rescue detectability, and E7B's null
  stands as the finding. Layers 1 and 3 of the learning-as-control defense do not depend
  on this channel, so this remains non-fatal to the extension.

## What a positive would and would not establish

A positive shows that the representational channel carries behavior-matched manipulation
information *once subjects share a frame*, i.e. detectability is a representational-
alignment problem, not an impossibility. It would NOT yet show this for real neural data
(where the "frame" is a real brain, not a shared init), nor that a real adversary's
manipulation lives in the detectable subspace. Those remain downstream. The claim earned
is bounded to: "on real behavior, a behavior-matched learning-rule manipulation is
representationally detectable under an identifiability constraint, above a sham at
matched behavior and matched clone accuracy."

## Freeze

Before running: fix the 6 sessions (the E7B manifest), the constraint grid
`beta in {0, 0.01, 0.03, 0.1, 0.3}` under shared init, the Procrustes and ridge-CCA
alignment recipes with their held-out split, the sham definition (label-shuffled attack
objective), the behavior-residual matching rule, the clone-accuracy tolerance (0.02
below unconstrained), and the estimand `rescue_gap` with its surrogate null. Council-
review this pre-registration before the first run and again after the sham control.
No threshold or knob is chosen after seeing `rescue_gap`.

## References

Reuses the E7B stack: `e7b_model_subject/model_subject.py` (model-subject),
`run_e7b.py` (readouts + controls), real IBL data via `fetch_ibl.py`. Alignment methods:
orthogonal Procrustes and CCA are standard representational-similarity tools; the
rich/lazy order parameter follows Liu, Baratin et al. (ICLR 2024).
