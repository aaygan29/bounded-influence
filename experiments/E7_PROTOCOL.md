# E7. Real-data test of the Lambda learning-rule detector (the kill-criterion run)

E6 showed the Lambda detector is calibrated and powered *in simulation*, where we control the
ground truth. E7 is the honest test: on real trial-by-trial learning data with a known external
signal, does the detector decide whether that signal enters the learning rule, without
false-positing on placebos? This mirrors E3 (real-data kill criterion for the belief monitor),
one level up, on the slow rule of `theory/learning_as_control.md`.

## Name the failure modes first (/litadapt discipline)

Stated before any data is touched, so a null is read honestly rather than tuned away:

1. **Over-call.** Real influence `u` is confounded with everything (arousal, difficulty, time
   on task). The detector conditions on the `f_0` (reward-prediction-error) channel and is
   surrogate-calibrated, but real confounds are richer than the E6 single confound, so a
   false-positive is a live risk. The placebo control (below) is the guard.
2. **Under-call.** Real learning is slow and noisy and sessions are short, so the inferred
   `theta`-trajectory may not carry enough increments to detect a true `u` term. E6 shows power
   scales with trajectory length `T`; real `T` may be below the detectable regime.
3. **Non-identifiability.** The inferred learning rule may not separate "u changed the rule"
   from "the rule drifted" at achievable `N`. The detector must then *abstain*, not guess.

## Estimand

Whether the external signal `u_t` enters the inferred learning rule, conditioned on the
unmanipulated channel `f_0`: the nested surrogate-calibrated Lambda test of E6, on real
increments. Reported as a call / no-call / abstain at level alpha, with the effect sizes from
`learning_as_control.md` (barrier susceptibility, DiD rule-shift, persistence) where the data
support them.

## Data requirements

A dataset qualifies only if it has all three:
- **Trial-by-trial de novo learning:** repeated choices while an agent acquires a task, so a
  weight trajectory `theta_t` can be inferred (PsyTrack / dynamic Bernoulli GLM [1], GLM-HMM [2]).
- **A known external signal `u_t`:** an experimentally set or logged influence that accumulates
  over trials and that we can point at as the candidate manipulation.
- **A reward / feedback channel** to form the `f_0` regressor (RPE * feature). A neural value
  proxy (NAcc reward-anticipation) is a bonus, not a requirement (see the neural arm).

## Candidate datasets (honest about the "external influence" mapping)

Tiered by how cleanly `u` maps to a real influence; none is a perfect "AI manipulates learning"
set, and that gap is stated, not hidden.

- **Tier 1, animal de novo learning with a designed manipulation.** IBL mouse decision-making:
  de novo acquisition, PsyTrack-fittable, and the biased-block structure is a real external
  signal `u_t`. Question: does the block manipulation enter the *learning rule*, not just the
  instantaneous bias. Clean positive control because the manipulation is known and strong.
- **Tier 2, human reinforcement learning with an external advice / framing signal.** Public
  OpenNeuro / behavioral sets where advice, social information, or framing is manipulated during
  a learning task; `u_t` is the manipulated advice/framing stream. Closest behavioral analog of
  external influence on human learning.
- **Tier 3, AI-in-the-loop learning (the target regime, emerging).** Studies where an LLM gives
  advice over trials, including the program's own `steering-in-the-wild` / SPAR agent-steering
  data. This is the regime the theory is about; data is scarce and small, so it is the hardest-
  power case and should be run last, after the apparatus proves out on Tiers 1-2.

## Controls (each reported alongside the number)

- **Naive detector (the E6 control):** omit the `f_0` term. It must be worse-calibrated than the
  proper detector on real data too, or the conditioning is not buying what E6 says it buys.
- **Placebo influence channel:** a real, logged covariate that is NOT the manipulation (e.g. a
  time-of-day or trial-index regressor with matched autocorrelation). The detector must return
  no-call on the placebo. A placebo that calls is an over-call and fires failure mode 1.
- **Surrogate calibration:** circular-shift `u` as in E6, preserving autocorrelation.
- **Positive control:** on the Tier-1 set where the manipulation is known and strong, the
  detector must call. A no-call there means it is underpowered (failure mode 2), not that the
  influence is absent.

## Kill criterion

If, on the Tier-1 positive control, the proper detector cannot distinguish the known
manipulation from the placebo and the surrogate at achievable `N`, the real-data Lambda detector
**fails** and we report that. Layers 1 and 3 of the learning-as-control defense (rule-robustness,
rule-relaxation) do not depend on this detector, so a Layer-2 null here is not fatal to the
extension, exactly as in `model.md`.

## Procedure

1. Fit the dynamic-GLM weight trajectory `theta_t` per subject (PsyTrack), holding out sessions.
2. Form increments `dtheta_t`, the `f_0` regressor (RPE * feature), and the `u_t` channel.
3. Run the proper and naive detectors with surrogate calibration (reuse `e6_lambda_power.py`'s
   `detect_p`); add the placebo channel as a third arm.
4. Where a call survives, report the effect sizes and their persistence (does the inferred rule
   stay shifted after `u` stops within the session).
5. Neural arm (optional, where NAcc / value-axis data exist): add the reward-anticipation
   regressor as an observed proxy for the value channel [3, 4] (or a CANlab value signature as
   the axis), and report whether it increases detection power / reduces the placebo call rate,
   i.e. whether the neural grounding buys identifiability as the theory predicts.

## Reporting rules

- Council-review this protocol before running, and again after the Tier-1 positive control.
- No threshold tuning until the positive control passes and the placebo returns no-call.
- Every number ships with its placebo, its naive-detector control, and a one-line statement of
  what result would have falsified it.

## References

[1] Roy, Bak, Akrami, Brody, Pillow. *Extracting the dynamics of behavior in sensory
decision-making experiments.* Neuron, 2021 (PsyTrack).

[2] Ashwood, Roy, Stone, et al., Pillow. *Mice alternate between discrete strategies during
perceptual decision-making.* Nature Neuroscience, 2022 (GLM-HMM).

[3] Knutson, Adams, Fong, Hommer. *Anticipation of increasing monetary reward selectively
recruits nucleus accumbens.* Journal of Neuroscience, 2001.

[4] Genevsky, Yoon, Knutson. *When brain beats behavior: neuroforecasting crowdfunding
outcomes.* Journal of Neuroscience, 2017.
