# Black-box learning-rule influence detector

A simple, self-contained, mathematically-grounded detector for external influence on a
LEARNING RULE, observable in **black box** (behaviour only) and validated entirely in
simulation, so it needs no real training data. This is the tractable core the project
refocused on after the white-box representational route hit an honest null on real IBL
model-subjects (`../e7b_model_subject/`) and the identifiability-constrained rescue
(`../E7B_R1_IDENTIFIABILITY_CONSTRAINED_PREREG.md`) required data we do not have.

## The premise and the question

We assume, by default, that external influence CAN affect a learning rule (the thing the
bounded-influence program exists to defend against). Given only an agent's behaviour, can
we detect it, and tell it apart from mere output bias?

Observables per trial: stimulus x_t, binary choice y_t, reward r_t, candidate influence
u_t. Hidden: the agent's internal state and the form of its update rule.

## The identifying idea (why black box suffices)

Track the agent's policy as a time-varying logistic map, P(y_t=1)=sigmoid(a_t + b_t x_t),
recovering the weight trajectory theta_hat_t with a dynamic GLM (PsyTrack; Roy, Bak,
Akrami, Brody & Pillow 2021 [1]). Two kinds of influence leave DIFFERENT fingerprints:

- **Choice/bias influence** adds u_t to the logit LEVEL each trial. The weight tracks u_t
  contemporaneously: `level ~ u_t`.
- **Learning-rule influence** adds u_t to the UPDATE. The weight then carries the INTEGRAL
  of influence: `level ~ sum_{s<=t} u_s`.

So the test is: does the *integrated* influence explain the weight trajectory, once the
*contemporaneous* influence (pure bias) is controlled for? That level-vs-integral split is
exactly the "does the external signal enter the update operator" estimand of Liu, Geadah &
Pillow (2025) [3], and it clears the non-identifiability their own method flags.

## Method (`bb_detector.py`)

1. Windowed logistic tracking of theta_hat_t (light dynamic GLM [1,2]).
2. Leaky integrals of u_t and of the reward-prediction-error channel (the agent's own
   learning signal).
3. Nested test on the LEVEL trajectory: integrated-u added to {endogenous drivers,
   contemporaneous u_t}. The contemporaneous term absorbs pure bias, leaving only
   accumulated (rule) influence.
4. Joint statistic over the (a, b) channels; surrogate null by circular-shifting u_t
   (preserves autocorrelation) before integrating. call / no-call / **abstain** at alpha,
   abstaining when the trajectory is unidentifiable (a "no power" guard, not "no effect").

## Validation (`validate_bb.py`, simulation, no real data)

Three ground-truth conditions: H0 (u present, does not enter rule or choice), rule (u
enters the update), bias (u shifts the choice logit each trial but not the update; the
detector must NOT fire). 40 seeds, T=800.

```
calibration  FPR(H0)        = 0.027   (target <= 0.05)
power        TPR(rule)      = 0.850
specificity  FPR(bias-only) = 0.050   (fires on the RULE, not on bias)
```

Dose-response over influence strength kappa (`bb_dose_response.json`):

| kappa | FPR(H0) | power(rule) | FPR(bias) |
|---|---|---|---|
| 0.05 | 0.027 | 0.487 | 0.029 |
| 0.10 | 0.027 | 0.825 | 0.000 |
| 0.15 | 0.027 | 0.850 | 0.050 |
| 0.25 | 0.027 | 0.750 | 0.075 |

Calibration is stable; power rises with influence strength then plateaus (the slight dip
at large kappa is weight-clipping in the simulated agent); specificity holds, creeping to
0.075 only at extreme bias. **The detector is calibrated, powered, and specific.**

## Honest scope

- Validated in simulation with a logistic-agent ground truth. It establishes the
  detector's mathematical properties (calibration / power / specificity), not that any
  particular real agent has been manipulated.
- The specificity that matters (bias vs rule) rests on the level-vs-integral distinction,
  which assumes the tracked policy is a reasonable summary of the agent. Richer agents
  (multi-strategy, history-dependent) would need the GLM-HMM extension [2].
- On real behavioural data the same test applies unchanged (track, integrate, nested test,
  surrogate null); the E7B kill-criterion controls (placebo channel, behavioral-only
  baseline) in `../E7B_PROTOCOL.md` then govern a real call.

## Run

```
python validate_bb.py --seeds 40 --T 800 --kappa 0.15
```

Requires numpy + scipy. Runtime ~1 min on CPU.

## References

[1] N. A. Roy, J. H. Bak, A. Akrami, C. D. Brody, J. W. Pillow. *Extracting the dynamics of
behavior in sensory decision-making experiments.* Neuron, 2021 (PsyTrack). Dynamic-GLM
weight-trajectory tracking, the behavioural readout here.

[2] Z. C. Ashwood, N. A. Roy, I. R. Stone, et al., J. W. Pillow. *Mice alternate between
discrete strategies during perceptual decision-making.* Nature Neuroscience, 2022
(GLM-HMM). The multi-strategy extension for richer agents.

[3] Y. H. Liu, V. Geadah, J. W. Pillow. *Flexible inference for animal learning rules using
neural networks.* NeurIPS, 2025. The "does the external signal enter the update operator"
estimand and the static-vs-drifting-rule non-identifiability this detector clears.

[4] Y. H. Liu, A. Baratin, J. Cornford, S. Mihalas, E. Shea-Brown, G. Lajoie. *How
connectivity structure shapes rich and lazy learning in neural circuits.* ICLR, 2024
(arXiv:2310.08513). The rich/lazy regime of the recovered trajectory.

[5] Y. H. Liu, S. Mihalas, E. Shea-Brown, et al. *Cell-type-specific neuromodulation guides
synaptic credit assignment in a spiking neural network.* PNAS, 2021. Why a learned rule
leaves a readable trace at all.

[6] D. Gütlin, D. Kittelmann, R. Auksztulewicz. *Predictive coding networks capture human
neural representations missing in supervised DNNs.* bioRxiv, 2026,
doi:10.1101/2026.09.18.752626. Motivates the representational companion channel
(`../gutlin_repro/`).

[7] B. Bagley; bounded-influence program (`../../theory/learning_as_control.md`). The
control-theoretic two-timescale framing (fast belief, slow rule), the surrogate-calibrated
detection with a kill criterion, and the dose-budget defense this detector plugs into.
