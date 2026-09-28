# Learning as control: manipulating the landscape, not just the ball

Status: proposed extension. Not worked out to the standard of `model.md`, and nothing here is a
result. This document states the direction, the measurable claims it would have to make, and the
kill criterion that would end it, so it can be picked up without re-deriving the framing.

## The gap this fills

`model.md` manipulates a belief state `x` inside a potential `U(x; a)`. The attacker lowers a
barrier and drives `x` over a separatrix; hysteresis makes the crossing stick. But `U` is treated
as fixed on the timescale of an attack. That is the acute-persuasion regime.

The regime this document adds is slower and more serious: an attacker who does not push the ball
over a hill but **reshapes the hill**. Concretely, an AI that shapes a person's (or a model's)
*learning* changes how future beliefs form at all. The object of attack is the learning rule, not
the current belief. A defense program built only on state manipulation is blind to it.

## The one equation, one level up

Let the landscape be parameterized by slow variables `theta` (barrier height, well positions,
coupling), and let those evolve by a learning rule driven by experience `e(t)` and an attacker
input `u(t)`:

```
fast:   dx/dt   = -dU/dx (x; theta) + g s(t) + sqrt(2D) xi(t)      (the belief, as before)
slow:   dtheta/dt = f_phi(theta, e(t), u(t))                        (the learning rule)
```

This is a two-timescale (singular-perturbation) system: `x` relaxes fast within the current
landscape, `theta` drifts slowly as the agent learns. `model.md` is the `dtheta/dt = 0` special
case. The manipulation question is whether `u` enters `f_phi` non-trivially, i.e. whether an
external input can bend the learning rule itself. This is exactly the "static vs a rule that is
itself drifting" distinction, made into a control problem: the attacker is a controller on the
slow manifold.

## Why this is the right frame, not a metaphor

The slow/fast split is the same center-manifold logic `foundations.md` already uses for `x`,
applied one level up: near a learning transition the many synaptic degrees of freedom are slaved
to a few slow collective coordinates (`theta`), and those govern how the landscape moves. The
pieces are borrowed from established behavioral-learning mathematics, not invented:

- The belief/weight trajectory `theta_t` is the smoothly evolving weight vector of a dynamic
  Bernoulli GLM (Roy, Bak, Akrami, Brody & Pillow, PsyTrack [1]; discrete-strategy GLM-HMM [2]).
- The learning rule `f_phi` is inferable from behavior without assuming its form (Liu, Geadah &
  Pillow [3]). Their own limitation, that a static inferred rule cannot be told apart from a rule
  that is itself drifting, is precisely the opening for a manipulation term `u`.
- The regime of the landscape, shallow-and-linear vs deep-and-feature-shaped, is the rich/lazy
  order parameter (Liu, Baratin, Cornford, Mihalas, Shea-Brown & Lajoie [4]). Rich vs lazy maps
  onto barrier geometry: a lazy regime is a shallow, easily-reshaped landscape; a rich regime is a
  deep, latched one. Moving an agent along the lazy<->rich axis IS reshaping `U`.
- Why manipulating a credit/reward signal reshapes the rule at all is the credit-assignment
  picture of Liu et al. [5].
- That changing the learning rule itself, both the objective and whether credit is assigned
  locally or globally, causally changes which representations form is shown directly by Gütlin,
  Kittelmann & Auksztulewicz [9]. Holding architecture, task, and capacity fixed in
  parameter-matched recurrent networks, they vary only the learning objective (predictive vs
  contrastive vs supervised) and the mechanism (local vs global), and find that the resulting
  representations diverge and that a local-predictive rule, not a supervised one, is what human
  neural representations converge to over the course of learning (RSA against EEG; the brain
  attenuates category-specific structure and retains predictive structure). This is the empirical
  warrant for treating "move the agent along the learning-rule axis" as a real reshaping of `U`
  with a measurable neural signature rather than a modeling convenience, and it supplies a clean
  parameter-matched paradigm as prior art for isolating a rule change from an architecture change.
  It is a single-dataset preprint with moderate Bayes factors, so it is cited as motivating, not
  settled, evidence.

## The estimand: did the input change the learning rule?

The measurement is a **difference in the inferred learning-rule functional**, not a difference in
choices. Fit `f_phi` on a control arm and a manipulated arm (matched experience, attacker input
present in one), and test whether the inferred rules differ (a difference-in-differences on the
update operator). Two agents can make identical choices while one's update rule has been reshaped;
the estimand is built to separate "changed what they chose" from "changed how they learn."

## Worked dynamics: detecting and measuring external influence on learning

This section turns the framing into equations built on the existing double-well bifurcation, so
"is there external influence, and what is its impact on the learning" becomes a set of estimators
with a stated false-report rate.

### The two-timescale system, with a reward-driven rule

Keep the fast belief of `model.md` and give the slow landscape a reward-prediction-error learning
rule, the standard credit-assignment form and the analyzable backbone of the flexible rule in [3]:

```
fast:   dx/dt      = -U'(x; theta) + g s(t) + sqrt(2D) xi(t)
slow:   theta_{t+1} = theta_t + eta_t * delta_t * z_t
        delta_t     = r_t - v_t          (reward prediction error)
        v_t         = anticipated value  (the credit/value channel)
        z_t         = credit / eligibility vector (which past states get updated)
```

`theta` are the landscape parameters: barrier height `DU(theta)` and attractor curvature
`kappa(theta) = U''(x*; theta)`. Learning reshapes `U` through `theta`; `model.md` is the frozen
case `eta = 0`.

### Neural grounding: the value channel is observable (Genevsky and Knutson)

The one latent that makes or breaks identifiability is `v_t`. Nucleus-accumbens reward-anticipation
activity is a readout of it: NAcc anticipatory signal `a_t` tracks anticipated value [6], and
crucially it forecasts choices and aggregate outcomes beyond self-report and past behavior
(neuroforecasting) [7, 8]. So add an observation model:

```
v_t = beta * a_t + e_t            (NAcc anticipation as an observed proxy for the value channel)
```

This is what lets the detector separate "the value signal was manipulated" from "the rule drifted
on its own": `a_t` carries manipulable, decision-relevant value information that behavior alone
does not, exactly the neuroforecasting result.

### External influence as a control input on the slow manifold

An attacker enters through the reward/value/credit channel, in three concrete places that map to
the three defenses:

```
(i)   value bias:      v_t   -> v_t + b(u_t)          affect-charged framing inflates anticipation
(ii)  plasticity gain: eta_t -> eta_t * (1 + c(u_t))  urgency/arousal raises the learning rate
(iii) credit reshape:  z_t   -> z_t + w(u_t)          misattribution of what caused the outcome
```

Each adds a `u`-proportional term to the update. Linearizing, the slow dynamics become a controlled
system:

```
dtheta/dt = f_0(theta, delta_t) + B(theta) u_t
```

`f_0` is the unmanipulated learning rule; `B(theta) u_t` is the influence. "Is there external
influence" is the question of whether `B != 0`.

### Impact on the learning theory: barrier susceptibility

The impact is the induced change in the landscape geometry. By the chain rule the barrier moves as

```
d(DU)/dt = grad_theta(DU) . f_0  +  grad_theta(DU) . B(theta) u_t
                                    \_________________________/
                                     chi_DU  =  barrier susceptibility to influence
```

`chi_DU = grad_theta(DU) . B` is the headline measurable: how much a unit of influence lowers the
belief barrier per unit time, i.e. how much easier a future belief flip becomes per unit of
manipulation. Cumulatively, the induced barrier deficit is `integral chi_DU u dt`, which is the
learning-level analog of the belief-dose of `model.md`.

### Detection: two critical-slowing-down signals plus a specificity test

Bifurcation theory gives the early-warning basis directly.

- Fast CSD (already Layer 2 of `model.md`): as the barrier flattens, `kappa(theta) -> 0`, the
  belief relaxation time `tau_x = 1/kappa` diverges, seen as rising lag-1 autocorrelation and
  variance of `x` (choice variability, response latency).
- Slow CSD (new): as `theta` approaches a rule regime shift, the leading eigenvalue of the update
  Jacobian `partial f / partial theta` approaches zero, seen as rising variance and AR(1) of the
  inferred `theta`-increments. This is the early warning of a rule change, one level up.

Neither CSD says the change is externally driven. For that, a nested-model specificity test:

```
H0: dtheta/dt = f_0(theta, delta_t)                 (no influence)
H1: dtheta/dt = f_0(theta, delta_t) + B(theta) u_t  (influence enters the rule)

Lambda = 2 [ log L(H1) - log L(H0) ]  on held-out trajectory
```

Calibrate `Lambda` against a null built by surrogate-shuffling `u_t` against the trajectory (breaks
the `u`-`theta` association, preserves marginals), giving a finite-sample p-value and a call /
no-call decision at level alpha, with abstention when underpowered. This is the same calibration
layer the program already uses (`whisper_audit` `cal-v1`, WARDEN abstention). The NAcc regressor
`a_t` enters `f_0` as an observed value proxy, which shrinks the `H0`/`H1` confound and is where the
neuroforecasting result buys detection power.

### Measurement: four effect sizes on the learning theory

Once influence is detected, its impact is reported as:

```
1. Barrier susceptibility     chi_DU   = d(DU) / d(integral u dt)      (flip-ease per unit dose)
2. Rule-shift magnitude       D_rule   = E_x || f_1(x) - f_0(x) ||     (DiD on the update operator)
3. Persistence (hysteresis)   after u -> 0 at T, residual DU(inf) - DU_baseline; loop area in (u, DU)
4. Regime shift               change in the rich/lazy order parameter (effective rank / curvature
                              spectrum of the learned representation [4])
```

Effect 3 is the serious one and the learning-level analog of belief hysteresis: a bounded dose that
leaves a permanent barrier deficit means the *rule* latched, not just the belief.

### Simulation status (E6)

`experiments/e6_lambda_power.py` is a first gate on this detector. On simulated two-timescale
trajectories with the reward-timed confound (`u` correlated with the `f_0` channel), the proper
nested test holds its false-positive rate at alpha at `B = 0` and gains power with influence
strength and trajectory length (power to 1.0 by `B = 0.4`, `T = 300`), while the naive detector
that omits `f_0` false-positives at ~0.99. That is apparatus validation, not a result on real
data; the real-influence analog of E3, with the kill criterion armed, is still owed.

### Defense (Layer 2, one level up): a rule-dose budget

Cap cumulative influence so the predicted barrier reduction stays below a safety margin:

```
integral chi_DU u dt  <=  DU_min
```

a per-session budget on rule manipulation, the direct analog of the belief-dose budget in E4.

## Measurable claims (the defense taxonomy, one level up)

The three-defense geometry of `model.md` is forced again, now on the slow variable:

| Layer | Handle on the learning rule | Measurable claim | Honest limit |
|---|---|---|---|
| 1 Rule-robustness | regularize `f_phi` against `u` | an intervention is protective iff it measurably reduces the DiD rule-shift out of sample | may blunt legitimate learning too |
| 2 Rule-dose detection + bound | watch drift of inferred `theta`/`f_phi`; cap cumulative rule shift | early warning fires before a **rule regime shift** (lazy->rich or a latched update); per-session cap on inferred-rule drift | fails if rule change is not identifiable from data (kill criterion) |
| 3 Rule-relaxation | engineer return of `theta` to baseline | designed review/consolidation shrinks a rule-hysteresis loop | needs a designed relaxation period |

The headline analog of hysteresis is **rule hysteresis**: a bounded dose of `u` induces a change
in `f_phi` (equivalently in `U`) that persists after `u` stops. A transient nudge to belief is
Layer-2-of-`model.md`; a persistent change to the update rule is the new harm, and its signature
is persistence of the DiD rule-shift past the end of the input.

## Attacker-side fingerprint

An AI optimizing to reshape how a target learns has a different signature from one optimizing to
flip a single belief: low-and-slow, spread across many interactions, aimed at the credit/reward
channel rather than acute near-critical dosing. That is a distinct tripwire from the persuasion-
optimizer fingerprint in `model.md`, and it ties to the same covert-agenda line: a system quietly
driving another system's learning is a driven-agenda model.

## The crux: identifiability

The whole direction lives or dies on whether "manipulation changed the learning rule" can be told
apart from "the rule was drifting anyway" or from noise. This is the same non-identifiability Liu,
Geadah & Pillow flag, and it is why the calibrated-abstention discipline from the program (WARDEN,
conformal) is load-bearing here, not decorative: the monitor must abstain when the data do not
constrain the inferred rule, and a rule-shift is only reported when it clears a stated false-report
rate.

## Worked demonstration: reproduction and a two-channel detector

The identifiability crux above motivates a **second, independent readout** of the same rule change,
and Gütlin, Kittelmann & Auksztulewicz [9] supplies it: the learning objective leaves a
*representational* signature (position on a predictive-vs-category RSA axis), not only a behavioral
one. `experiments/gutlin_repro/` builds this out on a controlled system.

- **Reproduction (`run_repro.py`).** Parameter-matched two-layer RNNs, seven conditions (objective x
  mechanism), RSA against a synthetic early/late neural ground truth. Across 5 seeds the paper's
  headline reproduces robustly: a **predictive objective uniquely captures predictive structure**
  (Spearman rho 0.46-0.66) that supervised and contrastive objectives do not (rho ~0 or negative).
  The category-stage half (supervised best on category) is seed-fragile (3/5), because the untrained
  architecture is the ceiling on raw category geometry. Reported as a partial, mechanism-level
  reproduction; the neural target is synthetic, not the McDermott EEG (which we do not have).

- **Two-channel detector (`detect.py`).** A control arm learns the true predictive rule; a
  manipulated arm has, for a fraction of trials, its predictive *target* bent to an attacker-chosen
  partner (matched experience, only the learning target moved: an input `u` on the slow manifold).
  Two difference-in-differences (manipulated minus control): a **behavioral** channel (fidelity of
  the projected next-step prediction to the true partner) and a **representational** channel (rho of
  the leading-grouped RDM to the true predictive target). Against a null of two clean arms:
  - behavioral channel separates attack from null cleanly at every scale tested;
  - the representational channel is **underpowered at 8 categories** (28-cell RDMs, bootstrap
    variance swamps the effect, one seed flips sign) and becomes **reliable at 20 categories**
    (attack repr DiD -0.13 +/- 0.02 vs null +0.04 +/- 0.05, both channels move under attack, both
    separated from null across 5 seeds). The named failure mode (small-RDM variance) was diagnosed
    before the fix (more categories), and the fix rescued it.

- **Behavior-matched manipulation (`behavior_matched.py`): the identifiability test.** The attack
  above changes behavior, so it does not test the crux. This experiment does. A teacher learns the
  true rule; two students are distilled to reproduce the teacher's outputs (behavior pinned), and one
  student additionally gets an attacker objective that bends its latent geometry through a separate
  head. Against a noise floor of two independently-initialised clean students:
  - behavior is matched in both arms (output agreement 1.000, choice-fidelity DiD 0.000): the
    behavioral channel is blind to the manipulation;
  - the attack arm's representation drifts **below the clean-student noise floor** (floor rho
    0.81 +/- 0.02; attack sits 0.25 +/- 0.03 below it at lam=0.5), positive on 5/5 seeds;
  - dose-response: over a 50x range of attack strength (lam 0.1 -> 5.0) behavior stays matched
    (agreement 1.000 throughout) while the representational separation grows monotonically
    (0.10 -> 0.41 -> 0.52).
  This is the existence result the crux needs: a rule manipulation that is invisible to behavior is
  still detectable representationally, beyond clean-student init noise. The second channel breaks the
  non-identifiability **on controlled synthetic data**.

- **What is still open.** The break is shown on the synthetic paradigm, not on real neural/behavioral
  data. The remaining step is to run the same behavior-matched-vs-noise-floor test on real data (IBL
  biased-block, human RL-with-advice) using the inferred-rule machinery from `whisper_audit` /
  `decision_phenotype`, where the representational channel is a probed internal direction rather than
  a synthetic RDM. The protocol for this is `experiments/E7B_BEHAVIOR_MATCHED_PROTOCOL.md` (the
  behavior-matched companion to the Lambda-detector test in `experiments/E7_PROTOCOL.md`): match
  behavior across a real manipulation, then test whether a representational channel separates beyond
  a matched-control noise floor and beyond a behavioral-only classifier.

## Instrument

Behavioral side: inferred `f_phi` from trial-by-trial data (the decision-phenotype and
value-decision-stack lines already do learning-rule / decision-process inference). Internal side:
once validated on loyalties, the `whisper_audit` reading-and-steering machinery becomes the causal
probe on representational dynamics, testing whether steering an internal direction changes the
*learning trajectory* and not just the instantaneous output. The rich/lazy read is the landscape-
regime measurement.

## Kill criterion

If the DiD rule-shift is not identifiable at a stated false-report rate in real behavior-plus-probe
data, or if inferred-rule drift shows no critical slowing down before a regime shift, the Layer-2
rule-monitor fails and we report that. Layers 1 and 3 (rule-robustness, rule-relaxation) survive
independently, so this is not all-or-nothing, exactly as in `model.md`.

## Relation to the rest of the model

This is a third distinction alongside the individual/collective "Two scales" of `model.md`: not a
new spatial scale but a new **temporal/meta** one. The individual model manipulates the fast belief
state; this manipulates the slow landscape that generates beliefs. A per-session dose budget on
rule drift is the natural bridge to the collective-scale erosion story, since a population whose
*learning* has been slowly reshaped is agency-eroded in a way no single belief flip captures.

## References

[1] N. A. Roy, J. H. Bak, A. Akrami, C. D. Brody, J. W. Pillow. *Extracting the dynamics of
behavior in sensory decision-making experiments.* Neuron, 2021 (PsyTrack).

[2] Z. C. Ashwood, N. A. Roy, I. R. Stone, et al., J. W. Pillow. *Mice alternate between discrete
strategies during perceptual decision-making.* Nature Neuroscience, 2022 (GLM-HMM).

[3] Y. H. Liu, V. Geadah, J. W. Pillow. *Flexible inference for animal learning rules using neural
networks.* NeurIPS, 2025.

[4] Y. H. Liu, A. Baratin, J. Cornford, S. Mihalas, E. Shea-Brown, G. Lajoie. *How connectivity
structure shapes rich and lazy learning in neural circuits.* ICLR, 2024. arXiv:2310.08513.

[5] Y. H. Liu, S. Mihalas, E. Shea-Brown, et al. *Cell-type-specific neuromodulation guides
synaptic credit assignment in a spiking neural network.* PNAS, 2021.

[6] B. Knutson, C. M. Adams, G. W. Fong, D. Hommer. *Anticipation of increasing monetary reward
selectively recruits nucleus accumbens.* Journal of Neuroscience, 2001.

[7] A. Genevsky, B. Knutson. *Neural affective mechanisms predict market-level microlending.*
Psychological Science, 2015.

[8] A. Genevsky, C. Yoon, B. Knutson. *When brain beats behavior: neuroforecasting crowdfunding
outcomes.* Journal of Neuroscience, 2017.

[9] D. Gütlin, D. Kittelmann, R. Auksztulewicz. *Predictive coding networks capture human neural
representations missing in supervised DNNs.* bioRxiv, 2026. doi:10.1101/2026.09.18.752626.
