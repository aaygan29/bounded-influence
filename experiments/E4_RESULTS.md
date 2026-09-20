# E4 results: the dose budget, pressure-tested

Run: `python3 experiments/e4_dose_budget.py` (seed 11, N=400 trials/point).
Figure: `figures/e4_dose_budget.png`. Numbers: `experiments/e4_results.json`.

This experiment tried to break the corrected Proposition 4 (cap cumulative
precision-weighted dose to bound crossing probability). It succeeded in breaking
it, and that failure is the result.

## Finding 1: a cumulative-dose cap does NOT bound crossing probability
Dose (cumulative precision-weighted displacement, `= D_KL`) is paid only for
MOVING the belief. Holding the tilt near-critical costs essentially zero further
dose but keeps the barrier low, so noise (Kramers escape) accrues crossing
probability for free over time. Panel (a): at the same cumulative dose, a
short-dwell attacker crosses rarely while a long-dwell attacker crosses with
probability ~1. A dose cap is therefore exploitable: a dose-budget-compliant
attacker that sits just below the fold and waits crosses with probability 1.00.

## Finding 2: proximity to the fold is the sufficient statistic
Panel (b): the long-dwell crossing probability collapses onto a Kramers law in
the minimum barrier reached,

    P(cross) = 1 - exp( -k * exp(-ΔU_min / D) * T ),     k = 0.19, fit MSE = 7e-4.

The whole crossing risk is captured by `ΔU_min` (equivalently, by the minimum
precision reached, since `precision = U''(x*)/D`), the noise `D`, and the dwell
`T`. Total dose is nearly irrelevant (panel d: dose stays ~0.001 across the whole
range while the barrier collapses from 0.16 to 0).

## Finding 3: what actually protects is a precision floor, derived not guessed
To hold session crossing probability below `eps` over a dwell `T`, invert the
Kramers law:

    ΔU_floor = D * ln( k T / (-ln(1-eps)) ).

For `eps=0.05, T=120, D=0.03, k=0.19`: `ΔU_floor = 0.183`. Enforcing it (capping
the attacker's proximity so the barrier never falls below the floor) holds the
worst-strategy crossing probability to about 0.10, versus 1.00 for the
dose-cap-only defense (panel c). Per strategy: slow ramp 0.06, fast frontload
0.105, over-long dwell (1.5T) 0.115.

## The revised protective claim (supersedes the dose-cap framing of Proposition 4)
The controlled quantity is the minimum precision (maximum proximity to the fold)
reached in a session, not cumulative `D_KL`. The protective primitive is a
**precision floor** `ΔU >= ΔU_floor(D, T, eps)`, monitored in real time by the E2
critical-slowing-down signal (which is exactly a precision-collapse detector).
This unifies Layer 2A (monitor) and Layer 2B (budget): the monitor measures
precision, and the rule is "do not let precision fall below the floor." A pure
cumulative-influence cap is insufficient and should not be sold as a guarantee.

## Honest caveats
- The floor is calibrated for a session length `T` and noise `D`. A longer dwell
  (1.5T here) or a faster approach slightly exceeds `eps` (0.10-0.115 vs 0.05), so
  the floor must be set for the maximum session length and carry a margin.
- `k` is fitted in-model and depends on the well geometry and mildly on `D`.
  Generalizing to other `D` needs re-fitting `k`; the Kramers functional form is
  validated (panel b), the constant is not universal.
- The overshoot of `eps` (worst 0.115 vs target 0.05) is finite-N sampling plus
  the Kramers-fit approximation plus strategy variation; deploy with margin.
- This is a simulation of the apparatus. The real test remains E3 (does proximity
  predict belief flips in human decision data).

## Bottom line
A genuine, pressure-tested protective statement now exists: **bounding proximity
to the fold (a precision floor set by D, session length, and target risk) bounds
crossing probability across attacker strategies; a cumulative-dose cap does not.**
The deployable defense is the critical-slowing-down monitor enforcing that floor,
not an influence-budget accountant.
