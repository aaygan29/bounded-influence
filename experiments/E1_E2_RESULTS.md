# E1 + E2 results

Run: `python3 experiments/e1_e2_dissociation.py` (seed 7, 200 crossings per regime).
Figure: `figures/e1_e2_dissociation.png`. Numbers: `experiments/e1_e2_results.json`.

## What was tested
E1 built the belief double-well `U(x;a) = x^4/4 - x^2/2 - a x` and confirmed the
fold at `a* = 2/(3*sqrt(3)) = 0.385` (panel a). E2 asked the key
question directly: does the critical-slowing-down early warning (trailing
residual variance of fluctuations about the moving well bottom) fire before a
crossing, and does it do so only in the regime where theory says it should?

Three conditions, same noise `D = 0.025`:
- B, bifurcation ramp: `a` ramped `0 -> 0.6` past `a*`.
- A, noise-activated: `a` held at 0.24 (barrier intact); crossings are rare noise escapes.
- S, surrogate: `a = 0.24` non-crossing trials, used to set the warning threshold (95th percentile of the surrogate variance peak) and as the negative control.

## Result: the dissociation holds
| | advance-warning rate | median lead time |
|---|---|---|
| B bifurcation | 0.375 | 5.8 time units |
| A noise-activated | 0.035 | (see caveat) |

The warning fires before 37.5% of bifurcation crossings with a median lead of
about 6 time units, and before only 3.5% of noise-activated crossings. Panel (c)
shows why: in B the variance rises above threshold roughly 10 time units before
the crossing; in A it stays at baseline until the crossing itself.

This is exactly what the Proposition 3 split predicted: usable lead
time exists in the bifurcation-tipping regime, not in the noise-activated regime.
The two regimes are physically different (a foreseeable slide over a vanishing
barrier vs an abrupt random escape over an intact one), and the early-warning
defense only works for the first.

## Honest caveats
- A's 3.5% is at the false-alarm floor. A 95th-percentile threshold trips on ~5%
  of surrogate trials by construction, so A's detections are statistically
  indistinguishable from chance. The few A "warnings" are false alarms, not
  precursors, which is why their lead times are scattered and large in panel (d)
  rather than concentrated like B's. Panel (e) is the unambiguous summary.
- B's rate is 37.5%, not near 100%. The detector is a first-pass, conservatively
  calibrated single statistic. The result is the dissociation (10x separation
  from the null floor), not the absolute rate, which depends on detector tuning,
  window length, and how far past `a*` the ramp goes.
- Fast ramps were not stress-tested here. Rate-induced tipping (a fast push
  through the fold) is expected to shrink the lead time; that is the next probe
  and a known failure mode of critical-slowing-down warnings.
- This is a simulation of the apparatus, not a real belief dataset. The real
  kill-criterion test is E3 (does critical slowing down precede belief flips in
  human decision data). E1/E2 only establish that the detector works where the
  theory says it must and stays silent where it must, which is the precondition
  for E3 to be meaningful.

## Verdict against the spec
E1/E2 pass: the apparatus is real and the early-warning signal dissociates the
two crossing regimes as predicted. This licenses proceeding to E3. It does not
yet establish the deployable defense; that needs real data and a fast-drive
stress test.
