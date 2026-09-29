# Collaborating on bounded-influence

This project is looking for a second pair of eyes and a second set of skills. It is easiest to help if you know what is solid, what is not, and what is missing.

## What is solid, what is not

- **Solid:** the simulation reproduces exactly from its seeds, and the numbers in the write-ups match the committed JSON. See [PROVENANCE.md](PROVENANCE.md).
- **Not yet solid:** the model itself. The bistable potential is a modelling choice. Whether human belief change actually shows critical slowing down before a flip is untested.
- **Known weaknesses in what exists:** the detector is a single first-pass statistic and catches only 37.5% of forced crossings; fast drives (rate-induced tipping) have not been tested; the fitted constant in E4 applies to one noise level and one well shape; the noise level was not calibrated to any dataset.

## Open problems, roughly in order of value

1. **E3: the real-data test.** Find a public dataset with repeated choices under accumulating influence, build belief-state proxies (choice, latency, variability), and test whether critical slowing down precedes observed belief flips. Plan in [experiments/README.md](experiments/README.md). This decides whether the early-warning idea survives.
2. **Fast-drive stress test.** Measure how much lead time shrinks under rate-induced tipping.
3. **Calibrate noise to behaviour.** Replace the chosen `D` with a value estimated from real choice variability.
4. **Add citations for the dynamical-systems background** (critical slowing down as an early-warning signal, Kramers escape) and audit the outside sources listed in PROVENANCE.md.
5. **E5: attacker fingerprint.** Can a persuasion-optimising agent be told apart from a matched non-optimising one?
6. **Stronger detectors.** Anything better than one trailing-variance statistic, with the same false-alarm control.
7. **Connection to manipulation studies.** If you work on manipulation or steering experiments, the most useful contribution is probably a mapping from your paradigm to the state variable `x`, the drive `a`, and an observable response.

## Running it

```
python3 experiments/e1_e2_dissociation.py
python3 experiments/e4_dose_budget.py
```
Needs Python 3, `numpy`, `matplotlib`.

## Ground rules

- Report negative results. The repository already contains one (a cumulative dose cap does not bound crossing risk), and it made the design better.
- Every reported number should come with a control and a note on what would have falsified it.
- Dual-use: work here is defensive. Do not publish or share an optimal attack vector; outputs should be lead times and stop or disclose triggers. See the dual-use section of [SPEC.md](SPEC.md).
