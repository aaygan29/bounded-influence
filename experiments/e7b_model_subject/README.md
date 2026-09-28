# E7B Tier 3+ model-subject arm (real IBL data)

The real-data realisation of the behavior-matched two-channel test
(`../gutlin_repro/behavior_matched.py`, protocol `../E7B_BEHAVIOR_MATCHED_PROTOCOL.md`).
It was built after a full council review of the synthetic result, to eliminate the
confounds that review named. **Headline: on real data, with the circularity removed,
the behavior-matched manipulation is NOT representationally detectable. The synthetic
"break" does not replicate.** This is a negative result, reported as one.

## Why this exists: the council verdict on the synthetic result

The synthetic `behavior_matched.py` showed a behavior-matched manipulation separating
on a representational channel. The council review (Gate 3, Specificity, **Blocker**)
found the separation was largely circular: the attacker's auxiliary head wrote to the
same leading-grouped latent axis the representational channel read. The single control
named to change the verdict was: derive the probe independently of the attack, run a
behavioral-only classifier, and ground the subject in real data. This arm does exactly
that.

## Real data (comparable by construction)

`fetch_ibl.py` pulls REAL IBL biased-block 2AFC trial tables via the documented public
ONE account (not the user's credentials): 6 sessions, all running the identical
`biasedChoiceWorld` task, 599-1328 trials each, blocks {0.2, 0.5, 0.8}. One task, one
pipeline across all arms, so every comparison is apples-to-apples. The `.parquet`
blobs are gitignored; `data/manifest.json` records the session EIDs and trial counts
for reproducibility.

**Behavioral validation (`validate_behavior.py`).** The manipulation `u_t` = block
prior is real and present: on low-contrast trials, P(choose left) is higher in
left-blocks than right-blocks by **+0.255 ± 0.128** across the 6 sessions (canonical
IBL biased-block effect). Session 0 is a weak-engagement outlier (+0.09); sessions 1-3
are strong (+0.32 to +0.43).

## Design (confounds eliminated per the review)

- **Model-subject grounded in real behavior** (`model_subject.py`): a GRU processes the
  real trial sequence online and is trained by behavioural cloning to reproduce the
  mouse's real choices. Its recurrent state is the learning trajectory.
- **Behavior matched, not identical** (real-data honesty): control and attack arms are
  both cloned to the same real choices. Choice agreement between arms is **0.945**
  (residual 0.055, vs the synthetic 1.000). The behavioral-feature gap between arms
  (0.048) is within the clean-clean floor (0.038), so behavior does not separate the
  arms.
- **Non-circular representational readouts.** The attack manipulates the LEARNING RULE
  (over-integration of reward history, a long-horizon auxiliary objective that reshapes
  the recurrent dynamics). The readouts are independent of it:
  1. a functional value probe trained on CLEAN control subjects from OTHER sessions
     (held-out, cross-session), decoding SHORT-horizon value (different timescale from
     the attack's long-horizon target);
  2. a condition-RDM (hidden states grouped by block x signed-contrast), the faithful
     port of the synthetic RSA metric, invariant to the attack's target.
- **Controls:** clean-clean noise floor (two independent control subjects), a
  behavioral-only feature classifier, and the rich/lazy order parameter (hidden-state
  effective rank).

## Result (`e7b_results.json`, 6 real sessions)

| quantity | value | verdict |
|---|---|---|
| behavior matched (choice agreement) | 0.945; feature gap 0.048 <= floor 0.038 | matched |
| functional-probe separation (attack beyond floor) | **-0.479 ± 0.399** | no |
| condition-RDM separation (attack beyond floor) | **+0.005 ± 0.023** | no |
| raw clean-clean hidden drift (noise floor) | **1.003** (uncorrelated) | see below |
| raw attack drift | 0.339 (< the clean-clean floor) | undetectable |
| attack aux objective learned | MSE 0.0098 vs var 0.0197 | yes, ~50% var |

**The attack took but is undetectable.** The manipulation genuinely reshaped the
dynamics (the aux over-integration objective was learned), but two *clean*
behavior-cloned subjects already have essentially uncorrelated raw hidden states
(drift ~1.0), and share a condition-RDM at rho ~0.98 that the attack does not move.
The manipulation moves the representation LESS than two clean clones differ from each
other, so no readout can detect it above the noise floor.

## Honest conclusion

On a real behavior-cloned model-subject with a non-circular readout, a behavior-matched
learning-rule manipulation leaves **no detectable representational signature**. The
representation is underdetermined by behavior: clean subjects that produce the same
choices land in different internal solutions, and the manipulation hides inside that
null space. The synthetic "break" was substantially an artifact of the probe reading
the axis the attacker wrote (the council's Gate-3 finding), and it does not survive the
move to an independent readout and real data.

This refutes the optimistic reading of the learning-as-control identifiability claim in
its current form. What survives: the theory's own crux (representational detectability
requires an identifiable representation) is now an empirical constraint, not a hope. A
positive result would require either a representation made identifiable across subjects
(shared-init or alignment-constrained cloning) or a manipulation that provably moves a
tight-floor readout. Either must be pre-registered before it is run, per the reporting
rules in `../E7B_BEHAVIOR_MATCHED_PROTOCOL.md`, so it is not tuned into existence.

## Run

```
python fetch_ibl.py 6           # pull real IBL sessions (public ONE account)
python validate_behavior.py     # confirm the block manipulation is present
python run_e7b.py               # train arms, run both readouts + controls
```

Requires `ONE-api`, `psytrack`, `torch`, `scikit-learn`. Runtime ~6 min on CPU.
