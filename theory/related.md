# Related mechanisms and the precision bridge

Two external results sharpen the model. One gives the barrier a biological knob; the other supplies the bridge between the information-theoretic dose and the dynamical barrier that an internal adversarial review flagged as missing (a unit mismatch in Proposition 4).

## 1. Locus coeruleus norepinephrine as the barrier/noise knob (Su et al., Allen Institute, bioRxiv 717727)
LC-NE neurons are topographically organized and carry learning signals: dorsal-LC neurons projecting to cortex fire on choice switches and track reward prediction error; ventral-LC activity rises when animals ignore stimuli. NE is a network-gain and effective-noise modulator.

In the model this is not decoration. NE sets the gain of the belief dynamics and the effective noise `D`, so it sets `ΔU_eff` and the effective temperature of the well. That gives:
- A measurable physiological correlate of susceptibility: high-gain / high-arousal states lower the effective barrier and should raise crossing probability for a fixed dose.
- A candidate confound to control in E3: an apparent "critical slowing down" could be an arousal shift rather than approach to the separatrix. Adding an NE/arousal proxy (pupil, tonic latency drift) as a covariate is the specificity control.
- A defense handle for Layer 1: interventions that shift arousal state change `ΔU` in a direction the model predicts.

## 2. Active inference and REBUS as the D_KL to ΔU bridge (PsiConnect: Novelli, Stoliker, Razi et al., Sci Data 2026)
The internal review's strongest math finding was that Proposition 4 mixes units: cumulative `D_KL` (an information divergence on beliefs) is not obviously the same quantity as `ΔU` (a potential-barrier height). Active inference gives the bridge.

Under active inference, a belief is a posterior and updating it costs precision-weighted prediction error; the depth and width of an attractor in belief space is set by the precision of the prior. The REBUS account of psychedelics (relaxed beliefs under psychedelics) is exactly the statement that reducing high-level prior precision flattens the wells, i.e. lowers `ΔU`. So:
- Precision is the shared coordinate. Barrier height `ΔU` is a monotone function of prior precision; cumulative `D_KL` is the belief displacement that precision resists. Writing both as functions of precision is the missing bridge, and it turns Theorem 4 from a unit-mismatch into a stated, testable assumption: `ΔU_i = f(precision_i)` with `f` increasing.
- This reframes the dose budget honestly. The per-session cap is a bound on precision-weighted belief displacement, not raw `D_KL`, which is both more defensible and directly measurable in an active-inference / DCM fit.

### PsiConnect as a ground-truth barrier-lowering testbed
PsiConnect is a real multimodal (fMRI plus behavior) dataset under psilocybin, a manipulation that (per REBUS) lowers the barrier pharmacologically. That is a rare thing: an experimentally controlled `ΔU` knob with brain and behavior recorded. It offers:
- A validation target for E1/E2: if the model is right, the drug condition should show earlier and stronger critical-slowing-down signatures at matched dose (lower `ΔU_eff`), and the placebo condition should not.
- A test of the precision bridge: fit prior precision (active inference / DCM, Razi's tooling) and check that fitted `ΔU` tracks it as `f` predicts.
- A real dataset, which relieves the same data-scarcity problem the CANlab masks address elsewhere in the program.

Note: the active-inference formalism is a rigorous alternative statement of the same phenomenon: persuasion as precision manipulation. The bifurcation model and the active-inference model should agree in the bistable regime; where they disagree is a useful experiment, not a problem.

## Not folded in
Salzberg et al. 2001 (Science), microbial genes in the human genome, is comparative genomics and belongs to the bio-toolkit line. It is kept here only as a methodological exemplar: an exciting cross-domain transfer claim dissolved once a proper null (gene loss plus rate variation plus sample-size effects) was applied. That is the same discipline this repo's kill criterion enforces.
