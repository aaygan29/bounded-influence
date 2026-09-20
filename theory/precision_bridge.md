# The precision bridge, derived from scratch

The council review's sharpest finding: the model measures the attacker's **dose**
in information units (`D_KL`, a divergence between belief distributions) but the
**barrier** in energy units (`ΔU`, a height in a potential). Capping one in terms
of the other is a category error unless something connects them. This document
builds that connection from first principles. The connecting quantity is **prior
precision**. Assumed background: multivariable calculus, Taylor expansion,
integrals, `exp`. Everything else is built here.

## 0. Where we are going

We will show, in six steps, that

    precision  =  U''(x*) / D          (well curvature over noise)
    ΔU         =  f(precision),  f increasing
    dose D_KL  ≈  (precision / 2) * (belief shift)^2

so barrier height and information dose are two views of one quantity, precision.
That makes the dose budget dimensionally honest and gives it an experimental
knob (psilocybin lowers precision).

## 1. A belief is a distribution, and precision is confidence

Stop thinking of a belief as a single number and think of it as a **bell curve**
over possible values. The Gaussian (normal) distribution is

    p(x) = (1 / sqrt(2 pi sigma^2)) * exp( -(x - mu)^2 / (2 sigma^2) ).

`mu` is where you think the truth is; `sigma^2` (the variance) is how unsure you
are. A wide curve is an open mind; a narrow spike is a firm conviction.

Define **precision** as the reciprocal of variance:

    tau = 1 / sigma^2.

High precision = narrow curve = confident. Low precision = wide = uncertain.
Precision, not variance, is the natural currency here, and the next step shows
why.

## 2. Updating a belief means adding precisions

Suppose your current belief (the **prior**) is Gaussian with mean `mu0` and
precision `tau0`, and a persuader presents evidence that is itself Gaussian with
mean `mu_e` and precision `tau_e`. Bayes' rule multiplies the two curves and
renormalizes. Multiply two Gaussians and complete the square in the exponent
(pure algebra), and the result is another Gaussian (the **posterior**) with

    tau_post = tau0 + tau_e                       (precisions add)
    mu_post  = (tau0 * mu0 + tau_e * mu_e) / (tau0 + tau_e)   (precision-weighted mean).

Read the mean update. The belief moves from `mu0` toward `mu_e` by an amount

    Delta_mu = mu_post - mu0 = (tau_e / (tau0 + tau_e)) * (mu_e - mu0).

If your prior precision `tau0` is large (firm belief), the fraction is small and
you barely move no matter how loud the evidence. If `tau0` is small (open mind),
you move a lot. This is the first appearance of the theme: **precision is what
resists persuasion.** It is exactly "barrier height," but in the language of
distributions instead of landscapes. We will make that identification exact in
step 5.

## 3. The dose is how far the belief moved, measured in information

To bound an attacker we need a number for "how much did you just move someone's
belief." The standard information-theoretic measure of the gap between two
distributions is the **Kullback-Leibler divergence**,

    D_KL(q || p) = integral q(x) * log( q(x) / p(x) ) dx.

It is zero when `q = p` and grows as they separate; it is measured in nats (or
bits). For two Gaussians with the same variance `1/tau`, whose means differ by
`Delta_mu`, the integral collapses to a clean closed form:

    D_KL = (tau / 2) * Delta_mu^2.

This is the key line for the dose. The information delivered by shifting a belief
is **precision times the squared shift**, halved. A shift through a
high-precision belief costs more information than the same shift through a vague
one. So `D_KL` is not a free-floating quantity: it already contains precision.
That is the first half of the bridge.

## 4. The landscape is a log-probability (Boltzmann)

Now connect the dynamics to the distribution. The belief obeys the stochastic
gradient equation

    dx = -U'(x) dt + sqrt(2 D) dW.

A standard fact about such gradient-plus-noise systems (the Ornstein-Uhlenbeck /
Fokker-Planck result) is that, left alone, the belief settles into a
**stationary distribution** that is not uniform but shaped by the landscape:

    p(x) ∝ exp( -U(x) / D ).

This is the Boltzmann form. Read it: deep valleys (low `U`) are high-probability
(confident) beliefs; the noise `D` plays the role of a temperature that smears
the distribution out. The landscape `U` and the belief distribution `p` are the
same object seen two ways, related by a logarithm: `U(x) = -D * log p(x)` up to a
constant. This is what lets an energy (`ΔU`) and an information quantity (`D_KL`)
ever be compared.

## 5. The bridge: precision = curvature over noise (Laplace)

Near the bottom of a valley `x*`, Taylor-expand the landscape (this is the Calc-3
move). Since `x*` is a minimum, `U'(x*) = 0`, so

    U(x) ≈ U(x*) + (1/2) * U''(x*) * (x - x*)^2.

Substitute into the Boltzmann form:

    p(x) ∝ exp( -U''(x*) * (x - x*)^2 / (2 D) ).

Compare to the Gaussian in step 1: this is a bell curve centered at `x*` with
variance `D / U''(x*)`. Therefore its precision is

    tau = U''(x*) / D.                          <-- THE BRIDGE

The precision of a belief equals the **curvature of the valley it sits in,
divided by the noise.** A steep, narrow valley is a high-precision, hard-to-move
belief. A flat valley is a low-precision, easily-moved one. This is the exact
statement that "barrier resists persuasion" and "precision resists persuasion"
are the same fact. (This is the Laplace approximation; it is exact for a
quadratic well and a good approximation near any smooth minimum.)

## 6. Barrier height and precision move together; slowing down is precision loss

Two payoffs fall out immediately.

**(a) `ΔU = f(precision)`, increasing.** As the attacker tilts the landscape
toward the fold, the valley flattens: `U''(x*) -> 0`. By the bridge, precision
`tau -> 0`. At the same time the barrier `ΔU` shrinks to zero (the valley is
merging with the hilltop at the fold). Both are monotone in "how established the
belief is," so barrier height is an increasing function of precision. That is the
assumption the corrected Proposition 4 needs, and here it is derived rather than
asserted, under the stated approximations.

**(b) Critical slowing down is precision going to zero.** The stationary variance
is `D / U''(x*) = 1 / tau`. As precision falls toward the fold, the variance
diverges and the belief decorrelates ever more slowly. So the early-warning
signal E2 measured (rising variance) is literally **watching precision collapse.**
The detector and the bridge are the same physics.

## 7. The corrected dose budget, now in coherent units

Combine step 3 and the bridge. A crossing requires moving the belief across a
displacement comparable to the well separation, against precision that itself
falls as the attack proceeds. The honest budget caps the **cumulative
precision-weighted displacement**,

    sum over the session of  (tau_t / 2) * (Delta_mu_t)^2   =   cumulative D_KL,

and requires it to stay below `beta * ΔU_i` for a margin `beta < 1`. Because
`ΔU_i = f(tau_i)` and the dose is already precision-weighted, both sides now live
in the same (information) units through precision. This replaces the original
"cap `D_KL` at `beta * ΔU`" which silently equated nats with energy.

Why not call it "conformal." Conformal prediction gives coverage guarantees under
**exchangeability** (the data order does not matter). An adversarial persuasion
stream is the opposite of exchangeable: each move is chosen in response to the
last. So the guarantee must be a **sequential** one. The correct tools are
supermartingale / e-value bounds (Ville's inequality), which do hold under
adaptive, non-exchangeable data, or simply an empirically measured miss-rate on
held-out attacks. The budget is calibrated, not conformal.

## 8. Assumptions, and where this breaks (honesty)

The bridge is clean but not free. It assumes:
- **Gaussian / near-well.** Step 5 is a quadratic approximation. It is accurate
  in a valley and degrades near the hilltop, exactly where a crossing happens. So
  the bridge describes the approach to the edge well and the passage over it
  poorly. Good enough for the budget (which is about staying away from the edge),
  not for the instant of crossing.
- **Gradient, stationary dynamics.** The Boltzmann form (step 4) needs the
  dynamics to be gradient (true here by construction) and to have had time to
  settle (quasi-static). Under fast drive the stationary picture lags, which is
  the same rate-dependence caveat that limits Proposition 2.
- **Conjugate Gaussian updating.** Steps 2 and 3 use Gaussian priors and
  evidence. Real belief updates are not exactly Gaussian; the precision-adds rule
  is then an approximation (a local/Laplace one), not an identity.

State these as the bridge's domain of validity. Within it, the units are honest
and the budget is well-posed.

## 9. How to test the bridge (not just assert it)

The bridge makes a falsifiable prediction: a manipulation that lowers prior
precision should lower the barrier and make beliefs move more per unit dose.
Psilocybin is exactly such a manipulation. The REBUS account states that
psychedelics reduce the precision of high-level priors, and PsiConnect (Novelli,
Stoliker, Razi et al.) records brain and behavior under psilocybin. So:
- Fit prior precision `tau` (via an active-inference / DCM model) in drug vs
  placebo.
- Check that a proxy for the barrier `ΔU` (resistance to belief change, or the
  critical-slowing-down onset) falls with `tau` as `f` predicts.
If `ΔU` does not track `tau`, the bridge is wrong and Proposition 4 loses its
foundation. That is the bridge's own kill criterion. In simulation, E4 tests the
budget directly: cap cumulative precision-weighted `D_KL` and measure the
realized crossing rate against the bound.
