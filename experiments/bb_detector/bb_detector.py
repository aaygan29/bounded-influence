"""Black-box detector for external influence on a LEARNING RULE.

Setting. We observe only behaviour: stimulus x_t, binary choice y_t, reward r_t, and a
candidate external influence u_t. We never see the agent's internal state or the form of
its update rule. We decide whether u_t enters the agent's LEARNING RULE (how its policy
updates over trials) rather than merely biasing its instantaneous choices.

Why this is possible in black box (the identifying idea). Model the policy as a
time-varying logistic map, P(y_t = 1) = sigmoid(a_t + b_t x_t), and recover the weight
trajectory theta_hat_t = (a_hat_t, b_hat_t) from behaviour with a dynamic GLM
(Roy, Bak, Akrami, Brody & Pillow, PsyTrack, Neuron 2021 [1]; discrete-strategy variant
Ashwood, Roy, Stone et al. & Pillow, Nat. Neurosci. 2022 [2]). Two kinds of influence
leave DIFFERENT fingerprints on that trajectory:

  * Influence on CHOICE/BIAS adds u_t to the logit LEVEL each trial (a_t^eff = a_t + k u_t).
    The tracked weight then tracks u_t contemporaneously: level ~ u_t.
  * Influence on the LEARNING RULE adds u_t to the UPDATE (theta_{t+1} = theta_t +
    f(endogenous) + k u_t). The weight then carries the INTEGRAL of u: level ~ sum_s<=t u_s.

This level-vs-integral distinction is what makes a learning-rule change identifiable
from behaviour alone, and it is exactly the "does the external signal enter the update
operator" estimand of Liu, Geadah & Pillow (NeurIPS 2025 [3]), whose own limitation (a
static inferred rule cannot be told from a drifting one) is the non-identifiability this
test is built to clear. The two-timescale framing (fast choice, slow rule), the surrogate
calibration, and the call/no-call/abstain discipline follow the bounded-influence
control-theoretic program (Bagley; `theory/learning_as_control.md` [7]). The rich-vs-lazy
regime of the recovered trajectory (Liu, Baratin, Cornford, Mihalas, Shea-Brown & Lajoie,
ICLR 2024 [4]) and the credit-assignment trace that makes a learned rule readable at all
(Liu, Mihalas, Shea-Brown et al., PNAS 2021 [5]) are the wider grounding.

Method.
1. Track theta_hat_t by windowed logistic regression of y on x (a light dynamic GLM [1]).
2. Form a leaky integral I_u(t) = sum_s<=t lambda^(t-s) u_s of the influence, and the same
   for the endogenous reward-prediction-error channel I_rpe(t) (the agent's own learning).
3. Nested test on the LEVEL trajectory: does I_u explain theta_hat_t beyond the endogenous
   drivers AND the contemporaneous u_t (the bias control)? The contemporaneous-u term
   absorbs pure choice-bias, so only accumulated (rule) influence is left to detect.
4. Statistic Lambda = summed log RSS-ratio over the (a, b) channels; surrogate null by
   circular-shifting u_t (preserving its autocorrelation) before integrating.
   p = (1 + #{surrogate >= observed}) / (1 + B).
5. Abstain when the trajectory is not identifiable (endogenous level model explains almost
   nothing), so a null reads as "no power" not "no effect" (the [3] non-identifiability
   guard). Decision: call / no-call / abstain at level alpha.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import lfilter


def _logistic_fit(x, y, iters=25, ridge=1e-3):
    """Tiny IRLS logistic regression of y on [1, x]; returns (intercept, slope)."""
    X = np.column_stack([np.ones_like(x), x]); w = np.zeros(2)
    for _ in range(iters):
        eta = np.clip(X @ w, -30, 30); p = 1 / (1 + np.exp(-eta))
        W = np.clip(p * (1 - p), 1e-6, None)
        H = X.T @ (W[:, None] * X) + ridge * np.eye(2)
        g = X.T @ (y - p) - ridge * w
        try:
            w = w + np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
    return w[0], w[1]


def track_policy(x, y, window=41):
    """Windowed dynamic-GLM tracking of theta_t = (a_t, b_t) [1]. Centered windows."""
    n = len(x); h = window // 2
    idx, A, B = [], [], []
    for t in range(h, n - h):
        sl = slice(t - h, t + h + 1)
        a, b = _logistic_fit(x[sl], y[sl]); idx.append(t); A.append(a); B.append(b)
    return np.array(idx), np.array(A), np.array(B)


def leaky_integral(v, lam=0.98):
    """I(t) = sum_s<=t lam^(t-s) v_s, vectorised (IIR filter)."""
    return lfilter([1.0], [1.0, -lam], v)


def _rss_gain(target, E, extra):
    """log(RSS_E / RSS_[E, extra]) for one channel (>= 0)."""
    def rss(D):
        beta, *_ = np.linalg.lstsq(D, target, rcond=None)
        res = target - D @ beta
        return float(res @ res) + 1e-12
    return np.log(rss(E) / rss(np.column_stack([E, extra])))


def detect(x, y, r, u, *, window=41, lam=0.98, n_surrogate=500, alpha=0.05,
           min_endog_r2=0.05, seed=0):
    """Black-box test: does u_t enter the learning rule (accumulate into the weight
    trajectory), as opposed to biasing choices? Returns a decision dict."""
    rng = np.random.default_rng(seed)
    idx, A, B = track_policy(x, y, window)
    if len(idx) < 40:
        return {"decision": "abstain", "reason": "too few tracked trials",
                "lambda": None, "p": None}

    xt = x[idx].astype(float); ut = u[idx].astype(float)
    p_right = 1 / (1 + np.exp(-np.clip(A + B * xt, -30, 30)))
    exp_reward = np.where(xt > 0, p_right, 1 - p_right)
    rpe = r[idx].astype(float) - exp_reward

    tt = np.arange(len(idx), dtype=float)
    I_rpe = leaky_integral(rpe, lam)
    I_u = leaky_integral(ut, lam)
    # endogenous LEVEL model + contemporaneous u_t (the bias control): what the weight
    # trajectory should look like if u does NOT enter the update.
    E = np.column_stack([np.ones(len(idx)), tt, I_rpe, ut])

    def endog_r2(target):
        beta, *_ = np.linalg.lstsq(E, target, rcond=None)
        return 1 - np.var(target - E @ beta) / (np.var(target) + 1e-12)
    if max(endog_r2(A), endog_r2(B)) < min_endog_r2:
        return {"decision": "abstain", "reason": "trajectory unidentifiable",
                "lambda": None, "p": None,
                "endog_r2": float(max(endog_r2(A), endog_r2(B)))}

    lam_obs = _rss_gain(A, E, I_u) + _rss_gain(B, E, I_u)      # joint over both channels
    null = np.empty(n_surrogate)
    for i in range(n_surrogate):
        us = np.roll(ut, rng.integers(1, len(ut) - 1))
        Ius = leaky_integral(us, lam)
        null[i] = _rss_gain(A, E, Ius) + _rss_gain(B, E, Ius)
    p = (1 + np.sum(null >= lam_obs)) / (1 + n_surrogate)

    return {"decision": "call" if p < alpha else "no-call",
            "lambda": float(lam_obs), "p": float(p),
            "endog_r2": float(max(endog_r2(A), endog_r2(B))), "n_tracked": int(len(idx))}
