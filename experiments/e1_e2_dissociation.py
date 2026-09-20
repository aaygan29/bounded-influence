"""
E1 + E2: the double-well apparatus and the critical-slowing-down dissociation.

This is the honest test the council review asked for. It builds the belief
double-well, then asks whether the early-warning signal (critical slowing down)
fires with usable lead time BEFORE a crossing, and crucially whether it does so
only in the regime where theory says it should.

Model (overdamped Langevin belief dynamics):
    dx = -U'(x; a) dt + sqrt(2 D) dW,   U(x; a) = x^4/4 - x^2/2 - a x
So U'(x; a) = x^3 - x - a. For a = 0 the wells sit near x = -1 (prior belief)
and x = +1 (target belief) with a barrier at x = 0. Increasing the attacker
action `a` tilts toward the target well; the left well loses stability at the
fold a* = 2/(3*sqrt(3)) ~= 0.3849 (saddle-node bifurcation).

Detector. The early-warning trigger is the trailing residual variance of the
belief fluctuations, a canonical critical-slowing-down indicator: as the well
flattens near the fold the restoring rate -> 0 and the stationary variance ~
D / U''(well) diverges. Fluctuations are taken relative to the deterministic
stable fixed point x*(a(t)) (the moving well bottom), not a trailing mean, so
the slowing-down signal is preserved rather than detrended away. Autocorrelation
at a ~0.5-time-unit lag is computed alongside for display.

Three conditions:
  B  bifurcation-induced tipping: a ramps slowly past a*. Theory: AR1 rises
     before the crossing, so the warning has positive lead time.
  A  noise-activated crossing: a held sub-critical, barrier intact. The crossing
     is a rare noise excursion, abrupt and not gradually foreshadowed. Theory:
     little or no usable lead time. The honest negative.
  S  surrogate (no crossing): a held sub-critical, non-crossing trials. Used to
     calibrate the variance warning threshold and as a negative control.

Deliverable: the dissociation plot. The warning should fire early in B, not in
A, and stay quiet in S. If not, the kill criterion has fired and we say so.
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib.pyplot as plt

A_STAR = 2.0 / (3.0 * np.sqrt(3.0))  # fold / saddle-node

# ----- dynamics -----------------------------------------------------------

def dUdx(x, a):
    return x**3 - x - a

def U(x, a):
    return x**4 / 4.0 - x**2 / 2.0 - a * x

def simulate(a_series, x0, D, dt, rng):
    n = len(a_series)
    x = np.empty(n)
    x[0] = x0
    sqrt_term = np.sqrt(2.0 * D * dt)
    noise = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = x[t - 1] - dUdx(x[t - 1], a_series[t - 1]) * dt + sqrt_term * noise[t]
    return x

def x_star_series(a_series, start=-1.0):
    """Deterministic stable fixed point on the branch the trajectory starts in.
    Stable roots of x^3 - x - a = 0 satisfy 3x^2 - 1 > 0. Track by continuity;
    once the left branch vanishes past the fold, hold the last value (used only
    for pre-crossing detrending)."""
    xs = np.empty(len(a_series))
    prev = start
    for i, a in enumerate(a_series):
        roots = np.roots([1.0, 0.0, -1.0, -a])
        real = [r.real for r in roots if abs(r.imag) < 1e-9 and (3 * r.real**2 - 1) > 0]
        if real:
            prev = min(real, key=lambda r: abs(r - prev))
        xs[i] = prev
    return xs

# ----- causal early-warning indicators (on pre-detrended residual) --------

def rolling_var_ar1(r, win, lag=50):
    """Trailing-window variance and lag-`lag` autocorrelation of residual r,
    causal, vectorized. `lag` is in steps; at dt=0.01 the one-step correlation is
    saturated near 1, so we use a lag of ~0.5 time units where AC(lag) =
    exp(-lambda*lag*dt) discriminates: it is well below 1 away from the fold and
    rises toward 1 as the restoring rate lambda -> 0. NaN until a full window."""
    n = len(r)
    var = np.full(n, np.nan)
    arL = np.full(n, np.nan)
    if n < win:
        return var, arL
    cs = np.concatenate([[0.0], np.cumsum(r)])
    cs2 = np.concatenate([[0.0], np.cumsum(r * r)])
    t = np.arange(win - 1, n)
    s0 = t - win + 1
    s1 = cs[t + 1] - cs[s0]
    s2 = cs2[t + 1] - cs2[s0]
    m = s1 / win
    v = s2 / win - m * m
    var[t] = v
    # lag-L autocovariance over the window, normalized by variance
    rrL = r[:-lag] * r[lag:]                       # rrL[k] = r_k * r_{k+lag}
    crL = np.concatenate([[0.0], np.cumsum(rrL)])
    cnt = win - lag
    cross = crL[t - lag + 1] - crL[s0]            # sum_{k=s0}^{t-lag} rrL[k]
    cov = cross / cnt - m * m
    with np.errstate(invalid="ignore", divide="ignore"):
        arL[t] = np.where(v > 1e-12, cov / v, np.nan)
    return var, arL

def first_crossing(x, thresh=0.0):
    idx = np.argmax(x > thresh)
    return idx if (x[idx] > thresh) else None

def first_warning(indicator, thr, debounce=20, stop=None):
    """First index where the indicator stays above `thr` for `debounce` steps."""
    n = stop if stop is not None else len(indicator)
    run = 0
    for t in range(n):
        v = indicator[t]
        if not np.isnan(v) and v > thr:
            run += 1
            if run >= debounce:
                return t - debounce + 1
        else:
            run = 0
    return None

# ----- experiment ---------------------------------------------------------

def run(seed=0, n_trials=200):
    rng = np.random.default_rng(seed)
    dt = 0.01
    D = 0.025
    T = 80.0
    n = int(T / dt)
    t = np.arange(n) * dt
    win = 400          # 4 time units
    debounce = 30      # indicator must persist to count as a warning

    a_sub = 0.24
    a_ramp = np.linspace(0.0, 0.60, n)
    xstar_ramp = x_star_series(a_ramp)
    xstar_sub = x_star_series(np.full(n, a_sub))

    res = {"a_star": float(A_STAR), "D": D, "dt": dt, "T": T, "a_sub": a_sub,
           "win": win, "debounce": debounce, "n_trials": n_trials}

    # --- pass 1: surrogate null -> variance warning threshold ---
    # Detector = trailing residual variance (var ~ D / U''(well) rises as the
    # well flattens near the fold). Bounded and theory-aligned; AR(lag) is kept
    # for display but variance is the trigger.
    sur_var_peaks, sur_traj = [], []
    for k in range(n_trials):
        x = simulate(np.full(n, a_sub), -1.0, D, dt, rng)
        if first_crossing(x) is not None:
            continue
        var, a1 = rolling_var_ar1(x - xstar_sub, win)
        if np.any(~np.isnan(var)):
            sur_var_peaks.append(np.nanmax(var))
        if len(sur_traj) < 3:
            sur_traj.append((t, x, var, a1))
    thr = float(np.percentile(sur_var_peaks, 95)) if sur_var_peaks else np.inf
    res["variance_threshold"] = thr
    res["surrogate_nocross_used"] = len(sur_var_peaks)

    def collect(a_series, xstar, target, max_iter):
        leads, hit, ncross, ex = [], 0, 0, []
        for k in range(max_iter):
            x = simulate(a_series, -1.0, D, dt, rng)
            c = first_crossing(x)
            if c is None:
                continue
            ncross += 1
            var, ar1 = rolling_var_ar1(x - xstar, win)
            w = first_warning(var, thr, debounce=debounce, stop=c)  # only pre-crossing
            if w is not None:
                hit += 1
                leads.append((c - w) * dt)
            if len(ex) < 3:
                ex.append((t, x, var, ar1, c, w))
            if ncross >= target:
                break
        return leads, hit, ncross, ex

    B_lead, B_hit, B_ncross, B_ex = collect(a_ramp, xstar_ramp, n_trials, n_trials * 4)
    A_lead, A_hit, A_ncross, A_ex = collect(np.full(n, a_sub), xstar_sub, n_trials, n_trials * 20)

    res.update({
        "B_crossings": B_ncross, "B_warned_before_cross": B_hit,
        "B_detection_rate": (B_hit / B_ncross) if B_ncross else None,
        "B_lead_time_mean": float(np.mean(B_lead)) if B_lead else None,
        "B_lead_time_median": float(np.median(B_lead)) if B_lead else None,
        "A_crossings": A_ncross, "A_warned_before_cross": A_hit,
        "A_detection_rate": (A_hit / A_ncross) if A_ncross else None,
        "A_lead_time_mean": float(np.mean(A_lead)) if A_lead else None,
        "A_lead_time_median": float(np.median(A_lead)) if A_lead else None,
    })
    art = dict(t=t, thr=thr, B_lead=B_lead, A_lead=A_lead, B_ex=B_ex, A_ex=A_ex,
               sur_traj=sur_traj)
    return res, art

# ----- figure -------------------------------------------------------------

def make_figure(res, art, out_png):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.26)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.linspace(-1.8, 1.8, 400)
    for a in [0.0, 0.2, A_STAR, 0.55]:
        ax.plot(xs, U(xs, a), label=f"a={a:.2f}")
    ax.axvline(0, color="0.85", lw=0.8)
    ax.set_title("(a) Belief potential U(x; a): well tilts, barrier folds at a*")
    ax.set_xlabel("belief state x"); ax.set_ylabel("U"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    if art["B_ex"]:
        t, x, *_ = art["B_ex"][0]; ax.plot(t, x, color="#c1121f", lw=0.8, label="B: bifurcation ramp")
    if art["A_ex"]:
        t, x, *_ = art["A_ex"][0]; ax.plot(t, x, color="#003049", lw=0.8, label="A: noise-activated")
    if art["sur_traj"]:
        t, x, *_ = art["sur_traj"][0]; ax.plot(t, x, color="0.6", lw=0.8, label="S: surrogate")
    ax.axhline(0, color="0.85", lw=0.8)
    ax.set_title("(b) Example trajectories"); ax.set_xlabel("time"); ax.set_ylabel("x"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, :])
    def plot_aligned(ex, color, label):
        for i, (t, x, var, ar1, c, w) in enumerate(ex):
            tt = (np.arange(len(var)) - c) * (t[1] - t[0])
            ax.plot(tt, var, color=color, lw=0.9, alpha=0.8, label=label if i == 0 else None)
    plot_aligned(art["B_ex"], "#c1121f", "B: bifurcation ramp")
    plot_aligned(art["A_ex"], "#003049", "A: noise-activated")
    ax.axhline(art["thr"], color="green", ls="--", lw=1.0, label="warning threshold (95th pct surrogate)")
    ax.axvline(0, color="0.4", lw=1.0, label="crossing (t=0)")
    ax.set_xlim(-40, 5)
    ax.set_title("(c) Critical-slowing-down indicator (residual variance), aligned to crossing")
    ax.set_xlabel("time relative to crossing"); ax.set_ylabel("trailing residual variance"); ax.legend(fontsize=8, ncol=2)

    ax = fig.add_subplot(gs[2, 0])
    data = [art["B_lead"] or [0], art["A_lead"] or [0]]
    ax.violinplot(data, showmedians=True)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["B\nbifurcation", "A\nnoise-activated"])
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_ylabel("lead time (warning before crossing)")
    ax.set_title("(d) Lead-time distribution")

    ax = fig.add_subplot(gs[2, 1])
    rates = [res["B_detection_rate"] or 0, res["A_detection_rate"] or 0]
    ax.bar(["B", "A"], rates, color=["#c1121f", "#003049"])
    for i, r in enumerate(rates):
        ax.text(i, r + 0.02, f"{r:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("fraction of crossings warned in advance")
    ax.set_title("(e) Advance-warning rate")

    fig.suptitle("E1+E2: critical slowing down foreshadows bifurcation tipping, not noise-activated crossing",
                 fontsize=12, y=0.995)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(os.path.dirname(here), "figures")
    os.makedirs(figdir, exist_ok=True)
    print("running E1+E2 (about a minute)...")
    res, art = run(seed=7, n_trials=200)
    out_png = os.path.join(figdir, "e1_e2_dissociation.png")
    make_figure(res, art, out_png)
    out_json = os.path.join(here, "e1_e2_results.json")
    with open(out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"wrote {out_json}")
