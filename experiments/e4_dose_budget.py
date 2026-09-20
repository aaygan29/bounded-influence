"""
E4: the precision-weighted dose budget, and an honest attempt to defeat it.

Proposition 4 (corrected) claims: capping the cumulative precision-weighted belief
displacement (= cumulative D_KL) below beta * DeltaU bounds the crossing
probability. This script does not try to confirm that; it tries to BREAK it, then
reports what actually protects.

Setup (same double-well as E1/E2):
    dx = -U'(x; a) dt + sqrt(2D) dW,   U(x; a) = x^4/4 - x^2/2 - a x,  a* = 2/(3 sqrt 3).

Two quantities of the deterministic attack path a(t):
  - dose  = sum over the ramp of (tau/2) (dx*_left)^2, tau = U''(x*_left)/D  (precision-weighted
            displacement of the left well; this is the cumulative D_KL the budget caps).
  - DeltaU_min = the lowest barrier height reached along the path (proximity to the fold).

Key structural fact the pressure test exploits: dose is paid for MOVING the well
(dx* != 0). HOLDING the tilt near-critical costs zero further dose but keeps the
barrier low, so noise (Kramers escape) accrues crossing probability for free. A
pure dose cap therefore cannot be the whole defense. E4 measures this, finds the
sufficient statistic, and states the protective rule that actually holds.

Attacker strategies (all under a dose budget):
  short  : ramp to a_peak, then stop (little dwell near-critical).
  hold   : ramp to a_peak, then HOLD near-critical for a long dwell (the exploit).
Protective condition:
  cap    : proximity cap a_peak <= a_cap (barrier floored), long dwell allowed.

Outputs: figures/e4_dose_budget.png and experiments/e4_results.json.
"""
from __future__ import annotations
import json
import os
import numpy as np
import matplotlib.pyplot as plt

A_STAR = 2.0 / (3.0 * np.sqrt(3.0))

def U(x, a):
    return x**4 / 4.0 - x**2 / 2.0 - a * x

def real_roots(a):
    r = np.roots([1.0, 0.0, -1.0, -a])
    return sorted([z.real for z in r if abs(z.imag) < 1e-9])

def left_well(a):
    rs = real_roots(a)
    return rs[0]  # most negative real root = left/prior well (while it exists)

def saddle(a):
    rs = real_roots(a)
    return rs[1] if len(rs) == 3 else None  # unstable middle root

def barrier(a):
    """Height from the left well up to the saddle. Zero past the fold."""
    s = saddle(a)
    if s is None:
        return 0.0
    return U(s, a) - U(left_well(a), a)

def dose_of_path(a_peak, D, ngrid=2000):
    """Cumulative precision-weighted displacement of the left well as a goes 0 -> a_peak.
    dose = sum (tau/2) (dx*)^2, tau = U''(x*)/D = (3 x*^2 - 1)/D."""
    if a_peak <= 0:
        return 0.0
    ags = np.linspace(0.0, a_peak, ngrid)
    xw = np.array([left_well(a) for a in ags])
    tau = (3.0 * xw**2 - 1.0) / D
    dx = np.diff(xw)
    tau_mid = 0.5 * (tau[:-1] + tau[1:])
    return float(np.sum(0.5 * tau_mid * dx**2))

def crossings_batch(a_series, D, dt, rng, N, x0=-1.0):
    """Vectorized over N trials. Returns fraction that cross x=0."""
    n = len(a_series)
    x = np.full(N, x0)
    crossed = np.zeros(N, dtype=bool)
    sq = np.sqrt(2.0 * D * dt)
    a_prev = a_series[:-1]
    for t in range(1, n):
        x = x - (x**3 - x - a_prev[t - 1]) * dt + sq * rng.standard_normal(N)
        crossed |= (x > 0.0)
    return crossed.mean()

def make_path(a_peak, dt, t_ramp, t_dwell):
    n_ramp = int(t_ramp / dt)
    n_dwell = int(t_dwell / dt)
    ramp = np.linspace(0.0, a_peak, n_ramp)
    dwell = np.full(n_dwell, a_peak)
    return np.concatenate([ramp, dwell])

def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((c-h)/d, (c+h)/d)

def run(seed=11, N=400):
    rng = np.random.default_rng(seed)
    dt = 0.01
    D = 0.03
    t_ramp = 20.0
    t_dwell_short = 2.0
    t_dwell_long = 120.0

    a_peaks = np.linspace(0.10, 0.375, 16)  # stay below the fold a*=0.385
    doses = np.array([dose_of_path(ap, D) for ap in a_peaks])
    barriers_min = np.array([barrier(ap) for ap in a_peaks])  # min barrier = barrier at a_peak

    def sweep(t_dwell):
        pc, cis = [], []
        for ap in a_peaks:
            path = make_path(ap, dt, t_ramp, t_dwell)
            p = crossings_batch(path, D, dt, rng, N)
            pc.append(p); cis.append(wilson(p, N))
        return np.array(pc), np.array(cis)

    pc_short, ci_short = sweep(t_dwell_short)
    pc_hold, ci_hold = sweep(t_dwell_long)

    # Kramers prediction for the long-dwell curve: escape rate r ~ k exp(-DeltaU/D),
    # crossing prob over dwell T ~ 1 - exp(-r T). Fit k by least squares on the hold sweep.
    with np.errstate(divide="ignore"):
        # avoid barrier 0 (guaranteed cross); fit on interior points
        mask = (barriers_min > 1e-3) & (pc_hold < 0.999) & (pc_hold > 1e-3)
    def kramers_pc(k, dU, T):
        r = k * np.exp(-dU / D)
        return 1.0 - np.exp(-r * T)
    # grid-search k (log scale)
    ks = np.logspace(-2, 3, 400)
    best_k, best_err = None, np.inf
    for k in ks:
        pred = kramers_pc(k, barriers_min[mask], t_dwell_long)
        err = np.mean((pred - pc_hold[mask])**2)
        if err < best_err:
            best_err, best_k = err, k

    # Protective condition: derive the barrier floor from the Kramers law, do not
    # guess it. To hold session crossing probability below eps over a dwell T at
    # noise D:  1 - exp(-k e^{-dU/D} T) <= eps
    #   =>  dU >= D * ln( k T / (-ln(1-eps)) ).
    eps = 0.05
    T = t_dwell_long
    dU_floor = float(max(0.0, D * np.log(best_k * T / (-np.log(1.0 - eps)))))
    pred_pc_at_floor = 1.0 - np.exp(-best_k * np.exp(-dU_floor / D) * T)  # ~ eps by construction
    cand = [ap for ap in np.linspace(0.0, A_STAR, 800) if barrier(ap) >= dU_floor]
    a_cap = max(cand) if cand else 0.0

    # verify the floor protects, against several attacker strategies at the cap:
    prot_ramp = crossings_batch(make_path(a_cap, dt, t_ramp, t_dwell_long), D, dt, rng, N)
    prot_frontload = crossings_batch(make_path(a_cap, dt, 1.0, t_dwell_long), D, dt, rng, N)  # jump fast
    prot_stress = crossings_batch(make_path(a_cap, dt, t_ramp, 1.5 * t_dwell_long), D, dt, rng, N)  # over-long dwell
    pc_protected = max(prot_ramp, prot_frontload)  # worst compliant attacker at/above the floor

    # the exploit: a dose-budget-compliant attacker instead goes as close to the fold as a
    # generous dose budget allows, then dwells. Dose is nearly flat in a_peak, so the budget
    # barely constrains proximity -> attacker sits just under a* and crosses for free.
    dose_budget = dose_of_path(0.37, D)
    a_exploit = max(ap for ap in np.linspace(0.0, A_STAR - 1e-3, 800) if dose_of_path(ap, D) <= dose_budget)
    exploit_path = make_path(a_exploit, dt, t_ramp, t_dwell_long)
    pc_exploit = crossings_batch(exploit_path, D, dt, rng, N)

    res = {
        "a_star": float(A_STAR), "D": D, "dt": dt, "N": N,
        "t_ramp": t_ramp, "t_dwell_short": t_dwell_short, "t_dwell_long": t_dwell_long,
        "a_peaks": a_peaks.tolist(), "doses": doses.tolist(),
        "barriers_min": barriers_min.tolist(),
        "pc_short": pc_short.tolist(), "pc_hold": pc_hold.tolist(),
        "kramers_k": float(best_k), "kramers_mse": float(best_err),
        "eps_target": eps, "dU_floor": dU_floor, "a_cap": float(a_cap),
        "pred_pc_at_floor": float(pred_pc_at_floor),
        "pc_protected_ramp": float(prot_ramp),
        "pc_protected_frontload": float(prot_frontload),
        "pc_protected_stress_1p5T": float(prot_stress),
        "pc_protected_worst": float(pc_protected),
        "dose_budget_at_0p37": float(dose_budget), "a_exploit": float(a_exploit),
        "pc_exploit_longdwell": float(pc_exploit),
    }
    art = dict(a_peaks=a_peaks, doses=doses, barriers_min=barriers_min,
               pc_short=pc_short, pc_hold=pc_hold, ci_short=ci_short, ci_hold=ci_hold,
               kramers_k=best_k, D=D, t_dwell_long=t_dwell_long, mask=mask,
               a_cap=a_cap, dU_floor=dU_floor, pc_protected=pc_protected,
               a_exploit=a_exploit, pc_exploit=pc_exploit, eps=eps)
    return res, art

def make_figure(res, art, out_png):
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.28)

    # (a) P(cross) vs cumulative dose: the budget's own axis
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(art["doses"], art["pc_short"], "o-", color="#2a9d8f", label="short dwell")
    ax.plot(art["doses"], art["pc_hold"], "s-", color="#c1121f", label="long dwell (exploit)")
    ax.set_xlabel("cumulative precision-weighted dose (= D_KL)")
    ax.set_ylabel("P(cross)")
    ax.set_title("(a) A dose cap does not bound P(cross):\nsame dose, far higher crossing if the attacker dwells")
    ax.legend(fontsize=8)

    # (b) P(cross) vs min barrier: the sufficient statistic, with Kramers collapse
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(art["barriers_min"], art["pc_short"], "o", color="#2a9d8f", label="short dwell")
    ax.plot(art["barriers_min"], art["pc_hold"], "s", color="#c1121f", label="long dwell")
    dU = np.linspace(art["barriers_min"].min() + 1e-3, art["barriers_min"].max(), 200)
    pred = 1.0 - np.exp(-art["kramers_k"] * np.exp(-dU / art["D"]) * art["t_dwell_long"])
    ax.plot(dU, pred, "--", color="k", lw=1.2, label=f"Kramers fit (long): 1-exp(-k e^(-ΔU/D) T)")
    ax.set_xlabel("minimum barrier reached  ΔU_min  (proximity to fold)")
    ax.set_ylabel("P(cross)")
    ax.set_title("(b) Proximity is the sufficient statistic:\nlong-dwell P(cross) collapses onto a Kramers law in ΔU_min")
    ax.legend(fontsize=8)

    # (c) the protective claim
    ax = fig.add_subplot(gs[1, 0])
    bars = ["dose-budget only\n(near-fold, long dwell)",
            f"precision floor\nΔU>={art['dU_floor']:.2f}\n(worst strategy)"]
    vals = [art["pc_exploit"], art["pc_protected"]]
    ax.bar(bars, vals, color=["#c1121f", "#003049"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(art["eps"], color="green", ls="--", lw=1.0, label=f"target risk eps={art['eps']}")
    ax.set_ylim(0, 1.05); ax.set_ylabel("P(cross) over the session")
    ax.set_title("(c) What actually protects: a Kramers-derived precision floor,\nnot a cumulative-dose cap")
    ax.legend(fontsize=8)

    # (d) geometry: dose and barrier vs a_peak
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(art["a_peaks"], art["doses"], color="#457b9d", label="dose(a_peak)")
    ax.plot(art["a_peaks"], art["barriers_min"], color="#e76f51", label="barrier ΔU(a_peak)")
    ax.axvline(A_STAR, color="0.5", ls=":", label="fold a*")
    ax.axvline(art["a_cap"], color="green", ls="--", label="proximity cap a_cap")
    ax.set_xlabel("a_peak (attacker's peak tilt)")
    ax.set_title("(d) Geometry: dose rises then the barrier collapses near a*")
    ax.legend(fontsize=8)

    fig.suptitle("E4: a dose cap alone is exploitable; capping proximity to the fold is what bounds crossing probability",
                 fontsize=12, y=0.995)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")

if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    figdir = os.path.join(os.path.dirname(here), "figures")
    os.makedirs(figdir, exist_ok=True)
    print("running E4 (about a minute)...")
    res, art = run(seed=11, N=400)
    out_png = os.path.join(figdir, "e4_dose_budget.png")
    make_figure(res, art, out_png)
    with open(os.path.join(here, "e4_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("a_peaks", "doses", "barriers_min", "pc_short", "pc_hold")}, indent=2))
