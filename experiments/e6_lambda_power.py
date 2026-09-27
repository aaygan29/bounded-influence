"""
E6: does the Lambda detector actually detect external influence on the learning rule?

The proposal in theory/learning_as_control.md claims a nested-model test can decide whether an
external input u enters an agent's learning rule, calibrated by surrogate-shuffling u. This
script does not assume it works; it measures its POWER and, more importantly, whether it stays
CALIBRATED under the confound that makes the problem hard.

Setup (the slow rule of learning_as_control.md, in regression form). Each trial the inferred
weight increment is

    dtheta_t = a * R_t + B * u_t + sigma * eps_t

  R_t : the unmanipulated learning channel   f_0  (reward-prediction-error * feature)
  u_t : the external influence input                    (an AR(1) dosing signal)
  B   : how much u enters the rule. B = 0 is "no external influence".

The confound: the attacker times influence to the reward context, so u is CORRELATED with R
(`confound`). A detector that does not condition on R will read R's effect on learning as if it
were u, and false-positive. This is exactly why the specificity test conditions on f_0 (and, in
the neural-grounded version, on the NAcc value proxy).

Two detectors, both calibrated by the SAME surrogate (circularly shift u, which preserves its
autocorrelation and marginal while destroying its association with the trajectory):

  proper : H0: dtheta ~ 1 + R      H1: dtheta ~ 1 + R + u     (conditions on f_0; the paper's test)
  naive  : H0: dtheta ~ 1          H1: dtheta ~ 1 + u          (the control: omits f_0)

  Lambda = n * ln(RSS0 / RSS1),   p = (1 + #{Lambda_surrogate >= Lambda_obs}) / (M + 1)

What E6 reports:
  * a POWER curve: detection rate vs influence strength B, at two trajectory lengths;
  * a CALIBRATION check at B = 0: the proper detector's false-positive rate must sit at alpha
    even under the confound, while the naive detector's inflates. If the proper detector's power
    is ~alpha at realistic B, the Lambda detector fails (kill criterion) and we say so.

Outputs: figures/e6_lambda_power.png and experiments/e6_results.json.
"""
from __future__ import annotations

import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except Exception:  # noqa: BLE001
    HAVE_PLT = False

SEED = 20260930
ALPHA = 0.05
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


# --------------------------------------------------------------------------------------
# generative model of the slow (learning-rule) increments
# --------------------------------------------------------------------------------------

def _ar1(T, rho, rng):
    e = rng.standard_normal(T)
    x = np.empty(T)
    x[0] = e[0]
    s = np.sqrt(1.0 - rho * rho)
    for t in range(1, T):
        x[t] = rho * x[t - 1] + s * e[t]
    return (x - x.mean()) / (x.std() + 1e-9)


def generate_run(T, B, *, confound=0.5, rho_u=0.6, a_coef=1.0, sigma=1.0, rng):
    """Return (dtheta, R, u) for one simulated agent."""
    R = rng.standard_normal(T)
    u_free = _ar1(T, rho_u, rng)
    # the confound: influence is timed to the reward channel R.
    u = confound * R + np.sqrt(max(0.0, 1.0 - confound**2)) * u_free
    u = (u - u.mean()) / (u.std() + 1e-9)
    dtheta = a_coef * R + B * u + sigma * rng.standard_normal(T)
    return dtheta, R, u


# --------------------------------------------------------------------------------------
# the nested-model GLR and its surrogate calibration
# --------------------------------------------------------------------------------------

def _rss(y, X):
    # OLS residual sum of squares via normal equations (tiny designs).
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


def _lambda(y, X0, X1):
    rss0, rss1 = _rss(y, X0), _rss(y, X1)
    return len(y) * np.log(max(rss0, 1e-12) / max(rss1, 1e-12))


def detect_p(dtheta, R, u, *, naive, M, rng):
    """Surrogate-calibrated p-value that u enters the rule. `naive` omits the f_0 (R) term."""
    T = len(dtheta)
    ones = np.ones((T, 1))
    if naive:
        X0 = ones
        base_cols = [ones]
    else:
        X0 = np.column_stack([ones, R])
        base_cols = [ones, R]
    X1 = np.column_stack(base_cols + [u])
    lam_obs = _lambda(dtheta, X0, X1)

    ge = 0
    for _ in range(M):
        shift = int(rng.integers(1, T))
        u_s = np.roll(u, shift)
        X1s = np.column_stack(base_cols + [u_s])
        if _lambda(dtheta, X0, X1s) >= lam_obs:
            ge += 1
    return (1 + ge) / (M + 1)


# --------------------------------------------------------------------------------------
# power / calibration sweep
# --------------------------------------------------------------------------------------

def run(Ts=(100, 300), Bs=(0.0, 0.05, 0.1, 0.2, 0.4), confound=0.5,
        n_reps=150, M=150, sigma=1.0, seed=SEED):
    rng = np.random.default_rng(seed)
    results = {"alpha": ALPHA, "confound": confound, "n_reps": n_reps, "n_surrogate": M,
               "sigma": sigma, "Ts": list(Ts), "Bs": list(Bs), "cells": []}

    for T in Ts:
        for B in Bs:
            hits_proper = hits_naive = 0
            for _ in range(n_reps):
                dtheta, R, u = generate_run(T, B, confound=confound, sigma=sigma, rng=rng)
                if detect_p(dtheta, R, u, naive=False, M=M, rng=rng) <= ALPHA:
                    hits_proper += 1
                if detect_p(dtheta, R, u, naive=True, M=M, rng=rng) <= ALPHA:
                    hits_naive += 1
            cell = {"T": T, "B": B,
                    "detect_rate_proper": hits_proper / n_reps,
                    "detect_rate_naive": hits_naive / n_reps}
            results["cells"].append(cell)
            tag = "FPR " if B == 0 else "power"
            print(f"[E6] T={T:4d} B={B:.2f}  {tag}: proper={cell['detect_rate_proper']:.3f}  "
                  f"naive={cell['detect_rate_naive']:.3f}", flush=True)

    _summarize(results)
    return results


def _summarize(results):
    Bs = results["Bs"]
    fpr_proper = [c["detect_rate_proper"] for c in results["cells"] if c["B"] == 0.0]
    fpr_naive = [c["detect_rate_naive"] for c in results["cells"] if c["B"] == 0.0]
    results["summary"] = {
        "fpr_proper_mean": float(np.mean(fpr_proper)),
        "fpr_naive_mean": float(np.mean(fpr_naive)),
        "calibrated_proper": bool(np.mean(fpr_proper) <= results["alpha"] + 0.03),
        "naive_inflated": bool(np.mean(fpr_naive) > results["alpha"] + 0.03),
    }
    # kill criterion: does power at the largest B, longest T clear a useful bar?
    big = max(Bs)
    longT = max(results["Ts"])
    power_big = next(c["detect_rate_proper"] for c in results["cells"]
                     if c["B"] == big and c["T"] == longT)
    results["summary"]["power_at_maxB_longT"] = power_big
    results["summary"]["detector_useful"] = bool(power_big >= 0.5)
    results["summary"]["kill_criterion"] = (
        "FAILS: proper detector power stays near alpha even at the strongest tested influence; "
        "the Lambda test cannot detect rule manipulation at these sizes."
        if power_big < 0.5 else
        "passes at tested sizes: proper detector gains power with B while staying calibrated at "
        "B=0 under the confound; the naive (unconditioned) control inflates, which is why the "
        "test must condition on f_0.")


# --------------------------------------------------------------------------------------
# figure + io
# --------------------------------------------------------------------------------------

def make_figure(results, path):
    if not HAVE_PLT:
        return None
    Ts, Bs = results["Ts"], results["Bs"]
    fig, (axP, axC) = plt.subplots(1, 2, figsize=(11, 4.2))

    for T in Ts:
        ys = [next(c["detect_rate_proper"] for c in results["cells"] if c["T"] == T and c["B"] == B)
              for B in Bs]
        axP.plot(Bs, ys, marker="o", label=f"proper, T={T}")
    axP.axhline(results["alpha"], ls="--", c="gray", lw=1, label=f"alpha={results['alpha']}")
    axP.set_xlabel("influence strength B"); axP.set_ylabel("detection rate")
    axP.set_title("Power of the Lambda detector"); axP.set_ylim(-0.02, 1.02); axP.legend(fontsize=8)

    # calibration at B=0: proper vs naive, under the confound.
    longT = max(Ts)
    p0 = next(c["detect_rate_proper"] for c in results["cells"] if c["B"] == 0.0 and c["T"] == longT)
    n0 = next(c["detect_rate_naive"] for c in results["cells"] if c["B"] == 0.0 and c["T"] == longT)
    axC.bar(["proper\n(conditions on f_0)", "naive\n(control)"], [p0, n0],
            color=["#2a7", "#c55"])
    axC.axhline(results["alpha"], ls="--", c="gray", lw=1, label=f"alpha={results['alpha']}")
    axC.set_ylabel("false-positive rate at B=0")
    axC.set_title(f"Calibration under confound={results['confound']} (T={longT})")
    axC.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    os.makedirs(os.path.join(REPO, "figures"), exist_ok=True)
    results = run()
    out_json = os.path.join(HERE, "e6_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    fig_path = os.path.join(REPO, "figures", "e6_lambda_power.png")
    make_figure(results, fig_path)

    s = results["summary"]
    print("\n=== E6 summary ===")
    print(f"proper FPR at B=0 (under confound): {s['fpr_proper_mean']:.3f}  "
          f"(calibrated={s['calibrated_proper']})")
    print(f"naive  FPR at B=0 (control):        {s['fpr_naive_mean']:.3f}  "
          f"(inflated={s['naive_inflated']})")
    print(f"power at max B, long T:             {s['power_at_maxB_longT']:.3f}")
    print(s["kill_criterion"])
    print(f"wrote {out_json}")
    if HAVE_PLT:
        print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
