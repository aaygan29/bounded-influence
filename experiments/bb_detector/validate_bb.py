"""Validate the black-box learning-rule influence detector in simulation.

No real data required: we simulate an agent whose learning rule we control, so the
ground truth (does u enter the rule?) is known. Three conditions:

  H0    : u_t present but does NOT enter the rule or the choice. FPR should ~ alpha.
  rule  : u_t enters the UPDATE (b_{t+1} += kappa u_t). Power (TPR) should be high.
  bias  : u_t adds to the choice logit each trial but NOT to the update. The detector
          must NOT fire (specificity); FPR should ~ alpha. This is the control that
          proves the method detects influence on the LEARNING RULE, not mere bias.

Reports calibration, power, and specificity across seeds.
Usage:  python validate_bb.py [--seeds 40] [--T 800] [--kappa 0.15]
"""
from __future__ import annotations
import argparse, json
import numpy as np
from bb_detector import detect


def simulate(T, condition, *, kappa=0.15, eta=0.04, phi=0.95, seed=0):
    """Agent: P(y=1)=sigmoid(a + b x). Endogenous rule: reward sharpens slope b via
    a delta term; a drifts slightly. u_t is a mean-reverting AR(1) external signal."""
    rng = np.random.default_rng(seed)
    x = rng.choice([-1.0, 1.0], size=T)
    # AR(1) influence, mean-reverting so level and increment decorrelate
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = phi * u[t - 1] + rng.normal(0, 1) * np.sqrt(1 - phi ** 2)
    a, b = 0.0, 0.8
    y = np.zeros(T, int); r = np.zeros(T)
    for t in range(T):
        logit = a + b * x[t]
        if condition == "bias":
            logit += kappa * 3.0 * u[t]          # instantaneous choice bias only
        p = 1 / (1 + np.exp(-np.clip(logit, -30, 30)))
        y[t] = int(rng.random() < p)
        correct = (x[t] > 0) == (y[t] == 1)
        r[t] = float(correct)
        rpe = r[t] - (p if x[t] > 0 else 1 - p)
        # endogenous learning rule: reward sharpens stimulus sensitivity
        b += eta * rpe
        a += eta * 0.3 * rpe * (1 if y[t] == 1 else -1)
        a += rng.normal(0, 0.01); b += rng.normal(0, 0.01)   # small process noise
        if condition == "rule":
            b += kappa * u[t]                    # u enters the UPDATE (learning rule)
        b = float(np.clip(b, -3, 3)); a = float(np.clip(a, -3, 3))
    return x, y, r, u


def run(condition, seeds, T, kappa, alpha=0.05):
    calls = abst = 0; ps = []
    for s in range(seeds):
        x, y, r, u = simulate(T, condition, kappa=kappa, seed=s)
        out = detect(x, y, r, u, alpha=alpha, seed=1000 + s)
        if out["decision"] == "abstain":
            abst += 1
        else:
            ps.append(out["p"])
            calls += int(out["decision"] == "call")
    n_dec = seeds - abst
    rate = calls / n_dec if n_dec else float("nan")
    return {"condition": condition, "n": seeds, "abstain": abst,
            "call_rate": rate, "median_p": float(np.median(ps)) if ps else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--T", type=int, default=800)
    ap.add_argument("--kappa", type=float, default=0.15)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="bb_results.json")
    args = ap.parse_args()

    res = {}
    for cond in ("H0", "rule", "bias"):
        r = run(cond, args.seeds, args.T, args.kappa, args.alpha)
        res[cond] = r
        print(f"{cond:5s}: call_rate={r['call_rate']:.3f}  abstain={r['abstain']}/{r['n']}  "
              f"median_p={r['median_p']}", flush=True)

    fpr, power, spec = res["H0"]["call_rate"], res["rule"]["call_rate"], res["bias"]["call_rate"]
    verdict = {
        "calibrated": fpr <= args.alpha + 0.05,        # FPR near alpha
        "powered": power >= 0.7,                        # detects real rule-influence
        "specific": spec <= args.alpha + 0.05,          # does NOT fire on bias-only
    }
    verdict["works"] = all(verdict.values())
    out = {"args": vars(args), "results": res, "summary": {
        "FPR_H0": fpr, "power_rule": power, "FPR_bias_specificity": spec, **verdict}}
    json.dump(out, open(args.out, "w"), indent=2)
    print("\n=== BLACK-BOX RULE-INFLUENCE DETECTOR ===")
    print(f"calibration  FPR(H0)          = {fpr:.3f}   (target <= {args.alpha})")
    print(f"power        TPR(rule)        = {power:.3f}   (want high)")
    print(f"specificity  FPR(bias-only)   = {spec:.3f}   (must be low: fires on RULE not bias)")
    print(f"=> detector works (calibrated & powered & specific): {verdict['works']}")


if __name__ == "__main__":
    main()
