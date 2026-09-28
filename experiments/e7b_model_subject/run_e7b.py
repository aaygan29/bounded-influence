"""E7B Tier 3+ model-subject test on REAL IBL data.

For each real session, train three model-subjects cloned to the same real choices:
control-A, control-B (the noise-floor pair), and attack (learning-rule manipulated).
The representational channel is a functional short-horizon value probe trained on
CLEAN control subjects from OTHER sessions (held-out, cross-session, never sees the
attack). Confound controls, per the council review:
  - probe circularity: probe target (short-horizon value) != attack target
    (long-horizon reward integration); probe trained cross-session on clean subjects.
  - behavioral-only classifier: can arms be told apart from choices alone? must be
    ~chance, else "behavior matched" failed.
  - noise floor: control-A vs control-B (two clean subjects), the null the attack
    must beat.
  - order parameter: hidden-state effective rank (rich/lazy), reported per arm.
Headline: does the probe separate attack from the noise floor, BEYOND the behavioral
classifier's accuracy? Reported per session and aggregated across the real sessions.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score

from model_subject import clone_subject, subject_outputs

DATA = Path(__file__).parent / "data"


def short_horizon_value(reward, w=3):
    r = reward; out = np.zeros_like(r)
    for t in range(len(r)):
        out[t] = r[max(0, t - w):t + 1].mean()
    return out


def effective_rank(H):
    Hc = H - H.mean(0, keepdims=True)
    s = np.linalg.svd(Hc, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))


def condition_rdm(o, n_cbin=5):
    """RDM of hidden states grouped by task condition (block x signed-contrast bin).
    This is the faithful port of the synthetic representational readout, and it is
    independent of the attack (which manipulates integration timescale, not the
    block x contrast geometry). Returns the upper triangle of the correlation-distance
    RDM over the occupied conditions, plus the condition key so RDMs are comparable."""
    H, blk = o["H"], o["block"]
    sc = o["signed_contrast"]
    edges = np.quantile(sc, np.linspace(0, 1, n_cbin + 1))
    cbin = np.clip(np.digitize(sc, edges[1:-1]), 0, n_cbin - 1)
    conds, means = [], []
    for b in (0.2, 0.5, 0.8):
        for c in range(n_cbin):
            m = np.isclose(blk, b) & (cbin == c)
            if m.sum() >= 3:
                conds.append((b, c)); means.append(H[m].mean(0))
    return conds, np.array(means)


def rsa_between(mA, mB, condsA, condsB):
    """RSA (Spearman) between two condition-mean sets over their SHARED conditions."""
    from scipy.stats import spearmanr
    shared = [c for c in condsA if c in condsB]
    if len(shared) < 4:
        return np.nan
    ia = [condsA.index(c) for c in shared]; ib = [condsB.index(c) for c in shared]
    def rdm(M):
        Mc = M - M.mean(1, keepdims=True)
        Mc /= (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-9)
        C = Mc @ Mc.T
        iu = np.triu_indices(len(M), 1)
        return (1 - C)[iu]
    ra, rb = rdm(mA[ia]), rdm(mB[ib])
    return float(spearmanr(ra, rb).correlation)


def behavioral_features(o):
    """Choice-only summary features of an arm (block-conditioned psychometric etc.).
    If behavior is matched, these cannot separate attack from control."""
    ch, blk = o["choice_hat"], o["block"]
    feats = []
    for b in (0.2, 0.5, 0.8):
        m = np.isclose(blk, b)
        feats.append(ch[m].mean() if m.any() else 0.5)
    # simple lag-1 choice autocorrelation
    feats.append(np.corrcoef(ch[:-1], ch[1:])[0, 1] if len(ch) > 2 else 0.0)
    return np.array(feats)


def main():
    man = json.load(open(DATA / "manifest.json"))
    dfs = {m["idx"]: pd.read_parquet(DATA / m["file"]) for m in man}
    ids = sorted(dfs)

    # train the three arms per session once
    arms = {}
    for i in ids:
        arms[i] = {
            "A": clone_subject(dfs[i], attack=False, seed=100 + i),
            "B": clone_subject(dfs[i], attack=False, seed=200 + i),
            "atk": clone_subject(dfs[i], attack=True, lam=1.0, seed=100 + i),
        }
        print(f"trained arms for session {i}", flush=True)

    rows = []
    for i in ids:
        # probe trained on CLEAN control-A subjects of all OTHER sessions
        Xtr, ytr = [], []
        for j in ids:
            if j == i:
                continue
            o = subject_outputs(arms[j]["A"], dfs[j])
            Xtr.append(o["H"]); ytr.append(short_horizon_value(o["reward"]))
        probe = Ridge(alpha=1.0).fit(np.vstack(Xtr), np.concatenate(ytr))

        oA = subject_outputs(arms[i]["A"], dfs[i])
        oB = subject_outputs(arms[i]["B"], dfs[i])
        oK = subject_outputs(arms[i]["atk"], dfs[i])
        vA, vB, vK = (probe.predict(o["H"]) for o in (oA, oB, oK))

        def dist(x, y):  # representational distance between decoded-value trajectories
            r = pearsonr(x, y)[0]
            return 1.0 - (r if np.isfinite(r) else 0.0)

        noise_floor = dist(vA, vB)
        attack_dist = dist(vA, vK)
        repr_sep = attack_dist - noise_floor

        # SECOND representational readout: condition-RDM RSA (faithful port of the
        # synthetic metric, non-circular). Higher RSA = more similar reps.
        cA, mA_ = condition_rdm(oA); cB, mB_ = condition_rdm(oB); cK, mK_ = condition_rdm(oK)
        rsa_AB = rsa_between(mA_, mB_, cA, cB)     # noise floor (clean-clean similarity)
        rsa_AK = rsa_between(mA_, mK_, cA, cK)     # attack similarity to control
        rdm_sep = (rsa_AB - rsa_AK) if np.isfinite(rsa_AB) and np.isfinite(rsa_AK) else np.nan

        # raw representational identifiability: cosine drift between clean subjects
        # (the noise floor in RAW hidden space) vs attack. If clean-clean drift is
        # ~1 (uncorrelated), the representation is underdetermined by behavior and no
        # readout can beat it. This is the core diagnostic of the real-data null.
        def raw_drift(hx, hy):
            a = hx / (np.linalg.norm(hx, axis=1, keepdims=True) + 1e-9)
            b = hy / (np.linalg.norm(hy, axis=1, keepdims=True) + 1e-9)
            return float((1 - (a * b).sum(1)).mean())
        raw_floor = raw_drift(oA["H"], oB["H"])
        raw_attack = raw_drift(oA["H"], oK["H"])

        # behavior-matched check: agreement of choices between arms
        agree_AB = float((oA["choice_hat"] == oB["choice_hat"]).mean())
        agree_AK = float((oA["choice_hat"] == oK["choice_hat"]).mean())
        # behavioral-only classifier: separate {A,B}=control from {atk}=attack by
        # choice features. With 1 sample/class this is degenerate, so we score
        # whether the attack's behavioral features fall within the control spread.
        fA, fB, fK = (behavioral_features(o) for o in (oA, oB, oK))
        beh_gap = float(np.linalg.norm(fK - (fA + fB) / 2))     # attack vs control mean
        beh_floor = float(np.linalg.norm(fA - fB))              # control-control spread

        rows.append({
            "session": i,
            "noise_floor": noise_floor, "attack_dist": attack_dist,
            "repr_sep": repr_sep,
            "rsa_AB": rsa_AB, "rsa_AK": rsa_AK, "rdm_sep": rdm_sep,
            "raw_floor": raw_floor, "raw_attack": raw_attack,
            "agree_AB": agree_AB, "agree_AK": agree_AK,
            "beh_gap": beh_gap, "beh_floor": beh_floor,
            "erank_ctrl": effective_rank(oA["H"]),
            "erank_atk": effective_rank(oK["H"]),
        })
        print(f"session {i}: probe_sep={repr_sep:+.3f} | rdm_sep={rdm_sep:+.3f} "
              f"(rsa AB/AK {rsa_AB:.2f}/{rsa_AK:.2f}) "
              f"| choice agree AK {agree_AK:.3f} "
              f"| beh gap/floor {beh_gap:.3f}/{beh_floor:.3f} "
              f"| erank c/a {rows[-1]['erank_ctrl']:.1f}/{rows[-1]['erank_atk']:.1f}", flush=True)

    def ms(k):
        v = np.array([r[k] for r in rows if np.isfinite(r[k])])
        return (float(v.mean()), float(v.std())) if len(v) else (float("nan"), float("nan"))

    rs, nf, ad = ms("repr_sep"), ms("noise_floor"), ms("attack_dist")
    rdm = ms("rdm_sep")
    rawf, rawa = ms("raw_floor"), ms("raw_attack")
    ag = ms("agree_AK"); bg, bf = ms("beh_gap"), ms("beh_floor")
    behavior_matched = ag[0] > 0.90 and bg[0] <= bf[0] + bf[1]
    probe_separates = rs[0] - 2 * rs[1] > 0 and all(r["repr_sep"] > 0 for r in rows)
    rdm_separates = (rdm[0] - 2 * rdm[1] > 0 and
                     all(r["rdm_sep"] > 0 for r in rows if np.isfinite(r["rdm_sep"])))
    repr_separates = probe_separates or rdm_separates
    # incremental: representational separation exists WHERE behavior did not separate
    identifiability_broken = behavior_matched and repr_separates

    result = {
        "n_sessions": len(ids),
        "probe_readout": {"repr_sep": rs, "noise_floor": nf, "attack_dist": ad,
                          "separates": bool(probe_separates)},
        "rdm_readout": {"rdm_sep": rdm, "separates": bool(rdm_separates)},
        "raw_identifiability": {"clean_clean_drift": rawf, "attack_drift": rawa,
                                "note": "clean-clean drift ~1 => representation underdetermined by behavior"},
        "choice_agree_attack_vs_control": ag,
        "behavioral_gap": bg, "behavioral_floor": bf,
        "behavior_matched": bool(behavior_matched),
        "repr_channel_separates": bool(repr_separates),
        "identifiability_broken": bool(identifiability_broken),
        "per_session": rows,
    }
    json.dump(result, open(Path(__file__).parent / "e7b_results.json", "w"), indent=2)
    print("\n=== E7B (real IBL model-subject) ===")
    print(f"behavior matched (choice agree {ag[0]:.3f}, beh gap {bg[0]:.3f} vs floor {bf[0]:.3f}): {behavior_matched}")
    print(f"probe readout   separation: {rs[0]:+.3f}±{rs[1]:.3f}  separates={probe_separates}")
    print(f"RDM readout     separation: {rdm[0]:+.3f}±{rdm[1]:.3f}  separates={rdm_separates}")
    print(f"raw identifiability: clean-clean hidden drift {rawf[0]:.3f} (>~1 = underdetermined); "
          f"attack drift {rawa[0]:.3f}")
    print(f"=> identifiability broken on REAL data (either readout): {identifiability_broken}")


if __name__ == "__main__":
    main()
