"""Two-channel detector for a learning-RULE manipulation.

This is the extension the reproduction exists to enable. `learning_as_control.md`
proposes detecting an attacker who reshapes the *learning rule* (not the current
belief) via a difference-in-differences on the inferred rule, and concedes a
non-identifiability: a rule change and a drifting-anyway rule can look the same
from behavior alone. Gütlin et al. (2026) supplies a second, independent readout
of the same rule change — the representational signature (position on the
predictive RSA axis). Two channels break the tie.

Setup (reuses the reproduction's paradigm and models):
  control arm    : predictive objective trained on the TRUE trailing feature.
  manipulated arm: identical experience, but for a fraction `atk` of trials the
                   predictive TARGET is swapped to an attacker-chosen trailing
                   category. This bends the learned predictive rule toward a false
                   association without changing the input distribution (matched
                   experience), i.e. an attacker input `u` on the slow manifold.

Channels, each measured as a difference-in-differences (manipulated - control):
  1. representational: rho of the post-leading, leading-grouped model RDM to the
     TRUE predictive target. Manipulation should DROP it (the rule now predicts
     the wrong partner).
  2. behavioral: fraction of leading images for which the model's projected
     next-step prediction is nearest the TRUE valid partner. Manipulation should
     drop this too.

Null control: two clean arms (no manipulation). Both DiDs should sit near zero;
their spread is the empirical false-report rate.

Usage:  python detect.py [--epochs 40] [--atk 0.4] [--seeds 5]
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch

from paradigm import Paradigm
from train import train_condition, ConditionModel, LOSSES
from rsa import model_rdm_leading, synthetic_neural_rdms, align
import torch.nn.functional as F


def train_predictive(para, data, *, attack=0.0, atk_seed=0, epochs=40, seed=0):
    """Train a predictive-objective model. If attack>0, swap the predictive target
    to an attacker-chosen trailing category on that fraction of trials."""
    d = dict(data)
    if attack > 0:
        rng = np.random.default_rng(atk_seed)
        trails = data["trails"].copy()
        n = len(trails)
        hit = rng.random(n) < attack
        # attacker maps every category k -> (k + 3) mod n_trailing (a fixed false
        # association); the manipulated target feature is that category's embedding
        false_cat = (trails + 3) % para.n_trailing
        atk_trails = np.where(hit, false_cat, trails)
        d = dict(data)
        # The predictive loss consumes only the leading frame and regresses to
        # trail_x, so bending trail_x bends ONLY the learning target; the observed
        # experience (leading distribution + transition structure) is unchanged.
        d["trail_x"] = para.trail_emb[atk_trails]
    model = train_condition("predictive", "global", para, d, epochs=epochs,
                            seed=seed, verbose=False)
    return model


def behavioral_score(model, para, data):
    """Fraction of leading categories whose projected next-step prediction is
    nearest the TRUE valid partner embedding (predictor-rule fidelity)."""
    lead_ids = np.arange(para.n_leading)
    x = torch.tensor(para.lead_emb[lead_ids])
    B = x.shape[0]
    s1 = torch.zeros(B, model.net.h1); s2 = torch.zeros(B, model.net.h2)
    with torch.no_grad():
        s1, s2 = model.net.step(x, s1, s2)
        pred = model.pred_head(s2).numpy()          # (n_leading, dim)
    true_partner = para.T.argmax(axis=1)            # valid trailing per leading
    # nearest trailing embedding to each prediction
    d = ((pred[:, None, :] - para.trail_emb[None]) ** 2).sum(-1)
    nearest = d.argmin(axis=1)
    return float((nearest == true_partner).mean())


def repr_score(model, para, data):
    """rho of the leading-grouped model RDM to the TRUE predictive target."""
    _, late = synthetic_neural_rdms(para, data)
    return align(model_rdm_leading(model, para, data), late)


def run_pair(para, data, *, attack, epochs, seed):
    ctrl = train_predictive(para, data, attack=0.0, epochs=epochs, seed=seed)
    manp = train_predictive(para, data, attack=attack, atk_seed=seed + 100,
                            epochs=epochs, seed=seed)
    out = {}
    out["repr_ctrl"] = repr_score(ctrl, para, data)
    out["repr_manp"] = repr_score(manp, para, data)
    out["beh_ctrl"] = behavioral_score(ctrl, para, data)
    out["beh_manp"] = behavioral_score(manp, para, data)
    out["repr_did"] = out["repr_manp"] - out["repr_ctrl"]
    out["beh_did"] = out["beh_manp"] - out["beh_ctrl"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--atk", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ncat", type=int, default=8, help="categories per side (RDM size)")
    ap.add_argument("--out", default="detect_results.json")
    args = ap.parse_args()

    attacked, null = [], []
    for s in range(args.seeds):
        para = Paradigm(n_leading=args.ncat, n_trailing=args.ncat, seed=s)
        data = para.dataset(5000, seed=s + 1)
        a = run_pair(para, data, attack=args.atk, epochs=args.epochs, seed=s)
        # null: two clean arms (attack=0), different init seeds
        n_ctrl = train_predictive(para, data, attack=0.0, epochs=args.epochs, seed=s)
        n_alt = train_predictive(para, data, attack=0.0, epochs=args.epochs, seed=s + 50)
        null.append({
            "repr_did": repr_score(n_alt, para, data) - repr_score(n_ctrl, para, data),
            "beh_did": behavioral_score(n_alt, para, data) - behavioral_score(n_ctrl, para, data),
        })
        attacked.append(a)
        print(f"seed {s}: attack repr_DiD={a['repr_did']:+.3f} beh_DiD={a['beh_did']:+.3f} "
              f"| null repr_DiD={null[-1]['repr_did']:+.3f} beh_DiD={null[-1]['beh_did']:+.3f}")

    def stat(rows, key):
        v = np.array([r[key] for r in rows])
        return float(v.mean()), float(v.std())

    ar_m, ar_s = stat(attacked, "repr_did")
    ab_m, ab_s = stat(attacked, "beh_did")
    nr_m, nr_s = stat(null, "repr_did")
    nb_m, nb_s = stat(null, "beh_did")
    # both channels move under attack, and both stay near zero under null
    channels_agree = all(a["repr_did"] < 0 and a["beh_did"] < 0 for a in attacked)
    separated = (ar_m + 2 * ar_s < nr_m - 0) and (ab_m + 2 * ab_s < nb_m - 0)

    result = {
        "attack": args.atk, "epochs": args.epochs, "seeds": args.seeds,
        "attack_repr_did": [ar_m, ar_s], "attack_beh_did": [ab_m, ab_s],
        "null_repr_did": [nr_m, nr_s], "null_beh_did": [nb_m, nb_s],
        "both_channels_move_under_attack": bool(channels_agree),
        "attack_separated_from_null": bool(separated),
        "per_seed": {"attacked": attacked, "null": null},
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print("\n=== DETECTOR ===")
    print(f"attack:  repr DiD {ar_m:+.3f}±{ar_s:.3f}   beh DiD {ab_m:+.3f}±{ab_s:.3f}")
    print(f"null:    repr DiD {nr_m:+.3f}±{nr_s:.3f}   beh DiD {nb_m:+.3f}±{nb_s:.3f}")
    print(f"both channels move under attack: {channels_agree}")
    print(f"attack separated from null (2sd): {separated}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
