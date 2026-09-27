"""Behavior-matched learning-rule manipulation: the decisive identifiability test.

The two-channel detector (`detect.py`) used an attack that CHANGES behavior, so the
behavioral channel alone caught it. That does not test the case the identifiability
crux is about (matched behavior, different rule). This experiment does.

Construction (distillation pins behavior; an auxiliary head bends the rule):
  teacher       : a control predictive model that learned the TRUE rule.
  student-null   : distilled to reproduce the teacher's OUTPUTS exactly. Behavior
                   matched to teacher, no rule manipulation.
  student-attack : distilled to the SAME teacher outputs (behavior matched), PLUS
                   an attacker objective applied through a SEPARATE head that pulls
                   the shared latent geometry toward a false partner map. The
                   behavioral readout still reproduces the teacher (over-parameterised
                   latent absorbs the aux constraint), so choices are unchanged while
                   the internal learning target differs.

Channels (each arm vs the teacher):
  behavioral      : output agreement with the teacher (should be ~1 for BOTH arms,
                    by construction) and true-partner fidelity DiD (~0 for both).
  representational: rho of the arm's leading-grouped RDM to the teacher's RDM.
                    student-null should stay high; student-attack should drift.

Decision:
  identifiability_broken = behavior is matched in both arms (behavioral channel
  cannot separate them) AND the representational channel separates attack from null
  across seeds. If instead behavior differs, or the representational channel does not
  separate, we report that honestly: the second channel did NOT break the tie.

Runs locally on CPU, synthetic paradigm only.
Usage:  python behavior_matched.py [--epochs 60] [--seeds 5] [--ncat 20] [--lam 0.5]
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from paradigm import Paradigm
from train import train_condition, ConditionModel
from rsa import model_rdm_leading, align, upper


def teacher_predictions(teacher, seq):
    """Teacher's projected next-step prediction from the post-leading state."""
    B = seq.shape[0]
    s1 = torch.zeros(B, teacher.net.h1); s2 = torch.zeros(B, teacher.net.h2)
    with torch.no_grad():
        s1, s2 = teacher.net.step(seq[:, 0], s1, s2)
        return teacher.pred_head(s2)


def train_student(teacher, para, data, *, attack, lam, epochs, seed,
                  match_weight=10.0):
    """Distill teacher outputs into a student. If attack, also bend the latent
    through a separate aux head toward a false partner map."""
    torch.manual_seed(seed)
    student = ConditionModel("predictive", para.feat_dim, para.n_trailing)
    aux = nn.Linear(student.net.h2, para.feat_dim)  # attacker head (latent-only)
    params = list(student.parameters()) + (list(aux.parameters()) if attack else [])
    opt = torch.optim.Adam(params, lr=1e-3)

    seq = torch.tensor(np.stack([data["lead_x"], data["trail_x"]], axis=1))
    tgt = teacher_predictions(teacher, seq).detach()  # behavioral target
    # attacker false-partner embedding per trial (map k -> (k+3) mod n on the LEAD)
    false_lead_partner = (data["leads"] + 3) % para.n_trailing
    atk_emb = torch.tensor(para.trail_emb[false_lead_partner])

    n = seq.shape[0]; batch = 256
    for ep in range(epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch):
            b = idx[start:start + batch]
            sb = seq[b]
            B = sb.shape[0]
            s1 = torch.zeros(B, student.net.h1); s2 = torch.zeros(B, student.net.h2)
            s1, s2 = student.net.step(sb[:, 0], s1, s2)
            pred = student.pred_head(s2)
            loss = match_weight * F.mse_loss(pred, tgt[b])   # pin behavior to teacher
            if attack:
                loss = loss + lam * F.mse_loss(aux(s2), atk_emb[b])  # bend the latent
            opt.zero_grad(); loss.backward(); opt.step()
    return student


def output_agreement(student, teacher, seq):
    """Cosine agreement between student and teacher predictions (1 = identical)."""
    sp = teacher_predictions(student, seq)
    tp = teacher_predictions(teacher, seq)
    return float(F.cosine_similarity(sp, tp, dim=1).mean())


def behavioral_fidelity(model, para):
    """Fraction of leads whose predicted partner is nearest the TRUE valid partner."""
    x = torch.tensor(para.lead_emb)
    B = x.shape[0]
    s1 = torch.zeros(B, model.net.h1); s2 = torch.zeros(B, model.net.h2)
    with torch.no_grad():
        s1, s2 = model.net.step(x, s1, s2)
        pred = model.pred_head(s2).numpy()
    true_partner = para.T.argmax(axis=1)
    d = ((pred[:, None, :] - para.trail_emb[None]) ** 2).sum(-1)
    return float((d.argmin(1) == true_partner).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ncat", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--out", default="behavior_matched_results.json")
    args = ap.parse_args()

    rows = []
    for s in range(args.seeds):
        para = Paradigm(n_leading=args.ncat, n_trailing=args.ncat, seed=s)
        data = para.dataset(5000, seed=s + 1)
        seq = torch.tensor(np.stack([data["lead_x"], data["trail_x"]], axis=1))

        teacher = train_condition("predictive", "global", para, data,
                                  epochs=args.epochs, seed=s, verbose=False)
        # two independently-initialised CLEAN students (the noise floor) + attack
        s_null = train_student(teacher, para, data, attack=False, lam=args.lam,
                               epochs=args.epochs, seed=s + 10)
        s_null2 = train_student(teacher, para, data, attack=False, lam=args.lam,
                                epochs=args.epochs, seed=s + 20)
        s_atk = train_student(teacher, para, data, attack=True, lam=args.lam,
                              epochs=args.epochs, seed=s + 10)

        t_rdm = model_rdm_leading(teacher, para, data)
        n_rdm = model_rdm_leading(s_null, para, data)
        n2_rdm = model_rdm_leading(s_null2, para, data)
        a_rdm = model_rdm_leading(s_atk, para, data)
        row = {
            # behavioral channel: agreement with teacher (should be high for both)
            "agree_null": output_agreement(s_null, teacher, seq),
            "agree_attack": output_agreement(s_atk, teacher, seq),
            # true-partner fidelity, to show choices themselves are matched
            "fid_teacher": behavioral_fidelity(teacher, para),
            "fid_null": behavioral_fidelity(s_null, para),
            "fid_attack": behavioral_fidelity(s_atk, para),
            # representational channel: rho of arm RDM to teacher RDM (1 = no drift)
            "repr_null": align(n_rdm, t_rdm),
            "repr_attack": align(a_rdm, t_rdm),
            # NOISE FLOOR: two clean students vs each other, and null2 vs teacher.
            # The attack only counts if it drifts BEYOND clean-vs-clean variation.
            "noise_floor_nullnull": align(n_rdm, n2_rdm),
            "repr_null2": align(n2_rdm, t_rdm),
        }
        row["beh_did"] = row["fid_attack"] - row["fid_null"]     # behavioral separation
        # attack drift measured against the clean-student noise floor, not teacher:
        row["repr_sep"] = row["noise_floor_nullnull"] - row["repr_attack"]
        rows.append(row)
        print(f"seed {s}: agree_atk {row['agree_attack']:.3f} "
              f"| fid t/n/a {row['fid_teacher']:.2f}/{row['fid_null']:.2f}/{row['fid_attack']:.2f} "
              f"| repr null/null2/atk {row['repr_null']:.2f}/{row['repr_null2']:.2f}/{row['repr_attack']:.2f} "
              f"| floor(n-n2) {row['noise_floor_nullnull']:.2f} "
              f"beh_DiD {row['beh_did']:+.2f} atk_below_floor {row['repr_sep']:+.2f}")

    def ms(key):
        v = np.array([r[key] for r in rows]); return float(v.mean()), float(v.std())

    agree_a = ms("agree_attack"); beh = ms("beh_did"); rsep = ms("repr_sep")
    floor = ms("noise_floor_nullnull")
    # behavior matched: both arms agree with teacher AND choice-fidelity DiD ~ 0
    behavior_matched = agree_a[0] > 0.95 and abs(beh[0]) < 0.10
    # representational channel separates ONLY if the attack drifts beyond the
    # clean-student noise floor on every seed, with a mean gap > 2 sd of that gap.
    repr_separates = rsep[0] - 2 * rsep[1] > 0 and all(r["repr_sep"] > 0 for r in rows)
    identifiability_broken = behavior_matched and repr_separates

    result = {
        "epochs": args.epochs, "seeds": args.seeds, "ncat": args.ncat, "lam": args.lam,
        "agree_attack": agree_a, "behavioral_did": beh,
        "attack_below_noise_floor": rsep, "noise_floor_nullnull": floor,
        "behavior_matched": bool(behavior_matched),
        "repr_channel_separates": bool(repr_separates),
        "identifiability_broken": bool(identifiability_broken),
        "per_seed": rows,
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print("\n=== BEHAVIOR-MATCHED MANIPULATION ===")
    print(f"output agreement with teacher (attack arm): {agree_a[0]:.3f}±{agree_a[1]:.3f}")
    print(f"behavioral DiD (attack-null fidelity):      {beh[0]:+.3f}±{beh[1]:.3f}")
    print(f"clean-student noise floor (rho null vs null2): {floor[0]:.3f}±{floor[1]:.3f}")
    print(f"attack drift BELOW noise floor (floor-attack): {rsep[0]:+.3f}±{rsep[1]:.3f}")
    print(f"behavior matched in both arms:      {behavior_matched}")
    print(f"representational channel separates: {repr_separates}")
    print(f"=> identifiability broken by 2nd channel: {identifiability_broken}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
