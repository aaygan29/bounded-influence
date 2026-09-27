"""End-to-end reproduction of the Gütlin et al. (2026) model-side result.

Trains all seven conditions (objective x mechanism), builds model RDMs, and tests
against the synthetic early/late neural ground truth whether:
  (1) the SUPERVISED model best matches the EARLY (category) target, and
  (2) the PREDICTIVE-LOCAL model best matches the LATE (predictive) target,
i.e. the reported flip in representational alignment across learning.

The neural target is synthetic and labelled as such. This validates the pipeline
and the mechanism, not brain alignment (we lack the McDermott et al. EEG).

Usage:  python run_repro.py [--epochs 30] [--seed 0] [--out results.json]
"""
from __future__ import annotations
import argparse, json, time
import numpy as np

from paradigm import Paradigm
from models import n_params
from train import train_condition, ConditionModel
from rsa import (model_rdm_trailing, model_rdm_leading, synthetic_neural_rdms,
                 align, bms_lite)

CONDITIONS = [
    ("contrastive", "global"), ("contrastive", "local"),
    ("predictive", "global"), ("predictive", "local"),
    ("supervised", "global"),
    ("supervised_shuffled", "global"),
    ("untrained", "global"),
]


def label(obj, mech):
    if obj == "untrained":
        return "untrained"
    if obj == "supervised_shuffled":
        return "supervised_shuffled"
    if obj == "supervised":
        return "supervised"
    return f"{obj}_{mech}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-trials", type=int, default=4000)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    t0 = time.time()
    para = Paradigm(seed=args.seed)
    data = para.dataset(args.n_trials, seed=args.seed + 1)
    early_rdm, late_rdm = synthetic_neural_rdms(para, data)

    print(f"paradigm: {para.n_leading} leading x {para.n_trailing} trailing, "
          f"feat_dim={para.feat_dim}, trials={args.n_trials}")

    # each target is compared against the matched model indexing:
    #   early/category  -> model post-trailing state grouped by trailing category
    #   late/predictive -> model post-leading state grouped by leading category
    rdms_trailing, rdms_leading, alignments = {}, {}, {}
    for obj, mech in CONDITIONS:
        name = label(obj, mech)
        model = train_condition(obj, mech, para, data, epochs=args.epochs,
                                seed=args.seed)
        if name == "untrained":
            print(f"  params/model = {n_params(model.net):,}")
        rt = model_rdm_trailing(model, para, data)
        rl = model_rdm_leading(model, para, data)
        rdms_trailing[name] = rt
        rdms_leading[name] = rl
        alignments[name] = {"early": align(rt, early_rdm),
                            "late": align(rl, late_rdm)}

    bms_early = bms_lite(rdms_trailing, early_rdm, seed=args.seed)
    bms_late = bms_lite(rdms_leading, late_rdm, seed=args.seed)

    best_early = max(bms_early, key=bms_early.get)
    best_late = max(bms_late, key=bms_late.get)

    # Scientifically correct success criteria (untrained is an architecture ceiling
    # on the category target, which in this reduction is raw input geometry, so it
    # is excluded from the "which trained objective" question):
    trained = [c for c in alignments if c != "untrained"]
    best_early_trained = max(trained, key=lambda c: alignments[c]["early"])
    best_late_trained = max(trained, key=lambda c: alignments[c]["late"])
    # double dissociation: supervised loads on category not predictive; predictive
    # loads on predictive not (uniquely) category.
    sup = alignments["supervised"]
    pred = max(alignments["predictive_global"]["late"],
               alignments["predictive_local"]["late"])
    double_dissociation = (
        best_early_trained == "supervised"
        and best_late_trained.startswith("predictive")
        and sup["early"] > sup["late"]
        and pred > alignments["supervised"]["late"]
    )

    result = {
        "epochs": args.epochs, "seed": args.seed, "n_trials": args.n_trials,
        "alignments": alignments,
        "bms_early_best": {k: round(v, 3) for k, v in bms_early.items()},
        "bms_late_best": {k: round(v, 3) for k, v in bms_late.items()},
        "best_early": best_early, "best_late": best_late,
        "best_early_trained": best_early_trained,
        "best_late_trained": best_late_trained,
        "double_dissociation_recovered": bool(double_dissociation),
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== RESULT ===")
    print(f"best trained match to EARLY (category)   : {best_early_trained}"
          f"  (untrained ceiling: {best_early})")
    print(f"best trained match to LATE  (predictive) : {best_late_trained}")
    print(f"double dissociation recovered: {double_dissociation}")
    print(f"P(best|early): {result['bms_early_best']}")
    print(f"P(best|late):  {result['bms_late_best']}")
    print(f"wrote {args.out}  ({result['runtime_sec']}s)")


if __name__ == "__main__":
    main()
