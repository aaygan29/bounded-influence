"""Replicate the Roy/Pillow PsyTrack dynamic-GLM on real IBL biased-block data.

Behavioral channel of the E7B test. Fits time-varying weights on
{bias, signed-contrast, previous-choice} to each session's real choices. Validation
(the basis-replication check): the dynamic BIAS weight should track the real block
prior probabilityLeft -- the known IBL biased-block effect. If it does not, our
behavioral readout is not capturing the manipulation and nothing downstream is valid.
"""
import numpy as np, pandas as pd
from pathlib import Path
import psytrack as psy

DATA = Path(__file__).parent / "data"


def prep(df):
    """Real IBL trials -> PsyTrack dataset dict. Keep decided trials (choice != 0)."""
    df = df[df["choice"].isin([-1, 1])].reset_index(drop=True)
    cL = df["contrastLeft"].fillna(0).to_numpy()
    cR = df["contrastRight"].fillna(0).to_numpy()
    signed = cR - cL                                  # signed contrast, right positive
    prev = np.zeros(len(df)); prev[1:] = df["choice"].to_numpy()[:-1]
    # PsyTrack y in {1,2}: 1 = chose left(-1), 2 = chose right(+1)
    y = np.where(df["choice"].to_numpy() == 1, 2, 1).astype(int)
    inputs = {"signed": signed[:, None].astype(float),
              "prev": prev[:, None].astype(float)}
    D = {"inputs": inputs, "y": y, "dayLength": np.array([len(df)])}
    return D, df["probabilityLeft"].to_numpy()


def fit_session(df, seed=0):
    D, pL = prep(df)
    weights = {"bias": 1, "signed": 1, "prev": 1}
    K = sum(weights.values())
    hyper = {"sigInit": 2**4, "sigma": [2**-5] * K, "sigDay": None}
    opt = ["sigma"]
    hyp, ev, wMode, hess = psy.hyperOpt(D, hyper, weights, opt, showOpt=0)
    # wMode rows: [bias, signed, prev] (order = alphabetical-ish per psytrack)
    return wMode, weights, pL


if __name__ == "__main__":
    import json, sys
    man = json.load(open(DATA / "manifest.json"))
    rows = []
    for m in man:
        df = pd.read_parquet(DATA / m["file"])
        wMode, weights, pL = fit_session(df, seed=m["idx"])
        names = list(weights.keys())
        bias = wMode[names.index("bias")]
        # validation: does the dynamic bias track the block prior?
        # block prior signal centered: higher probabilityLeft -> bias toward left
        block = (pL - 0.5)
        # align lengths (decided trials)
        r = np.corrcoef(bias, block[:len(bias)])[0, 1]
        rows.append({"session": m["idx"], "n": len(bias), "bias_block_r": float(r)})
        print(f"session {m['idx']}: n={len(bias)} corr(dynamic bias, block prior) = {r:+.3f}", flush=True)
    rr = np.array([x["bias_block_r"] for x in rows])
    print(f"\nMEAN corr(bias, block) = {rr.mean():+.3f} ± {rr.std():.3f}  (validation: should be clearly nonzero, sign = bias follows block)")
    json.dump(rows, open(DATA / "psytrack_validation.json", "w"), indent=2)
