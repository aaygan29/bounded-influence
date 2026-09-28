"""Model-free validation that the real IBL manipulation (biased block) is present.
Canonical biased-block effect: on low-contrast trials, P(choose left) is higher in
left-blocks (pL=0.8) than right-blocks (pL=0.2). This is the ground-truth check that
the behavioral channel and the manipulation u are real before any modelling."""
import json, numpy as np, pandas as pd
from pathlib import Path
DATA = Path(__file__).parent / "data"
man = json.load(open(DATA / "manifest.json")); rows = []
for m in man:
    df = pd.read_parquet(DATA / m["file"]); df = df[df["choice"].isin([-1, 1])]
    low = (df["contrastLeft"].fillna(0).abs() <= 0.0625) & (df["contrastRight"].fillna(0).abs() <= 0.0625)
    d = df[low]
    pl8 = (d[d["probabilityLeft"].round(2) == 0.8]["choice"] == 1).mean()
    pl2 = (d[d["probabilityLeft"].round(2) == 0.2]["choice"] == 1).mean()
    rows.append({"session": m["idx"], "n_low": int(low.sum()),
                 "p_left_block": float(pl8), "p_right_block": float(pl2),
                 "block_effect": float(pl8 - pl2)})
eff = np.array([r["block_effect"] for r in rows])
out = {"per_session": rows, "mean_block_effect": float(eff.mean()),
       "std_block_effect": float(eff.std())}
json.dump(out, open(DATA / "block_effect_validation.json", "w"), indent=2)
print(f"mean block effect {eff.mean():+.3f} ± {eff.std():.3f} across {len(rows)} sessions")
