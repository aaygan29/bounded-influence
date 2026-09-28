"""Fetch REAL IBL biased-block 2AFC trial tables via the public ONE account.

Public read-only IBL data (openalyx.internationalbrainlab.org), documented public
credentials (username intbrainlab) -- not the user's credentials. Behavioural trial
tables only (~100 KB each). One shared task (biasedChoiceWorld) for comparability.
The real manipulation u_t = probabilityLeft (the block prior), which shifts the
choice bias; the question is whether it also enters the learning rule.
"""
import sys, json
import pandas as pd
from pathlib import Path
from one.api import ONE

OUT = Path(__file__).parent / "data"; OUT.mkdir(exist_ok=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
KEEP = ["choice", "contrastLeft", "contrastRight", "probabilityLeft",
        "feedbackType", "response_times", "stimOn_times", "firstMovement_times"]

one = ONE(base_url="https://openalyx.internationalbrainlab.org",
          password="international", silent=True)
eids = list(one.search(task_protocol="biasedChoiceWorld", dataset="trials.table"))
print(f"{len(eids)} biasedChoiceWorld sessions available; taking first {N}", flush=True)

manifest = []
for i, eid in enumerate(eids):
    if len(manifest) >= N:
        break
    try:
        df = one.load_dataset(eid, "_ibl_trials.table.pqt")
        df = df[[c for c in KEEP if c in df.columns]].copy()
        # a usable session: enough trials and real block structure (both 0.2 and 0.8)
        pl = set(df["probabilityLeft"].round(2).unique())
        if len(df) < 400 or not ({0.2, 0.8} <= pl):
            continue
        f = OUT / f"session_{len(manifest):02d}.parquet"
        df.to_parquet(f)
        manifest.append({"idx": len(manifest), "eid": str(eid),
                         "n_trials": int(len(df)), "file": f.name,
                         "blocks": sorted(pl)})
        print(f"  saved {f.name}: {len(df)} trials, blocks {sorted(pl)}", flush=True)
    except Exception as e:
        print(f"  skip {eid}: {type(e).__name__}", flush=True)

json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
print(f"DONE: {len(manifest)} comparable sessions", flush=True)
