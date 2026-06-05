"""Read 09w training metrics via kernel_sources (no big download).

Mounts the 09w training kernel's output and extracts the small metric JSONs
(final mAP, per-epoch history, split stats), printing them and writing a tiny
results.json that we download locally — avoids pulling the ~1.4GB full output.
"""

import glob
import json
import os

res = {}
for name in [
    "resnext101ibn_synth_metadata.json",
    "stage1_history.json",
    "resnext101ibn_synth_history.json",
]:
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if hits:
        try:
            with open(hits[0]) as fh:
                res[name] = json.load(fh)
        except Exception as e:  # noqa: BLE001
            res[name] = f"ERR {e}"
    else:
        res[name] = "NOT_FOUND"

# Compact summary
summary = {}
meta = res.get("resnext101ibn_synth_metadata.json")
if isinstance(meta, dict):
    summary["final_metrics"] = meta.get("final_metrics")
    summary["stage1_best"] = meta.get("stage1_best")
    summary["num_train_ids"] = meta.get("num_train_ids")
    summary["num_synthetic_imgs"] = meta.get("num_synthetic_imgs")
    summary["split_stats"] = meta.get("split_stats")
hist = res.get("stage1_history.json")
if isinstance(hist, list):
    summary["epochs_logged"] = len(hist)
    summary["eval_points"] = [
        {"epoch": r.get("epoch"), "mAP": r.get("mAP"), "rank1": r.get("rank1")}
        for r in hist
        if r.get("mAP") is not None
    ]

print("================ 09w RESULTS SUMMARY ================")
print(json.dumps(summary, indent=2))
with open("/kaggle/working/results.json", "w") as fh:
    json.dump({"summary": summary, "raw": res}, fh, indent=2)
