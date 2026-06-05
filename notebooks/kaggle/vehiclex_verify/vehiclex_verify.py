"""Verify the mrkdagods VehicleX probe output is intact and mountable.

Reads the probe's output via kernel_sources (no local download), checks the
synthetic zip is a real archive with the expected ~1,362 ids / ~113k images,
and writes a tiny verify.json. Guards a 2h GPU run from a Google-Drive quota
wall (which would leave a 0-byte file) or a wrong kernel_sources mount path.
"""

import json
import os
import zipfile

INP = "/kaggle/input/vehiclex-probe"
res = {"input_dir": INP, "input_exists": os.path.isdir(INP), "files": {}}

# Diagnostic: enumerate everything actually mounted under /kaggle/input
KIN = "/kaggle/input"
res["kaggle_input_listing"] = {}
if os.path.isdir(KIN):
    for d in sorted(os.listdir(KIN)):
        dp = os.path.join(KIN, d)
        if os.path.isdir(dp):
            sample = sorted(os.listdir(dp))[:8]
            res["kaggle_input_listing"][d] = sample
        else:
            res["kaggle_input_listing"][d] = os.path.getsize(dp)
print("KAGGLE INPUT:", json.dumps(res["kaggle_input_listing"], indent=2))

if os.path.isdir(INP):
    for f in sorted(os.listdir(INP)):
        p = os.path.join(INP, f)
        res["files"][f] = os.path.getsize(p) if os.path.isfile(p) else "<dir>"
    cand = [
        os.path.join(INP, "vehiclex_vehicleid_adapted"),
        os.path.join(INP, "vehiclex_vehicleid_adapted.zip"),
    ]
    z = next((c for c in cand if os.path.isfile(c)), None)
    res["zip_path"] = z
    res["zip_size"] = os.path.getsize(z) if z else 0
    if z and zipfile.is_zipfile(z):
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
        imgs = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png"))]
        ids = {
            os.path.basename(n).split("_")[0].split(".")[0]
            for n in imgs
            if os.path.basename(n).split("_")[0].split(".")[0].isdigit()
        }
        res.update(
            {"is_zip": True, "n_entries": len(names), "n_images": len(imgs), "n_ids": len(ids)}
        )
        res["VERDICT"] = "OK" if len(imgs) > 50_000 and len(ids) > 1000 else "SUSPECT"
    else:
        res["is_zip"] = False
        res["VERDICT"] = "BAD_OR_MISSING_ZIP"
else:
    res["VERDICT"] = "NO_INPUT_MOUNT"

print(json.dumps(res, indent=2))
with open("/kaggle/working/verify.json", "w") as fh:
    json.dump(res, fh, indent=2)
