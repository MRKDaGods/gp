"""Fix 10a and 10c notebooks for v48 Kaggle push.

Changes:
  10a:
    - Remove 09c kernel source from metadata (KD model is garbage)
    - Replace KD cell with a no-op (baseline TransReID is 78% mAP, KD is 22%)
  10c:
    - Disable FAC (confirmed harmful)
    - camera_bias already defaults to false in default.yaml — no change needed
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── 10a: kernel-metadata.json ──────────────────────────────────────────────────
meta_10a = ROOT / "notebooks/kaggle/10a_stages012/kernel-metadata.json"
m = json.loads(meta_10a.read_text(encoding="utf-8"))
old_ks = m.get("kernel_sources", [])
m["kernel_sources"] = [s for s in old_ks if "09c" not in s]
if m["kernel_sources"] != old_ks:
    meta_10a.write_text(json.dumps(m, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"[10a] kernel-metadata.json: removed 09c from kernel_sources")
    print(f"       was: {old_ks}")
    print(f"       now: {m['kernel_sources']}")
else:
    print(f"[10a] kernel-metadata.json: 09c already absent")

# ── 10a: notebook — disable KD cell ──────────────────────────────────────────
nb_10a = ROOT / "notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb"
nb = json.loads(nb_10a.read_text(encoding="utf-8"))

kd_patched = False
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "_KD_SEARCH" in src and "KD model integration" in src:
        nb["cells"][i]["source"] = [
            "# --- KD model integration (DISABLED: 09c model is 22% mAP, baseline is 78%) ---\n",
            "print('\\u26a0 KD model override SKIPPED (09c dead end) \\u2014 using baseline TransReID weights')\n",
        ]
        kd_patched = True
        print(f"[10a] Cell {i}: replaced KD integration with skip message")
        break

if not kd_patched:
    print("[10a] WARNING: could not find KD cell to patch!")

nb_10a.write_text(json.dumps(nb, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
print(f"[10a] notebook saved")

# ── 10c: notebook — disable FAC ─────────────────────────────────────────────
nb_10c = ROOT / "notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb"
nb2 = json.loads(nb_10c.read_text(encoding="utf-8"))

fac_patched = False
for i, cell in enumerate(nb2["cells"]):
    src_lines = cell["source"]
    new_lines = []
    changed = False
    for line in src_lines:
        # Disable FAC in the pipeline command
        if '"stage4.association.fac.enabled=true"' in line:
            line = line.replace("fac.enabled=true", "fac.enabled=false")
            changed = True
        new_lines.append(line)
    if changed:
        nb2["cells"][i]["source"] = new_lines
        fac_patched = True
        print(f"[10c] Cell {i}: FAC disabled (fac.enabled=true → false)")

if not fac_patched:
    print("[10c] WARNING: could not find FAC override to patch!")

nb_10c.write_text(json.dumps(nb2, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
print(f"[10c] notebook saved")

print("\nDone. Review changes with: git diff --stat")
