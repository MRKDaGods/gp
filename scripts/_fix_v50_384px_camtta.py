"""v50: Add 384px + CamTTA to 10a, disable scan/AB in 10c (v49 proved optimal)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
changes = 0

# ── 10a: Add 384px + CamTTA overrides ────────────────────────────────────
nb10a_path = ROOT / "notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb"
nb10a = json.loads(nb10a_path.read_text(encoding="utf-8"))

for i, cell in enumerate(nb10a["cells"]):
    src = "".join(cell["source"])
    if 'stage2.power_norm.alpha=0.5",' in src and "input_size" not in src:
        # Add 384px and CamTTA overrides after power_norm line
        new_lines = []
        for line in cell["source"]:
            new_lines.append(line)
            if 'stage2.power_norm.alpha=0.5",' in line:
                indent = line[:len(line) - len(line.lstrip())]
                new_lines.append(indent + '# v50: 384px resolution + camera TTA\n')
                new_lines.append(indent + '"--override", "stage2.reid.vehicle.input_size=[384,384]",\n')
                new_lines.append(indent + '"--override", "stage2.camera_tta.enabled=true",\n')
        nb10a["cells"][i]["source"] = new_lines
        print(f"[10a Cell {i}] Added 384px + CamTTA overrides")
        changes += 1
        break

if changes > 0:
    nb10a_path.write_text(json.dumps(nb10a, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")

# ── 10c: Disable scan + A/B (v49 scan found no improvements) ─────────────
nb10c_path = ROOT / "notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb"
nb10c = json.loads(nb10c_path.read_text(encoding="utf-8"))

for i, cell in enumerate(nb10c["cells"]):
    src = "".join(cell["source"])

    if "SCAN_ENABLED = True" in src:
        nb10c["cells"][i]["source"] = [
            line.replace("SCAN_ENABLED = True", "SCAN_ENABLED = False")
            for line in cell["source"]
        ]
        print(f"[10c Cell {i}] SCAN_ENABLED = False")
        changes += 1

    if "FEATURE_TEST_ENABLED = True" in src:
        nb10c["cells"][i]["source"] = [
            line.replace("FEATURE_TEST_ENABLED = True", "FEATURE_TEST_ENABLED = False")
            for line in cell["source"]
        ]
        print(f"[10c Cell {i}] FEATURE_TEST_ENABLED = False")
        changes += 1

nb10c_path.write_text(json.dumps(nb10c, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")

print(f"\nTotal: {changes} changes applied")
