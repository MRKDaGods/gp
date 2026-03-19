"""Fix hardcoded mrkdagods paths in notebooks for account migration."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "notebooks" / "kaggle"


def fix_cell_source(nb, cell_idx, old, new):
    src = "".join(nb["cells"][cell_idx]["source"])
    if old not in src:
        return False
    src = src.replace(old, new)
    lines = src.split("\n")
    nb["cells"][cell_idx]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return True


def save_nb(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
        f.write("\n")


# --- 10a: dataset path fallback ---
p = ROOT / "10a_stages012" / "mtmc-10a-stages-0-2-tracking-reid-features.ipynb"
with open(p) as f:
    nb = json.load(f)

old_10a = 'WEIGHTS_INPUT = Path("/kaggle/input/datasets/mrkdagods/mtmc-weights/models")\nassert WEIGHTS_INPUT.exists()'
new_10a = (
    'WEIGHTS_INPUT = None\n'
    'for _p in [\n'
    '    Path("/kaggle/input/mtmc-weights/models"),\n'
    '    Path("/kaggle/input/datasets/mrkdagods/mtmc-weights/models"),\n'
    ']:\n'
    '    if _p.exists():\n'
    '        WEIGHTS_INPUT = _p\n'
    '        break\n'
    'assert WEIGHTS_INPUT is not None'
)
if fix_cell_source(nb, 7, old_10a, new_10a):
    save_nb(nb, p)
    print("Fixed 10a cell 7: dataset path fallback")
else:
    print("10a cell 7: pattern not found (may already be fixed)")

# --- 10b: kernel source slug ---
p = ROOT / "10b_stage3" / "mtmc-10b-stage-3-faiss-indexing.ipynb"
with open(p) as f:
    nb = json.load(f)
if fix_cell_source(nb, 7,
    "mrkdagods/mtmc-10a-stages-0-2-tracking-reid-features",
    "yahiaakhalafallah/mtmc-10a-stages-0-2-tracking-reid-features"):
    save_nb(nb, p)
    print("Fixed 10b cell 7: kernel source slug")
else:
    print("10b cell 7: already correct or pattern not found")

# --- 10c: kernel source slug ---
p = ROOT / "10c_stages45" / "mtmc-10c-stages-4-5-association-eval.ipynb"
with open(p) as f:
    nb = json.load(f)
if fix_cell_source(nb, 7,
    "mrkdagods/mtmc-10b-stage-3-faiss-indexing",
    "yahiaakhalafallah/mtmc-10b-stage-3-faiss-indexing"):
    save_nb(nb, p)
    print("Fixed 10c cell 7: kernel source slug")
else:
    print("10c cell 7: already correct or pattern not found")

# --- 09b: search paths (they have fallback already, but update comment + git URL) ---
p = ROOT / "09b_vehicle_reid_384px" / "09b_vehicle_reid_384px.ipynb"
with open(p) as f:
    nb = json.load(f)
changed = False
# Fix git clone URL
if fix_cell_source(nb, 3,
    "https://github.com/mrkdagods/gp.git",
    "https://github.com/yahiaakhalafallah/gp.git"):
    changed = True
    print("Fixed 09b cell 3: git clone URL")
# Search paths already have fallback (/kaggle/input/mtmc-weights/...) so the
# mrkdagods-prefixed path is just a secondary option. Leave it as-is.
# Fix upload instructions
if fix_cell_source(nb, 24,
    "mrkdagods/mtmc-weights",
    "yahiaakhalafallah/mtmc-weights or mrkdagods/mtmc-weights"):
    changed = True
    print("Fixed 09b cell 24: upload instructions")
if changed:
    save_nb(nb, p)

# --- 09c: search paths ---
p = ROOT / "09c_kd_vitl_teacher" / "09c_kd_vitl_teacher.ipynb"
with open(p) as f:
    nb = json.load(f)
# Add slug-only fallback before mrkdagods-prefixed paths
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "/kaggle/input/datasets/mrkdagods/mtmc-weights" in src and "/kaggle/input/mtmc-weights" not in src:
        # Add fallback paths
        src = src.replace(
            'Path("/kaggle/input/datasets/mrkdagods/mtmc-weights',
            'Path("/kaggle/input/mtmc-weights'
        )
        lines = src.split("\n")
        nb["cells"][i]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        save_nb(nb, p)
        print(f"Fixed 09c cell {i}: dataset path (slug-only)")
        break
else:
    print("09c: paths already have fallback or pattern not found")

print("\nDone. Verify with: python -c \"import json; ...\"")
