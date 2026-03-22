"""Fix the corrupted comment in scan cell."""
import json

nb_path = "notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Fix the corrupted line in scan cell (cell 15)
for j, line in enumerate(nb["cells"][15]["source"]):
    if "Current best v71: INTRA_MERGE_THRESH=0.75, INTRA_MERGE_GAP" in line:
        nb["cells"][15]["source"][j] = "# Current best v71: thresh=0.75, gap=60 -> v72 best: thresh=0.80, gap=30\n"
        print(f"Fixed line {j}")
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print("Done")
