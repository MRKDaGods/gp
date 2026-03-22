"""Edit 10a notebook: PCA 384→512."""
import json

nb_path = "notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

changed = 0
for i, cell in enumerate(nb["cells"]):
    for j, line in enumerate(cell.get("source", [])):
        if "n_components=384" in line:
            cell["source"][j] = line.replace("n_components=384", "n_components=512")
            print(f"Cell {i}, line {j}: 384 -> 512")
            changed += 1
        if "PCA 256D features" in line:
            cell["source"][j] = line.replace("PCA 256D", "PCA 512D")
            print(f"Cell {i}, line {j}: updated comment 256->512")
            changed += 1

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"Done: {changed} changes")
