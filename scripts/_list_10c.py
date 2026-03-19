"""List all cells in the 10c notebook with their type and first line."""
import json

nb = json.load(open("notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb"))
print("Total cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))[:80].replace("\n", " ")
    print(f"  [{i}] {c['cell_type']:8s}: {src}")
