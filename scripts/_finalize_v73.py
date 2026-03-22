"""
v73 finalize: Update 10c with app_w=0.70 win + disable scan.
"""
import json

nb_path = "notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

changes = 0
for i, cell in enumerate(nb["cells"]):
    for j, line in enumerate(cell.get("source", [])):
        # Update appearance weight 0.75 -> 0.70
        if "APPEARANCE_WEIGHT = 0.75" in line:
            cell["source"][j] = "APPEARANCE_WEIGHT = 0.70   # v73: +0.76pp over 0.75 (more ST weight helps cross-cam)\n"
            print(f"Cell {i}: APPEARANCE_WEIGHT -> 0.70")
            changes += 1
        # Update the ST_WEIGHT auto-compute (it should auto-derive but let's verify comments)
        # Disable scan
        if "SCAN_ENABLED = True" in line:
            cell["source"][j] = line.replace("SCAN_ENABLED = True", "SCAN_ENABLED = False")
            print(f"Cell {i}: Disabled scan")
            changes += 1

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"\nDone: {changes} changes")
