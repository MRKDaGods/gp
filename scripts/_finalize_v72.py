"""
v72 finalize: Update 10c notebook with intra-merge wins + disable scan for PCA 512D run.
- INTRA_MERGE_THRESH: 0.75 -> 0.80 (+0.14pp)
- INTRA_MERGE_GAP: 60 -> 30
- SCAN_ENABLED: True -> False
"""
import json

nb_path = "notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

changes = 0
for i, cell in enumerate(nb["cells"]):
    for j, line in enumerate(cell.get("source", [])):
        # Update intra-merge threshold
        if "INTRA_MERGE_THRESH = 0.75" in line or "INTRA_MERGE_THRESH = 0.80" in line:
            cell["source"][j] = line.split("INTRA_MERGE_THRESH")[0] + "INTRA_MERGE_THRESH = 0.80   # v72: +0.14pp (was 0.75)\n"
            print(f"Cell {i}: Updated INTRA_MERGE_THRESH -> 0.80")
            changes += 1
        # Update intra-merge gap
        if "INTRA_MERGE_GAP" in line and "=" in line and "intra_camera_merge" not in line and "intra_gap" not in line:
            cell["source"][j] = line.split("INTRA_MERGE_GAP")[0] + "INTRA_MERGE_GAP   = 30     # v72: tighter gap (was 60)\n"
            print(f"Cell {i}: Updated INTRA_MERGE_GAP -> 30")
            changes += 1
        # Disable scan
        if "SCAN_ENABLED = True" in line:
            cell["source"][j] = line.replace("SCAN_ENABLED = True", "SCAN_ENABLED = False")
            print(f"Cell {i}: Disabled scan")
            changes += 1

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"\nDone: {changes} changes applied")
