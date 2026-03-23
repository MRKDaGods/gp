#!/usr/bin/env python3
"""v78: More aggressive tracker fragmentation reduction.

v77 (max_gap=50, merge_time=40) → 78.2% NEW BEST, id_sw 131→99
v78: push further:
  - interpolation.max_gap: 50 → 80 (fill even larger detection gaps)
  - intra_merge.max_time_gap: 40 → 60 (merge across even longer gaps)
  - match_thresh: 0.85 → 0.80 (more flexible IoU matching in BoTSORT)
"""
import json, pathlib

NB_10A = pathlib.Path("notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb")

nb = json.load(open(NB_10A, encoding="utf-8"))

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "--override" in src and "run_pipeline.py" in src:
        lines = cell["source"]

        # Replace v77 values with v78 values
        new_lines = []
        for line in lines:
            if "v77:" in line:
                new_lines.append(line.replace("v77:", "v78:").replace("42 fragmented GT IDs", "push further from v77 78.2%"))
            elif "max_gap=50" in line:
                new_lines.append(line.replace("max_gap=50", "max_gap=80"))
            elif "max_time_gap=40" in line:
                new_lines.append(line.replace("max_time_gap=40.0", "max_time_gap=60.0"))
            else:
                new_lines.append(line)

        # Add match_thresh override before the closing ]
        insert_idx = None
        for j in range(len(new_lines) - 1, -1, -1):
            if '"--override"' in new_lines[j]:
                insert_idx = j + 1
                break

        new_lines.insert(insert_idx, '    "--override", "stage1.tracker.match_thresh=0.80",\n')

        cell["source"] = new_lines
        print(f"Updated Cell {i}")
        print("".join(new_lines))
        break

json.dump(nb, open(NB_10A, "w", encoding="utf-8"), ensure_ascii=True)
print(f"\nSaved {NB_10A}")
