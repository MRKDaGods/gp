#!/usr/bin/env python3
"""v79: Measured tracker refinement between v77 (50/40) and v78 (80/60).

v77 (max_gap=50, merge_time=40): 78.2% NEW BEST
v78 (max_gap=80, merge_time=60, match_thresh=0.80): 75.0% TOO AGGRESSIVE
v79: try the middle ground:
  - interpolation.max_gap: 60 (between 50 and 80)
  - intra_merge.max_time_gap: 50 (between 40 and 60)
  - match_thresh: 0.85 (keep original)
"""
import json, pathlib

NB_10A = pathlib.Path("notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb")

nb = json.load(open(NB_10A, encoding="utf-8"))

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "--override" in src and "run_pipeline.py" in src:
        lines = cell["source"]

        new_lines = []
        for line in lines:
            # Replace v78 comment with v79
            if "v78:" in line:
                new_lines.append(line.replace("v78:", "v79:").replace("push further from v77 78.2%", "measured refinement between v77 and v78"))
            # Update max_gap from 80 to 60
            elif "max_gap=80" in line:
                new_lines.append(line.replace("max_gap=80", "max_gap=60"))
            # Update max_time_gap from 60 to 50
            elif "max_time_gap=60" in line:
                new_lines.append(line.replace("max_time_gap=60.0", "max_time_gap=50.0"))
            # Remove match_thresh override (keep 0.85 default)
            elif "match_thresh=0.80" in line:
                continue
            else:
                new_lines.append(line)

        cell["source"] = new_lines
        print(f"Updated Cell {i}")
        print("".join(new_lines))
        break

json.dump(nb, open(NB_10A, "w", encoding="utf-8"), ensure_ascii=True)
print(f"\nSaved {NB_10A}")
