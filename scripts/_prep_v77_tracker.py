#!/usr/bin/env python3
"""v77: Tracker parameter tuning — reduce tracklet fragmentation.

Changes to 10a run cell:
- REMOVE v76 quality_temperature/laplacian_min_var overrides (hurt -0.7pp)
- ADD stage1 interpolation.max_gap=50 (was 30) — fill larger detection gaps
- ADD stage1 intra_merge.max_time_gap=40 (was 25) — merge across longer gaps

Targets: 42 fragmented GT IDs from error analysis.
"""
import json, pathlib

NB_10A = pathlib.Path("notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb")

nb = json.load(open(NB_10A, encoding="utf-8"))

# Find the run cell (Cell 16) with --override
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "--override" in src and "run_pipeline.py" in src:
        lines = cell["source"]

        # Remove v76 overrides
        new_lines = []
        for line in lines:
            if "quality_temperature" in line or "laplacian_min_var" in line:
                continue
            # Remove v76 comment
            if "v76: feature quality" in line:
                continue
            new_lines.append(line)

        # Find the insertion point: after the last --override line, before the closing ]
        insert_idx = None
        for j in range(len(new_lines) - 1, -1, -1):
            if '"--override"' in new_lines[j]:
                insert_idx = j + 1
                break

        # Add v77 tracker overrides
        v77_lines = [
            '    # v77: tracker params to reduce fragmentation (42 fragmented GT IDs)\n',
            '    "--override", "stage1.interpolation.max_gap=50",\n',
            '    "--override", "stage1.intra_merge.max_time_gap=40.0",\n',
        ]
        for k, line in enumerate(v77_lines):
            new_lines.insert(insert_idx + k, line)

        cell["source"] = new_lines
        print(f"Updated Cell {i} (run cell)")
        print("".join(new_lines))
        break

json.dump(nb, open(NB_10A, "w", encoding="utf-8"), ensure_ascii=True)
print(f"\nSaved {NB_10A}")
