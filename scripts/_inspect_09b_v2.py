"""Inspect 09b v2 notebook for error outputs."""
import json, sys

path = "data/09b_v2/09b-vehicle-reid-384px-fine-tune.ipynb"
nb = json.load(open(path, encoding="utf-8", errors="replace"))

cells = nb["cells"]
exec_cells = [c for c in cells if c["cell_type"] == "code"]
print(f"Total cells: {len(cells)}, code cells: {len(exec_cells)}")
print(f"Cells with outputs: {sum(1 for c in cells if c.get('outputs'))}")
print(f"Last executed cell: {max((c.get('execution_count') or 0) for c in exec_cells)}")

for i, c in enumerate(cells):
    outs = c.get("outputs", [])
    if not outs:
        continue
    for o in outs:
        t = "".join(o.get("text", []) + o.get("traceback", []))
        if t:
            print(f"\n--- Cell {i} ({o.get('output_type')}) ---")
            print(t[-2000:])
