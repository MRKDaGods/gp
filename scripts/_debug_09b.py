"""Debug script for 09b notebook validation."""
import json
import ast

nb_path = "notebooks/kaggle/09b_vehicle_reid_384px/09b_vehicle_reid_384px.ipynb"

with open(nb_path) as f:
    nb = json.load(f)

print(f"JSON valid. Cells: {len(nb['cells'])}")
code_cells = [(i, c) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
print(f"Code cells: {len(code_cells)}")

for i, c in code_cells:
    src = "".join(c.get("source", []))
    # Skip magic commands
    lines = [l for l in src.split("\n") if not l.startswith("!") and not l.startswith("%")]
    filtered = "\n".join(lines)
    try:
        ast.parse(filtered)
    except SyntaxError as e:
        print(f"Cell {i}: SYNTAX ERROR: {e}")
        print("  Source preview:", src[:200])

# Print all cells briefly
print("\n=== Cell Summary ===")
for i, c in code_cells:
    src = "".join(c.get("source", []))
    print(f"Cell {i}: {src[:100].replace(chr(10), ' ')}")
