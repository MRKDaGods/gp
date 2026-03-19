"""Validate all code cells in 09c notebook."""
import json, ast

nb = json.load(open("notebooks/kaggle/09c_kd_vitl_teacher/09c_kd_vitl_teacher.ipynb"))
errors = []
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    # Remove magic commands
    lines = [l for l in src.split("\n") if not l.startswith("!") and not l.startswith("%")]
    try:
        ast.parse("\n".join(lines))
        print(f"Cell {i:2d}: OK  ({src[:60].replace(chr(10),' ')})")
    except SyntaxError as e:
        errors.append((i, str(e), src[:200]))
        print(f"Cell {i:2d}: SYNTAX ERROR: {e}")

if errors:
    print(f"\n{len(errors)} ERRORS")
else:
    print(f"\nAll {sum(1 for c in nb['cells'] if c['cell_type']=='code')} code cells OK")
