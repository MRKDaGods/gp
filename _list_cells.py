import json
nb = json.load(open(r'e:\dev\src\gp\notebooks\kaggle\10c_stages45\mtmc-10c-stages-4-5-association-eval.ipynb'))
cells = nb['cells']
print(f"{len(cells)} cells")
for i, c in enumerate(cells):
    src = c.get('source', [])
    first = src[0][:90].strip() if src else 'EMPTY'
    print(f"Cell {i}: type={c['cell_type']}, lines={len(src)}, first={first}")
