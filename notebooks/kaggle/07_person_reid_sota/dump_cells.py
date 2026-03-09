"""Dump key cells from NB07 for analysis."""
import json

nb = json.load(open(r'e:\dev\src\gp\notebooks\kaggle\07_person_reid_sota\07_person_reid_sota.ipynb', 'r', encoding='utf-8'))
for ci in [0, 3, 5, 6, 8, 10, 14, 19, 20, 24, 25]:
    src = ''.join(nb['cells'][ci]['source'])
    lines = src.split('\n')
    print(f'=== CELL {ci} ({len(lines)} lines) ===')
    for i, l in enumerate(lines):
        print(f'{i:3d}: {l}')
    print()
