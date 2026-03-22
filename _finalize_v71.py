"""
Finalize v71 best config:
- Cell 15: disable scan (SCAN_ENABLED = False)
- Verify Cell 9 + Cell 11 have correct params
"""
import json

NB_PATH = r'e:\dev\src\gp\notebooks\kaggle\10c_stages45\mtmc-10c-stages-4-5-association-eval.ipynb'
nb = json.load(open(NB_PATH, encoding='utf-8'))

# --- Disable scan in Cell 15 ---
cell15_src = nb['cells'][15]['source']
new_cell15 = []
for line in cell15_src:
    if line.strip() == 'SCAN_ENABLED = True':
        new_cell15.append(line.replace('SCAN_ENABLED = True', 'SCAN_ENABLED = False'))
    else:
        new_cell15.append(line)
nb['cells'][15]['source'] = new_cell15

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

# Verify all settings
nb2 = json.load(open(NB_PATH, encoding='utf-8'))
cell9 = ''.join(nb2['cells'][9]['source'])
cell11 = ''.join(nb2['cells'][11]['source'])
cell15 = ''.join(nb2['cells'][15]['source'])

print("=== FINAL CONFIG VERIFICATION ===")
checks = [
    ('ALGORITHM = "conflict_free_cc"', cell9),
    ('AQE_K             = 3', cell9),
    ('SIM_THRESH        = 0.53', cell9),
    ('APPEARANCE_WEIGHT = 0.75', cell9),
    ('FUSION_WEIGHT     = 0.10', cell9),
    ('CAMERA_BIAS       = False', cell9),
    ('ZONE_MODEL        = False', cell9),
    ('HIERARCHICAL      = False', cell9),
    ('fic.regularisation=0.1', cell11),
    ('cross_id_nms_iou=0.40', cell11),
    ('min_trajectory_frames=40', cell11),
    ('SCAN_ENABLED = False', cell15),
]
all_ok = True
for check, src in checks:
    ok = check in src
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {check}")
    if not ok:
        all_ok = False

if all_ok:
    print("\nAll checks passed! Ready to commit.")
else:
    print("\nSOME CHECKS FAILED!")
