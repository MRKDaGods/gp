import json

nb_path = 'notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][16]
new_source = ''.join(cell['source'])

# Remove v82 denoise lines, add v83 max_iou_distance
old_v82 = (
    '    # v82: denoise=true (default=false) -- bilateral filter noise reduction\n'
    '    "--override", "stage0.denoise=true",'
)
new_source = new_source.replace(old_v82,
    '    # v83: tighter intra-merge spatial matching (default=0.7)\n'
    '    "--override", "stage1.intra_merge.max_iou_distance=0.5",'
)

lines = new_source.split('\n')
cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print('OK')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)
src = ''.join(nb2['cells'][16]['source'])
for line in src.split('\n'):
    if '--override' in line or line.strip().startswith('#'):
        print(line)
