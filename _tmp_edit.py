import json

nb_path = 'notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][16]
new_source = ''.join(cell['source'])

old = '    "--override", "stage1.intra_merge.max_time_gap=40.0",'
new_text = old + '\n    # v80: min_hits=2 (default=3) -- recover short tracklets\n    "--override", "stage1.tracker.min_hits=2",'

new_source = new_source.replace(old, new_text)

lines = new_source.split('\n')
cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print('OK')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)
print(''.join(nb2['cells'][16]['source']))
