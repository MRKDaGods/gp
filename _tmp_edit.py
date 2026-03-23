import json

nb_path = 'notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][16]
new_source = ''.join(cell['source'])

# Add confidence_threshold=0.20 after min_hits=2
# Also need to lower track_high_thresh and new_track_thresh to match (BotSort docs say they MUST match)
old = '    "--override", "stage1.tracker.min_hits=2",'
new_text = (old + '\n'
    '    # v81: confidence_threshold=0.20 (default=0.25) -- more detections\n'
    '    "--override", "stage1.detector.confidence_threshold=0.20",\n'
    '    "--override", "stage1.tracker.track_high_thresh=0.20",\n'
    '    "--override", "stage1.tracker.new_track_thresh=0.20",')

new_source = new_source.replace(old, new_text)

lines = new_source.split('\n')
cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print('OK')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb2 = json.load(f)
src = ''.join(nb2['cells'][16]['source'])
# Show just the override lines
for line in src.split('\n'):
    if '--override' in line or line.strip().startswith('#'):
        print(line)
