"""Update kernel-metadata.json version for v67."""
import json

META_PATH = r'e:\dev\src\gp\notebooks\kaggle\10c_stages45\kernel-metadata.json'
meta = json.load(open(META_PATH))
print(f"Current title: {meta.get('title')}")

# Check current version
import re
m = re.search(r'v(\d+)', meta.get('title', ''))
if m:
    cur_v = int(m.group(1))
    new_v = cur_v + 1
    meta['title'] = re.sub(r'v\d+', f'v{new_v}', meta['title'])
    print(f"Updated title: {meta['title']} (v{cur_v} -> v{new_v})")
else:
    print("No version found in title")

with open(META_PATH, 'w') as f:
    json.dump(meta, f, indent=2, ensure_ascii=True)
