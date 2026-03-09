"""Fix cell 14: remove duplicate prints, fix elapsed bug."""
import json, pathlib

path = pathlib.Path(__file__).resolve().parent / '08b_vehicle_reid_triplet.ipynb'
nb = json.load(open(path, 'r', encoding='utf-8'))
src = ''.join(nb['cells'][14]['source'])
lines = src.split('\n')

# Find and remove duplicate lines
# Line 27 is a duplicate "Metric loss" print right after "Label smoothing"
new_lines = []
seen_metric = False
seen_losses_banner = False
for i, line in enumerate(lines):
    stripped = line.strip()
    
    # Skip duplicate "Metric loss" print (keep first, drop second)
    if 'Metric loss: TripletLoss' in stripped:
        if seen_metric:
            print(f"  Removing duplicate line {i}: {stripped}")
            continue
        seen_metric = True
    
    # Skip duplicate "Losses:" banner (keep first, drop second)
    if 'Losses: CE' in stripped and 'Triplet' in stripped and 'Center' in stripped:
        if seen_losses_banner:
            print(f"  Removing duplicate line {i}: {stripped}")
            continue
        seen_losses_banner = True
    
    # Skip duplicate "v8→v8+" line
    if 'same Triplet' in stripped and '0.15' in stripped and '180' in stripped:
        if seen_losses_banner and i > 55:
            print(f"  Removing duplicate line {i}: {stripped}")
            continue
    
    new_lines.append(line)

# Fix elapsed bug: find the two lines and swap them
for i, line in enumerate(new_lines):
    if 'done in {elapsed/3600' in line and i + 1 < len(new_lines) and 'elapsed = time.time()' in new_lines[i + 1]:
        print(f"  Fixing elapsed bug: swapping lines {i} and {i+1}")
        new_lines[i], new_lines[i + 1] = new_lines[i + 1], new_lines[i]
        break

nb['cells'][14]['source'] = [l + '\n' for l in new_lines[:-1]] + [new_lines[-1]]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done! Cell 14 fixed.")
