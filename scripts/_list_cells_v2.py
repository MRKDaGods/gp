"""List cells in a notebook."""
import json, sys
nb = json.load(open(sys.argv[1]))
for i, c in enumerate(nb["cells"]):
    src = c["source"]
    first = (src[0][:80].strip() if src else "(empty)")
    print(f"Cell {i}: {c['cell_type']}, {len(src)} lines | {first}")
