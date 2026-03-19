"""Show cell outputs and errors from pulled notebook."""
import json

path = "data/outputs/09b_pull_v2/09b-vehicle-reid-384px-fine-tune.ipynb"
nb = json.load(open(path))

cells = nb["cells"]
print(f"Cells: {len(cells)}")

for i, c in enumerate(cells):
    outs = c.get("outputs", [])
    if not outs:
        continue
    print(f"\n=== Cell {i} [{c['cell_type']}] ===")
    for o in outs:
        if o.get("output_type") in ("stream", "error"):
            text = o.get("text", []) or o.get("traceback", [])
            joined = "".join(text)
            # Show last 1000 chars of each output
            if len(joined) > 1000:
                print("...[truncated]...")
                print(joined[-1000:])
            else:
                print(joined)
