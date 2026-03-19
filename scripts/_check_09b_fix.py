"""Check 09b v2 notebook cell 3 for the is_file fix."""
import json

nb = json.load(open("data/09b_v2/09b-vehicle-reid-384px-fine-tune.ipynb",
                    encoding="utf-8", errors="replace"))
cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if "shutil.copy2" in src:
        print(f"=== Code Cell {i} (has shutil.copy2) ===")
        # Show the relevant part
        lines = src.split("\n")
        for j, l in enumerate(lines):
            if "copy2" in l or "is_file" in l or "iterdir" in l:
                start = max(0, j-2)
                end = min(len(lines), j+3)
                print(f"  lines {start}-{end}:")
                for ll in lines[start:end]:
                    print(f"    {ll}")
                print()
