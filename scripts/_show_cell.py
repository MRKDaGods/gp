"""Show cell 4 (crop extraction) from 09b source."""
import json

nb = json.load(open("data/09b_v2/09b-vehicle-reid-384px-fine-tune.ipynb",
                    encoding="utf-8", errors="replace"))
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
# Cell 4 = crop extraction (5th code cell)
if len(code_cells) > 4:
    print("=== Code Cell 4 (crop extraction) ===")
    print("".join(code_cells[4]["source"]))
