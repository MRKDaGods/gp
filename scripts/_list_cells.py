"""List cells in a notebook with type and first-line preview."""
import json, sys

nb_path = sys.argv[1]
nb = json.load(open(nb_path, encoding="utf-8"))
for i, cell in enumerate(nb["cells"]):
    ct = cell["cell_type"]
    cid = cell.get("id", "?")
    first = "".join(cell["source"][:2]).replace("\n", " | ")[:100]
    print(f"Cell {i:2d}: {ct:8s} id={cid:20s} | {first}")
