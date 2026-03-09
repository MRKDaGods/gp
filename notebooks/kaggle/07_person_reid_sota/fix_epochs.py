"""Fix EPOCHS from 120 to 140 in the notebook."""
import json

NB = "07_person_reid_sota.ipynb"
nb = json.load(open(NB, "r", encoding="utf-8"))

for ci in range(len(nb["cells"])):
    src = "".join(nb["cells"][ci]["source"])
    changed = False
    for old, new in [
        ("EPOCHS = 120", "EPOCHS = 140"),
        ("epochs=120", "epochs=140"),
        ("120 epochs", "140 epochs"),
        ("120ep", "140ep"),
        ('"120"', '"140"'),
    ]:
        if old in src:
            src = src.replace(old, new)
            changed = True
    if changed:
        lines = src.split("\n")
        nb["cells"][ci]["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        print(f"Fixed cell {ci}")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Done")
