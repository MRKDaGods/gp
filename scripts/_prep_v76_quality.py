"""v76: Feature quality improvements in 10a.

Changes:
  - quality_temperature: 3.0 -> 5.0 (sharper crop weighting → more emphasis on best crops)
  - laplacian_min_var: 30.0 -> 50.0 (stricter blur filter → fewer blurry crops)

These two changes are synergistic:
  - Fewer bad crops (stricter blur filter)
  - More weight on best remaining crops (sharper temperature)

No detection/tracking changes (low risk).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_10A = ROOT / "notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb"


def update_10a():
    nb = json.load(open(NB_10A, encoding="utf-8"))

    # Find the pipeline run cell (has "run_pipeline.py" and "--stages", "0,1,2")
    run_cell_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if "run_pipeline.py" in src and '"0,1,2"' in src:
            run_cell_idx = i
            break

    if run_cell_idx is None:
        print("[ERROR] Could not find 10a run cell")
        return

    cell = nb["cells"][run_cell_idx]
    src = cell["source"]

    # Find the insertion point — after the last --override line and before the closing ]
    new_src = []
    inserted = False
    for line in src:
        # Insert our new overrides just before "stage2.power_norm.alpha=0.5" line
        # or at the end of overrides
        if '"stage2.power_norm.alpha=0.5"' in line and not inserted:
            new_src.append(line)
            new_src.append('    # v76: feature quality improvements\n')
            new_src.append('    "--override", "stage2.reid.quality_temperature=5.0",\n')
            new_src.append('    "--override", "stage2.crop.laplacian_min_var=50.0",\n')
            inserted = True
            continue
        new_src.append(line)

    if not inserted:
        print("[WARN] Could not find insertion point, appending before ]")
        # Find the ] closing line and insert before it
        new_src2 = []
        for line in src:
            if line.strip() == "]":
                new_src2.append('    # v76: feature quality improvements\n')
                new_src2.append('    "--override", "stage2.reid.quality_temperature=5.0",\n')
                new_src2.append('    "--override", "stage2.crop.laplacian_min_var=50.0",\n')
            new_src2.append(line)
        new_src = new_src2

    cell["source"] = new_src

    with open(NB_10A, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print("[OK] 10a updated with v76 overrides")

    # Print the run cell for verification
    print("\n--- Run cell (full) ---")
    for i, line in enumerate(new_src):
        print(f"  {i:2d}: {line}", end="")


if __name__ == "__main__":
    update_10a()
