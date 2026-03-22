"""v51: Revert 384px + CamTTA, add multi-scale TTA.

Changes to 10a:
  - Remove stage2.reid.vehicle.input_size=[384,384]  (384px hurt -1.3% IDF1)
  - Remove stage2.camera_tta.enabled=true  (CamTTA hurt with small batches)
  - Add stage2.reid.multiscale_sizes=[[224,224],[288,288]]  (multi-scale TTA)

10c unchanged (SCAN_ENABLED=False, intra_merge=True already).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NB_10A = ROOT / "notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb"

def fix_10a():
    nb = json.load(open(NB_10A, encoding="utf-8"))
    cell = nb["cells"][16]  # pipeline execution cell
    src = cell["source"]

    new_src = []
    for line in src:
        # Remove v50 384px override
        if 'stage2.reid.vehicle.input_size=[384,384]' in line:
            continue
        # Remove v50 CamTTA override
        if 'stage2.camera_tta.enabled=true' in line:
            continue
        # Remove v50 comment
        if '# v50: 384px resolution + camera TTA' in line:
            # Replace with v51 comment + multiscale TTA override
            new_src.append('    # v51: multi-scale TTA (average features at 224px and 288px)\n')
            new_src.append('    "--override", "stage2.reid.multiscale_sizes=[[224,224],[288,288]]",\n')
            continue
        new_src.append(line)

    cell["source"] = new_src

    with open(NB_10A, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"[OK] 10a: replaced 384px+CamTTA with multiscale TTA")

    # Print the new cell for verification
    print("\n--- Cell 16 source ---")
    for i, line in enumerate(new_src):
        print(f"  {i}: {line}", end="")


if __name__ == "__main__":
    fix_10a()
    print("\n[DONE] v51 notebook changes applied")
