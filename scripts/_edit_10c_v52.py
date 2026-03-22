"""Edit 10c notebook to enable score-level fusion (v52)."""
import json
from pathlib import Path

path_10c = Path(r"e:\dev\src\gp\notebooks\kaggle\10c_stages45\mtmc-10c-stages-4-5-association-eval.ipynb")

with open(path_10c, "r", encoding="utf-8") as f:
    nb = json.load(f)

# === Edit cell 9: Add FUSION_WEIGHT to tuning params ===
cell9 = nb["cells"][9]
src9 = cell9["source"]

new_src9 = []
for li, line in enumerate(src9):
    new_src9.append(line)
    if li == 19:  # after INTRA_MERGE_GAP line
        new_src9.append("\n")
        new_src9.append("# v52: Score-level fusion with OSNet secondary embeddings\n")
        new_src9.append("FUSION_WEIGHT     = 0.3   # 30% secondary (OSNet 512D), 70% primary (TransReID)\n")

# Update print to include fusion weight
new_src9_final = []
for line in new_src9:
    new_src9_final.append(line)
    if "intra_gap={INTRA_MERGE_GAP}" in line:
        new_src9_final.append('print(f"  fusion_weight={FUSION_WEIGHT}")\n')

cell9["source"] = new_src9_final

# === Edit cell 11: Add secondary_embeddings overrides to cmd ===
cell11 = nb["cells"][11]
src11 = cell11["source"]

new_src11 = []
for li, line in enumerate(src11):
    # Insert BEFORE the intra_camera_merge lines (index 24)
    if li == 24:
        new_src11.append("    # v52: score-level fusion with OSNet secondary embeddings\n")
        new_src11.append('    "--override", f"stage4.association.secondary_embeddings.path={RUN_DIR}/stage2/embeddings_secondary.npy",\n')
        new_src11.append('    "--override", f"stage4.association.secondary_embeddings.weight={FUSION_WEIGHT}",\n')
    new_src11.append(line)

cell11["source"] = new_src11

with open(path_10c, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
    f.write("\n")

print("10c saved with fusion overrides")

# Verify
with open(path_10c, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

c9_text = "".join(nb2["cells"][9]["source"])
c11_text = "".join(nb2["cells"][11]["source"])
assert "FUSION_WEIGHT" in c9_text, "FUSION_WEIGHT not found in cell 9"
assert "secondary_embeddings.path" in c11_text, "secondary_embeddings.path not found in cell 11"
assert "secondary_embeddings.weight" in c11_text, "secondary_embeddings.weight not found in cell 11"
print("Verification passed!")
