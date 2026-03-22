"""v54: Enable SOTA features — camera bias, zone model, hierarchical association."""
import json
from pathlib import Path

path_10c = Path(r"e:\dev\src\gp\notebooks\kaggle\10c_stages45\mtmc-10c-stages-4-5-association-eval.ipynb")

with open(path_10c, "r", encoding="utf-8") as f:
    nb = json.load(f)

# === Edit cell 9: Add SOTA tuning params ===
cell9 = nb["cells"][9]
src9 = cell9["source"]

# Find the line after FUSION_WEIGHT (line 22) to insert SOTA params
new_src9 = []
for li, line in enumerate(src9):
    new_src9.append(line)
    if li == 22:  # after FUSION_WEIGHT
        new_src9.append("\n")
        new_src9.append("# v54: SOTA techniques (AIC21 1st place)\n")
        new_src9.append("# Camera bias: learn per-camera-pair similarity offsets from clusters\n")
        new_src9.append("CAMERA_BIAS       = True\n")
        new_src9.append("CAMERA_BIAS_ITERS = 2     # iterative: cluster → learn bias → re-cluster\n")
        new_src9.append("# Zone model: penalize impossible transitions, boost valid ones\n")
        new_src9.append("ZONE_MODEL        = True\n")
        new_src9.append("ZONE_BONUS        = 0.06  # boost valid zone transitions\n")
        new_src9.append("ZONE_PENALTY      = 0.04  # penalize invalid transitions\n")
        new_src9.append("# Hierarchical: multi-pass centroid expansion to recover orphans\n")
        new_src9.append("HIERARCHICAL      = True\n")
        new_src9.append("HIER_CENTROID_TH  = 0.35  # orphan → cluster threshold\n")
        new_src9.append("HIER_MERGE_TH     = 0.35  # cluster ↔ cluster merge threshold\n")
        new_src9.append("HIER_ORPHAN_TH    = 0.30  # orphan ↔ orphan final threshold\n")

# Update print statements
new_src9_final = []
for line in new_src9:
    new_src9_final.append(line)
    if "fusion_weight={FUSION_WEIGHT}" in line:
        new_src9_final.append('print(f"  camera_bias={CAMERA_BIAS}  zone_model={ZONE_MODEL}  hierarchical={HIERARCHICAL}")\n')
        new_src9_final.append('print(f"  zone_bonus={ZONE_BONUS}  zone_penalty={ZONE_PENALTY}")\n')
        new_src9_final.append('print(f"  hier: centroid={HIER_CENTROID_TH}  merge={HIER_MERGE_TH}  orphan={HIER_ORPHAN_TH}")\n')

cell9["source"] = new_src9_final

# === Edit cell 11: Add SOTA overrides to cmd ===
cell11 = nb["cells"][11]
src11 = cell11["source"]

new_src11 = []
for li, line in enumerate(src11):
    # Insert BEFORE the intra_camera_merge lines
    # Find the intra_camera_merge.enabled line
    if "intra_camera_merge.enabled" in line:
        # Add camera bias overrides
        new_src11.append("    # v54: SOTA — camera distance bias\n")
        new_src11.append('    "--override", f"stage4.association.camera_bias.enabled={str(CAMERA_BIAS).lower()}",\n')
        new_src11.append('    "--override", f"stage4.association.camera_bias.iterations={CAMERA_BIAS_ITERS}",\n')
        # Add zone model overrides
        new_src11.append("    # v54: SOTA — zone-based transition scoring\n")
        new_src11.append('    "--override", f"stage4.association.zone_model.enabled={str(ZONE_MODEL).lower()}",\n')
        new_src11.append('    "--override", "stage4.association.zone_model.zone_data_path=configs/datasets/cityflowv2_zones.json",\n')
        new_src11.append('    "--override", f"stage4.association.zone_model.bonus={ZONE_BONUS}",\n')
        new_src11.append('    "--override", f"stage4.association.zone_model.penalty={ZONE_PENALTY}",\n')
        # Add hierarchical overrides
        new_src11.append("    # v54: SOTA — hierarchical centroid expansion\n")
        new_src11.append('    "--override", f"stage4.association.hierarchical.enabled={str(HIERARCHICAL).lower()}",\n')
        new_src11.append('    "--override", f"stage4.association.hierarchical.centroid_threshold={HIER_CENTROID_TH}",\n')
        new_src11.append('    "--override", f"stage4.association.hierarchical.merge_threshold={HIER_MERGE_TH}",\n')
        new_src11.append('    "--override", f"stage4.association.hierarchical.orphan_threshold={HIER_ORPHAN_TH}",\n')
        new_src11.append('    "--override", "stage4.association.hierarchical.max_merge_size=12",\n')
    new_src11.append(line)

cell11["source"] = new_src11

with open(path_10c, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
    f.write("\n")

# Verify
with open(path_10c, "r", encoding="utf-8") as f:
    nb2 = json.load(f)

c9_text = "".join(nb2["cells"][9]["source"])
c11_text = "".join(nb2["cells"][11]["source"])
assert "CAMERA_BIAS" in c9_text, "CAMERA_BIAS not in cell 9"
assert "ZONE_MODEL" in c9_text, "ZONE_MODEL not in cell 9"
assert "HIERARCHICAL" in c9_text, "HIERARCHICAL not in cell 9"
assert "camera_bias.enabled" in c11_text, "camera_bias.enabled not in cell 11"
assert "zone_model.enabled" in c11_text, "zone_model.enabled not in cell 11"
assert "hierarchical.enabled" in c11_text, "hierarchical.enabled not in cell 11"
print("All 3 SOTA features added and verified!")
