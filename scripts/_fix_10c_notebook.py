"""Fix 10c notebook: add bridge_prune_margin=0.0 and camera_bias.enabled=false overrides."""
import json

path = "notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb"
nb = json.load(open(path, encoding="utf-8"))

# Fix cell 11 (main pipeline cmd)
cell11_src = nb["cells"][11]["source"]
new_src = []
for line in cell11_src:
    new_src.append(line)
    if "fic.enabled=true" in line:
        new_src.append("    # v49: override harmful cityflowv2.yaml defaults (bridge_prune +2.5pp, camera_bias)\n")
        new_src.append('    "--override", "stage4.association.graph.bridge_prune_margin=0.0",\n')
        new_src.append('    "--override", "stage4.association.camera_bias.enabled=false",\n')
nb["cells"][11]["source"] = new_src
print(f"Cell 11: added 3 lines after fic.enabled=true ({len(cell11_src)} -> {len(new_src)} lines)")

# Fix cell 15 (Phase 3 scan cmd) - it already has bridge_prune in one scan block
# but check if both dataset-config scan blocks need fixing
cell15_src = nb["cells"][15]["source"]
has_bridge_prune = any("bridge_prune_margin" in line for line in cell15_src)
if not has_bridge_prune:
    new_src15 = []
    for line in cell15_src:
        new_src15.append(line)
        if "fic.enabled=true" in line:
            new_src15.append('            "--override", "stage4.association.graph.bridge_prune_margin=0.0",\n')
            new_src15.append('            "--override", "stage4.association.camera_bias.enabled=false",\n')
    nb["cells"][15]["source"] = new_src15
    print(f"Cell 15: added overrides ({len(cell15_src)} -> {len(new_src15)} lines)")
else:
    print(f"Cell 15: bridge_prune_margin already present, checking camera_bias...")
    if not any("camera_bias" in line for line in cell15_src):
        new_src15 = []
        for line in cell15_src:
            new_src15.append(line)
            if "bridge_prune_margin" in line:
                new_src15.append('            "--override", "stage4.association.camera_bias.enabled=false",\n')
        nb["cells"][15]["source"] = new_src15
        print(f"  Added camera_bias.enabled=false after bridge_prune_margin")
    else:
        print(f"  Both already present, skipping")

# Verify
for ci in [11, 15]:
    src = "".join(nb["cells"][ci]["source"])
    bp = "bridge_prune" in src
    cb = "camera_bias" in src
    print(f"Cell {ci}: bridge_prune={bp}, camera_bias={cb}")

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print("Done - 10c notebook fixed!")
