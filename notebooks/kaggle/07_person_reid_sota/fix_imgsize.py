"""Fix img_size in TransReID model cell — pass (256, 128) to timm.create_model."""
import json

NB_PATH = r"e:\dev\src\gp\notebooks\kaggle\07_person_reid_sota\07_person_reid_sota.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 14 (0-indexed) is the TransReID model cell
cell = nb["cells"][14]
src = "".join(cell["source"])

# Fix 1: Add img_size param to __init__
old1 = '                 sie_camera=True, jpm=True):'
new1 = '                 sie_camera=True, jpm=True, img_size=(256, 128)):'
assert old1 in src, f"Could not find __init__ signature"
src = src.replace(old1, new1, 1)

# Fix 2: Pass img_size to timm.create_model
old2 = '        self.vit = timm.create_model(vit_model, pretrained=pretrained, num_classes=0)'
new2 = '''        # Pass img_size so timm creates correct patch_embed and interpolates
        # pos_embed for 256x128 person images (16x8 grid instead of 14x14)
        self.vit = timm.create_model(vit_model, pretrained=pretrained,
                                     num_classes=0, img_size=img_size)'''
assert old2 in src, f"Could not find timm.create_model call"
src = src.replace(old2, new2, 1)

# Fix 3: Add grid_size print after pretrained print
old3 = '''        print(f"  Pretrained: {cfg.get('hf_hub_id', cfg.get('url', 'unknown'))[:80]}")

        # Detect architecture features'''
new3 = '''        print(f"  Pretrained: {cfg.get('hf_hub_id', cfg.get('url', 'unknown'))[:80]}")
        grid = self.vit.patch_embed.grid_size
        print(f"  Patch grid: {grid[0]}x{grid[1]} = {grid[0]*grid[1]} patches (img_size={img_size})")

        # Detect architecture features'''
assert old3 in src, "Could not find pretrained print"
src = src.replace(old3, new3, 1)

# Fix 4: Pass img_size=(H, W) at model instantiation
old4 = '''model = TransReID(
    num_classes=num_classes, num_cameras=num_cameras,
    embed_dim=768, vit_model=VIT_MODEL, sie_camera=True, jpm=True,
).to(DEVICE)'''
new4 = '''model = TransReID(
    num_classes=num_classes, num_cameras=num_cameras,
    embed_dim=768, vit_model=VIT_MODEL, sie_camera=True, jpm=True,
    img_size=(H, W),
).to(DEVICE)'''
assert old4 in src, "Could not find model instantiation"
src = src.replace(old4, new4, 1)

# Write back as line-based source
lines = src.split('\n')
cell["source"] = [l + '\n' for l in lines[:-1]] + [lines[-1]]

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Fixed! Verifying...")
# Verify
with open(NB_PATH, "r", encoding="utf-8") as f:
    nb2 = json.load(f)
src2 = "".join(nb2["cells"][14]["source"])
assert "img_size=(256, 128)" in src2, "img_size param not found"
assert "img_size=img_size" in src2, "img_size not passed to timm"
assert "img_size=(H, W)" in src2, "img_size not passed at instantiation"
print("All 3 img_size changes verified in notebook.")
