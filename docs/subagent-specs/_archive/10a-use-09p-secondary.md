# 10a -> 10b -> 10c Chain: Use 09p as Secondary ReID

## Goal
Deploy the newly trained 09p v5 checkpoint as the vehicle2 secondary ReID model in 10a, with a reliable source path and safe fallback behavior.

Checkpoint to deploy:
- 09p kernel slug: gumfreddy/09p-fastreid-r50-extended-cityflowv2
- Expected artifact: /kaggle/working/fastreid_r50_ibn_cityflowv2_extended_final.pth

## Findings from Current 10a

### 1) Where secondary source/path is configured
In notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb:
- Secondary source is currently hardcoded to 09n:
  - FASTREID_FINETUNED_SRC = /kaggle/input/09n-fastreid-r50-finetune-cityflowv2/fastreid_r50_ibn_cityflowv2_final.pth
- Destination used by stage2 is:
  - models/reid/fastreid_r50_ibn_cityflowv2.pth
- Current fallback is direct VeRi-776 download URL.

### 2) Kernel metadata wiring
In notebooks/kaggle/10a_stages012/kernel-metadata.json:
- Current kernel_sources includes 09n (plus 09l and 09o)
- Current dataset_sources does not include a 09p checkpoint dataset

### 3) Stage2 override already points to correct destination
In 10a run command overrides:
- stage2.reid.vehicle2.weights_path=models/reid/fastreid_r50_ibn_cityflowv2.pth
No path change is needed in stage2 override if we keep copying the selected checkpoint into that same destination.

## Safest ingestion strategy
Recommended: publish a small Kaggle dataset containing the 09p checkpoint and make 10a read from that dataset first.

Why this is safer than kernel output as primary:
- Avoids coupling 10a to kernel output mount behavior and output retention timing
- Allows explicit versioned artifact for reproducible reruns
- Lets 10a resolve both flat and nested Kaggle input mount variants

Keep kernel output mount as secondary fallback, not primary dependency.

## Exact 10a edits

### Edit A: notebooks/kaggle/10a_stages012/kernel-metadata.json
Add the 09p dataset artifact to dataset_sources. Keep existing sources.

Expected shape:
```json
{
  "dataset_sources": [
    "gumfreddy/mtmc-weights",
    "thanhnguyenle/data-aicity-2023-track-2",
    "gumfreddy/09p-r50-ibn-cityflowv2-extended-checkpoint"
  ],
  "kernel_sources": [
    "gumfreddy/09l-transreid-laion-2b-training",
    "gumfreddy/09n-fastreid-r50-finetune-cityflowv2",
    "gumfreddy/09o-eva02-vit-reid-cityflowv2"
  ]
}
```

Note: if the published dataset slug differs, replace only that one line.

### Edit B: replace 10a secondary checkpoint copy block
File:
- notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb

Replace the current FASTREID_FINETUNED_SRC block with this resolver-based block:
```python
import shutil
import urllib.request
from pathlib import Path

# Keep destination unchanged so existing stage2 override still works.
FASTREID_SECONDARY_PATH = PROJECT / "models" / "reid" / "fastreid_r50_ibn_cityflowv2.pth"

# Preferred: 09p dataset artifact mount.
# Fallbacks: alternate Kaggle mount forms, then 09p kernel output mount.
CKPT_ROOT_CANDIDATES = [
    Path("/kaggle/input/09p-r50-ibn-cityflowv2-extended-checkpoint"),
    Path("/kaggle/input/datasets/gumfreddy/09p-r50-ibn-cityflowv2-extended-checkpoint"),
    Path("/kaggle/input/09p-fastreid-r50-extended-cityflowv2"),
]

CKPT_FILE_CANDIDATES = [
    "fastreid_r50_ibn_cityflowv2_extended_final.pth",
    "fastreid_r50_ibn_cityflowv2_extended_epoch_200.pth",
    "checkpoints/fastreid_r50_ibn_cityflowv2_extended_final.pth",
    "checkpoints/fastreid_r50_ibn_cityflowv2_extended_epoch_200.pth",
]

FALLBACK_09N = Path(
    "/kaggle/input/09n-fastreid-r50-finetune-cityflowv2/fastreid_r50_ibn_cityflowv2_final.pth"
)
FASTREID_SECONDARY_URL = (
    "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth"
)


def resolve_secondary_ckpt() -> tuple[Path | None, str]:
    for root in CKPT_ROOT_CANDIDATES:
        for rel in CKPT_FILE_CANDIDATES:
            p = root / rel
            if p.exists():
                return p, f"09p:{p}"
    if FALLBACK_09N.exists():
        return FALLBACK_09N, f"09n:{FALLBACK_09N}"
    return None, "veri-url"


src_ckpt, src_tag = resolve_secondary_ckpt()
FASTREID_SECONDARY_PATH.parent.mkdir(parents=True, exist_ok=True)

if src_ckpt is not None:
    shutil.copy2(src_ckpt, FASTREID_SECONDARY_PATH)
    print(f"Copied secondary checkpoint ({src_tag}) -> {FASTREID_SECONDARY_PATH}")
else:
    urllib.request.urlretrieve(FASTREID_SECONDARY_URL, str(FASTREID_SECONDARY_PATH))
    print(f"Downloaded fallback VeRi checkpoint (veri-url) -> {FASTREID_SECONDARY_PATH}")
```

### Edit C: keep stage2 vehicle2 wiring unchanged
Do not change these functional values in 10a run overrides:
- stage2.reid.vehicle2.enabled=true
- stage2.reid.vehicle2.model_name=fastreid_sbs_r50_ibn
- stage2.reid.vehicle2.weights_path=models/reid/fastreid_r50_ibn_cityflowv2.pth
- stage2.reid.vehicle2.embedding_dim=2048
- stage2.reid.vehicle2.input_size=[256,256]
- stage2.reid.vehicle2.clip_normalization=false
- stage2.reid.vehicle2.save_separate=true

## Fallback behavior (required)
The resolver order must be:
1. 09p dataset artifact (flat mount)
2. 09p dataset artifact (nested mount)
3. 09p kernel output mount
4. 09n final checkpoint
5. VeRi-776 URL download

This ensures 10a runs even if 09p is temporarily unavailable.

## 10b and 10c impact
No metadata/code change needed in 10b or 10c for this switch.
- 10b still reads 10a checkpoint.tar.gz output
- 10c still reads 10b output

## Verification checklist
1. kernel-metadata.json contains the 09p dataset slug in dataset_sources.
2. 10a notebook contains CKPT_ROOT_CANDIDATES and CKPT_FILE_CANDIDATES.
3. 10a notebook still writes selected checkpoint to models/reid/fastreid_r50_ibn_cityflowv2.pth.
4. 10a run overrides still point vehicle2 weights_path to models/reid/fastreid_r50_ibn_cityflowv2.pth.
5. Fallback order includes 09n before VeRi URL.

## Optional publish step (outside 10a code)
Publish a minimal dataset containing only:
- fastreid_r50_ibn_cityflowv2_extended_final.pth
Recommended dataset slug:
- gumfreddy/09p-r50-ibn-cityflowv2-extended-checkpoint