# Ensemble Experiment Spec — Fix 10c Fusion Bug + 256px Revert

## Problem
1. 10c notebook has `secondary_embeddings.weight=0.0` HARDCODED in 3 places, overriding the Python variable `FUSION_WEIGHT`
2. 10a notebook v31 was set to 384px (which regressed IDF1). Needs reverting to 256px.
3. The 0.784 IDF1 baseline was achieved with ZERO secondary model contribution. Fixing the fusion bug + using VeRi-776 pretrained ResNet could yield significant gains.

## Changes Required

### 10a Notebook
- Remove/revert `vehicle.weights_path` override that pointed to 384px model
- Remove/revert `vehicle.input_size=[384,384]` override
- Keep primary at 256px ViT: `transreid_cityflowv2_best.pth`
- Update secondary (vehicle2) weights_path to use `resnet101ibn_cityflowv2_veri776_best.pth` (when available from 09d)
- ESSENTIAL weights list: both 256px ViT and VeRi-776 ResNet models

### 10c Notebook
- **CRITICAL BUG FIX**: Find all 3 places where `secondary_embeddings.weight=0.0` is hardcoded
- Replace with `{FUSION_WEIGHT}` or the Python variable equivalent
- This enables the fusion weight sweep to actually work

## Fusion Weight Sweep Plan
| Weight | Expected behavior |
|--------|-------------------|
| 0.0 | Pure 256px ViT — replicates v80 baseline |
| 0.1 | 90% ViT + 10% ResNet |
| 0.2 | 80/20 blend |
| 0.3 | 70/30 blend — most likely sweet spot |
| 0.4 | 60/40 blend |
| 0.5 | 50/50 blend |

## Expected Impact
- If 09d achieves ~65% mAP: +1.0–1.5pp at optimal weight
- If 09d achieves ~72% mAP: +1.5–2.5pp at optimal weight