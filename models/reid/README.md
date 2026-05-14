# ReID Checkpoints

Place model weights here for local path resolution. Do not commit `.pth` files.

| Local filename | Role | Source | Size / hash hint |
| --- | --- | --- | --- |
| `transreid_vitb16_clip_cityflowv2.pth` | Primary CLIP TransReID vehicle stream | `mrkdagods/mtmc-weights` (09v v17 lineage) | ~347 MB; existing local aliases include `transreid_cityflowv2_best.pth` / `transreid_cityflowv2_384px_best.pth` |
| `dinov2_large_cityflowv2.pth` | Tertiary DINOv2 ViT-L/14 stream | `gumfreddy/09s-dinov2-large-cityflowv2` (`vehicle_transreid_dinov2_large_cityflowv2_final.pth`) | ~1.2 GB expected; verify SHA256 after download |

`configs/datasets/cityflowv2.yaml` references local paths so Kaggle-only `/kaggle/input/...` paths are not required at runtime.