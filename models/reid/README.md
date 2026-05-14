# ReID Checkpoints

Place model weights here for local path resolution. Do not commit `.pth` files.

| Local filename | Role | Source | Size / hash hint |
| --- | --- | --- | --- |
| `transreid_cityflowv2_384px_best.pth` | Primary CLIP TransReID vehicle stream | CityFlowV2 TransReID ViT-B/16 CLIP checkpoint used by `configs/datasets/cityflowv2.yaml` | Current config feeds 256x256 crops despite the legacy `384px` filename; verify before renaming or changing YAML |
| `transreid_cityflowv2_best.pth` | Primary CLIP TransReID fallback | Local fallback alias for the same CityFlowV2 TransReID lineage | Used only if the primary path is unavailable |
| `resnet101ibn_cityflowv2_384px_best.pth` | Optional secondary ResNet101-IBN-a stream | CityFlowV2-trained secondary model retained for disabled ensemble experiments | Disabled in the promoted 14e B1 config |
| `dinov2_large_cityflowv2.pth` | Tertiary DINOv2 ViT-L/14 stream | `gumfreddy/09s-dinov2-large-cityflowv2` (`vehicle_transreid_dinov2_large_cityflowv2_final.pth`) | Enabled tertiary stream for 14e B1 fusion |
| `pca_transform_ensemble.pkl` | Primary PCA whitening transform | Fitted for the primary vehicle embedding stream | Referenced by `stage2.pca.pca_model_path` |
| `pca_transform_secondary.pkl` | Secondary PCA whitening transform | Fitted for the optional secondary stream | Referenced by `stage2.pca.secondary_pca_model_path` |
| `pca_transform_tertiary.pkl` | Tertiary PCA whitening transform | Fitted for the DINOv2 tertiary stream | Referenced by `stage2.pca.tertiary_pca_model_path` |

`configs/datasets/cityflowv2.yaml` references local paths so Kaggle-only `/kaggle/input/...` paths are not required at runtime.