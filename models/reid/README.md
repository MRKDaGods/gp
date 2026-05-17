# ReID Checkpoints

Place model weights here for local path resolution. Do not commit `.pth` files.

Preferred setup is automated from the repo root:

```bash
python scripts/download_assets.py --reid-only
python scripts/verify_assets.py
```

The manual source notes below are retained as a fallback when Kaggle access or a hosted artifact changes.

| Local filename | Role | Source | Size / hash hint |
| --- | --- | --- | --- |
| `transreid_cityflowv2_best.pth` | Canonical primary CLIP TransReID vehicle stream | `gumfreddy/mtmc-weights` or `yahiaakhalafallah/mtmc-weights`, member `reid/transreid_cityflowv2_best.pth` | 346,518,635 bytes; MD5 `bee2e2bc7e733d8eb3574abcc10ef5ed` |
| `resnet101ibn_cityflowv2_384px_best.pth` | Optional secondary ResNet101-IBN-a stream | CityFlowV2-trained secondary model retained for disabled ensemble experiments | Disabled in the promoted 14e B1 config |
| `vehicle_transreid_dinov2_large_cityflowv2_final.pth` | Tertiary DINOv2 ViT-L/14 stream | Producing kernel `yahiaakhalafallah/09s-dinov2-large-cityflowv2` | Enabled tertiary stream for 14e B1 fusion |
| `pca_transform_ensemble.pkl` | Primary PCA whitening transform | Fitted for the primary vehicle embedding stream | Referenced by `stage2.pca.pca_model_path` |
| `pca_transform_secondary.pkl` | Secondary PCA whitening transform | Fitted for the optional secondary stream | Referenced by `stage2.pca.secondary_pca_model_path` |
| `pca_transform_tertiary.pkl` | Tertiary PCA whitening transform | Fitted for the DINOv2 tertiary stream | Referenced by `stage2.pca.tertiary_pca_model_path` |
| `vehicle_transreid_vit_base_veri776.pth` | VeRi-776 TransReID 09v v17 single-camera reference | `mrkdagods/mtmc-weights`, member `reid/vehicle_transreid_vit_base_veri776.pth`; mirrored in other `mtmc-weights` datasets | 346,889,637 bytes; MD5 `1ddd5b60cf6071a7794be169b41f63e1` |
| `clipsenet_v6_veri776_best.pth` | CLIP-SENet v6 VeRi-776 single-camera reference | Kernel `yahiaakhalafallah/13-clip-senet-train`, output `checkpoints/best.pth` | 1,111,738,778 bytes; MD5 `a55f55f60d404c7df5cb62690d7213dc` |

`configs/datasets/cityflowv2.yaml` references local paths so Kaggle-only `/kaggle/input/...` paths are not required at runtime.

## 14t VeRi-776 Fusion Scope

`scripts/eval/eval_14t_fusion_veri776.py` evaluates a single-camera VeRi-776 score-level fusion of CLIP-SENet v6 and TransReID 09v v17. The expected verifier target is mAP 0.9330 and R1 0.9845 with CLIP-SENet weight 0.7, TransReID weight 0.3, AQE k=3, and rerank k1=80/k2=15/lambda=0.2.

This fusion is not a CityFlow MTMC feature stream or a replacement for `transreid_cityflowv2_best.pth`. The CityFlow port was tested separately in 14u and documented as a dead end in `docs/findings.md`; keep this model path limited to VeRi-776 single-camera evaluation and verification.