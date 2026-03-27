# Person MTMC Pipeline — WILDTRACK Dataset Spec

## Summary
Run the full 7-stage MTMC pipeline on WILDTRACK with TransReID ViT-B/16 CLIP person model (Market-1501: mAP=81.77%, R1=92.87%). All data is available locally.

## CLI Command
```
python scripts/run_pipeline.py --config configs/default.yaml --dataset-config configs/datasets/wildtrack.yaml --stages 0,1,2,3,4,5
```

## Config Fix Required
wildtrack.yaml stage2.reid.person must be updated from ResNet50-IBN-a to TransReID ViT.

## Expected Metrics
- MODA: 65-80% (first run)
- IDF1: 50-70%
- Published SOTA: MODA 88-92% (uses different multi-view detection approach)