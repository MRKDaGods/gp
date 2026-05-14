# Vehicle Pipeline - CityFlowV2 Reproduction

> Canonical headline: MTMC IDF1 0.7793596227569698 from 14e B1. This recipe is Kaggle-first; do not run Stage 0, 1, or 2 locally.

## Inputs

- Dataset: CityFlowV2 / AI City Challenge 2022 Track 1, mounted on Kaggle as `thanhnguyenle/data-aicity-2023-track-2` in the canonical kernels.
- Weights dataset: `yahiaakhalafallah/mtmc-weights` or `gumfreddy/mtmc-weights` for YOLO26m and primary TransReID.
- DINOv2 tertiary checkpoint: kernel output from `yahiaakhalafallah/09s-dinov2-large-cityflowv2`, expected filename `vehicle_transreid_dinov2_large_cityflowv2_final.pth`.
- Baseline Stage 0/1 source: kernel output from `yahiaakhalafallah/mtmc-10a-stages-0-2`.

## Canonical Kaggle Chain

1. Stage 0-2 baseline tracking/features: `notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb` (`yahiaakhalafallah/mtmc-10a-stages-0-2`).
2. Stage 2 TTA refresh: `notebooks/kaggle/14c_tta_stage2/14c_tta_stage2.ipynb` (`yahiaakhalafallah/14c-tta-stage2`). This reuses the 10a Stage 1 tracklets, extracts primary 4-view TTA features, extracts tertiary DINOv2 2-view TTA features, and verifies a control MTMC IDF1 of 0.770846.
3. CPU-only Stage 3-5 sweep: `notebooks/kaggle/14e_tta_sweep_expanded/14e_tta_sweep_expanded.ipynb` (`yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep`). This reuses 14c features and produces the promoted B1 result.

## Verified 14e B1 Settings

The downloaded `14e_summary.json` verifies:

```yaml
stage4.association.tertiary_embeddings.weight: 0.525
stage4.association.graph.similarity_threshold: 0.48
stage4.association.query_expansion.k: 2
stage4.association.fic.regularisation: 0.5
```

B1 result: `mtmc_idf1=0.7793596227569698`, `trackeval_idf1=0.7946139234279311`, `id_switches=154`.

The same settings are encoded in [configs/datasets/cityflowv2.yaml](../configs/datasets/cityflowv2.yaml).

## Push / Monitor Commands

Use the appropriate account token before pushing or polling:

```powershell
$env:KAGGLE_API_TOKEN = (Get-Content $HOME\.kaggle\yahiaakhalafallah_access_token -Raw).Trim()
kaggle kernels push -p notebooks/kaggle/14c_tta_stage2
kaggle kernels push -p notebooks/kaggle/14e_tta_sweep_expanded
kaggle kernels status yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep
```

Metadata-only checks used for this verification:

```powershell
kaggle datasets files yahiaakhalafallah/mtmc-weights
kaggle datasets files gumfreddy/mtmc-weights
kaggle kernels output yahiaakhalafallah/14c-tta-stage2 --file-pattern '^14c_summary\.json$' -p <tmp>
kaggle kernels output yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep --file-pattern '^14e_summary\.json$' -p <tmp>
```

No checkpoint download is required to reproduce the documentation trail. For full experiment history and dead ends, see [docs/findings.md](findings.md) and [docs/experiment-log.md](experiment-log.md).