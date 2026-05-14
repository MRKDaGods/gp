# 14m-extract — OSNet-IBN-x1.0 CityFlowV2 Stage-2 Tracklet Feature Extraction

**Date**: 2026-05-08
**Parent spec**: docs/subagent-specs/14m-osnet-ibn-train.md
**Status**: READY FOR IMPLEMENTATION (gated on 14m training PASS)
**Goal**: Extract a 512-d quinary OSNet-IBN tracklet embedding stream over the same 929 14h v3 tracklets so the 14n 5-way score-fusion sweep has its fifth stream.

## Pre-condition (eligibility gate)
- Do NOT push this kernel until the 14m training kernel `yahiaakhalafallah/14m-osnet-ibn-cityflowv2-train` has emitted the final-eval log line `OSNet-IBN-x1.0 CityFlowV2 mAP=XX.XX% R1=XX.XX% gate=PASS`.
- `final_metrics.json` must show CityFlowV2 mAP >=75% AND R1 >=90%.
- If `gate=FAIL`, abort 14m-extract and 14n entirely; log the dead end in `docs/findings.md` per the parent spec.

## Kernel Identity
- **Slug**: `<active-account>/14m-extract-osnet-features`
- The Coder MUST verify the active Kaggle CLI account before push. `yahiaakhalafallah` is the current account per the running 14m kernel; keep alignment unless the CLI account proves otherwise.
- `title`: `14m Extract OSNet-IBN Features`
- `code_file`: `14m_extract_osnet_features.ipynb`
- `language`: `python`
- `kernel_type`: `notebook`
- `is_private`: `true`
- `enable_gpu`: `true`
- `enable_internet`: `true`
- `enable_tpu`: `false`
- `machine_shape`: `NvidiaTeslaP100` (matches the 14j R50-IBN extraction precedent)
- `dataset_sources`: `[
  "thanhnguyenle/data-aicity-2023-track-2"
]`
- `competition_sources`: `[]`
- `kernel_sources`:
  - `<active-account>/14m-osnet-ibn-cityflowv2-train` (provides `osnet_ibn_cityflowv2_v1_final.pth` under `/kaggle/input/notebooks/<owner>/14m-osnet-ibn-cityflowv2-train/`)
  - `yahiaakhalafallah/14h-robust-tracklet-pooling` (provides `checkpoint.tar.gz` with the exact 929-tracklet Stage-1 output)
  - `yahiaakhalafallah/mtmc-10a-stages-0-2` (fallback for `checkpoint.tar.gz` if the 14h mount is missing)
- `model_sources`: `[]`

## Workspace Files
- Notebook: [notebooks/kaggle/14m_extract_osnet_features/14m_extract_osnet_features.ipynb](notebooks/kaggle/14m_extract_osnet_features/14m_extract_osnet_features.ipynb)
- Metadata: [notebooks/kaggle/14m_extract_osnet_features/kernel-metadata.json](notebooks/kaggle/14m_extract_osnet_features/kernel-metadata.json)
- Builder script (optional, mirroring `_build_14j_4way_fusion_sweep.py` style): `_build_14m_extract_osnet_features.py`

## Output Contract
Write Stage-2 outputs under `/kaggle/working/outputs/14m_extract_v1/stage2/`:

- `embeddings_quinary.npy`: shape `(929, 512)`, dtype `float32`, L2-unit-norm rows except zero-filled rows at the dropped indices `[280, 286, 481]`.
- `embeddings_secondary.npy`: exact byte-identical alias of `embeddings_quinary.npy` to satisfy the score-fusion code path, matching the 14j v4 alias trick.
- `embedding_index.json`: byte-identical to `embedding_index.json` produced by 14h v3 / 14j v4, with 929 rows and identical `track_id` / `camera_id` / `class_id` ordering. Compare against the 14h source under `<14h_extract>/stage2/embedding_index.json` and assert equality at extraction time.
- `dropped_indices.json`: `{"indices": [280, 286, 481], "count": 3, "fill_value": 0.0}` or the indices the OSNet checkpoint actually drops. The hard assertion is that the dropped set must EQUAL the 14j v4 set; if it differs, raise and halt.
- Root summary: `/kaggle/working/14m_extract_summary.json` with `experiment`, `kernel`, `run_name`, `source_run_name`, `checkpoint_path`, `shape`, `dtype`, `per_camera`, `dropped`, `norm_min`, `norm_max`, `norm_mean`, `has_nan`, `has_inf`, `aggregation`, `samples_per_tracklet`, `input_size`, `elapsed_minutes`, `mean`, and `std`. Mirror the 14j summary schema, swapping `quaternary` for `quinary`.

## Model Loading
- Load `osnet_ibn_cityflowv2_v1_final.pth` from `/kaggle/input/notebooks/<owner>/14m-osnet-ibn-cityflowv2-train/osnet_ibn_cityflowv2_v1_final.pth` using the same `find_input_dir` discovery idiom as 14j v4, because Kaggle mount layout varies.
- Read `model_name`, `feat_dim`, `image_size`, `mean`, and `std` from checkpoint metadata.
- Assert `feat_dim==512`, `model_name=="osnet_ibn_x1_0"`, and `image_size==[256,256]`.
- Construct ReID via `torchreid.models.build_model(name="osnet_ibn_x1_0", num_classes=666, loss="softmax", pretrained=False)` and then `load_state_dict(state_dict, strict=False)`; BNNeck features are exposed via the `src.training.model.ReIDModelBoT` reconstruction described in the parent spec Output Contract.
- Wrap the model in the existing `src.stage2_features.reid_model.ReIDModel` interface so `get_tracklet_embedding_from_scored_crops(scored_crops, quality_temperature=3.0)` is callable.
- If `ReIDModel` does not natively support `osnet_ibn_x1_0`, the Coder must add a thin adapter. Any code change to `src/stage2_features/` must be a separate PR-style workspace edit and re-pushed to Kaggle via `git pull` in the kernel; do NOT inline-monkey-patch.

## Crop & Aggregation Contract
This must match 14j v4 byte-for-byte, otherwise the W0 drift gate in 14n will fail.

- Use `CropExtractor(samples_per_tracklet=48, min_crop_size=32, min_quality=0.05)` with identical args to 14j v4.
- Feed per-frame crops to `r50_reid.get_tracklet_embedding_from_scored_crops(scored_crops, quality_temperature=3.0)`; in the OSNet kernel this is the OSNet-wrapped ReID object but the aggregation call shape must remain the same.
- Aggregation is the softmax-quality-weighted multi-query mean with `temperature=3.0`.
- No TTA at extraction time. TTA was already baked into the 14h v3 detection-time crops via the 14c v2 lineage; adding HFlip or scale crops at the OSNet stage would double-apply TTA and decorrelate dropped-index alignment from the existing primary, tertiary, and quaternary streams.
- Preserve the 14h v3 tracklet ordering by iterating `sorted(tracklets_by_camera)` and, within each camera, iterating `tracklets` in the order returned by `load_tracklets_by_camera`. Do not sort by `track_id` and do not shuffle.
- On empty `extract_crops` results or `embedding is None`, append `np.zeros(512, dtype=np.float32)` and record the index, consistent with 14j v4 dropped-index handling.

## L2-renormalization & Zero-fill Idiom
Literal copy of 14j v4 cell `#VSC-244f0df8`, adapted to 512-d:

```python
norms = np.linalg.norm(quinary, axis=1, keepdims=True)
nonzero_mask = norms[:, 0] > 1e-8
quinary[nonzero_mask] = quinary[nonzero_mask] / norms[nonzero_mask]
quinary[~nonzero_mask] = 0.0
```

After renormalization, zero rows must be exactly zero and non-zero rows must be unit-norm to within 1e-4. Persist both `embeddings_quinary.npy` and `embeddings_secondary.npy` as byte-identical arrays.

## Drift-gate Contract for downstream 14n
- The Stage-1 source must be the 14h v3 `checkpoint.tar.gz`.
- Assert `total_tracklets == 929` and the expected 6-camera list: `S01_c001`, `S01_c002`, `S01_c003`, `S02_c006`, `S02_c007`, `S02_c008`.
- The dropped-indices set must equal `[280, 286, 481]`. If different, raise `RuntimeError` and halt; do NOT write outputs.
- The output `embedding_index.json` must compare-equal with Python `==` to the 14h v3 / 14j v4 index. Load both and assert.
- Save a small validation cell at the bottom that prints `embedding_matrix.shape`, `len(index_map)`, `norm_min`, `norm_max`, `dropped_indices`, and a four-line ASCII recap.

## ETA
- ~30-50 min on P100. OSNet is ~2.2M params versus R50-IBN's ~25M, so per-image inference is ~5x faster, but 48 crops x 929 tracklets remains the same upper bound.

## Pre-flight Checks (Coder MUST run BEFORE pushing)
1. **CPU shape smoke** in `.venv`:
   ```python
   from src.stage2_features.reid_model import ReIDModel
   # Loads checkpoint metadata via the same discovery code; asserts feat_dim, model_name, image_size, mean, std
   ```
   If the existing `ReIDModel` factory cannot handle `osnet_ibn_x1_0`, fail the smoke locally and add the adapter before pushing.
2. **AST validation** of the on-disk notebook:
   ```bash
   python -c "import json,ast; nb=json.load(open('notebooks/kaggle/14m_extract_osnet_features/14m_extract_osnet_features.ipynb')); [ast.parse(''.join(c['source'])) for c in nb['cells'] if c['cell_type']=='code']"
   ```
3. **Name simulation**: dry-run the cell ordering against a fresh Python process using `exec` of the import-only cells; ensure no `NameError` before pushing.
4. **Static contracts**: assert these literal strings are present in the notebook source: `"embeddings_quinary.npy"`, `"embeddings_secondary.npy"`, `"feat_dim==512"`, `"osnet_ibn_x1_0"`, `"samples_per_tracklet=48"`, `"quality_temperature=3.0"`, `"929"`, and `"[280, 286, 481]"`.
5. **Active account check**: `kaggle config view | grep username` must match the slug owner.
6. **GPU slot check**: require 0 active GPU runs on the account before push. The 14m training kernel must be COMPLETE, not still running.
7. **Push-once rule**: validate metadata locally, push exactly once, then poll `kaggle kernels status <slug>` every ~60s until `complete` or `error`. If the push log shows `The following are not valid dataset sources` or any other input warning, immediately cancel via `kaggle kernels cancel <slug>` or status-poll until cancelled if the CLI lacks the subcommand. Do NOT iterate-and-push.

## Hard Constraints
- Do NOT alter the 14h v3 tracklet ordering.
- Do NOT change aggregation; it must remain softmax-quality mean with `temperature=3.0` and 48 samples.
- Do NOT introduce TTA, HFlip, scale jitter, or multi-crop at the OSNet extraction stage.
- Do NOT enable AFLink, CSLS, reranking, FAC, or feature concatenation in any cell; this kernel writes Stage-2 features only.
- Do NOT use 384px crops; the OSNet checkpoint is 256x256 only.
- Do NOT change the 0.77936 headline regardless of single-camera metrics.

## Handoff Note
Once `14m_extract_summary.json` shows an exact `embedding_index` match and a 929-row L2-unit embedding matrix, hand off to 14n. The next kernel `<active-account>/14n-5way-fusion-sweep` consumes the quinary stream as a fifth score-fusion path.