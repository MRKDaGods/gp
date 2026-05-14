# 14m — OSNet-IBN-x1.0 CityFlowV2 Training Spec

**Date**: 2026-05-08
**Parent spec**: docs/subagent-specs/post-14k-next.md (Candidate 1)
**Status**: RESOLVED FAIL — COMPLETED ON GUMFREDDY, DEAD END
**Goal**: Train an OSNet-IBN-x1.0 vehicle ReID model on CityFlowV2 to serve as a fifth, architecturally-diverse score-fusion stream on top of the current 4-way (CLIP TransReID primary, DINOv2 tertiary, R50-IBN quaternary) ensemble that plateaued at 0.78079 in 14k.

## Verdict Bands (inherited from 14l parent spec)
- **WIN**: ≥0.7920 MTMC IDF1 in 14n fusion → promote
- **MARGINAL**: ≥0.7820 and <0.7920 → document only
- **FAIL**: <0.7820 → close architecture-diverse-stream branch, escalate to GNN

## Eligibility Gate (single-camera, before any MTMC fusion)
- **Required**: CityFlowV2 mAP ≥75% AND R1 ≥90%
- **Preferred**: mAP ≥78% AND R1 ≥91%
- If single-cam metrics fail: do NOT extract Stage-2 features, do NOT run 14n fusion, log the failure and stop.

## Kernel Identity
- **Slug**: `<active-account>/14m-osnet-ibn-cityflowv2-train`
- The Coder MUST verify the currently active Kaggle CLI account before push (`kaggle config view`) and align the slug. The user's prompt suggested `yahiaakhalafallah/...`; copilot-instructions list `gumfreddy` as primary. Either account is acceptable as long as it has access to the inputs and a free GPU slot.
- `enable_gpu=true`, `enable_internet=true`, `is_private=true`
- `machine_shape="NvidiaTeslaT4"` (matches 09p precedent; OSNet is small enough)
- `dataset_sources=[]` (CityFlowV2 downloaded in-kernel; mirror 09p)

## Architecture
- **Backbone**: `torchreid.models.osnet_ibn_x1_0` with `pretrained=True` (ImageNet weights via torchreid's gdown)
- **Head**: `src.training.model.ReIDModelBoT` with:
  - `model_name="osnet_ibn_x1_0"`
  - `feat_dim=512` ← **CRITICAL: OSNet outputs 512-d, not 2048-d**
  - `num_classes=666` (CityFlowV2 train IDs; assert at runtime)
  - `last_stride=1` (no-op for OSNet; harmless)
  - `neck="bnneck"`
- **Forward path**: relies on `backbone.featuremaps(x)` which torchreid OSNet provides natively.

## Training Recipe (cloned from 09p with OSNet-appropriate adjustments)
| Hyperparameter | Value | Source |
|:--|:--|:--|
| Image size | 256×256 | 09p |
| Batch size | 128 (32 ids × 4 instances via PKSampler P=32, K=4) | 09p adjusted |
| Optimizer | AdamW; backbone lr ×0.1 | 09p |
| Base LR | 3.5e-4 | BoT default; 09p extended ran at 1.75e-4 from checkpoint |
| Weight decay | 5e-4 | 09p |
| Schedule | LinearLR warmup 5 epochs → CosineAnnealingLR | 09p |
| Epochs | **120** (from-scratch budget; not 200) | adjusted |
| Loss | CE label-smoothing (ε=0.1) + Triplet (margin=0.3, soft) + Center (weight=5e-4, dim=512) | 09p, dim adjusted |
| Center optimizer | SGD lr=0.5 (separate optimizer) | 09p |
| Augmentation | Resize → HFlip(0.5) → Pad(10)+RandomCrop → ToTensor → Normalize(IN) → RandomErasing(0.5) | `build_train_transforms` |
| AMP | bf16 if available else fp16 | 09p |
| Checkpoint cadence | epochs 30 / 60 / 90 / 120 + best-mAP + best-joint | adjusted from 09p |

**Forbidden** (per copilot-instructions dead-ends):
- ArcFace (DEAD END on ResNet101-IBN; do NOT add to OSNet either)
- CircleLoss
- 384px deployment crops
- CSLS

## Data Pipeline
1. Reuse 09p's CityFlowV2 GDrive download + crop extraction code (cells around line 444 `extract_crops_from_camera` and line 537 `build_prepared_splits` in 09p notebook). Cache under `/tmp/cityflowv2_raw_crops` and reuse across kernel restarts within the same Kaggle session.
2. Build `train/query/gallery` splits at `/tmp/cityflowv2_crops` matching `parse_cityflowv2` in `src/training/datasets.py`.
3. **Assert** `NUM_CLASSES==666` after re-labelling.
4. `MAX_CROPS_PER_ID_CAM=20` (matches 09p; bounds dataset size to ~40K imgs → fits 12h on T4).

## Output Contract (must satisfy for downstream Stage-2 extraction)
- **Final checkpoint**: `/kaggle/working/osnet_ibn_cityflowv2_v1_final.pth` containing:
  - `state_dict` (model weights, `module.` prefix stripped)
  - `num_classes`, `feat_dim=512`, `model_name="osnet_ibn_x1_0"`, `image_size=(256,256)`, `mean`, `std`
- **Per-cadence checkpoints**: `osnet_ibn_cityflowv2_v1_epoch_{030,060,090,120}.pth` with same schema
- **Best checkpoints**: `osnet_ibn_cityflowv2_v1_best_map.pth`, `osnet_ibn_cityflowv2_v1_best_joint.pth`
- **Metrics**: `/kaggle/working/14m_osnet_ibn/training_history.json` and `final_metrics.json`
- Loadable in Stage-2 the same way 09p's R50-IBN was consumed by 14j v4: a small wrapper that does `torchreid.models.build_model(name="osnet_ibn_x1_0", num_classes=666, loss="softmax", pretrained=False)` then `load_state_dict(state_dict, strict=False)` then takes the BN-feature output via `ReIDModelBoT` reconstruction.

## Eval Contract (in-notebook, runs after final epoch)
- Use `src.training.evaluate_reid.evaluate_reid` against the CityFlowV2 query/gallery splits built above.
- Report: mAP, R1, R5, R10 — both on raw BN features and after AQE(k=3)+rerank diagnostics (rerank is for diagnostic comparison only; deployment uses raw BN features).
- Print and JSON-dump a one-line gate summary: `OSNet-IBN-x1.0 CityFlowV2 mAP=XX.XX% R1=XX.XX% gate=PASS|FAIL`.
- Do NOT extract Stage-2 tracklet embeddings in this kernel — that is a separate 14m-extract kernel scheduled only if gate=PASS.

## Pre-flight Checks (Coder MUST run BEFORE pushing)
1. **Active account check**: `kaggle config view | grep username`. Confirm token is for the account in the slug.
2. **GPU slot check**: `kaggle kernels status <active-account>/<existing-kernel>` for any currently-running kernel; ensure ≤1 active GPU run on the account.
3. **CPU shape smoke** (local, in `.venv`):
   ```python
   from src.training.model import ReIDModelBoT
   import torch
   m = ReIDModelBoT(model_name="osnet_ibn_x1_0", num_classes=666,
                   feat_dim=512, last_stride=1, pretrained=False, neck="bnneck")
   m.eval()
   x = torch.randn(2, 3, 256, 256)
   y = m(x)
   assert y.shape == (2, 512), f"Got {y.shape}"
   m.train()
   cls, gf, bn = m(x)
   assert cls.shape == (2, 666) and gf.shape == (2, 512) and bn.shape == (2, 512)
   print("OK")
   ```
   If this fails, STOP — do not push the kernel. Fix model wiring first.
4. **In-kernel cell-0 ImageNet-weights preflight** (must be the first GPU cell):
   ```python
   import torchreid
   m = torchreid.models.build_model(name="osnet_ibn_x1_0", num_classes=1000,
                                    loss="softmax", pretrained=True)
   # If torchreid raised on download → cell errors out, kernel fails fast,
   # GPU slot is freed within minutes instead of after a 12h dead run.
   ```
   If this cell fails on Kaggle (gdown quota), the Coder must immediately `kaggle kernels cancel` per the safety rules and switch to candidate 3 (CLIP TransReID L/14) instead.

## Quota Budget (target)
- CityFlowV2 download + crop build: ≤90 min
- Training 120 epochs on T4: ~6–8h
- In-kernel single-cam eval: ~30 min
- Total target: **<10h**, leaves >2h buffer under Kaggle's 12h GPU limit.

## Hard Constraints
- Do NOT change the 0.77936 headline regardless of single-cam metrics.
- Do NOT begin a 5-way fusion sweep in this kernel — that is a separate 14n CPU kernel.
- Do NOT add ArcFace, CircleLoss, CSLS, or 384px crops.
- Do NOT pre-stage a custom Kaggle dataset for CityFlowV2; reuse 09p's in-kernel GDrive download.

## Fallback (if eligibility gate fails)
- If single-cam mAP <75% OR R1 <90% after 120 epochs:
  1. Log result in `docs/findings.md` and `docs/experiment-log.md` as a 14m FAIL.
  2. Do NOT promote, do NOT extract Stage-2 features, do NOT run 14n fusion.
  3. Per parent spec, jump to Candidate 3 (CLIP TransReID L/14) before EVA-02-L/14.

## Handoff Note
This spec is implementation-ready: every hyperparameter, file path, output schema, and gate is concrete. The Coder should be able to produce a kernel notebook by templating off 09p, swapping the model construction block, adjusting `feat_dim=512` and `CenterLoss(num_classes, 512)`, removing the resume-from-checkpoint logic, and adding the cell-0 preflight + final eval gate.

## Update Log

### Session of 2026-05-08 — v1 -> v2 -> v3

- **v1 pushed**: `yahiaakhalafallah/14m-osnet-ibn-cityflowv2-train` v1 used the intended 120-epoch OSNet-IBN recipe with batch 128, `P=32, K=4`, bfloat16 AMP, and DataParallel on 2x T4. It crashed on the first training batch in `src/training/losses.py:69` TripletLoss: `dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)` raised `RuntimeError: self and mat2 must have the same dtype, but got Float and Half`.
- **Root cause**: under bfloat16 autocast plus DataParallel, `global_feat` exited as half precision. The training cell already cast `.float()` for center loss but passed the raw half tensor into triplet loss.
- **v2 patch**: changed only the notebook training cell from `loss_tri = triplet_loss_fn(global_feat, pids)` to `loss_tri = triplet_loss_fn(global_feat.float(), pids)`. Shared `src/training/losses.py` was not changed. All hyperparameters stayed identical: 120 epochs, batch 128, `P=32, K=4`, AdamW `lr=3.5e-4`, bfloat16 AMP, DataParallel on 2x T4.
- **v2 result**: AMP fix worked. v2 trained successfully through **44/120 epochs** with healthy convergence (`total_loss` 3.93 -> 2.28, `id_loss` 3.64 -> 2.05, triplet steady around 0.71, center loss 151 -> 37) and saved `/kaggle/working/osnet_ibn_cityflowv2_v1_epoch_030.pth`. It then crashed around epoch 44 with Kaggle system RAM OOM (`Your notebook tried to allocate more memory than is available`; kernel died waiting for execute reply). Total wall time was about 65 min after about 24 min setup.
- **Warm-start recovery failed**: `kaggle kernels output yahiaakhalafallah/14m-osnet-ibn-cityflowv2-train` did not recover the epoch-030 checkpoint. Kaggle did not persist `/kaggle/working/` checkpoint outputs from the ERROR'd run; only `training_history.json` and repo/config files came back. The next attempt must restart from scratch.
- **v3 memory-defense patches applied locally, not pushed**: `BATCH_SIZE 128 -> 64`, `EVAL_BATCH_SIZE 16 -> 8`, `P_IDS 32 -> 16` with `K=4` unchanged, DataParallel removed (`base_model = model` single GPU), per-epoch `gc.collect(); torch.cuda.empty_cache()` inserted at the top of the training loop, and `kernel-metadata.json` description updated to `v3 memory defense: single-GPU OSNet-IBN training, batch 64, eval batch 8, per-epoch gc cleanup.` Local `_validate_14m_kernel.py` passes with the new contracts.
- **Former blocker resolved**: v3 was initially blocked by Kaggle account state (`yahiaakhalafallah` weekly GPU cap and `ali369` push failures). The run was later pushed and completed on gumfreddy using the `KAGGLE_API_TOKEN` access-token workflow.
- **Ready-to-push state superseded**: the notebook and metadata under `notebooks/kaggle/14m_osnet_ibn_train/` were patched and validation-passing, then successfully executed on gumfreddy. The final result below supersedes the earlier handoff state.

### Resolution — completed on gumfreddy, FAILED gate

- **Completed run**: `gumfreddy/14m-osnet-ibn-cityflowv2-train` v1 ran successfully to completion after the v3 memory-defense patches. It trained **120 epochs** in about **6.5h on T4** with single-GPU batch 64, eval batch 8, `P=16/K=4`, no DataParallel, and per-epoch memory cleanup.
- **Final eval (epoch 120)**: **mAP=23.80%, R1=43.89%, R5=53.72%, R10=60.28%**.
- **Best mAP checkpoint (epoch 60)**: **mAP=24.27%, R1=43.59%, R5=53.91%, R10=60.65%**.
- **Best joint checkpoint (epoch 90)**: **mAP=23.90%, R1=43.97%, R5=53.89%, R10=60.32%**.
- **Eligibility gate**: required **mAP >=75% AND R1 >=90%** before any Stage-2 extraction or 14n fusion. 14m fails by about **50pp mAP**, so do **not** extract features, do **not** run 14n, and do **not** add this stream to any fusion sweep.
- **Verdict**: **DEAD END**. OSNet-IBN-x1.0 in-domain CityFlowV2 from-scratch is much too weak for the intended architecture-diverse stream. Its final **23.80% mAP** is below the **52.77% R101-IBN floor** and far below the **80%+ TransReID CLIP primary**; fusion would likely hurt.
- **Training behavior**: performance peaked at epoch 60 and then slowly degraded through epoch 120, so the run over-trained rather than needing more epochs.
- **Lessons for future OSNet attempts**: do not retry OSNet-IBN-x1.0 from scratch on CityFlowV2 as-is. Any future OSNet branch must address (a) the small 666-class CityFlowV2 train set, likely via strong VeRi pretraining; (b) the changed batch/sampler dynamics introduced by the v3 single-GPU batch-64 memory defense; and (c) the likely mismatch between OSNet and the BoT LR schedule.
- **Operational lesson**: Kaggle multi-account auth works with `KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/<account>_access_token -Raw).Trim()`, proven on gumfreddy.