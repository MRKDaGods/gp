# Vehicle Breakthrough — TODAY Spec

> **Deadline**: URGENT. Target: materially move vehicle MTMC IDF1 above the current reproducible 0.7736 floor (10c v61, fusion w=0.10 with 09p R50-IBN secondary). Historical best 0.784. SOTA 0.8486 (gap 7.36pp). Association tuning is fully exhausted (225+ configs). This spec focuses exclusively on feature-side and untested fusion-geometry moves.

## TL;DR — Phase 1 (start immediately)

**3-way score-level ensemble sweep**: primary OpenAI CLIP ViT-B/16 (80.14 mAP) + secondary LAION-2B CLIP ViT-B/16 09l v3 (78.61 mAP) + tertiary fine-tuned R50-IBN 09p (63.64 mAP). Every pairwise combination has been tested and either capped at +0.06pp (09p) or regressed -0.5pp (09l v3). The 3-way combination has **never been tested** and is the only remaining uncombinatorially-exhausted fusion geometry on the current codebase. Expected delta: neutral to +0.5pp. Infrastructure already supports 3-way fusion via `stage2.reid.vehicle2` + `stage2.reid.vehicle3` and `stage4.association.secondary_embeddings` + `stage4.association.tertiary_embeddings`. Wall clock: one 10a run (~90 min) + one 10b run (~10 min) + one 10c two-axis sweep (~30 min). No new training.

## Avenues, Ranked by Expected Delta / Effort

### Avenue 1 — 3-Way Score-Level Ensemble (OpenAI × LAION × R50-IBN) — PHASE 1
- **Expected MTMC IDF1 delta**: +0.0 to +0.5pp. Justification: each pairwise was flat-to-slightly-harmful, but 3-way diversity has never been measured. R50-IBN CNN brings architectural diversity that the two CLIP ViTs lack (10c v56 regressed because primary+secondary were both CLIP ViT-B/16, too correlated). A CNN tertiary with even partial complementary signal could flip the 3-way fusion positive where 2-way was flat.
- **Risk**: Medium. Could also be flat. Worst case: ~-0.2pp if R50-IBN CNN noise dominates LAION ViT agreement.
- **Wall-clock**: ~2.5h end-to-end on Kaggle (gumfreddy).
- **Not a dead end**: pairwise fusion with 63.64% R50-IBN was tested (10c v61: +0.06pp). Pairwise fusion with 78.61% LAION ViT was tested (10c v56: -0.5pp). 3-way has **not** been tested on the current codebase.
- **Files to modify**:
  - `notebooks/kaggle/10a_stages012/mtmc-10a-stages-0-2-tracking-reid-features.ipynb` — repurpose the existing `vehicle3` slot to load the **09l v3 LAION-2B CLIP** TransReID checkpoint (NOT EVA02, which is a dead end at 48.17% mAP). The slot already exists; swap the weights path and `vit_model` field. Also confirm `vehicle2` remains the **09p** (not 09n) R50-IBN fine-tuned checkpoint.
  - `notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb` — add a 2D fusion-weight sweep over `(w_secondary, w_tertiary)` with pairs:
    - `(0.10, 0.00)` → reproduces 10c v61 baseline (floor, ≈0.7736)
    - `(0.00, 0.00)` → no-fusion control (≈0.7730)
    - `(0.10, 0.10)`, `(0.10, 0.15)`, `(0.10, 0.20)`
    - `(0.15, 0.10)`, `(0.05, 0.15)`, `(0.05, 0.20)`
    - `(0.20, 0.10)`, `(0.00, 0.15)`, `(0.00, 0.20)`
    - Keep every other association parameter at the v80-restored recipe: `sim_thresh=0.50`, `appearance_weight=0.70`, `fic.regularisation=0.50`, `query_expansion.k=3`, `gallery_expansion.threshold=0.48`, `gallery_expansion.orphan_match_threshold=0.38`.
- **Exact Kaggle inputs required on the 10a run**:
  - Primary OpenAI CLIP 09 v2 weights (already attached)
  - Secondary **09p** fine-tuned R50-IBN (already attached in 10a `run_kaggle_20260420_201401`)
  - Tertiary **09l v3** LAION-2B CLIP `.pth` — attach dataset `gumfreddy/09l-transreid-laion-2b-training/output` (version that exported the 300-epoch resumed checkpoint)
- **Config overrides on 10a**:
  - `stage2.reid.vehicle.input_size=[256,256]`
  - `stage2.reid.vehicle2.enabled=true` `stage2.reid.vehicle2.model_name=fastreid_sbs_r50_ibn` `stage2.reid.vehicle2.weights_path=models/reid/fastreid_r50_ibn_cityflowv2.pth` `stage2.reid.vehicle2.embedding_dim=2048` `stage2.reid.vehicle2.input_size=[256,256]` `stage2.reid.vehicle2.save_separate=true`
  - `stage2.reid.vehicle3.enabled=true` `stage2.reid.vehicle3.model_name=transreid` `stage2.reid.vehicle3.weights_path=models/reid/laion2b_clip_vitb16_09lv3.pth` `stage2.reid.vehicle3.embedding_dim=768` `stage2.reid.vehicle3.input_size=[256,256]` `stage2.reid.vehicle3.vit_model=vit_base_patch16_clip_224.laion2b` `stage2.reid.vehicle3.clip_normalization=true` `stage2.reid.vehicle3.num_cameras=59` `stage2.reid.vehicle3.save_separate=true`
- **Config overrides on 10c per sweep point**:
  - `stage4.association.secondary_embeddings.path=<10b/out>/embeddings_secondary.npy`
  - `stage4.association.secondary_embeddings.weight=<w2>`
  - `stage4.association.tertiary_embeddings.path=<10b/out>/embeddings_tertiary.npy`
  - `stage4.association.tertiary_embeddings.weight=<w3>`
- **Fallback if it flatlines**: lock the 10c v61 recipe as the submission floor and advance Phase 2 (Avenue 2).

### Avenue 2 — 09q Extended Primary Training with SAFE Recipe — PHASE 2 (launch in parallel, overnight)
- **Expected MTMC IDF1 delta**: +0.3 to +1.0pp **if** the extended checkpoint reaches ≥82% mAP without the augoverhaul regression pattern. Justification: 09 v2 augoverhaul reached 81.59% mAP but crashed MTMC to 0.722 (10c v48/v49) because it simultaneously changed augmentations AND loss (stronger ColorJitter/RandomPerspective/RandomGrayscale/GaussianBlur + TripletLoss→CircleLoss). Extending the schedule of the ORIGINAL 09 recipe (baseline augmentations, TripletLoss) from the 80.14% checkpoint has **not** been tested. The only previously "extended" 09d run was the ResNet101-IBN-a (50.61% ceiling) — a different architecture.
- **Risk**: Medium-High. Prior evidence shows higher primary mAP does not always translate to MTMC. But the augoverhaul confound does not apply here because we keep baseline augs. Worst case: mAP plateaus at 80% and nothing moves.
- **Wall-clock**: 3-5h training on Kaggle T4 + 1.5h re-extraction + 0.5h association sweep. Fits overnight.
- **Not a dead end**: CircleLoss and augoverhaul augs are dead ends, but **resuming from the 80.14% checkpoint with the original TripletLoss recipe for 60-120 more epochs** has never been tried.
- **CRITICAL 1-line fix to `09q_transreid_extended.ipynb` v10 BEFORE launch**:
  - User reported v9 failed with `DataParallel.get_llrd_param_groups` AttributeError. The current notebook already has `raw_model = model.module if hasattr(model, "module") else model` and calls `raw_model.get_llrd_param_groups(...)` at lines 1022/1029 and 1154/1161, which should work. **Action for coder**: (a) push a fresh version (v10) of 09q to Kaggle and verify the traceback is gone by monitoring the first 3 epochs; (b) if the error persists, replace both calls with `(raw_model.module if isinstance(raw_model, nn.DataParallel) else raw_model).get_llrd_param_groups(...)` as a defensive fix.
- **SECOND critical patch to 09q v10 BEFORE launch (recipe fix)**: The notebook currently uses **CircleLoss (m=0.25, gamma=128)** — a confirmed dead end (09 v4 `inf` loss, 18.45% mAP; 09l v1 `inf` loss, 20.36% mAP; see findings.md). **Must replace with**:
  - `TripletLoss(margin=0.3)` + `CrossEntropyLabelSmooth(eps=0.05)` + delayed `CenterLoss(weight=5e-4, start_epoch=15)` (identical to what rescued 09l v2 → 09l v3).
  - Keep `backbone_lr=5e-5`, `head_lr=5e-4`, `LLRD=0.75`, `EMA(decay=0.9999)` enabled, 5-epoch warmup + cosine-120.
  - **Keep baseline augmentations only**: `RandomHorizontalFlip`, `Pad+RandomCrop`, weaker `ColorJitter(0.2, 0.15, 0.1, 0.0)`, `Normalize`, `RandomErasing`. **Do NOT enable** `RandomGrayscale`, `GaussianBlur`, `RandomPerspective`, or stronger jitter — these are the augoverhaul components that regressed MTMC.
- **Post-training deployment**: after 09q v10 finishes, copy the best checkpoint into a 10a override chain (same structure as the 10a run from Phase 1 but swap `stage2.reid.vehicle.weights_path` to the 09q extended checkpoint and keep vehicle2/vehicle3 identical). Re-run 10b and the same 10c 2-way association sweep.
- **Fallback if regressed**: roll back to the 09 v2 80.14% primary and lock 10c v61 as submission.

### Avenue 3 — Part-Level JPM Fusion in Stage 4 (cheap, untested)
- **Expected MTMC IDF1 delta**: +0.0 to +0.5pp. Justification: TransReID ViT already outputs a JPM multi-part token head during training (4 local tokens + global CLS). Our current pipeline only deploys the **global CLS** embedding at inference. AIC22 1st place used box-grained matching (part-level similarity). Exposing JPM local tokens as additional similarity streams and averaging part-wise cosines before the existing fusion could capture more identity signal without any new training.
- **Risk**: Medium. Requires code changes in `src/stage2_features/transreid_model.py` to also forward the JPM local tokens and in `src/stage2_features/pipeline.py` to persist them. Then `src/stage4_association/similarity.py` needs a part-wise similarity that reduces via max/mean. Not a 1-day slam-dunk; probably a 1-day prototype for first measurement.
- **Wall-clock**: 4-8h (code + one full pipeline run). Defer to Phase 3 if Phase 1/2 disappoint.
- **Not a dead end**: Multi-query (k>0) was tested as stage4 parameter and hurt, but that used top-K whole-crop embeddings, not part tokens. JPM part tokens have never been exposed at inference.

### Avenue 4 — Per-Camera-Pair FIC Regularization Sweep + Per-Pair Sim Threshold — CHEAP SAFETY NET
- **Expected MTMC IDF1 delta**: +0.0 to +0.2pp. Justification: current FIC uses one global `regularisation` value (0.50). Dead-end cataloguing confirmed additive CID_BIAS is harmful (10c v55), but **per-pair FIC regularization** (which stays multiplicative/geometric, not additive) has not been swept. This is borderline-exhausted but the specific knob of per-pair reg has not been logged.
- **Risk**: Low. If flat, costs ~20 min of a 10c re-run.
- **Wall-clock**: 20-40 min. Queue as free-side experiment alongside Phase 1.
- **NOT a must-do**: only run if Phase 1 / Phase 2 complete with time to spare. Expected delta is too small to be the main effort.

## Explicit DO NOT RETRY (dead ends confirmed in findings.md)

- CSLS (-34.7pp)
- 384px ViT deployment (-2.8pp vs 256px; higher single-cam mAP but cross-camera regression)
- AFLink (-3.8pp at tightest sweep, -13.2pp at wide gap; confirmed structural, not a threshold artifact)
- CID_BIAS (GT-learned -3.3pp; topology +0.02/-0.10 to +0.06/-0.20 all -1.0 to -1.2pp)
- Hierarchical centroid clustering (-1 to -5.1pp)
- FAC (cross-camera KNN consensus, -2.5pp)
- Reranking k-reciprocal (always hurts with current features)
- Feature concat ensemble (-1.6pp; use score-level only)
- Network flow / Hungarian solver (-0.24pp, conflation 27→30)
- VeRi-776 → CityFlowV2 ResNet101-IBN pretrain (09f v3: 42.7% mAP, worse than direct)
- Extended ResNet fine-tuning (09d gumfreddy v3: 50.61%, degraded from 52.77%)
- ArcFace on R101-IBN warm-started from CE (09i v1: 50.80%, overfit)
- ResNeXt101-IBN-a ArcFace (09j v2: 36.88%, partial weight load crippled)
- DMT camera-aware training (single-model regime, -1.4pp; 09g 43.8% mAP)
- CLIP RN50x4 CNN (09m v2: 1.55% mAP; QuickGELU + attention-pool incompatibility)
- EVA02 ViT-B/16 CLIP (09o v1: 48.17% mAP; under the 65% ensemble floor)
- ViT-Small IN-21k (09k v1: 48.66%)
- Score-level 2-way fusion with weak (<65% mAP) secondary (several runs; 10c v61 w=0.10 tops at +0.06pp)
- Score-level 2-way fusion with same-family CLIP ViT secondary (10c v56 LAION: -0.5pp)
- Circle loss on TransReID (09 v4 `inf` loss 18.45% mAP; 09l v1 `inf` loss 20.36% mAP)
- SGD for ResNet fine-tune (30.27% mAP)
- 09 v2/v3 augoverhaul stack (GaussianBlur + RandomPerspective + stronger ColorJitter + RandomGrayscale) — 10c v48/v49 both 0.722 MTMC IDF1
- Multi-query stage4 k>0 (tested v51 neutral/harmful)
- Multi-scale TTA at stage2 (always neutral/harmful across 4+ tests)
- CamTTA (helps global IDF1 but hurts MTMC)
- Track smoothing / edge trim (always harmful)
- `mtmc_only_submission=true` (-5pp)
- SAM2 foreground masking (10a v29 / 10c v50: -8.7pp, 105 min vs 65 min runtime)
- Feature concat of secondary/tertiary (use score-level instead)

## Submission Lock Plan

**Floor submission (non-negotiable)**: 10c v61 with `(w_secondary=0.10, w_tertiary=0.00)` = **MTMC IDF1 = 0.7736**. The 10a chain `run_kaggle_20260420_201401` + 10b v23 is the canonical input. Do NOT overwrite these kernel outputs until a strictly-better run lands.

**Secondary floor**: 10c v52 recipe (v80-restored, no fusion) = **MTMC IDF1 = 0.775**. Keep the kernel output preserved.

**Promotion rule**: any new candidate must clear **MTMC IDF1 ≥ 0.7760** on the clean CityFlowV2 eval split (gt_frame_clip + gt_zone_filter) before replacing the submission artifact. Anything in [0.7736, 0.7760) is informational only.

## Commands — Phase 1 (start immediately)

```powershell
# 1. Patch 10a notebook: swap vehicle3 from EVA02 to 09l v3 LAION-2B CLIP weights.
#    Use a small Python script (json.load → modify → json.dump, ensure_ascii=True) to:
#    - update the TERTIARY_WEIGHTS cell to point at models/reid/laion2b_clip_vitb16_09lv3.pth
#    - add the vehicle3 override block with vit_model=vit_base_patch16_clip_224.laion2b
#    - attach the 09l v3 Kaggle dataset as input (edit kernel-metadata.json `dataset_sources`)

# 2. Push 10a and launch.
kaggle kernels push -p notebooks/kaggle/10a_stages012/

# 3. Poll until 10a completes. Extract the run tag (e.g. run_kaggle_YYYYMMDD_HHMMSS).
python scripts/kaggle_logs.py mtmc-10a-stages-0-2-tracking-reid-features --tail 200

# 4. Patch 10b to consume vehicle3 embeddings as tertiary_embeddings. Launch.
kaggle kernels push -p notebooks/kaggle/10b_stage3/

# 5. Patch 10c to run the (w2, w3) sweep described above. Ensure baseline
#    (0.10, 0.00) and no-fusion (0.00, 0.00) points are included as controls.
kaggle kernels push -p notebooks/kaggle/10c_stages45/

# 6. Read results, log to docs/experiment-log.md under a new '2.8 3-way Fusion Follow-Up' subsection.
python scripts/kaggle_logs.py mtmc-10c-stages-4-5-association-eval --tail 400
```

## Commands — Phase 2 (queue in parallel after Phase 1 10a is launched)

```powershell
# 1. Apply SAFE recipe patch to 09q_transreid_extended.ipynb via python json script:
#    - replace CircleLoss(m=0.25, gamma=128) import/use with TripletLoss(margin=0.3)
#    - update CENTER_START and CENTER_WEIGHT (5e-4, start epoch 15)
#    - keep LLRD=0.75, backbone_lr=5e-5, head_lr=5e-4, EMA(0.9999)
#    - remove/disable RandomGrayscale, GaussianBlur, RandomPerspective, stronger ColorJitter cells
#    - verify raw_model.get_llrd_param_groups calls use the unwrapped model correctly

# 2. Push 09q v10 on gumfreddy. EXPECTED: 3-5h run.
kaggle kernels push -p notebooks/kaggle/09q_transreid_extended/

# 3. Monitor first 40 min to confirm:
#    - loss is FINITE (not inf) through epoch 1-5
#    - param group count matches expected LLRD bucket count
#    - no DataParallel traceback
python scripts/kaggle_logs.py 09q-transreid-extended --tail 200
```

## Phase Ordering Rationale

Phase 1 is launched first because (a) it requires zero new training, (b) it measures the only remaining uncombinatorially-exhausted fusion geometry on currently available trained assets, and (c) it finishes within ~2.5h, freeing the kernel slot for Phase 2 results. Phase 2 is launched second because it requires a 3-5h training run and cannot be evaluated until training + a new 10a/b/c chain completes — best queued overnight.

## 10-Line Executive Summary of Phase 1 (for immediate coder dispatch)

1. Patch `notebooks/kaggle/10a_stages012/*.ipynb`: repurpose `vehicle3` slot from EVA02 (dead end) to 09l v3 LAION-2B CLIP TransReID checkpoint, keeping `vehicle2` as fine-tuned R50-IBN (09p). Use a Python json.load/dump script (ensure_ascii=True), NEVER replace_string_in_file on .ipynb.
2. Add required 10a overrides: `stage2.reid.vehicle3.enabled=true`, `model_name=transreid`, `weights_path=models/reid/laion2b_clip_vitb16_09lv3.pth`, `embedding_dim=768`, `input_size=[256,256]`, `vit_model=vit_base_patch16_clip_224.laion2b`, `clip_normalization=true`, `num_cameras=59`, `save_separate=true`.
3. Attach the 09l v3 LAION-2B kernel output as a 10a Kaggle input dataset (edit `kernel-metadata.json`).
4. Push 10a on gumfreddy. Wait for completion (~90 min).
5. Patch 10b to consume and export tertiary embeddings alongside secondary. Push, wait (~10 min).
6. Patch 10c to run an 11-point `(w_secondary, w_tertiary)` sweep including `(0.10, 0.00)` and `(0.00, 0.00)` as controls. Keep all other association params at the v80-restored recipe. Push, wait (~30 min).
7. Confirm promotion rule: any new best must clear MTMC IDF1 ≥ 0.7760 on CityFlowV2 eval split (gt_frame_clip + gt_zone_filter) before replacing submission.
8. Log results under a new section in `docs/experiment-log.md` ("2.8 3-way Fusion Follow-Up") and update `docs/findings.md` with delta and verdict.
9. Regardless of outcome, preserve 10c v61 kernel output as the submission floor (MTMC IDF1 = 0.7736). Do NOT delete.
10. On completion, dispatch Phase 2 (09q v10 SAFE-recipe extended primary training).