# 14r Recovery - Stage 2 Only from Saved CLIP-ReID Prompts

**Date**: 2026-05-10  
**Status**: DIAGNOSED - do not push until approved  
**Failed kernel**: `mrkdagods/14r-clip-reid-veri-776-train`  
**Recommended path**: Option B, Stage 2 only from the saved Stage 1 prompts checkpoint.

## Diagnosis

The failure was not OOM, state-dict mismatch, missing data, or AMP scaler reuse. The notebook completed Stage 1 and saved prompts, then Stage 2 completed epoch 1. Immediately after epoch 1, the projected walltime guard fired:

```text
GPU memory free before Stage 2: 13.40 GB / 14.56 GB
{"stage": 2, "epoch": 1, "loss": 9.845594459365003, "elapsed_hours": 0.1032089078426361, "projected_stage2_hours": 12.385068941116332, "lr_head": 0.0003509}
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/local/lib/python3.12/dist-packages/papermill/execute.py", line 131, in execute_notebook
    raise_for_execution_errors(nb, output_path)
  File "/usr/local/lib/python3.12/dist-packages/papermill/execute.py", line 251, in raise_for_execution_errors
    raise error
papermill.exceptions.PapermillExecutionError:
---------------------------------------------------------------------------
Exception encountered at "In [11]":
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_24/2053603976.py in <cell line: 0>()
     33     if projected_total + (time.time() - stage1_start) / 3600 > MAX_TOTAL_TRAIN_HOURS:
     34         torch.save(model.state_dict(), LAST_PATH)
---> 35         raise RuntimeError("Projected walltime exceeds 14h hard cutoff; saved last checkpoint and aborted")
     36     if epoch in PERIODIC_EVAL_EPOCHS:
     37         result = evaluate_feature_row(f"epoch_{epoch}_single_flip_cls_base", concat_patch=False, aqe_k=None, rerank=False)

RuntimeError: Projected walltime exceeds 14h hard cutoff; saved last checkpoint and aborted
```

The downloaded `train_log.json` shows zero Stage 2 epochs because the notebook appends the Stage 2 row, prints it, checks the walltime guard, raises, and only writes `train_log.json` after the guard. The stdout log is authoritative for Stage 2 epoch 1.

## Notebook Review

- Stage 1 saves `clip_reid_vit_b16_veri776_stage1_prompts.pt` with `text_features`, `ctx`, `id_tokens`, and `recipe`.
- The defensive cleanup cell runs between stages: `del clip_model`, `gc.collect()`, and `torch.cuda.empty_cache()`.
- Stage 2 rebuilds `timm.create_model("vit_base_patch16_clip_224.openai", pretrained=True, num_classes=0, img_size=224)` through `TransReIDClipReID`, with all params trainable, SIE/JPM/BNNeck, and a 768-to-512 `i2t_projection` head.
- Stage 2 loads prompts as `cached_text_features = F.normalize(prompt_payload["text_features"].to(DEVICE).float(), dim=-1)`.
- Stage 2 optimizer is AdamW with LLRD backbone groups plus high-lr heads; batch remains `P=8/K=4`, size 32. It fits memory: Stage 2 started with 13.40 GB free on a 14.56 GB T4 and completed one epoch.
- Stage 1 and Stage 2 use separate AMP scalers, so scaler carryover is not the failure.

## Prompt Checkpoint Validation

Local checkpoint: `tmp_14r_primary_outputs/clip_reid_vit_b16_veri776_stage1_prompts.pt`

| Key | Shape | Dtype | Finite | Notes |
|---|---:|---|---|---|
| `text_features` | `[576, 512]` | `torch.float32` | yes | Mean feature norm is 1.0; matches Stage 2 cached text feature expectation. |
| `ctx` | `[4, 512]` | `torch.float32` | yes | Four shared context tokens. |
| `id_tokens` | `[576, 512]` | `torch.float32` | yes | One identity token per VeRi train ID. |
| `recipe` | dict | n/a | n/a | Embedded original recipe metadata. |

The saved prompts are valid and were written after Stage 1 completed all 120 epochs. They are suitable for a Stage-2-only recovery.

## Recovery Design - Option B

Create a new Stage-2-only Kaggle kernel that takes the saved prompts checkpoint as an input dataset and skips Stage 1 entirely.

Inputs:

- VeRi-776 dataset: `abhyudaya12/veri-vehicle-re-identification-dataset`
- New prompts dataset containing `clip_reid_vit_b16_veri776_stage1_prompts.pt`

Kernel behavior:

1. Install/import the same dependencies as 14r primary.
2. Build the same VeRi dataset, PK sampler, transforms, `TransReIDClipReID` model, optimizer, scheduler, losses, and eval suite.
3. Load the prompts checkpoint from the Kaggle input dataset.
4. Assert `text_features.shape == (576, 512)` and all prompt tensors are finite before allocating Stage 2 training.
5. Set `stage2_start = time.time()` and use a Stage-2-only walltime guard, not the failed total-run guard that includes the already-completed Stage 1 walltime.
6. Train Stage 2 for 120 epochs, saving `stage2_last`, `stage2_best_mAP`, `stage2_best_R1`, `train_log.json`, `eval_results.json`, `summary.json`, and `recipe.json`.
7. Keep the same verdict gates as 14r primary: WIN if best concat post-rerank/AQE row has `mAP >= 0.9154` and `R1 >= 0.9833`; MARGINAL if `mAP >= 0.905` or `R1 >= 0.980`; otherwise FAIL.

Walltime guard fix:

```python
projected_stage2_hours = elapsed_hours / epoch * STAGE2_EPOCHS
if projected_stage2_hours > MAX_STAGE2_TRAIN_HOURS:
    torch.save(model.state_dict(), LAST_PATH)
    TRAIN_LOG_PATH.write_text(json.dumps(train_log, indent=2), encoding="utf-8")
    raise RuntimeError("Projected Stage 2 walltime exceeds guard; saved last checkpoint and aborted")
```

Recommended guard: `MAX_STAGE2_TRAIN_HOURS = 13.5` if only training is guarded, or `12.5` if preserving room for final eval under Kaggle's practical runtime ceiling. The failed epoch-1 projection was about 12.39h for Stage 2 training, so a 12.5h guard is tight but likely viable; 13.5h is safer if quota/runtime allows final eval.

Do not reuse the failed `stage2_last.pth` as a resume point. It was saved after only one Stage 2 epoch and before any periodic evaluation; restarting Stage 2 from the saved prompts is cleaner and reproducible.

## Why Not Option A

Full restart would spend another Stage 1 run even though the prompts checkpoint is valid. With Stage 1 observed at 5.66h and Stage 2 projected at 12.39h, the full two-stage recipe is not viable under the remaining MRKDaGods quota or the existing 14h hard cutoff.

## Next Move

Push recovery only after approval. The recommended next action is to create a prompts Kaggle dataset, build a new Stage-2-only notebook/kernel, validate metadata locally, then push once under MRKDaGods. If the user wants to conserve MRKDaGods quota, pivot to another account or reduce Stage 2 schedule before pushing.

---

## LOCKED PLAN — 2026-05-10 (supersedes "Recovery Design - Option B" above)

**Chosen path**: Option 1 from planner shortlist — **Stage-2-only recovery, 60 epochs, batch 64 (P=16, K=4)**.
**Account**: gumfreddy (~6h quota; MRKDaGods kept as warm fallback).
**Spec status**: LOCKED. Coder may build the notebook directly from this section.

### Rationale (4 bullets)

1. **Stage 1 prompts are valid and complete** (`text_features [576,512]`, `ctx [4,512]`, `id_tokens [576,512]`, all finite, norm=1.0). Re-running Stage 1 is pure waste — Option B (Stage-2-only) is mandatory.
2. **The 12.4h Stage 2 projection is the binding constraint**. Original recipe at batch 32 = 6.2 min/epoch × 120 = 12.4h, which exceeds *any* available quota (MRKDaGods 6h, gumfreddy 6h). Sticking with batch 32 means we get walltime-killed at ~50–55 epochs *with no guarantee a periodic eval ran late enough* — high probability of zero recoverable Stage-2 best checkpoint.
3. **Batch 64 (P=16, K=4) matches 14q's proven config on identical hardware**. 14q ran ViT-B/16 @ 256² at 0.48 min/epoch with this batch on the same T4 (13.4 GB free at Stage 2 start ≫ 14q's footprint). Halving the step count is the single highest-leverage knob: projects Stage 2 to ~3.0–3.5 min/epoch × 60 epochs ≈ **3.0–3.5h**, comfortably inside the gumfreddy 6h quota with margin for final eval.
4. **60 epochs is the lower end of CLIP-ReID Stage 2 paper recipes** (paper uses 60 on Market/MSMT, 120 on VeRi). Slight underfit risk vs 120 ep, but with batch 64 the per-step gradient is stronger (16 IDs × 4 = 64 vs 8 × 4 = 32), partially compensating. Net EV strictly higher than (a) batch-32 60-ep run that gets walltime-killed mid-training, or (b) recipe-modification options that abandon the published anchor.

### Locked Hyperparameters (Stage 2 only)

| Param | Value | Note |
|---|---|---|
| Backbone | `vit_base_patch16_clip_224.openai` via timm | unchanged from 14r primary |
| Image size | 224 | unchanged |
| Batch | **P=16, K=4, size=64** | **CHANGED from primary's P=8/K=4=32** |
| Epochs | **60** | **CHANGED from primary's 120** |
| Optimizer | AdamW | unchanged |
| `backbone_lr` | **0.000495** (= 0.00035 × √2) | sqrt-scale for 2× batch |
| `head_lr` | **0.00495** (= 0.0035 × √2) | sqrt-scale for 2× batch |
| `llrd_factor` | 0.65 | unchanged |
| `warmup_epochs` | **5** (= 10 × 60/120) | scaled to new schedule |
| `min_lr` | 1e-6 | unchanged |
| Losses | `ce=1.0`, `triplet=1.0`, `i2tce=1.0`, `jpm_ce=1.0` | unchanged — do NOT drop i2tce; it's the whole point of CLIP-ReID |
| SIE / JPM / BNNeck | unchanged from primary | |
| Augmentation | resize_bicubic_224 → hflip 0.5 → pad 10 → random_crop 224 → random_erasing 0.5 | unchanged |
| AMP | fp16, separate scaler from Stage 1 | unchanged |
| Periodic eval epochs | **[20, 40, 50, 55, 60]** | front-load so an early walltime kill still leaves a usable best-mAP checkpoint |
| Save policy | `stage2_best_mAP.pth`, `stage2_best_R1.pth`, `stage2_last.pth` after every periodic eval AND every epoch's `stage2_last.pth` | aggressive, defensive |

### Walltime Guard (Stage-2-only, fixed)

```python
STAGE2_EPOCHS = 60
MAX_STAGE2_TRAIN_HOURS = 4.5   # leaves ~1.5h of 6h quota for final eval + safety margin
stage2_start = time.time()
# inside the epoch loop, AFTER appending the epoch row to train_log:
elapsed_hours = (time.time() - stage2_start) / 3600.0
projected_total = elapsed_hours / epoch * STAGE2_EPOCHS
if projected_total > MAX_STAGE2_TRAIN_HOURS:
  torch.save(model.state_dict(), LAST_PATH)
  TRAIN_LOG_PATH.write_text(json.dumps(train_log, indent=2), encoding="utf-8")
  raise RuntimeError(
    f"Projected Stage-2 walltime {projected_total:.2f}h exceeds {MAX_STAGE2_TRAIN_HOURS}h guard; "
    f"saved last checkpoint and aborted at epoch {epoch}"
  )
```

**Critical**: write `train_log.json` to disk on EVERY epoch (not just at the guard), so a hard kill never loses Stage 2 history. The 14r primary failure was made worse by deferred log writing.

### Pre-registered Verdict Bands

Evaluation matrix unchanged from 14r primary (4 rows: `single_flip_cls_base`, `single_flip_cls_aqe2_rerank`, `concat_patch_flip_aqe2_rerank`, `concat_patch_flip_aqe3_rerank`; rerank `k1=80, k2=15, λ=0.2`).

| Verdict | Condition (best concat-AQE row) | Action |
|---|---|---|
| **WIN** | `mAP ≥ 0.9154` AND `R1 ≥ 0.9833` | Promote: replace 09v v17 in production fusion stack; update `findings.md` headline; update `copilot-instructions.md` "Vehicle ReID single-cam" line. |
| **MARGINAL** | `mAP ≥ 0.905` OR `R1 ≥ 0.980` (but not WIN) | Keep checkpoint, add to ensemble pool for 14s/Stage-4 fusion experiments only; do NOT replace 09v v17 production. |
| **FAIL** | neither MARGINAL nor WIN | Mark CLIP-ReID family CLOSED for VeRi-776 at our compute scale. Pivot to 14s (CityFlow MTMC fusion of existing 5 models). Update `findings.md` dead-ends list. |

### Compute Budget

| Item | Estimate | Quota source |
|---|---|---|
| Setup + dataset mount + model build | ~5 min | gumfreddy |
| Stage 2 training (60 ep × ~3.0–3.5 min/ep) | **3.0–3.5h** | gumfreddy |
| Final 4-row eval (concat patch + flip + rerank) | ~25–35 min | gumfreddy |
| **Total expected** | **~3.5–4.2h** | well inside 6h gumfreddy quota |
| **Hard guard cutoff** | 4.5h training + ~0.5h eval = ~5.0h worst-case | leaves 1h margin |

If gumfreddy 409s at push time, fall back to MRKDaGods (also ~6h, same fit).

### Fast-Fail Abort Signals

Coder MUST abort the kernel and report back if ANY of the following occur during the run (no autonomous retry):

1. **Stage 2 epoch 1 elapsed > 5.0 min** → memory thrash or batch-64 doesn't fit on T4. Cancel via `kaggle kernels cancel` and revert to batch 32 in a follow-up push (NOT in this run).
2. **Stage 2 epoch 1 loss > 12.0** OR **NaN/Inf in loss** → LR scaling broken or AMP scaler issue. Cancel.
3. **Walltime guard fires before epoch 20** → projection blew up; means epoch-1 timing was misleading. Cancel and re-plan.
4. **Periodic eval at epoch 20 returns mAP < 0.55** → Stage 1 prompts not loading correctly OR i2tce auxiliary broken. Cancel; re-validate prompt loading code path. (Reference: 14r primary CPU smoke had loss ~6.97 with bogus data and still trended sane; real-data Stage 2 epoch 20 should be well above this floor on mAP.)
5. **OOM** → batch 64 doesn't fit despite 14q precedent. Cancel; coder reverts to batch 48 (P=12, K=4) in follow-up only after planner approval.

### Pre-Push Checklist (Coder)

- [ ] Create Kaggle dataset containing `clip_reid_vit_b16_veri776_stage1_prompts.pt` (under gumfreddy account).
- [ ] Build Stage-2-only notebook from 14r primary's Stage-2 cells; strip Stage 1 entirely; load prompts from new input dataset.
- [ ] Apply locked hyperparameter table above. Diff against 14r primary recipe.json and confirm exactly 4 changes: batch P, epochs, backbone_lr, head_lr, warmup_epochs.
- [ ] Wire the fixed Stage-2-only walltime guard with per-epoch log flush.
- [ ] CPU smoke test the notebook end-to-end with `--nb-fastsmoke` style flag (1 epoch, 8 IDs) to confirm no import / shape / load-prompt errors.
- [ ] `kaggle kernels push` ONCE under gumfreddy. Confirm no "not valid dataset sources" warnings.
- [ ] Do NOT re-push on first sign of slowness — let the in-kernel walltime guard do its job.
- [ ] Poll `kaggle kernels status` every ~15 min; tail logs at epoch 20 periodic eval mark.

### What This Spec Does NOT Authorize

- Modifying loss weights or dropping i2tce (Option 2 was explicitly rejected — published recipe must stay intact apart from batch/epochs/LR scaling).
- Pushing under any account other than gumfreddy → MRKDaGods fallback, in that order.
- Restarting from `stage2_last.pth` of the failed primary run (only 1 epoch, no eval baseline).
- Score-fusion / ensemble work (that's 14s territory; this spec is training-only).

### If Result is FAIL

Planner pre-commits: Coder will NOT autonomously re-push CLIP-ReID with different hyperparameters. Pivot decision (to 14s ensemble or 14k+ CityFlow MTMC fusion) goes back to planner. Three FAIL'd large-scale ReID training runs in a row (14m, 14p, 14r-recovery if it FAILs) constitutes confirmed evidence of a ~89–90% mAP feature ceiling at our compute scale, and further VeRi training spend is no longer EV-positive.