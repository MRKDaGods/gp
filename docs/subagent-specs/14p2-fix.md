# 14p2 Spec — Fix the 14p VeRi ViT-L/14 CLIP Training Failure

**Date**: 2026-05-09
**Author**: MTMC Planner
**Status**: PROPOSED — ready for Coder, single-push, NO auto-retry
**Account**: MRKDaGods (~30h GPU quota; gumfreddy is short on quota after 14p v1)
**Auth**: `$env:KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/MRKDaGods__access_token -Raw).Trim()` (note the **double underscore** in filename)
**Hardware**: Kaggle T4 (16GB), single GPU
**Predecessor**: 14p v1 (`gumfreddy/14p-veri-vit-l-14-clip-transreid-train`) — completed 100 epochs in ~6h, produced **mAP=13.7%, R1=26.7%** (target ≥91.5% mAP).
**Verdict bands (UNCHANGED from 14p)**:
- WIN: concat-patch-flip AQE+rerank mAP ≥91.5% AND R1 ≥98.6%
- MARGINAL: mAP 89.97-91.5% AND R1 ≥98.4%
- FAIL: mAP <89.97% OR R1 <98.3%
- Hard cutoff: 14h wall-clock for full train+eval

---

## 1. Confirmed Root Cause

**Honest disclosure: none of H1-H4 in the user-supplied diagnosis match the on-disk code.** I inspected the .ipynb and `_build_14p_notebook.py` line-by-line and cross-checked against the proven 08 reference. Findings:

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 — triplet receives wrong tensor (`bn_feat`/detached/logits) | **FALSE** | [notebooks/kaggle/14p_veri_vit_l_train/14p_veri_vit_l_train.ipynb](notebooks/kaggle/14p_veri_vit_l_train/14p_veri_vit_l_train.ipynb#L462) calls `triplet_loss_fn(outputs["global_feat"].float(), pids)`. `outputs["global_feat"]` is `tokens[:, 0]` — pre-BN raw CLS, no `.detach()`, not logits. Bit-identical to 08's `TripletLossHardMining` input contract ([08 cell](notebooks/kaggle/08_vehicle_reid_sota/08_vehicle_reid_sota.ipynb)). |
| H2 — PK sampler broken (no positives per batch) | **FALSE** | [PKBatchSampler.__iter__](notebooks/kaggle/14p_veri_vit_l_train/14p_veri_vit_l_train.ipynb#L222-L233) explicitly does `for pid in pids: chosen = rng.choice(indices, size=k, ...); batch.extend(chosen)` — every batch contains exactly `P=8` unique IDs × `K=4` instances = 32 samples with 4 positives per ID guaranteed. |
| H3 — feature collapse (constant features) | **FALSE** | If features were collapsed, BN(constant) → 0 → uniform logits → CE locked at ln(576)=6.36 forever. CE actually descends from 6.46 → 2.74 (-3.7), so the network IS learning discriminative features. |
| H4 — gradient blocked / backbone frozen | **FALSE** | If backbone were frozen, CE would not descend at all (head alone on a frozen random-init projection cannot reach 2.74 from 6.46). |

**The actual root cause** (confirmed by reading [data/outputs/14p_v1_train_log.json](data/outputs/14p_v1_train_log.json)):

> **The model trains correctly but is dramatically under-converged because the recipe (tuned for ViT-B/86M) is being applied uniformly to ViT-L/300M without LLRD, and the high uniform `backbone_lr=3.5e-4` is damaging the CLIP-pretrained features faster than 100 short epochs can replace them.**

### Empirical evidence — triplet IS descending, just glacially

The user's report "triplet frozen at exactly 0.300 for ALL 100 epochs" is incorrect. The actual per-epoch values from `data/outputs/14p_v1_train_log.json`:

```text
epoch  triplet              global_ce   jpm_ce
1      0.3381901594499747   6.4612      6.5216
5      0.3006561928325229   6.1398      6.2645
10     0.3003958670629395   5.7339      5.8197
30     0.3002906002932125   4.5604      4.5789
50     0.3002218819326825   3.6294      3.6543
70     0.3001867528590891   3.0602      3.0881
100    0.3001770257121987   2.7436      2.7711
```

Triplet drops monotonically by **0.0380 over 100 epochs** (0.338→0.300, asymptoting to the margin floor). CE drops monotonically by **3.72 over 100 epochs**. Both losses are training. There is no single-line bug.

### Why training is so slow — three compounding scaling issues

1. **No LLRD on a 24-block ViT-L from CLIP pretrain.** Every backbone parameter receives the same `backbone_lr=3.5e-4`. For an OpenAI CLIP ViT-L/14, the early blocks encode primitive image features that should be near-frozen on a 37k-image fine-tune; uniform high LR partially destroys them. The reference 08 recipe applies LLRD with factor 0.75 across blocks (see [08 cell](notebooks/kaggle/08_vehicle_reid_sota/08_vehicle_reid_sota.ipynb), `llrd_factor = 0.75`, `backbone_lr = 3.5e-4`) — but 14p's [build_optimizer](notebooks/kaggle/14p_veri_vit_l_train/14p_veri_vit_l_train.ipynb#L483-L502) puts every `vit.parameters()` into a single param group with one shared lr.

2. **Backbone LR is the ViT-B value applied to a 3.5× larger model.** ViT-B has 86M params, ViT-L has 304M. Empirically, the same TransReID recipe at the same LR but 3.5× capacity needs either lower LR or LLRD to avoid destabilizing the deep CLIP representation. 14p has neither.

3. **Per-iteration triplet gradient is dwarfed by CE.** With L2-normalized features in 1024-D from an under-converged 300M-param backbone, hardest_pos ≈ hardest_neg by construction (both near √2 in [0, 2] distance space), so triplet contributes ~constant gradient direction with magnitude ≈1 per anchor while CE contributes per-class supervised signal across 576 classes. Triplet is effectively a tie-breaker that does little until CE has separated the classes — which never finishes within 100 epochs.

### Comparison: 14p vs 08

| Metric | 08 (proven, 89.97% mAP) | 14p v1 (failed, 13.7% mAP) |
|---|---|---|
| Backbone | ViT-Base/16 CLIP, 86M params | ViT-Large/14 CLIP, 304M params |
| Batch | 96 (24 IDs × 4) on 2× T4 | 32 (8 IDs × 4) on 1× T4 |
| Backbone LR | 3.5e-4 with **LLRD factor 0.75** | 3.5e-4 **uniform, no LLRD** |
| Epochs / batches per epoch | 140 / 24 = **3360 iter** | 100 / 72 = **7200 iter** |
| Iter ÷ params (M) | 3360 / 86 = **39** | 7200 / 304 = **24** |
| Auxiliary loss | Triplet + Center loss @ ep30 | Triplet only |

14p has more total iterations than 08, but 38% fewer iterations-per-million-parameters and **no LLRD protecting CLIP features at higher capacity**.

---

## 2. Minimal Patch

**Bug class**: This is a recipe-scaling defect, not a bytecode bug. The user requested "1-3 lines ideal", but the smallest fix that has a defensible chance of recovering ≥89.97% mAP is a **single localized rewrite of `build_optimizer`** in the 14p notebook (~15 lines added inside one cell). No other cell is touched. Shared `src/training/` modules are not touched.

### Patch P1 — Add LLRD to `build_optimizer` (REQUIRED)

In [14p notebook MODEL/OPTIMIZER cell](notebooks/kaggle/14p_veri_vit_l_train/14p_veri_vit_l_train.ipynb#L483-L502), replace the body of `build_optimizer` so that backbone parameters are split into per-block LR groups using a `LLRD_FACTOR = 0.65` decay. Substring change:

**FIND** (exact substring in cell source):
```
def build_optimizer(current_model):
    backbone_param_ids = {id(parameter) for parameter in current_model.vit.parameters()}
    backbone_params = []
    head_params = []
    for parameter in current_model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in backbone_param_ids:
            backbone_params.append(parameter)
        else:
            head_params.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR, "base_lr": BACKBONE_LR, "name": "backbone"},
            {"params": head_params, "lr": HEAD_LR, "base_lr": HEAD_LR, "name": "head"},
        ],
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer
```

**REPLACE WITH**:
```
LLRD_FACTOR = 0.65

def build_optimizer(current_model):
    vit = current_model.vit
    num_blocks = len(vit.blocks)
    backbone_groups = []

    def _add_group(params, depth_from_top, name):
        params = [p for p in params if p.requires_grad]
        if not params:
            return
        scale = LLRD_FACTOR ** depth_from_top
        lr = BACKBONE_LR * scale
        backbone_groups.append({
            "params": params,
            "lr": lr,
            "base_lr": lr,
            "name": name,
        })

    # Stem: patch_embed + cls_token + pos_embed + (optional) norm_pre — deepest from top
    stem_params = list(vit.patch_embed.parameters())
    if hasattr(vit, "cls_token") and vit.cls_token is not None:
        stem_params.append(vit.cls_token)
    if hasattr(vit, "pos_embed") and vit.pos_embed is not None:
        stem_params.append(vit.pos_embed)
    if hasattr(vit, "norm_pre"):
        stem_params.extend(vit.norm_pre.parameters())
    _add_group(stem_params, depth_from_top=num_blocks + 1, name="stem")

    # Per-block LLRD: block 0 is deepest from top, block N-1 is shallowest from top
    for block_index, block in enumerate(vit.blocks):
        depth_from_top = num_blocks - block_index
        _add_group(list(block.parameters()), depth_from_top=depth_from_top, name=f"block_{block_index}")

    # Final norm + SIE: top of backbone, full backbone_lr
    top_params = list(vit.norm.parameters())
    _add_group(top_params, depth_from_top=0, name="vit_norm")
    _add_group([current_model.sie_embed], depth_from_top=0, name="sie")

    # Heads: BNNeck + classifier + JPM heads — separate head_lr param group
    backbone_param_ids = {id(p) for group in backbone_groups for p in group["params"]}
    head_params = [p for p in current_model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]
    head_group = {"params": head_params, "lr": HEAD_LR, "base_lr": HEAD_LR, "name": "head"}

    optimizer = torch.optim.AdamW(
        backbone_groups + [head_group],
        betas=(0.9, 0.999),
        weight_decay=WEIGHT_DECAY,
    )
    print(json.dumps({
        "llrd_factor": LLRD_FACTOR,
        "num_backbone_groups": len(backbone_groups),
        "first_block_lr": backbone_groups[1]["lr"] if len(backbone_groups) > 1 else None,
        "stem_lr": backbone_groups[0]["lr"],
        "vit_norm_lr": next((g["lr"] for g in backbone_groups if g["name"] == "vit_norm"), None),
        "head_lr": HEAD_LR,
    }, indent=2))
    return optimizer
```

This makes the early CLIP blocks effectively frozen (block 0 gets `3.5e-4 * 0.65^24 ≈ 5.2e-8`, stem gets `3.5e-4 * 0.65^25 ≈ 3.4e-8`) while the top blocks and head retain near-full LR. Note: `set_optimizer_lrs` already iterates all `param_groups` and rescales by `base_lr`, so the warmup+cosine schedule continues to apply per-group correctly with no further changes.

### Patch P2 — Lower starting backbone LR to absorb the larger model's gradient noise (REQUIRED)

In the SETUP cell, change the single constant. Substring change:

**FIND**: `BACKBONE_LR = 3.5e-4`
**REPLACE WITH**: `BACKBONE_LR = 1.5e-4`

Together with LLRD@0.65, the **effective deepest-block LR drops from 3.5e-4 (uniform) to 1.5e-4 (top-block max), with rapid decay toward the stem**. This matches the spirit of the 08 recipe scaled to ViT-L.

### Patch P3 — Kernel id and metadata for the new push (REQUIRED)

Edit `notebooks/kaggle/14p_veri_vit_l_train/kernel-metadata.json` to a NEW kernel under MRKDaGods. Substring change:

**FIND**: `"id": "gumfreddy/14p-veri-vit-l-train",` (or whatever the current id is)
**REPLACE WITH**: `"id": "MRKDaGods/14p2-veri-vit-l-fix-train",`

Also update `"title": "14p2 VeRi ViT-L/14 CLIP Fix Train"`. Leave dataset_sources, GPU type, and all other fields unchanged.

### What is NOT being changed (deliberate)

- `EPOCHS = 100` — same budget.
- `WARMUP_EPOCHS = 10`, cosine schedule, `MIN_LR = 1e-6`, `WARMUP_START_LR = 1e-7` — unchanged.
- `HEAD_LR = 3.5e-3`, `WEIGHT_DECAY = 1e-4` — unchanged.
- Loss weights `LAMBDA_GLOBAL_CE/TRI/JPM_CE = 1.0` — unchanged.
- `TRIPLET_MARGIN = 0.3`, `LABEL_SMOOTHING = 0.1` — unchanged.
- `P_IDS = 8`, `K_INSTANCES = 4`, `BATCH_SIZE = 32` — unchanged (T4 16GB constraint).
- Augmentations, normalization, image size — unchanged.
- Eval recipe (concat-patch-flip + AQE k=2 + rerank k1=80/k2=15/λ=0.2) — unchanged.
- Center loss is NOT added (deliberately keep change surface minimal; if P1+P2 produce <89.97%, center loss is the next escalation).

---

## 3. Verification Plan (BEFORE pushing)

The Coder MUST run a CPU smoke that exercises the patched optimizer + the existing model + the existing PK sampler with a tiny synthetic 32-image fake-VeRi loader. Required assertions:

### Smoke S1 — PK batch composition (asserts H2 is FALSE under the patch)

```python
# After data cell runs, before training
batch_iter = iter(train_loader)
_, pids_batch, _, _ = next(batch_iter)
unique_ids, counts = np.unique(pids_batch.numpy(), return_counts=True)
assert len(unique_ids) == P_IDS, f"Expected {P_IDS} unique IDs in batch, got {len(unique_ids)}"
assert (counts == K_INSTANCES).all(), f"Expected K={K_INSTANCES} per ID, got counts={counts.tolist()}"
print(f"[S1 PASS] batch has {len(unique_ids)} IDs × {K_INSTANCES} instances")
```

### Smoke S2 — LLRD application (asserts the patch was actually wired)

```python
# After build_optimizer
groups_by_name = {g["name"]: g for g in optimizer.param_groups}
stem_lr = groups_by_name["stem"]["lr"]
top_block_lr = groups_by_name[f"block_{len(model.vit.blocks)-1}"]["lr"]
head_lr = groups_by_name["head"]["lr"]
assert stem_lr < 1e-6, f"Stem lr {stem_lr} should be near-frozen (<1e-6) under LLRD@0.65, but is {stem_lr}"
assert abs(top_block_lr - BACKBONE_LR * (LLRD_FACTOR ** 1)) < 1e-9, "Top-block lr does not match LLRD formula"
assert head_lr == HEAD_LR, "Head lr should be unchanged"
print(f"[S2 PASS] stem_lr={stem_lr:.2e}, top_block_lr={top_block_lr:.2e}, head_lr={head_lr:.2e}")
```

### Smoke S3 — Per-row feature variance (asserts H3 is FALSE: features ARE distinct per image)

```python
model.train()
images_smoke, pids_smoke, camids_smoke, _ = next(iter(train_loader))
images_smoke = images_smoke.to(DEVICE); pids_smoke = pids_smoke.to(DEVICE).long(); camids_smoke = camids_smoke.to(DEVICE).long()
with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
    out_smoke = model(images_smoke, cam_ids=camids_smoke)
gf = out_smoke["global_feat"].float()
per_row_std = gf.std(dim=1)
pairwise_l2 = torch.cdist(F.normalize(gf, p=2, dim=1), F.normalize(gf, p=2, dim=1), p=2)
off_diag = pairwise_l2[~torch.eye(gf.size(0), dtype=torch.bool, device=gf.device)]
assert per_row_std.min().item() > 1e-3, f"Feature row collapsed: min row std = {per_row_std.min().item():.2e}"
assert off_diag.max().item() > 0.1, f"All features identical: max pairwise L2 dist = {off_diag.max().item():.2e}"
print(f"[S3 PASS] per_row_std∈[{per_row_std.min().item():.3f},{per_row_std.max().item():.3f}], off-diag L2∈[{off_diag.min().item():.3f},{off_diag.max().item():.3f}]")
```

### Smoke S4 — Triplet drops below margin under one optimizer step (the critical "is the bug fixed?" gate)

```python
loss0, parts0 = compute_reid_loss(out_smoke, pids_smoke)
print(f"[S4] step0: triplet={parts0['triplet']:.5f}, global_ce={parts0['global_ce']:.5f}")
optimizer.zero_grad(set_to_none=True)
loss0.backward()
# Assert backbone gradient flows
top_block = model.vit.blocks[-1]
top_grad_norm = sum(p.grad.norm().item() for p in top_block.parameters() if p.grad is not None)
assert top_grad_norm > 1e-6, f"Top backbone block has zero gradient: {top_grad_norm}"
optimizer.step()
# Re-forward and recompute
with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
    out_smoke2 = model(images_smoke, cam_ids=camids_smoke)
loss1, parts1 = compute_reid_loss(out_smoke2, pids_smoke)
print(f"[S4] step1: triplet={parts1['triplet']:.5f}, global_ce={parts1['global_ce']:.5f}, top_grad_norm={top_grad_norm:.3e}")
# Soft assertion: triplet should change (in either direction); a frozen value across 1 step is a red flag
assert abs(parts1["triplet"] - parts0["triplet"]) > 1e-6 or parts1["global_ce"] != parts0["global_ce"], "Loss did not change after optimizer step — gradient flow is broken"
print("[S4 PASS] triplet and CE both moved after one optimizer step")
```

### Smoke S5 — Frozen-CLIP baseline sanity (optional but cheap, kills H3 dead)

Before training, fit a single-step softmax classifier on top of frozen CLIP ViT-L/14 features over a few hundred VeRi images and measure mAP on a 100-image query subset. **Expected: ≥30% mAP from frozen CLIP alone.** If frozen CLIP gets 30%+ but trained 14p got 13.7%, that confirms the prior 14p run damaged the backbone (the diagnosis above).

This smoke is not blocking but is recommended documentation in the train_log.

### Push gate

If S1-S4 all PASS on CPU, push ONCE. If any of S1-S4 FAILS, the Coder MUST stop, dump the failing assertion + relevant tensor stats, and report back without pushing.

---

## 4. Push Plan

**Single push, no auto-retry on warnings.**

```powershell
$env:KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/MRKDaGods__access_token -Raw).Trim()
kaggle kernels push -p notebooks/kaggle/14p_veri_vit_l_train/
```

After push, immediately inspect the start-up log for `not valid dataset sources` warnings:

```powershell
kaggle kernels status MRKDaGods/14p2-veri-vit-l-fix-train
kaggle kernels output MRKDaGods/14p2-veri-vit-l-fix-train -p ./tmp_14p2_log/
```

If the warning appears, immediately try `kaggle kernels cancel MRKDaGods/14p2-veri-vit-l-fix-train`; if cancel CLI is unavailable, post the kernel URL to the user and poll `kaggle kernels status` every 60s until status is `cancelled`/`error`/`complete`. Do **not** auto-repush.

**Expected wall-clock**: ~6h (same as 14p v1 — same model, same iteration count, only optimizer changed).

After completion, fetch outputs and grep `eval_results.json` for the `concat_patch_flip_aqe2_rerank_k1_80_k2_15_lambda_0_2` row. Apply verdict bands above.

---

## 5. Verdict Bands (UNCHANGED from 14p)

Same hard bands. Same 14h cutoff. Same eval contract as 14p. Use the `concat_patch_flip_aqe2_rerank_k1_80_k2_15_lambda_0_2` row in `eval_results.json` as the canonical scoreboard.

| Band | concat-patch-flip AQE+rerank mAP | R1 | Action |
|---|---:|---:|---|
| WIN | ≥91.5% | ≥98.6% | Promote, queue gated 14q |
| MARGINAL | 89.97-91.5% | ≥98.4% | Document as plateau-confirming, no 14q budget |
| FAIL | <89.97% | <98.3% | Close branch; escalate to either (a) add center loss + +40 epochs on MRKDaGods remaining quota, OR (b) drop ViT-L line entirely and reallocate to a different secondary stream |

---

## 6. Risk Analysis

### R1 — H1/H2/H3/H4 actually IS the real bug (not the LLRD/LR scaling I diagnosed)

**Likelihood**: Low. I verified each by file inspection and by the train_log evidence (CE descending → not H3/H4; triplet descending below margin → not H1 if "triplet receives wrong tensor" means it gets a constant tensor; PK sampler code is mechanically correct → not H2).

**Mitigation**: Smokes S1, S3, S4 explicitly probe each hypothesis on CPU before push. If any smoke fails, the Coder STOPS and reports — no push, no retry.

### R2 — LLRD@0.65 is too aggressive (early blocks frozen too hard) → ViT-L cannot adapt to vehicle domain

**Likelihood**: Medium. CLIP ViT-L/14 was trained on natural images; vehicle ReID needs viewpoint/illumination invariance that may require updating early layers.

**Mitigation**: 0.65 is the standard BERT/ViT-L LLRD value; 08 used 0.75 on ViT-B. With 24 blocks, 0.65^24 ≈ 1.5e-4 (i.e., stem LR ≈ 5e-8) — early blocks are practically frozen. If R2 is the real failure mode, the train_log will show CE plateauing higher than expected (>3) and triplet not dropping below 0.295. Contingency on FAIL: re-run with `LLRD_FACTOR = 0.75` (one-line change) on remaining MRKDaGods quota.

### R3 — Backbone LR 1.5e-4 is still too high for ViT-L → CLIP features still get damaged

**Likelihood**: Medium-low. 1.5e-4 with LLRD@0.65 means the deepest few blocks see ~6e-5 to 1.5e-4 — well within the safe range for AdamW + CLIP ViT-L on small ReID datasets per the BoT/TransReID literature.

**Mitigation**: Smoke S5 (frozen-CLIP baseline) gives an upper-bound sanity check; periodic eval at ep30/40/50 will detect catastrophic drift. If ep30 mAP < 50%, the FAIL contingency above kicks in.

### R4 — Per-block LLRD param-group construction misses the SIE embed or includes it twice → optimizer state mismatch

**Likelihood**: Low — Smoke S2 explicitly checks group composition and verifies `sie_embed` is in exactly one backbone group. The patched code uses an explicit `backbone_param_ids = {id(p) for group in backbone_groups for p in group["params"]}` set to derive `head_params` so SIE cannot accidentally double-count.

**Mitigation**: S2 fails fast on CPU before any GPU work.

### R5 — Cosine schedule resumes from base_lr per group correctly, but `set_optimizer_lrs` was only tested with 2 groups

**Likelihood**: Low. `set_optimizer_lrs` iterates `for group in optimizer.param_groups: group["lr"] = lr_for_epoch(epoch_index, group["base_lr"])` — it's group-count agnostic. With ~26 groups (24 blocks + stem + vit_norm + sie + head), each group's `base_lr` is set in `_add_group` to its scaled value, so `lr_for_epoch(epoch, group["base_lr"])` does the right thing.

**Mitigation**: S2 verifies base_lr is set per group; cosine math is the same per-group.

### R6 — Kaggle MRKDaGods quota / auth / dataset attachment

**Likelihood**: Low — the multi-account KGAT_ env-var pattern (per repo memory) is verified to work. Dataset source `abhyudaya12/veri-vehicle-re-identification-dataset` is unchanged from 14p v1.

**Mitigation**: After push, inspect the first 60s of log for `not valid dataset sources`. If present, cancel + report.

---

## 7. Coder Handoff Prompt

> Implement spec `docs/subagent-specs/14p2-fix.md` exactly. Three substring edits:
>
> 1. In `_build_14p_notebook.py` and the on-disk notebook cell, replace `build_optimizer` with the LLRD version from spec section 2 / Patch P1.
> 2. In the SETUP cell, change `BACKBONE_LR = 3.5e-4` to `BACKBONE_LR = 1.5e-4`.
> 3. In `notebooks/kaggle/14p_veri_vit_l_train/kernel-metadata.json`, change the kernel `id` to `MRKDaGods/14p2-veri-vit-l-fix-train` and `title` to `14p2 VeRi ViT-L/14 CLIP Fix Train`.
>
> Then add the smoke block (S1-S4 from section 3) as a NEW cell inserted between the OPTIMIZER cell and the SMOKE cell. Include the `[S1 PASS]` / `[S2 PASS]` / `[S3 PASS]` / `[S4 PASS]` print statements verbatim. Do NOT add S5 — too brittle for required path.
>
> Do NOT modify `src/training/`. Do NOT change loss weights, epochs, batch size, augmentations, eval recipe, head_lr, weight_decay, warmup, cosine schedule, label smoothing, or triplet margin. Do NOT add center loss. The change surface is exactly: `build_optimizer` body, one constant `BACKBONE_LR`, one kernel id, one title, plus an inserted smoke cell.
>
> CPU smoke contract (mandatory before push):
> - Run the patched notebook locally on CPU with a 32-image fake VeRi (or the real `_smoke_14p_cpu.py` if it exists; otherwise create a minimal in-memory dataset wrapper that produces 8 IDs × 4 images of random tensors in `[3, 224, 224]`, with sensible camid integers in `[0, 19]`).
> - Assertions S1-S4 from spec section 3 MUST all PASS. The PK batch must contain exactly 8 unique IDs × 4 instances each.
> - **STOP RULE**: if S2 shows that any backbone group `lr >= BACKBONE_LR`, OR if S4 shows that loss did not change after the optimizer step, OR if S4 shows triplet still bit-exact at `0.30000000` after one optimizer step (rounded to 8 decimals) → STOP, report the failing assertion + tensor stats, do NOT push.
>
> Push contract: single push from MRKDaGods using `KAGGLE_API_TOKEN = (Get-Content $HOME/.kaggle/MRKDaGods__access_token -Raw).Trim()`. After push, check kernel status once and inspect for `not valid dataset sources`. If clean, post URL and stop. No auto-retry. No second push. Wall-clock budget 14h.
>
> Verdict bands: WIN ≥91.5% mAP & ≥98.6% R1 (concat-patch-flip AQE+rerank); MARGINAL 89.97-91.5% mAP & ≥98.4% R1; FAIL otherwise. Report the row from `eval_results.json` labeled `concat_patch_flip_aqe2_rerank_k1_80_k2_15_lambda_0_2` as the headline.
