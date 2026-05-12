# 14t — VeRi-776 Single-Cam Score-Fusion: CLIP-SENet v6 × TransReID 09v

**Status**: PROPOSED (planner spec, not yet implemented)
**Author**: MTMC Planner, 2026-05-11
**Type**: Eval-time experiment, single Kaggle kernel, **no training**
**Estimated wall time**: <2h on T4
**Account**: MRKDaGods (~15h quota available)

---

## TL;DR

Score-fuse two existing VeRi-776-trained vehicle ReID checkpoints — **CLIP-SENet v6** (best post-rerank mAP **91.54%**, R1 **97.32%**) and **TransReID 09v v17** (best post-rerank mAP **89.97%** / R1 **97.80%**; joint optimum **89.71% mAP / 98.15% R1**) — to see whether their independent error modes complement each other and clear both parents on the VeRi-776 single-camera benchmark. **WIN gate: post-fusion mAP ≥ 91.54% AND R1 ≥ 98.33%.**

---

## Section 1 — Prior evidence

This combination has **never** been tested. The full prior fusion table for these two models:

| # | Experiment | Domain | Streams | Result | Source |
|---|------------|--------|---------|--------|--------|
| 1 | 13d v2 | **CityFlowV2 MTMC** | TransReID + DINOv2 + CLIP-SENet v6 (3-way) | Monotonic degradation; standalone CLIP-SENet 0.6855 MTMC IDF1; fusion peak = control 0.7679 at `w_cs=0` | `findings.md` "CLIP-SENet × CityFlowV2 Score-Level Fusion" |
| 2 | 13f / 13h | **CityFlowV2 MTMC** | TransReID + DINOv2 + CLIP-SENet **fine-tuned on CityFlow IDs** | Peak 0.7691 at `w_cs_ft=0.30`; below production 0.7703 | `findings.md` "CLIP-SENet × CityFlowV2 Fine-Tune Fusion" |
| 3 | 10c v15 / production | **CityFlowV2 MTMC** | TransReID + DINOv2 (2-way) | 0.7703 MTMC IDF1 | `findings.md` "Final Result" |
| 4 | 14j / 14k | **CityFlowV2 MTMC** | TransReID + DINOv2 + R50-IBN (3-way, FastReID quaternary) | K7 peak 0.78079, below 0.7810 WIN bar | `experiment-log.md` 2.17 |
| 5 | OSNet score-fusion | **CityFlowV2 MTMC** | TransReID + OSNet-VeRi | -0.8pp at w=0.10 | `findings.md` Dead Ends |
| 6 | **CLIP-SENet × TransReID on VeRi-776 single-cam** | **VeRi-776** | **CLIP-SENet v6 + TransReID 09v** | **NEVER TESTED** | — |

**Key gap identified**: Every prior CLIP-SENet × TransReID fusion has been on **CityFlowV2 MTMC** (cross-camera, domain-mismatched for CLIP-SENet). The **VeRi-776 single-cam** axis — where both models are in-domain — is novel territory.

**Why VeRi-776 single-cam is different from CityFlow MTMC**:
- 13d's failure was attributed to *domain gap* (CLIP-SENet is VeRi-776-trained, evaluated cross-domain on CityFlow). On VeRi-776 itself, both models are *fully in-domain*.
- CityFlow MTMC uses tracklet-mean-pooled features over cross-camera matching; VeRi-776 uses per-image queries against a fixed gallery. The failure mode is different.
- AIC22 1st place (0.8486 IDF1 on CityFlow) used a **5-model ensemble** where every model exceeded 70% VeRi-776 mAP. That precedent supports same-domain VeRi-776 fusion working when MTMC fusion does not.

---

## Section 2 — Why this might work (concrete arguments)

1. **Orthogonal architectures**:
   - **TransReID 09v v17**: pure ViT-B/16 CLIP backbone, 768-d CLS + JPM patch grouping + SIE camera embeddings. Patch-token attention captures global vehicle shape and part-level layout.
   - **CLIP-SENet v6**: ResNet101-IBN-a **CNN** appearance branch (2048-d) + TinyCLIP-ViT-medium-patch32 semantic branch (512-d) → concat → FC → unified embedding + **AFEM (G=32) squeeze-excitation channel re-weighting** → BNNeck. CNN inductive bias + SE attention captures local texture (grilles, wheels, logos) very differently from ViT patch tokens.
2. **Different error modes plausible**: TransReID 09v post-rerank tops out at **89.97% mAP / 97.80% R1**; CLIP-SENet v6 tops out at **91.54% mAP / 97.32% R1**. CLIP-SENet wins mAP by **+1.57pp** but loses R1 by **−0.48pp**. The fact that the **R1-vs-mAP Pareto frontier is non-dominated** between these two checkpoints is direct evidence that the two models disagree on a non-trivial subset of queries.
3. **Joint optimum exists**: 09v v17 already shows the joint optimum (`98.15% R1 / 89.71% mAP`) and the R1-leader (`98.33% / 85.14%`) come from different feature constructions (concat-patch-flip+AQE+rerank vs single-flip+rerank). Adding a second-model stream is a natural extension of the eval-time recipe family.
4. **Standard practice in AIC22 winners**: Top teams routinely fuse 3-5 ReID backbones on VeRi-776 to reach paper-claim 92-95% mAP / 98.5-99% R1 ranges. Single-pair fusion is a much weaker variant of this, but should show *some* lift if signal-additive.

---

## Section 3 — Why this might not work (concrete risks)

1. **Both models share CLIP pretraining**: TransReID 09v uses OpenAI CLIP ViT-B/16 init; CLIP-SENet v6 uses TinyCLIP-ViT-medium as its semantic branch. They may share enough representation bias that errors correlate too much for fusion to add value (analogous to the LAION-2B CLIP × OpenAI CLIP score fusion that regressed on CityFlow).
2. **Rerank saturation**: CLIP-SENet's 91.54% mAP comes from `rerank(k1=50, k2=10, λ=0.1) + AQE(k=10)`; TransReID's 89.97% from `rerank(k1=80, k2=15, λ=0.2) + AQE(k=3)`. Rerank uses k-reciprocal neighbour sets, which are already aggressive smoothers on the gallery. The post-rerank distance landscape is highly non-linear, and naive score-level averaging may *break* the rerank's neighbourhood consistency rather than improve it.
3. **R1 ceiling at 98.33% is hard**: Pushing R1 from 98.15% (joint optimum) or 98.33% (R1 leader) to >98.33% requires fixing ~3-7 hard queries out of ~1,678 VeRi-776 query set. Fusion may improve mAP (averaging easy decisions) without moving the top-1 needle.
4. **Different rerank/AQE optima**: The two models prefer different `(k1, k2, λ, aqe_k)`. A single set of post-fusion rerank params may underfit both, requiring a small grid sweep (cheap, ~20 configs).
5. **Concat-fusion historically hurts on CityFlow** (-1.6pp documented dead end). It may also hurt on VeRi-776 single-cam if dimensional scales are mismatched.

---

## Section 4 — Experiment design (14t)

### 4.1 Single Kaggle kernel
- **Slug**: `mrkdagods/14t-veri-clipsenet-transreid-fusion`
- **Title**: `14t VeRi-776 Fusion: CLIP-SENet × TransReID`
- **Notebook path**: `notebooks/kaggle/14t_veri_clipsenet_transreid_fusion/14t_veri_clipsenet_transreid_fusion.ipynb`
- **Account**: MRKDaGods (~15h quota; estimated run <2h since no training)
- **Hardware**: T4 (GPU only for feature extraction; fusion + rerank + AQE are CPU)
- **enable_gpu**: true, **enable_internet**: true

### 4.2 Data sources
```
dataset_sources:
  - abhyudaya12/veri-vehicle-re-identification-dataset   # VeRi-776 splits
  - mrkdagods/mtmc-weights                               # vehicle_transreid_vit_base_veri776.pth (TransReID 09v)
kernel_sources:
  - yahiaakhalafallah/13-clip-senet-train                # CLIP-SENet v6 best_mAP.pth
```

### 4.3 Pipeline

**Step 1 — Load both models** (GPU):
- TransReID 09v: load `vehicle_transreid_vit_base_veri776.pth` into the existing 09v eval architecture (ViT-B/16 CLIP + JPM + SIE-20cam + BNNeck, 768-d output). Mirror the 09v v17 forward pass that produces both `single_flip_cls` (768-d) and `concat_patch_flip` (1536-d) features.
- CLIP-SENet v6: load the v6 checkpoint (ResNet101-IBN-a + TinyCLIP-ViT-medium + AFEM(G=32) + BNNeck, 2048-d output). Mirror the 13e eval forward pass.

**Step 2 — Extract features** (GPU):
- For each of 1,678 query and ~11,579 gallery images:
  - Extract `transreid_feat_768` (single_flip CLS BNNeck output)
  - Extract `transreid_feat_1536` (concat_patch_flip, the joint-optimum feature)
  - Extract `clipsenet_feat_2048` (single_flip BNNeck output, image_size=320)
- L2-normalize all three per-row.
- Save raw `.npy` arrays under `/kaggle/working/14t_features/` for reproducibility.

**Step 3 — Fusion sweep** (CPU):

For each fusion strategy below, compute query-gallery cosine similarity matrix `S` of shape (1678, 11579), convert to distance `D = 1 - S`, and pass through optional AQE + rerank.

**Strategy A — L2-norm concatenation**:
- For each pair `(transreid_dim, clipsenet)`:
  - `f_concat = [α · transreid_feat ; (1-α) · clipsenet_feat_2048]`, L2-renorm
  - Use transreid_dim ∈ {768, 1536}
  - Sweep `α ∈ {0.3, 0.5, 0.7}` (3 configs × 2 transreid_dim = 6 concat configs)

**Strategy B — Score-level weighted sum** (primary strategy):
- `S_fused = w · S_transreid + (1-w) · S_clipsenet`
- For each `transreid_dim ∈ {768, 1536}`:
  - Sweep `w ∈ {0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0}` (9 configs × 2 = 18 score configs)
- `w=0.0` and `w=1.0` are the standalone-baseline drift gates; they MUST reproduce 91.54% (CLIP-SENet alone) and 89.97% (TransReID alone) respectively, ±0.1pp.

**Strategy C — Rank fusion (optional, cheap)**:
- Convert each `S_*` to per-query rank arrays, then `rank_fused = β · rank_transreid + (1-β) · rank_clipsenet`.
- Sweep `β ∈ {0.4, 0.5, 0.6}` (3 configs × 2 transreid_dim = 6 rank configs). Skip if Strategy B already clears WIN.

**Step 4 — Post-fusion AQE + rerank**:

Apply to **each** fusion output above:
- `aqe_k ∈ {2, 3}` (2 values)
- `rerank` fixed at `k1=80, k2=15, λ=0.2` (matches 09v joint optimum); also test the CLIP-SENet-preferred `k1=50, k2=10, λ=0.1` as a sanity row at the best Strategy B point only

Total post-fusion eval rows: (6 concat + 18 score + 6 rank) × 2 AQE = **60 main configs**, plus ~4 sanity rows. Each row takes ~30-60s on T4 (rerank is the bottleneck). Total wall <2h with margin.

**Step 5 — Verdict & summary**:
- Report Pareto-best `(mAP, R1)` from all rows.
- Emit `14t_summary.json` with: best row, all 60 rows, parents' standalone reproductions (drift gates), verdict band, fusion-direction analysis (does fusion help mAP, R1, or both?).

### 4.4 Required outputs (saved to `/kaggle/working/`)
```
14t_features/
  query_transreid_768.npy        # (1678, 768)
  query_transreid_1536.npy       # (1678, 1536)
  query_clipsenet_2048.npy       # (1678, 2048)
  gallery_transreid_768.npy      # (~11579, 768)
  gallery_transreid_1536.npy     # (~11579, 1536)
  gallery_clipsenet_2048.npy     # (~11579, 2048)
  query_pids.npy, query_camids.npy
  gallery_pids.npy, gallery_camids.npy
14t_fusion_results.json          # all 60+ rows with (strategy, params, mAP, R1, R5, R10)
14t_summary.json                 # best row + verdict + drift gates
recipe.json                      # exact pipeline parameters
```

### 4.5 Reproduction sanity checks (MUST pass before reporting verdict)
- **Drift gate TR-only**: `w=1.0` score fusion with `transreid_1536 + aqe_k=2 + rerank(80,15,0.2)` must reproduce **89.71% mAP / 98.15% R1** (09v v17 joint optimum) ±0.15pp. If drift exceeds, abort and report bug.
- **Drift gate CS-only**: `w=0.0` score fusion with `clipsenet_2048 + aqe_k=10 + rerank(50,10,0.1)` must reproduce **91.54% mAP / 97.32% R1** (CLIP-SENet v6 best) ±0.15pp. If drift exceeds, abort.
- If either drift gate fails: stop, write `drift_check_failed.json`, do not run fusion sweep.

---

## Section 5 — Verdict gate

| Band | Condition | Action |
|------|-----------|--------|
| **WIN** | Best fused row has **mAP ≥ 91.54% AND R1 ≥ 98.33%** | Promote as new VeRi-776 single-cam SOTA-recreation milestone. Update `findings.md` and `experiment-log.md`. Consider port to CityFlow MTMC (with caveat that 13d/13f already showed CLIP-SENet hurts MTMC). |
| **MARGINAL** | mAP ≥ 91.54% **OR** R1 ≥ 98.33% (one parent cleared, not both) | Document the Pareto-improvement axis. Single 30-min `k1, k2, λ` rerank micro-sweep around the best row. Do NOT chase further unless the micro-sweep clears WIN. |
| **FAIL** | Neither parent metric is exceeded | Close the VeRi-776 single-cam pairwise-fusion axis. Update `findings.md` Dead Ends. Both models' rerank-saturated features are mutually redundant under naive fusion. |

---

## Section 6 — Required checkpoints (verified accessible)

| Checkpoint | Source | Path in Kaggle | Confirmed? |
|------------|--------|----------------|------------|
| TransReID 09v ViT-B/16 CLIP (576 IDs, VeRi-776) | `mrkdagods/mtmc-weights` dataset | `/kaggle/input/mtmc-weights/vehicle_transreid_vit_base_veri776.pth` | YES — same dataset used by 09v v10/12/14/15/v17 (kernel `yahiaakhalafallah/09v-veri-776-eval-transreid-rerank`); confirmed in `notebooks/kaggle/09v_veri776_eval/kernel-metadata.json` |
| CLIP-SENet v6 (ResNet101-IBN-a + TinyCLIP-ViT-medium) | `yahiaakhalafallah/13-clip-senet-train` kernel output v6 | `/kaggle/input/13-clip-senet-train/best_mAP.pth` (or `best.pth`; resolve in notebook) | YES — used as `kernel_source` by 13c (CityFlow features), 13d (CityFlow fusion), 13e (VeRi-776 eval), 13f (CityFlow finetune); confirmed reproducible across 4 downstream kernels |

---

## Section 7 — Risk: CLIP-SENet v6 checkpoint availability

**Concern (from copilot-instructions)**: the OSNet `vehicle_osnet_veri776.pth` checkpoint (the lost "v80 78.4% MTMC IDF1" enabler) was dropped from `mrkdagods/mtmc-weights` on 2026-03-30. Is CLIP-SENet v6 also at risk?

**Verification (2026-05-11)**:
- CLIP-SENet v6 is stored as a **kernel output**, not as an entry in `mrkdagods/mtmc-weights`. Specifically, it lives under `yahiaakhalafallah/13-clip-senet-train` version 6.
- Kaggle persists kernel outputs indefinitely (or until the kernel author deletes them); kernel-output retention is more durable than dataset-version retention.
- The same checkpoint was successfully consumed by 4 downstream kernels (13c, 13d, 13e, 13f) over April-May 2026 with no access errors logged. The 13d v2 fusion run (2026-05-07) loaded v6 features without issue.
- **However**, the v7 retraining (`yahiaakhalafallah/13-clip-senet-train` version **7**) may have overwritten v6 if the kernel was force-pushed. **Action item before 14t push**: run `kaggle kernels output yahiaakhalafallah/13-clip-senet-train -v 6 --path /tmp/14t_v6_check/` locally and verify `best_mAP.pth` is downloadable and ~370MB. If v6 is no longer retrievable, fall back to the most recent checkpoint matching the v6 recipe (320² P=8/K=8 24-epoch) before re-running 13c/13d.
- **Local backup recommendation**: download v6 once and add it to a private Kaggle dataset under MRKDaGods's account so 14t and any future fusion experiment are not dependent on the upstream kernel output retention.

**Fallback plan if v6 is lost**: re-run `notebooks/kaggle/13_clip_senet_train` v6 recipe (24 epochs, 320², P=8/K=8, Adam 5e-4, cosine + 5ep warmup, RandomErasing + HFlip + Pad + RandomCrop, AMP fp16) on MRKDaGods. ~4.5h on P100. Defer 14t until v6 is restored.

---

## Open questions for user / planner before push

1. Should we test concat-fusion at all (Strategy A), given concat has been a -1.6pp dead end on CityFlow MTMC? **Default**: include it on VeRi-776 since the single-cam failure mode is different; mark as low-confidence.
2. Should rank fusion (Strategy C) be in v1 or held for a v2 follow-up? **Default**: include in v1 since it costs only 6 extra rows.
3. Should we additionally probe a 3-way fusion (TransReID + CLIP-SENet + DINOv2 ViT-B/14 from 14r-probe at 89.27% / 98.15%) in v1? **Default**: NO — keep v1 strictly pairwise. If 14t scores WIN, queue a separate 14u spec for 3-way fusion.

---

## Files (to be created in implementation phase)

- `docs/subagent-specs/14t-veri-clipsenet-transreid-fusion.md` ← **this document**
- `notebooks/kaggle/14t_veri_clipsenet_transreid_fusion/14t_veri_clipsenet_transreid_fusion.ipynb` (not yet built)
- `notebooks/kaggle/14t_veri_clipsenet_transreid_fusion/kernel-metadata.json` (not yet built)
- `_build_14t_notebook.py` (notebook builder, follows the 14p/14q/14r pattern)

Implementation deferred. Planner recommendation in cover message.
