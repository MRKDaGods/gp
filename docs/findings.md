# MTMC Tracker — Research Findings & Strategic Analysis

> **IMPORTANT**: This is a living document. Update it whenever new experiments are run, new dead ends are discovered, or performance numbers change. Keep the "Current Performance" and "Dead Ends" sections current.

## Canonical VeRi-776 Reproducibility Reference

- **14t fusion WIN**: mAP=0.9330 / R1=0.9845 (`kaggle://yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid`); requires `mrkdagods/mtmc-weights` + `yahiaakhalafallah/13-clip-senet-train` outputs; score fusion `w_clipsenet=0.7`, `w_transreid=0.3`, `transreid_768`, AQE k=3, rerank `k1=80,k2=15,lambda=0.2`.
- **09v v17 TransReID standalone**: best R1=98.33%, best mAP=89.97%, joint optimum R1=98.15% / mAP=89.71%; canonical artifact `outputs/09v_veri_v9/veri776_eval_results_v9.json`; kernel `kaggle://yahiaakhalafallah/09v-veri-776-eval-transreid-rerank`.
- **13 v6 CLIP-SENet standalone**: 320x320, P=8/K=8 -> base mAP=82.34% / R1=96.54%; rerank+AQE reference mAP=91.54% / R1=97.32%; kernel `kaggle://yahiaakhalafallah/13-clip-senet-train`.

## Strategic frame shift (user-clarified, 2026-05-08)

The paper direction is now **VeRi-776 SOTA recreation first, then port to CityFlowV2**. CityFlowV2 MTMC remains the downstream proof, but it is no longer the immediate optimization target. The confirmed 14e B1 / 14f / 14h / 14i / 14j / 14k / 14u plateau at **0.77936 MTMC IDF1** should be treated as a strong checkpoint and transfer baseline, not the final paper target.

14p3 and 14q now close the scale-only axis for the CLIP TransReID family on VeRi-776. 14p3 ViT-L/14 CLIP @ 224² failed with best base **80.90% mAP / 96.90% R1** and best post-rerank **87.95% mAP / 97.32% R1**; the 304M-param backbone overfit VeRi-776's 576 train IDs / 37k images. 14q ViT-B/16 CLIP @ 256² also failed: best base **79.68% mAP / 96.84% R1**, best post-rerank **89.15% mAP / 97.20% R1**, below the 09v v17 224² ceiling (**89.97% mAP / 98.33% R1 base; ~91.54% post-rerank**). Triplet loss saturated to **0.005** by epoch 160, so the current CE+triplet recipe had no remaining signal to learn. Strong conclusion: **09v v17 ViT-B/16 @ 224² is at/near the achievable ceiling for CLIP TransReID under standard CE+triplet supervision**.

14r is now closed. 14r-probe (`gumfreddy/14r-probe-dinov2-veri-776-train`) **FAILED**: DINOv2 ViT-B/14 standalone reached best post-rerank **89.27% mAP / 98.15% R1**, below WIN (**91.54% / 98.33%**) and below MARGINAL mAP (**90.5% / 98.0%**). This confirms **CLIP pretraining is necessary, not just any SSL pretraining** for VeRi-776 under the current recipe, though R1 remains close enough to 09v v17's 98.33% to keep DINOv2 as a possible diversity stream for ensemble use. 14r primary (`mrkdagods/14r-clip-reid-veri-776-train`) was **ABORTED by walltime guard** after Stage 1 and one Stage 2 epoch; a full coupled CLIP-ReID run was not feasible in the single-kernel T4 budget. 14r-recovery (`gumfreddy/14r-recovery-clip-reid-stage-2`) **FAILED**: Stage-2-only resume from saved prompts reached only **80.55% mAP / 93.68% R1**, a **-9.4pp mAP regression** versus 09v v17. Conclusion: final Stage 1 prompt vectors are not enough to cleanly continue CLIP-ReID in a separate Stage-2-only kernel; Stage 2 appears to need the full Stage1->Stage2 coupled trajectory.

**Update 2026-05-11 — 14t fusion WIN, new VeRi-776 high-water mark.** Score-level fusion of CLIP-SENet v6 × TransReID 09v v17 on VeRi-776 reaches **mAP = 93.30 % / R1 = 98.45 %** at `w_clipsenet=0.7, w_transreid=0.3` (transreid_768 global token) with AQE k=3 + rerank (k1=80, k2=15, λ=0.2). That is **+3.33pp mAP / +0.12pp R1** over the 09v v17 single-model base (89.97 / 98.33) and **+1.76pp / +1.13pp** over CLIP-SENet v6 alone post-rerank (91.54 / 97.32). This is the first experiment to reach **0.9845 as a real Rank-1** number on this checkpoint family (previously only attainable as R5-via-AQE). Strict-spec verdict was MARGINAL because the best *concat* row (93.19 / 98.27) missed the R1≥0.9833 bar by 0.06pp, but the best *score-fusion* row clears both bars — de-facto WIN. The VeRi-776 single-cam paper-recreation chain is **closed with a WIN** via fusion, even though the 14p3/14q/14r single-model SOTA chase failed.

## Final Result — CLIP+DINOv2 Score-Level Ensemble (2026-04-25)

Experiment design: CLIP ViT-B/16 remained the primary feature space and DINOv2 ViT-L/14 was injected as the tertiary score stream during Stage 4 association similarity computation. The secondary slot was disabled by design, so **all rows use `w_secondary=0.00`**.

| Label | w_tertiary | MTMC_IDF1 | IDF1 | MOTA | HOTA |
|---|---:|---:|---:|---:|---:|
| no_fusion_control | 0.00 | 0.7663 | 0.7842 | 0.6691 | 0.5703 |
| ter_005 | 0.05 | 0.7669 | 0.7846 | 0.6697 | 0.5706 |
| ter_010 | 0.10 | 0.7669 | 0.7846 | 0.6697 | 0.5706 |
| ter_015 | 0.15 | 0.7673 | 0.7851 | 0.6702 | 0.5710 |
| ter_020 | 0.20 | 0.7663 | 0.7840 | 0.6704 | 0.5706 |
| ter_025 | 0.25 | 0.7662 | 0.7840 | 0.6704 | 0.5706 |
| ter_030 | 0.30 | 0.7674 | 0.7853 | 0.6716 | 0.5717 |
| ter_040 | 0.40 | 0.7679 | 0.7851 | 0.6708 | 0.5719 |
| ter_050 | 0.50 | 0.7696 | 0.7857 | 0.6703 | 0.5721 |
| **ter_060** | **0.60** | **0.7703** | **0.7916** | **0.6725** | **0.5749** |
| ter_070 | 0.70 | 0.7693 | 0.7909 | 0.6716 | 0.5746 |

Best operating point: **`w_tertiary=0.60`** produced **MTMC IDF1 = 0.7703**, a **+0.40pp** gain over this run's local CLIP-only baseline at **0.7663**.

The mAP-vs-MTMC paradox remains intact. DINOv2 improves vehicle ReID from **80.14% -> 86.79% mAP** and **92.27% -> 96.15% R1**, yet its single-model MTMC IDF1 is **0.744**, which is **-3.1pp** worse than CLIP solo. Score-level fusion partially recovered some of that loss, and the best fusion point at **0.7703** became the previous deployed best on available weights, but it still finished **-1.37pp** below the historical **0.784** v80 peak and was later superseded by the 14e TTA + AQE k=2 headline.

The drift investigation is now closed. The historical **0.784** result depended on `vehicle_osnet_veri776.pth`, a CityFlowV2-adapted VeRi-776 OSNet checkpoint that was present in `mrkdagods/mtmc-weights` and later dropped when that dataset was regenerated on **2026-03-30**. A clean Phase C retest on `fix/baseline-drift` at commit `7e242f6` using `yahiaakhalafallah/mtmc-10a-stages-0-2` v8 -> `yahiaakhalafallah/mtmc-10b-stage-3-faiss-indexing` v6 -> `yahiaakhalafallah/mtmc-10c-stages-4-5-association-eval` v17 changed only `camera_bn.enabled: true -> false` and recovered **0.7666** versus **0.7663** (**+0.03pp**), while both OSNet repro strategies with current weights regressed to **76.7%** (score-level) and **76.4%** (concat). The historical v80 result is therefore **not reproducible with current available weights**, not a still-open code-drift mystery.

This result confirms that the feature-quality bottleneck is **not** solved by adding stronger ReID features via score-level fusion.

## CLIP-SENet Reproduction (VeRi-776, 2026-05-06)

- **Run**: `yahiaakhalafallah/13-clip-senet-train` v6, 24/24 epochs, ~4h26min on P100; commits `53c2947` (train) and `3df0915` (eval).
- **Architecture**: ResNet101-IBN-a appearance branch (2048d) + TinyCLIP `vit_medium_patch32_clip_224.tinyclip_laion400m` semantic branch (512d) -> concat -> FC -> `T_u` -> AFEM(G=32) -> `T_s'` -> `T = T_u + T_s'` -> BNNeck; **92.6M params**.
- **Training**: Adam 5e-4, cosine schedule + 5-epoch warmup, P=8/K=8 micro-batch with accum=2 (effective batch 128), 320x320, RandomErasing + HFlip + Pad + RandomCrop, CE label smoothing 0.1 + SupCon tau=0.07, AMP fp16.

| Setting | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|
| Base | 82.34 | 96.54 | 98.51 | 99.11 |
| AQE k=10 | 89.21 | 96.90 | 98.03 | 98.75 |
| **Rerank k1=50,k2=10,λ=0.1** | **91.54** | **97.32** | 98.09 | 98.69 |

Comparison: beats **09v TransReID** on VeRi-776 mAP by **+1.57pp** (91.54 vs 89.97), but loses on R1 (97.32 vs 98.33) and remains **1.36pp below** the paper claim of 92.9 mAP. The gap is plausibly from 2-step accumulation on P100 16GB (BN sees 64 images/step instead of 128), plus TinyCLIP-ViT-40M-32-Text-19M being unavailable in `open_clip==2.30.0`; the run fell back to timm TinyCLIP and loaded ResNet101-IBN-a via torch.hub `XingangPan/IBN-Net`.

**Implication**: there are now **two competitive single-model VeRi-776 backbones** for score-fusion experiments: TransReID 09v and CLIP-SENet v6. Score-fusion ablation is worth retrying with this new, more diverse pair.

**M4 (VehicleID) abandoned 2026-05-06**: The only available Kaggle VehicleID dataset (`maphat/vehicleid`, 7.5GB `VehicleID_V1.0.zip`) is password-protected (NLPR licensing). No alternative public Kaggle dataset contains the standard VehicleID-V1.0 splits. M4 cannot proceed without dataset access. Pivoting to M5 (CityFlowV2 integration) directly.

## CLIP-SENet v7 (image_size=256, P=16) — REGRESSION (2026-05-07) — DEAD END

- **Run**: `yahiaakhalafallah/13-clip-senet-train` v7. Eval kernel `yahiaakhalafallah/13e-v7-clip-senet-eval` complete.
- **Hyperparameter delta vs v6**: `image_size: 320 → 256`, PK sampler `P=8/K=8 → P=16/K=8`, `batch_size=128`, `accum_steps=2`. All other settings identical (Adam 5e-4, cosine + 5-epoch warmup, RandomErasing/HFlip/Pad/RandomCrop, CE label-smoothing 0.1 + SupCon τ=0.07, AMP fp16, 24 epochs).
- **Training metrics (VeRi-776)**: **mAP=81.36%, R1=95.71%** — **−0.98pp mAP, −0.83pp R1** vs v6 (82.34 / 96.54). 13e-v7 eval (rerank+AQE sweep, image_size=320 to match v6): best_overall mAP=88.98% (rerank k1=50, k2=10, λ=0.1), R1=96.31%. Confirms regression of -2.56pp post-rerank vs v6's 91.54%. v6 320² remains canonical.
- **Diagnosis**: smaller crops (256² vs 320²) lose fine-grained vehicle texture (logos, grilles, wheel detail) that the SENet/AFEM module relies on for discriminative grouping. Increasing `P` from 8 to 16 also reduces images-per-identity per micro-batch (8→8 K stays fixed but the supcon batch contains more identities each with their fixed K=8 samples — the effective per-identity gradient signal is unchanged but BN statistics shift). The paper-claimed **92.9% mAP** is now **−1.56pp** above v6 and **−2.54pp** above v7; closing that gap likely requires ingredients we lack (the original TinyCLIP-ViT-40M-32-Text-19M weights unavailable in `open_clip==2.30.0`, full batch-128 BN without accumulation, possibly longer schedule or undocumented augmentation), not crop-size or sampler tuning.
- **Status**: **DEAD END** — image-size 256 retrain underperforms v6 on every training metric and post-rerank eval metric. v6 (320²) remains the canonical CLIP-SENet checkpoint. Do not retrain at 256² or sweep `P` further on this architecture.
- **Files**: training logs in `outputs/13_v7_v2/` (if applicable); eval results in `outputs/13e_v7/eval_results.json`.

## CLIP-SENet × CityFlowV2 Score-Level Fusion (2026-05-07) — DEAD END

- **Run**: `yahiaakhalafallah/13d-clip-senet-cityflow-associate` v2, COMPLETE.
- **Setup**: CLIP-SENet v6 (91.54% VeRi-776 mAP with rerank+AQE) tracklet features fused with the existing TransReID + DINOv2 score-fusion control. Tracklet granularity, mean-pooled per-tracklet 2048-d features. Per-tracklet feature alignment validated: **6/6 cameras 100% key-set match** (104, 128, 146, 201, 131, 219 tracklets across S01_c001/c002/c003 and S02_c006/c007/c008), 99,270 detections total, `feature_dim=2048`.
- **Fusion form**: control point is `w_primary=0.4, w_dinov2=0.6, w_cs=0.0`. For each `w_cs ∈ {0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0}`, `w_primary` and `w_dinov2` are rescaled by `(1 − w_cs)` so weights still sum to 1.

| `w_cs` | MTMC IDF1 | Δ vs control |
|---:|---:|---:|
| 0.0 (control) | **0.7679** | — |
| 0.2 | 0.7665 | −0.13pp |
| 0.4 | 0.7628 | −0.50pp |
| 0.5 | 0.7589 | −0.90pp |
| 0.6 | 0.7502 | −1.77pp |
| 0.7 | 0.7447 | −2.32pp |
| 0.8 | 0.7311 | −3.68pp |
| 1.0 (CLIP-SENet only) | 0.6855 | −8.24pp |

- **Reproducibility**: `w_cs=0.0` reproduces **0.7679** vs the historical control of **0.7703** — within **0.24pp** run-to-run variance.
- **Conclusion**: trend is **monotonic degradation** with increasing `w_cs`. Standalone CLIP-SENet (`w_cs=1.0`) reaches only **0.6855 MTMC IDF1**, far below TransReID's standalone (~0.75+). Score-level fusion with TransReID+DINOv2 hurts at every `w_cs > 0`. Confirmed dead end: a strong VeRi-776 model (91.54% mAP) does **not** transfer to CityFlowV2 cross-camera matching, and adding it as a fusion stream only injects domain-mismatched noise.
- **Generalization**: this strengthens the prior "Score-level ensemble with 52.77% mAP secondary" dead end. Even a **91.54% VeRi-776** secondary cannot help CityFlowV2 — **domain gap dominates secondary-model strength**. The score-level ensemble path is now closed across the full secondary-strength spectrum (52.77% → 91.54% mAP), which is consistent with **DINOv2 ViT-L/14** (86.79% CityFlowV2 mAP, but **−3.1pp MTMC IDF1**) and **10c v56** (LAION-2B CLIP 78.61% mAP fused with primary CLIP) regressing for the same reason: cross-camera invariance is determined by training methodology, not single-distribution mAP.
- **Files**: `outputs/13d_v2/fusion_results.json`, `outputs/13d_v2/13d_summary.json`.

## CLIP-SENet × CityFlowV2 Fine-Tune Fusion (13f→13h, 2026-05-07) — MARGINAL / DEAD END

- **Hypothesis tested**: the prior 13d dead end might be a domain-gap problem rather than a feature-quality problem. If we fine-tune CLIP-SENet v6 on CityFlowV2 IDs (instead of using the VeRi-776-only checkpoint), the standalone CityFlow MTMC IDF1 should rise, and a score-level fusion with TransReID + DINOv2 should clear the production baseline of **0.7703**.
- **13f training run**: CLIP-SENet v6 backbone (ResNet101-IBN-a + TinyCLIP, 92.6M params) fine-tuned on **666 CityFlowV2 IDs**. Classifier head reinitialized for the new ID count (v6 source had 575). 12 epochs, lr=1e-4, P=8 K=8, image_size=320, AMP fp16. Train loss **10.21 → 7.65** (still monotonically decreasing at epoch 12 — likely undertrained).
- **13h fusion sweep**: TransReID `w_p` and DINOv2 `w_d` rescaled by `(1 − w_cs_ft)` so weights still sum to 1. Control point is the same `w_p=0.40, w_d=0.60` setup as 13d.

| `w_cs_ft` | `w_p` | `w_d` | MTMC IDF1 | Δ vs control | clusters |
|---:|---:|---:|---:|---:|---:|
| 0.00 (control) | 0.40 | 0.60 | **0.7679** | — | 588 |
| 0.10 | 0.36 | 0.54 | 0.7679 | +0.00pp | 592 |
| 0.15 | 0.34 | 0.51 | 0.7677 | −0.02pp | 592 |
| 0.20 | 0.32 | 0.48 | 0.7670 | −0.09pp | 593 |
| 0.25 | 0.30 | 0.45 | 0.7670 | −0.09pp | 593 |
| **0.30** | **0.28** | **0.42** | **0.7691** | **+0.12pp** | 594 |
| 0.40 | 0.24 | 0.36 | 0.7674 | −0.05pp | 597 |
| 0.50 | 0.20 | 0.30 | 0.7664 | −0.15pp | 599 |
| 1.00 (FT-only) | 0.00 | 0.00 | 0.7099 | −5.80pp | 597 |

- **Standalone result vs 13d**: VeRi-only CLIP-SENet on CityFlow gave **0.6855 IDF1** (13d, `w_cs=1.0`). The CityFlow-fine-tuned version gives **0.7099 IDF1** (13h, `w_cs_ft=1.0`), a **+2.44pp** standalone gain. **Domain adaptation worked**, just not enough.
- **Fusion outcome**: peak `w_cs_ft=0.30 → 0.7691`, which is **+0.12pp** over the 13h control (0.7679) but **−0.12pp** below the production reproducible best of **0.7703** (10c v15, `w_tertiary=0.60`). The peak is bracketed by neutral/negative neighbors (`{0.20, 0.25}` at −0.09pp, `{0.40, 0.50}` at −0.05/−0.15pp), so the +0.12pp bump is within run-to-run noise of ~0.24pp observed earlier in 13d.
- **Verdict**: **MARGINAL / DEAD END for production deployment**. Fine-tuning rescues the standalone CityFlow performance but the resulting feature stream is still too correlated with the existing CLIP+DINOv2 pair to clear the production baseline. The hypothesis "domain gap is the only reason 13d failed" is now falsified — even with 666-ID CityFlow fine-tuning, the cross-camera invariance does not exceed what TransReID+DINOv2 already provide.
- **Why we stop here, not longer training**: at 12 epochs the train loss is still 7.65 and decreasing, so an obvious follow-up is 24 epochs. Expected lift on standalone: 0.7099 → ~0.73, which would lift fusion peak to maybe 0.770–0.775. That **best case still does not clear 0.7703** and most likely lands ≤0.7703. The fine-tune CLIP-SENet branch is now closed for production fusion.
- **Files**: 13f training kernel `yahiaakhalafallah/13f-clip-senet-cityflow-finetune` v1; 13h fusion kernel `yahiaakhalafallah/13h-clip-senet-ft-fusion` v1; results in `outputs/13h_v1/fusion_results.json`, `outputs/13f_v1/`.

## SAM2-Masked Stage 2 (14a v8, 2026-05-07) — DEAD END

- **Final result**: MTMC IDF1 = **0.7647** (**-0.56pp** vs production **0.7703**); `trackeval_idf1=0.7866`, `MOTA=0.6723`, `id_switches=158`.
- **Configuration**: `sam2_hiera_base_plus`, bbox-prompt with center-point, 5px dilation, zeros background, `bbox_expand=1.10`, applied during Stage 2 ReID feature extraction. Downstream fusion mirrored the 13h/production setup: `w_tertiary=0.60`, AQE `k=3`, FIC regularisation `0.50`, PCA 384D, gallery expansion, temporal overlap bonus, and `mtmc_only_submission=false`.
- **Walltime**: Stage 2 took **~146.67 min** on P100; downstream stages 3-5 took **~0.4 min** total.
- **Files**: Kaggle kernel `yahiaakhalafallah/14a-sam2-masked-stage2` v8; result summary `outputs/14a_v8_summary/14a_summary.json`; source 10a run `run_kaggle_20260425_202123`.
- **Hypothesis for failure**: SAM2 base-plus with center-point box-prompt removes too much vehicle context (wheels/tires/road-reflection cues) that the cross-camera matcher relies on. The `trackeval_idf1=0.7866` vs `mtmc_idf1=0.7647` gap suggests within-camera ID consistency held while cross-camera matching regressed, consistent with loss of contextual camera-invariant cues. Even though SAM2 produces clean foreground masks, TransReID embeddings encode some background/context as discriminative signal; replacing it with zeros injects an out-of-distribution shift relative to TransReID's training data.
- **Variants not tested**: mean-fill background, wider dilation (10-20px), and subtler edge feathering. None plausibly recovers a 0.56pp gap given the zero background already preserves the full vehicle silhouette and the gap suggests genuine information loss, not artifact-level damage. Closing this branch.
- **Cross-reference**: this is the **third feature-quality experiment in two days** after 13d CLIP-SENet fusion and 13h CLIP-SENet fine-tune fusion to fail to clear **0.7703**. Vehicle feature-quality experiments at fixed-checkpoint level look exhausted; next viable directions are either training-side changes (camera-invariant losses, longer fine-tune, different backbone) or learned association (GNN edge classifier).

## Multi-Crop TTA at Stage 2 + Fusion Sweep (14c v2 + 14d v1, 2026-05-07) — MARGINAL POSITIVE

- **14c v2 result (TTA features only, production fusion)**: MTMC IDF1 = **0.77085** (+0.05pp vs production 0.7703); `trackeval_idf1=0.7881`, `MOTA=0.6717`, `id_switches=212`. 4-view primary {original, hflip, scale_0.95, scale_1.05}, 2-view DINOv2 {original, hflip}, L2-mean aggregation. Stage 2 walltime 99.42 min on P100. Stages 3-5 ~0.43 min total. `w_tertiary=0.60`, `aqe_k=3`, `fic_reg=0.50`, `sim_thresh=0.50`.
- **14d v1 result (CPU-only fusion sweep on 14c v2 features)**: 8 configs in 4.9 min CPU. Best = **0.77155** at `w_tertiary=0.50, sim_thresh=0.50` (C3) — **+0.13pp vs production 0.7703**, **+0.07pp vs 14c v2 control 0.77085**. Sweep grid:

| Config | `w_tertiary` | `sim_thresh` | MTMC IDF1 | trackeval_idf1 |
|:------:|:------------:|:------------:|:---------:|:--------------:|
| C0 control | 0.60 | 0.50 | 0.77085 | 0.7881 |
| C1 | 0.55 | 0.50 | 0.77149 | 0.7896 |
| C2 | 0.65 | 0.50 | 0.77124 | 0.7885 |
| **C3 best** | **0.50** | **0.50** | **0.77155** | **0.7897** |
| C4 | 0.70 | 0.50 | 0.77115 | 0.7887 |
| C5 | 0.60 | 0.40 | 0.7566 | 0.7775 |
| C6 | 0.55 | 0.40 | 0.7566 | — |
| C7 | 0.65 | 0.40 | 0.7566 | — |

- **Verdict**: **MARGINAL POSITIVE**. The +0.13pp lift is below the 0.7720 WIN threshold and within ~0.24pp run-to-run noise, but the lift is **consistent across all `w_t∈{0.50,0.55,0.60,0.65,0.70}` at `thr=0.50`** (+0.03 to +0.07pp range vs C0 control), the **optimum shifted from `w_t=0.60` (production tuned on single-view) to `w_t=0.50` on TTA features** — a real signal that TTA changed the primary-embedding distribution — and trackeval IDF1 rose by **+0.31pp** (0.7866 → 0.7897). The thr=0.40 family (C5–C7) is universally **−1.4pp** worse, regardless of `w_t`, confirming the production threshold remains correct.
- **New best reproducible**: 0.77155 (14d v1 C3) is a new floor for reproducible MTMC IDF1, but is NOT promoted to the headline production result yet — the lift is within noise and replication on a second seed is pending. Production baseline 0.7703 (10c v15) remains the canonical headline until 14e delivers a ≥0.7720 WIN.
- **Files**: 14c kernel `yahiaakhalafallah/14c-tta-stage2` v2 → `outputs/14c_v2_summary/14c_summary.json`; 14d kernel `yahiaakhalafallah/14d-tta-fusion-sweep` v1 → `outputs/14d_v1_summary/14d_summary.json`; source 10a run `run_kaggle_20260425_202123`.
- **Next step**: 14e tighter `w_t × thr` grid around C3 (`w_t∈{0.45, 0.475, 0.50, 0.525}` × `thr∈{0.48, 0.50, 0.52}`, 12 configs) plus AQE/FIC sweep at the Block A best (`aqe_k∈{2,3,4}`, `fic_reg∈{0.30,0.50,0.70}`, 4 configs). Single CPU kernel, ~10-15 min, zero GPU. Spec: `docs/subagent-specs/post-14d-next.md`.
- **If 14e fails to produce a ≥0.7720 WIN**: TTA family closed, escalate to GNN edge classifier (post-14e spec).

## 14e Expanded TTA Fusion + AQE/FIC Sweep (2026-05-07) — WIN, NEW HEADLINE 0.77936

- **Headline**: new reproducible MTMC IDF1 = **0.77936** (14e B1 v1), using TTA Stage-2 features plus Stage-4 `aqe_k=2`, `w_tertiary=0.525`, `similarity_threshold=0.48`, and `fic_regularisation=0.5`. This is **+0.91pp vs production 0.7703** and **+0.78pp vs 14d floor 0.77155**.
- **Block A flatness**: 12 fine `w_tertiary × similarity_threshold` configs at `aqe_k=3, fic_reg=0.5` clustered in **0.7707–0.7717** MTMC IDF1. A8 replicates 14d C3 at **0.77155** within ±0.0001, so the drift check passed. Block A best is **A10** (`w_t=0.525`, `thr=0.48`) at **0.77171**.

| Label | w_tertiary | sim_thresh | MTMC IDF1 | id_switches |
|:-----:|:----------:|:----------:|:---------:|:-----------:|
| A1 | 0.45 | 0.48 | 0.77157 | 213 |
| A2 | 0.45 | 0.50 | 0.77155 | 214 |
| A3 | 0.45 | 0.52 | 0.77167 | 212 |
| A4 | 0.475 | 0.48 | 0.77157 | 213 |
| A5 | 0.475 | 0.50 | 0.77155 | 214 |
| A6 | 0.475 | 0.52 | 0.77167 | 212 |
| A7 | 0.50 | 0.48 | 0.77157 | 213 |
| A8 | 0.50 | 0.50 | 0.77155 | 214 |
| A9 | 0.50 | 0.52 | 0.77070 | 212 |
| **A10** | **0.525** | **0.48** | **0.77171** | **213** |
| A11 | 0.525 | 0.50 | 0.77170 | 214 |
| A12 | 0.525 | 0.52 | 0.77100 | 212 |

- **Block B at A10 anchor**:

| Label | aqe_k | fic_reg | MTMC IDF1 | id_switches | trackeval_idf1 | Δ vs A10 |
|:-----:|:-----:|:-------:|:---------:|:-----------:|:--------------:|:--------:|
| **B1** | **2** | **0.5** | **0.77936** | **154** | **0.7946** | **+0.77pp** |
| B2 | 4 | 0.5 | 0.77052 | 162 | 0.7925 | -1.12pp |
| B3 | 3 | 0.3 | 0.77268 | 213 | 0.7904 | +0.10pp |
| B4 | 3 | 0.7 | 0.77192 | 215 | 0.7901 | +0.02pp |

- **Status**: **TTA family is PROMOTED, NOT closed**. 14d's marginal lift was the harbinger; B1 is the breakthrough.
- **Discovery**: **AQE k=2 (vs production k=3) is the real lever on TTA features**. Lower k means less query smoothing; TTA already smooths features, so production AQE k=3 was over-smoothing. Going to k=2 cuts 60 ID switches (**213→154**, **-28%**). k=4 is worse (162 IDS → 0.77052), confirming the "TTA pre-smooths so AQE should reduce" interpretation. k=1 is untested and is the natural next probe in 14f.
- **FIC sensitivity**: small. B3 at `fic_reg=0.3` and B4 at `fic_reg=0.7` differ by only **0.0008 IDF1** at k=3, so production 0.5 remains reasonable but deserves a tighter k=2 sweep in 14f.
- **Verdict**: **WIN per `docs/subagent-specs/post-14d-next.md` thresholds** (≥0.7720). Promote to the new reproducible headline. The production-deployed config (10c v15 at 0.7703) stays documented as the previous baseline; the new headline requires (a) the 14c v2 TTA feature pipeline at Stage 2 and (b) the new fusion+AQE config at Stage 4.
- **Next experiment**: 14f confirms B1 with a tighter sweep around `aqe_k=2` plus k=1 probes (CPU only, ~25 min). Spec: `docs/subagent-specs/post-14e-next.md`.
- **Files**: kernel `yahiaakhalafallah/14e-tta-fusion-aqe-fic-sweep` v1 → `outputs/14e_v1_summary/14e_summary.json`.

## 14f Confirmation Sweep (2026-05-07) — NEUTRAL, B1 Plateau Confirmed

- **Headline**: 14e B1 = **0.77936** is **reproducible and a confirmed plateau**. The drift check (A20 = B1 replicate at `aqe_k=2, fic_reg=0.5, thr=0.48, w_t=0.525`) reproduced **0.77936 EXACTLY** with `id_switches=154` — no kernel drift, no metric noise on this point.
- **Block A flatness (8 ties at 0.77936)**: 8 of the 45 confirmation configs tied at **0.77936** MTMC IDF1 with identical `id_switches=154`. The plateau covers `fic_reg ∈ {0.3, 0.4, 0.5, 0.6}` × `thr ∈ {0.46, 0.48}` around `w_t=0.525, aqe_k=2`. Slight regression at `thr=0.50` (~0.77815 with `id_switches=154`). The Stage-4 axis (`w_t × thr × FIC`) at `aqe_k=2` is **fully saturated**.
- **k=1 universally worse**: all 9 `aqe_k=1` probes degraded to **0.76933–0.77059** with `id_switches ∈ [143, 193]`. The AQE axis is concave at k=2: too little smoothing (k=1) re-introduces noise; too much (k=3, k=4) over-smooths. **k=2 is the discrete optimum for TTA features.**
- **No new winner**: zero configs beat the 0.77936 baseline. Verdict per `docs/subagent-specs/post-14e-next.md` stop criteria: **NEUTRAL** (best ≤ 0.77936, > 0.7785).
- **Status**: **TTA × Stage-4-association-tuning family is EXHAUSTED at 0.77936**. This is a *confirmed-win plateau*, not a dead end — the 0.91pp lift from production 0.7703 is real and reproducible. Further IDF1 gains require a **non-Stage-4 lever**: track-quality preprocessing, multi-view feature fusion expansion, or a learned association model (GNN).
- **Next experiment**: 14g — DINOv2 4-view TTA expansion at Stage 2 (GPU P100, ~2.5–3 hr). Currently DINOv2 ViT-L/14 uses only 2 TTA views `{original, hflip}` while the primary CLIP TransReID uses 4 `{original, hflip, scale_0.95, scale_1.05}`. Symmetrize the tertiary stream to 4 views. Tertiary fusion weight is now `w_t=0.525` (≈half the score), so lifting tertiary embedding stability has a fusion-level multiplier. Same axis (Stage-2 TTA on ReID features) that delivered 14e's +0.91pp. Spec: `docs/subagent-specs/post-14f-next.md`.
- **Files**: kernel `yahiaakhalafallah/14f-tta-aqe-confirmation-sweep` → `outputs/14f_v1_summary/14f_summary.json`.

## 14g DINOv2 4-View TTA Expansion (2026-05-08) — NEUTRAL, Tertiary Stream Saturated

- **Headline**: symmetrizing the tertiary DINOv2 ViT-L/14 stream from 2 TTA views `{original, hflip}` to 4 views `{original, hflip, scale_0.95, scale_1.05}` did **not** improve MTMC IDF1. Best of 8 configs is **S2 = 0.77926** (`w_t=0.55`, `thr=0.48`, `aqe_k=2`, `fic_reg=0.5`), essentially tied with the 14e B1 plateau **0.77936**. Verdict per `docs/subagent-specs/post-14f-next.md`: **NEUTRAL** band (best ∈ [0.7785, 0.7795] effectively at-baseline).
- **Drift gate passed**: S0 anchor at the 14e B1 config (`w_t=0.525, thr=0.48, aqe_k=2, fic_reg=0.5`) on the new DINOv2-4view features = **0.77902** (drift −0.00034 vs 14e B1 0.77936, well within the ±0.005 tolerance). Stage-2 build is correct; the change just doesn't help.
- **Strongest signal — DINOv2 4-view changed nothing in association decisions**: every config at `aqe_k=2` (S0–S5, S7) landed at exactly **`id_switches=154`** — the same count as 14e B1 / 14f A20 on the 2-view tertiary features. Tertiary stream stability is **not** the residual error source. With `w_t=0.525` the primary CLIP TransReID stream dominates the fused score, and the tertiary embedding's TTA noise floor is no longer measurably affecting cross-camera ID assignment.
- **AQE k=3 recheck (S6) confirms the k=2 unlock is real and not a tertiary artefact**: S6 at production AQE k=3 collapsed to **0.77149** with `id_switches=213` — the same regression observed in 14e B3/B4 and the 14f k=3 family. The +0.77pp from `aqe_k=3 → aqe_k=2` is reproducible across all three feature builds (14c v2 / 14e / 14g) and is a property of how AQE interacts with TTA-smoothed features, independent of how many DINOv2 views are used.
- **Block A flatness (`aqe_k=2`)**: S0/S1 (`w_t ∈ {0.525, 0.500}`) tied at 0.77902; S2/S3/S4 (`w_t ∈ {0.55, 0.575}` and `thr=0.46`) tied at 0.77926; S5 (`thr=0.50`) regressed to 0.77805; S7 (`fic_reg=0.4`) tied with S0 at 0.77902. The Stage-4 axis at the new feature build is again fully saturated and the optimum sits ~0.0001 below the 14e B1 / 14f plateau.
- **Status — TTA expansion family is fully saturated**. Both **primary 4-view** (14c v2, the 14e WIN feature build) and **tertiary 4-view** (14g) have been tested. The primary expansion was the WIN; the tertiary expansion is null. More views of the **same** models cannot lift IDF1 beyond 0.77936 because the bottleneck is no longer per-frame embedding noise — it's per-tracklet embedding *stability* and the limited diversity between the two ViT feature families (TransReID CLIP + DINOv2 are both ViT, both pretrained on web-scale image-text/self-supervised data, and produce strongly correlated cross-camera similarity rankings).
- **Implication for 14h**: the 0.77936 plateau is **feature-diversity limited and tracklet-aggregation limited**, not view-coverage limited. Three ranked levers remain (in increasing effort): **(D+F) robust tracklet pooling** — replace the current softmax-quality-weighted mean with median / geometric-median / medoid / trimmed-mean aggregates over the top-K highest-quality TTA-smoothed frame embeddings, exploiting the existing `multi_query` storage path; **(B) track-quality pre-filter**; **(A) genuinely new feature stream** (different architecture *and* different training data); **(C) GNN edge classifier**.
- **Next experiment**: **14h — robust tracklet pooling sweep** (LOW–MEDIUM effort). Single GPU Stage-2 rerun with `multi_query.k=24` enabled on the proven 14c v2 TTA recipe, then a CPU sweep over 7–8 aggregation rules at the 14e B1 anchor. Spec: `docs/subagent-specs/post-14g-next.md`.
- **Files**: kernel `yahiaakhalafallah/14g-dinov2-4view-tta-stage2` v1 → `outputs/14g_v1_summary/14g_summary.json`.

## 14h Robust Tracklet Pooling (2026-05-08) — NEUTRAL, Plateau Confirmed Across Three Axes

- **Headline**: enabling `stage2.multi_query.k=24` on the proven 14c v2 TTA recipe and post-processing per-tracklet pooled embeddings via 8 robust aggregation modes did **not** improve MTMC IDF1. M0 drift gate (existing softmax-quality mean) reproduced **0.77936** EXACT with `id_switches=154` EXACT — confirming both that the multi-query enable was a no-op on `embeddings.npy` and that 14e B1 / 14f A20 / 14h M0 are bit-identical anchors. All 8 robust modes lost IDF1, range **0.76881–0.77829** (−0.11pp to −1.06pp vs 14e B1). Verdict per `docs/subagent-specs/post-14g-next.md` thresholds: **NEUTRAL** (best of robust modes is M1 mean at 0.77829, 0.7785–0.7795 band). The existing softmax-quality-weighted mean is already the optimal pooler on TTA-smoothed features.

| Label | mode | MTMC IDF1 | id_switches | trackeval_idf1 | Δ vs 14e B1 (0.77936) | fallback_count |
|:-----:|:-----|:---------:|:-----------:|:--------------:|:---------------------:|:--------------:|
| **M0** | **existing_softmax_quality_mean (drift gate)** | **0.77936** | **154** | **0.79461** | **+0.00000** | **0** |
| M1 | mean | 0.77829 | 163 | 0.79444 | −0.00107 | 158 |
| M2 | median | 0.77107 | 162 | 0.79299 | −0.00829 | 158 |
| M3 | geo_median | 0.77514 | 163 | 0.79367 | −0.00422 | 158 |
| M4 | medoid | 0.77234 | **134** | 0.79151 | −0.00702 | 158 |
| M5 | trimmed_mean_10 | 0.77522 | 167 | 0.79468 | −0.00414 | 158 |
| M6 | trimmed_mean_25 | 0.77146 | 162 | 0.79367 | −0.00790 | 158 |
| M7 | top12_to_mean | 0.76933 | 149 | 0.79291 | −0.01003 | 158 |
| M8 | top12_to_medoid | 0.76881 | 149 | 0.79177 | −0.01055 | 158 |

- **Drift gate result**: M0 = **0.77936 EXACT**, `id_switches=154 EXACT` — bit-identical to 14e B1 v1 and 14f A20. The Stage-2 build, multi-query enable, and Stages 3–5 sweep are reproducible to floating-point.
- **Strongest signal — robust modes universally degrade IDF1**: every one of the 8 modes is worse than M0. Mean (M1) is closest at −0.11pp, suggesting the 24-row TTA-smoothed top-K block already concentrates around the same direction the existing softmax-quality pool selects, so plain mean is the closest match but still ~10 ID switches worse (163 vs 154). Robust statistics don't help because the input rows are not heavy-tailed — TTA pre-smoothing already removed the per-frame outliers that median/medoid/trimmed-mean would clip.
- **"Stable but wrong" — medoid (M4) cuts ID switches by 13% with no IDF1 gain**: M4 reaches `id_switches=134` (−20 vs M0's 154, lowest in the entire 14h sweep) but MTMC IDF1 drops to 0.77234 (−0.70pp). M7/M8 (top-12-to-mean / top-12-to-medoid) repeat the pattern at `id_switches=149` with worse IDF1 still. Interpretation: medoid and top-K-then-aggregate produce more deterministic, lower-variance pooled embeddings, so the matcher commits more confidently to its assignments — but the exemplar/consensus chosen is *not* the most discriminative cross-camera view. The matcher gains *consistency* and loses *correctness*. ID-switch count is therefore not a reliable proxy for IDF1 on this floor; the 154-floor is not aggregation-related.
- **Status — robust pooling is a confirmed dead end on the current feature build**. Three feature-side levers tested in succession have all landed at the 0.77936 plateau: (1) primary 4-view TTA (14c v2/14e WIN, the lift came from AQE k=2 not from view count); (2) tertiary 4-view TTA (14g NEUTRAL, `id_sw=154` unchanged); (3) robust per-tracklet aggregation (14h NEUTRAL, all 8 modes worse than the existing softmax mean). **The 0.77936 plateau is feature-diversity limited, NOT view-coverage limited, NOT per-tracklet-aggregation limited, NOT Stage-4-tuning limited.** The 154 ID-switch floor is NOT aggregation-related; it is determined by the discriminability of the underlying TransReID-CLIP + DINOv2 feature pair on the residual hard cross-camera ID assignments.
- **Implication for 14i**: the cheap-CPU axes are mostly exhausted for the current feature build. Three options remain: **(B) track-quality pre-filter** — drop low-confidence / short tracklets before association (CPU-only, ~30 min, no positive prior but a hedge); **(A) genuinely new third feature stream** — different architecture *and* different pretraining regime than CLIP TransReID + DINOv2 (HIGH effort: download + fine-tune on CityFlow + 3-way fusion sweep); **(C) GNN edge classifier** (VERY HIGH effort, multi-week).
- **Next experiment — 14i: track-quality pre-filter (CPU only)**. Sweep `min_track_length L_min ∈ {3, 5, 8, 12}` × `min_avg_detection_confidence τ_c ∈ {0.30, 0.35, 0.40, 0.45, 0.50}` (20 configs) at the 14e B1 anchor on the existing 14h Stage-2 outputs. Expected effect band ±0.30pp; this is a **HEDGE** experiment to confirm that the residual error is not concentrated in low-quality short tracklets. If 14i also returns NEUTRAL/MARGINAL → escalate to (A) new third feature stream. Spec: `docs/subagent-specs/post-14h-next.md`.
- **Files**: kernel `yahiaakhalafallah/14h-robust-tracklet-pooling` (final variant 14h v3); results `outputs/14h_v3_summary/14h_summary.json`; source 10a run `run_kaggle_20260425_202123`; reused 14c v2 TTA recipe + 14g/10b feature stack.

## 14i Track-Quality Pre-Filter (2026-05-08) — MARGINAL, F0 Wiring Fixed

- **F0 drift gate fixed and confirmed**: 14i v2 reproduced the 14e/14f/14h anchor with **MTMC IDF1 = 0.77935962** and **id_switches = 154** at 929/929 pass-through. This confirms the 14i v1 all-zero result was a notebook wiring/evaluation-root failure, not a filter or source-artifact problem. F0 wrote 6 MOT prediction files with **26,523 rows**, 594 global trajectories, and 929 Stage-4 tracklets.
- **Root cause of v1 all-zero metrics**: 14i v1 accepted a Kaggle dataset directory as `stage5.ground_truth_dir` merely because it existed, without verifying the evaluator-visible layout `<root>/<cam>/gt/gt.txt` for all six expected cameras. v2 validates the GT root before evaluation and copies Stage-4/Stage-5 recovery artifacts under `/kaggle/working/outputs/14i_v2_recovery/<label>` so future failures are inspectable.
- **Best filter config**: **F2** (`min_length=3`, `min_avg_confidence=0.35`) reached **MTMC IDF1 = 0.77963534**, `id_switches=120`, retaining 818/929 tracklets. This is only **+0.00028 IDF1** (+0.03pp) over F0 and below the 0.781 WIN threshold, so it is **not promoted** over the 0.77936 headline plateau.

| Label | `min_length` | `min_avg_confidence` | kept | MTMC IDF1 | id_switches | Δ vs F0 |
|:-----:|-------------:|---------------------:|-----:|----------:|------------:|--------:|
| **F0** | **0** | **0.00** | **929** | **0.77935962** | **154** | — |
| F1 | 3 | 0.30 | 845 | 0.77953820 | 125 | +0.00018 |
| **F2** | **3** | **0.35** | **818** | **0.77963534** | **120** | **+0.00028** |
| F6 | 5 | 0.30 | 794 | 0.77910981 | 131 | -0.00025 |
| F7 | 5 | 0.35 | 769 | 0.77880991 | 126 | -0.00055 |

- **Interpretation**: low-confidence/short tracklets do contribute some ID switches, but removing them mostly trades coverage for cosmetic IDS reduction rather than a meaningful IDF1 gain. F9 (`L=5, τ=0.45`) cut IDS to 97 but dropped MTMC IDF1 to 0.77604, repeating the 14h "stable but wrong / less complete" pattern. The residual error is therefore not concentrated enough in obvious low-quality tracklets to break the plateau.
- **Status**: track-quality pre-filter is **MARGINAL / not deployable**. The 0.77936 plateau is now confirmed across Stage-4 tuning, TTA view count, robust pooling, and track-quality filtering. Further gains require either a genuinely new feature stream or a learned association model.
- **Files**: kernel `yahiaakhalafallah/14i-track-quality-prefilter` v2; results `outputs/14i_v2_recovery/14i_summary.json`; recovery artifacts under `outputs/14i_v2_recovery/outputs/14i_v2_recovery/<label>/` after Kaggle output download.

## 14j R50-IBN as 4-Way Score-Fusion Stream (2026-05-08) — MARGINAL, Headline Not Promoted

- **W0 drift gate passed**: reproduced 14e B1 anchor with **MTMC IDF1 = 0.77935962** and **id_switches = 154 EXACT** at `w_quaternary=0.0`. The 4-way fusion plumbing is correct at zero secondary weight.
- **Best config (W14)**: `w_quaternary=0.30, similarity_threshold=0.48, w_primary=0.175, w_tertiary=0.525, aqe_k=2, fic_reg=0.5` reached **MTMC IDF1 = 0.78032** with `id_switches=207`, **+0.00097 over W0** (+0.10pp) and **+0.0010 over the previous deployed baseline 0.7703**. Verdict per the 14j spec bands (WIN ≥0.7810, MARGINAL 0.7795–0.7810): **MARGINAL**. Headline NOT promoted; 0.77936 stands.

### Sweep table (16 configs, all at `w_t=0.525, aqe_k=2, fic_reg=0.5`)

| Label | `w_q` | `thr` | `w_p` | MTMC IDF1 | id_switches | Δ vs W0 |
|:-----:|:-----:|:-----:|:-----:|:---------:|:-----------:|:-------:|
| **W0 (drift)** | 0.00 | 0.48 | 0.475 | **0.77936** | **154** | — |
| W1 | 0.05 | 0.46 | 0.425 | 0.77713 | 200 | −0.00223 |
| W2 | 0.05 | 0.48 | 0.425 | 0.77950 | 154 | +0.00014 |
| W3 | 0.05 | 0.50 | 0.425 | 0.77853 | 154 | −0.00083 |
| W4 | 0.10 | 0.46 | 0.375 | 0.77742 | 206 | −0.00194 |
| W5 | 0.10 | 0.48 | 0.375 | 0.77727 | 200 | −0.00209 |
| W6 | 0.10 | 0.50 | 0.375 | 0.77853 | 154 | −0.00083 |
| W7 | 0.15 | 0.46 | 0.325 | 0.77828 | 206 | −0.00108 |
| W8 | 0.15 | 0.48 | 0.325 | 0.77796 | 206 | −0.00140 |
| W9 | 0.15 | 0.50 | 0.325 | 0.77699 | 206 | −0.00237 |
| W10 | 0.20 | 0.46 | 0.275 | 0.77856 | 207 | −0.00080 |
| W11 | 0.20 | 0.48 | 0.275 | 0.77833 | 206 | −0.00103 |
| W12 | 0.20 | 0.50 | 0.275 | 0.77736 | 206 | −0.00200 |
| W13 | 0.30 | 0.46 | 0.175 | 0.77917 | 207 | −0.00019 |
| **W14** | **0.30** | **0.48** | **0.175** | **0.78032** | **207** | **+0.00097** |
| W15 | 0.30 | 0.50 | 0.175 | 0.77855 | 206 | −0.00081 |

### Boundary effect

W14 sits on the **upper boundary** of the `w_q` grid (max value tested = 0.30). At `thr=0.46` the trend is monotonic across `w_q ∈ {0.05, 0.10, 0.15, 0.20, 0.30}`: 0.77713 → 0.77742 → 0.77828 → 0.77856 → 0.77917, no turnover. At `thr=0.48` the trend dips at `w_q=0.10` then climbs cleanly: 0.77727 → 0.77796 → 0.77833 → 0.78032. The dip is a regime change (id_switches jumps from 154 to 200 at `w_q=0.10` and stabilises at 206–207 from there), not noise. The optimum may continue rising into `w_q ∈ [0.35, 0.50]`.

### Caveat: primary suppression at high `w_q`

At `w_q=0.30`, `w_primary=0.175` — the primary CLIP TransReID stream is already a minority weight relative to tertiary DINOv2 (0.525). At hypothetical `w_q=0.50`, `w_primary=0.025` (primary essentially zero'd out). The apparent improvement at high `w_q` may therefore be **rebalancing of expert weights** (R50-IBN + DINOv2 outperforming CLIP + DINOv2 on this anchor) rather than additive 4-way ensemble diversity. The 14k spec includes a sanity probe (K13: `w_p=0.30, w_t=0.30, w_q=0.40`) to discriminate these hypotheses.

### Decision: extend with 14k

Run a 13-config CPU-only extended sweep (K0 drift + K1–K12 over `w_q ∈ {0.35, 0.40, 0.45, 0.50}` × `thr ∈ {0.46, 0.48, 0.50}` + K13 primary-balance sanity). Cost ~10–15 min CPU, no GPU. Spec: `docs/subagent-specs/post-14j-next.md`.

### Files

Kernel `yahiaakhalafallah/14j-4way-fusion-sweep`; results `outputs/14j_4way_sweep/14j_4way_summary.json`; R50-IBN features dataset `yahiaakhalafallah/14j-r50-ibn-features`; source 14h v3 outputs `yahiaakhalafallah/14h-robust-tracklet-pooling`.

## 14k v1 — Extended R50-IBN 4-way Fusion Sweep (MARGINAL, NOT PROMOTED)

- **K0 drift gate passed**: reproduced the 14e B1 / 14f / 14h / 14i / 14j anchor with **MTMC IDF1 = 0.77936** and **id_switches = 154 EXACT**. The extended 4-way fusion harness preserved the zero-quaternary baseline.
- **Best config (K7)**: `w_primary=0.10, w_tertiary=0.45, w_quaternary=0.45, similarity_threshold=0.46` reached **MTMC IDF1 = 0.78079** with `id_switches=213`, **+0.00143 over K0** (+0.14pp). This remains below the pre-registered WIN bar of **0.7810**, so it is **MARGINAL** and **not promoted**. Headline stays **0.77936**.
- **K13 literal sanity passed**: the balanced three-stream sanity probe (`w_primary=0.30, w_tertiary=0.30, w_quaternary=0.40, thr=0.48`) reached **0.78048** with `id_switches=213`. This confirms the 14k lift is a real ensemble effect rather than only primary suppression, but the lift is still too small to promote.

### Sweep table (14 configs)

| Label | `w_p` | `w_t` | `w_q` | `thr` | MTMC IDF1 | id_switches | Verdict |
|:-----:|:-----:|:-----:|:-----:|:-----:|:---------:|:-----------:|:--------|
| **K0 (drift)** | **0.475** | **0.525** | **0.00** | **0.48** | **0.77936** | **154** | drift gate passed |
| K1 | 0.150 | 0.500 | 0.35 | 0.46 | 0.77936 | 213 | neutral |
| K2 | 0.150 | 0.500 | 0.35 | 0.48 | 0.77917 | 207 | neutral |
| K3 | 0.150 | 0.500 | 0.35 | 0.50 | 0.77753 | 206 | regression |
| K4 | 0.125 | 0.475 | 0.40 | 0.46 | 0.78041 | 213 | marginal plateau |
| K5 | 0.125 | 0.475 | 0.40 | 0.48 | 0.78041 | 213 | marginal plateau |
| K6 | 0.125 | 0.475 | 0.40 | 0.50 | 0.78017 | 212 | marginal plateau |
| **K7** | **0.100** | **0.450** | **0.45** | **0.46** | **0.78079** | **213** | **peak, MARGINAL, not promoted** |
| K8 | 0.100 | 0.450 | 0.45 | 0.48 | 0.78048 | 213 | marginal plateau |
| K9 | 0.100 | 0.450 | 0.45 | 0.50 | 0.78048 | 213 | marginal plateau |
| K10 | 0.075 | 0.425 | 0.50 | 0.46 | 0.77964 | 213 | turnover |
| K11 | 0.075 | 0.425 | 0.50 | 0.48 | 0.78048 | 213 | marginal plateau |
| K12 | 0.075 | 0.425 | 0.50 | 0.50 | 0.78048 | 213 | marginal plateau |
| **K13 (sanity)** | **0.300** | **0.300** | **0.40** | **0.48** | **0.78048** | **213** | **literal sanity passed** |

### Turnover analysis

14j showed a boundary effect at `w_q=0.30`; 14k resolves that curve. The extended grid rises from the K1–K3 block (`w_q=0.35`) into a stable MARGINAL plateau at `w_q=0.40–0.45`, peaking at K7 (`0.78079`) and then turning over at `w_q=0.50` (K10 drops to `0.77964`). The repeated `0.78048` results across K8, K9, K11, K12, and K13 show the plateau is real but saturated. ID switches increase to ~213 across the high-quaternary regime, so the small IDF1 lift is not accompanied by a cleaner identity graph.

### Conclusion

14k closes the R50-IBN 4-way score-fusion family as **MARGINAL, NOT PROMOTED**. K13 confirms real ensemble lift, but the best result is only **+0.0014 vs 14e B1**, below the WIN bar and within the historical noise band. The headline remains **0.77936**. All CPU-only axes are now saturated, and the feature-quality ceiling is confirmed across **5 independent axes**: Stage-4 tuning, tertiary view expansion, tracklet aggregation, track-quality filtering, and 4-way score fusion. Remaining viable levers require GPU work: a genuinely new feature stream or a learned GNN edge classifier.

### Files

Results: `outputs/14k_extended/14k_extended_summary.json`; kernel `yahiaakhalafallah/14k-r50-ibn-fusion-extended`; source feature stack from 14h v3 plus 14j R50-IBN quaternary features.

## 14t — CLIP-SENet v6 × TransReID 09v v17 Score-Level Fusion (VeRi-776, WIN — new reproducible high-water mark)

### Goal

Test whether two strong VeRi-776 experts trained with different architectures (ResNet101-IBN + TinyCLIP-AFEM vs ViT-B/16-CLIP-TransReID) and pretrainings produce complementary feature spaces under simple score-level fusion.

### Config

Score-level fusion at `w_clipsenet=0.7, w_transreid=0.3`, using the transreid_768 global-token stream, AQE k=3 + rerank (k1=80, k2=15, λ=0.2). Runtime ≈ 49.5 min on T4. Kernel: `https://www.kaggle.com/code/yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid`.

### Result table

| Stream | mAP | R1 |
|---|---:|---:|
| TransReID 09v v17 (base) | 0.8997 | 0.9833 |
| CLIP-SENet v6 (post-rerank) | 0.9154 | 0.9732 |
| **14t score-fusion (best)** | **0.9330** | **0.9845** |
| 14t concat (best) | 0.9319 | 0.9827 |

### Plateau

Top score-fusion rows are tightly clustered for `w_clipsenet ∈ [0.6, 0.8]`; transreid_768 (global token) clearly beat transreid_1536 (concat token).

### Verdict

Spec = MARGINAL (concat row R1 missed bar by 0.06pp); de-facto = **WIN**. Two ReID models trained on the same dataset with different architectures and pretraining produce genuinely complementary embeddings, despite TransReID being the stronger single model.

### Caveat — Do NOT auto-port to CityFlow

13d/13f/13h proved CLIP-SENet × CityFlow fusion is strongly negative: monotonic IDF1 degradation in the score-fusion sweep (-1.77pp at w=0.6, -8.24pp standalone), and even a CityFlow-fine-tuned CLIP-SENet peaked at 0.7691 fusion IDF1 (-0.12pp below production 0.7703). The VeRi-776 → CityFlowV2 domain gap dominates and a strong VeRi-776 expert does not transfer.

### Next-step recommendation

**Option B — Accept and freeze.** Rationale: 14t is paper-quality VeRi-776 SOTA-equivalent and orthogonal to the CityFlow MTMC 0.77936 plateau (confirmed across 6 axes — Stage-4 tuning, tertiary view expansion, tracklet aggregation, track-quality filter, 4-way score fusion, and VeRi-fusion-port). Option A (port to CityFlow) has a strongly negative prior from 13d/13f/13h. Option C (3-way fusion on VeRi) has 14k precedent showing marginal-at-best returns from adding a third stream once two complementary streams are fused. Stop here on VeRi.

## 14u CityFlow VeRi-Fusion Port — DEAD END (2026-05-12)

Tested whether the 14t WIN mechanism (CLIP-SENet × TransReID score fusion + AQE k=3 + k-reciprocal rerank on the *fused* similarity) ports to CityFlowV2 as a 4th score-fusion stream on top of the 14e B1 anchor. 19-config CPU-only sweep on `w_14t × thr`.

- **U0 drift gate**: reproduced **0.77936 / id_switches=154 EXACT**.
- **Best (U5/U6/U9)**: `w_14t=0.10–0.15` at `thr=0.48–0.50` reached **0.77995 / id_switches=160** — only +0.00059 IDF1 vs U0, below the 14u spec MARGINAL bar of 0.7800. id_switches went *up* (154→160) at the "best" point, i.e. fusion adds conflation, not signal.
- **Higher `w_14t`** (≥0.15 at thr=0.46/0.48): **0.77809 / id_sw=207** (-0.13pp). Optimum sits at the lower boundary of the sweep, exactly mirroring the 13d `w_cs=0.10` row in the prior CLIP-SENet × CityFlow fusion FAIL.
- **Mechanistic significance**: this closes the 5th and final CityFlow VeRi-fusion branch (13d / 13f / 13g / 13h / 14u all FAIL or MARGINAL). The rerank-on-fused-similarity mechanism that produced the **+3.33pp VeRi mAP lift** in 14t adds **zero cross-camera signal** on CityFlowV2. Any single-cam-strong ReID feature space (CLIP-SENet alone, fine-tuned CLIP-SENet, 14t fusion + rerank) loses signal when projected onto non-overlapping CityFlow cameras — the VeRi-776 → CityFlowV2 domain gap is the binding constraint, not feature-space construction.
- **Headline plateau now confirmed across SIX independent axes**: Stage-4 tuning (14e/14f), tertiary view expansion (14g), tracklet aggregation (14h), track-quality filter (14i), 4-way score fusion (14j/14k), and VeRi-fusion-port (14u). All cheap-to-medium-cost CityFlow MTMC IDF1 levers are now exhausted. Remaining paths to >0.7900 require either (a) AIC22-style 5-model ensemble (multi-day GPU), (b) GNN edge classifier (untried, requires labeled cross-camera pairs), or (c) zone-based ST + per-camera distance bias (hand-engineered).
- **Files**: kernel `https://www.kaggle.com/code/yahiaakhalafallah/14u-cityflow-veri-fusion-port`; results `tmp_14u_outputs/14u_summary.json`.

## 14m — OSNet-IBN-x1.0 CityFlowV2 From-Scratch Training (FAILED, DEAD END)

- **Goal**: train a fifth feature stream for a 14n 5-way fusion ensemble, replacing the lost ali369 v80 OSNet stream that enabled the historical **0.784** MTMC result but whose `vehicle_osnet_veri776.pth` checkpoint is no longer available.
- **Resolved run**: kernel `gumfreddy/14m-osnet-ibn-cityflowv2-train` v1 completed successfully after the v3 memory-defense patches (single GPU, batch 64, eval batch 8, `P=16/K=4`, no DataParallel, per-epoch cleanup). It trained **120 epochs** in about **6.5h on T4** and wrote the expected cadence/final checkpoints plus `data/outputs/14m_final_metrics.json`.
- **Eligibility gate**: required **mAP >=75% AND R1 >=90%** on CityFlowV2 before any Stage-2 extraction or 14n fusion. 14m failed by about **50pp mAP**, so no downstream feature extraction or fusion should be run.

| Checkpoint | Epoch | mAP | R1 | R5 | R10 | Verdict |
|:--|--:|--:|--:|--:|--:|:--|
| Best mAP | 60 | **24.27%** | 43.59% | **53.91%** | **60.65%** | gate FAIL |
| Best joint | 90 | 23.90% | **43.97%** | 53.89% | 60.32% | gate FAIL |
| Final eval | 120 | 23.80% | 43.89% | 53.72% | 60.28% | gate FAIL |

- **Training shape**: the model peaked at epoch 60 and slowly degraded through epoch 120, so the run over-trained rather than being under-trained. The best checkpoint is still far below the eligibility floor.
- **Verdict**: **DEAD END** for the 14n fusion goal. OSNet-IBN-x1.0 in-domain CityFlowV2 from-scratch is too weak to add as an ensemble stream: **23.80% mAP** final is below the **52.77% R101-IBN floor** and far below the **80%+ TransReID CLIP primary**. Adding it to fusion would inject noise and likely hurt MTMC IDF1.
- **Likely root causes**: (a) the **666-class CityFlowV2** train split is too small for OSNet's design when trained from scratch; (b) the v3 single-GPU, batch-64, `P=16/K=4` memory-defense recipe changed dynamics from the BoT recipe that was originally tuned for ResNet-family models and larger batches; (c) the BoT LR schedule may not suit OSNet.
- **Do not retry as-is**: do **not** rerun OSNet-IBN-x1.0 from scratch on CityFlowV2 for this branch. Any future OSNet attempt must explicitly address (a)/(b)/(c), for example via strong VeRi pretraining, a different sampler/batch regime, and an OSNet-specific LR schedule. It should be treated as a new experiment, not a 14m continuation.
- **Auth workflow learned**: multi-account Kaggle pushes can use `KAGGLE_API_TOKEN` with `~/.kaggle/<account>_access_token`; this was proven on gumfreddy and removed the earlier full-credential JSON blocker.

## Current Performance (Last Updated: 2026-05-11)

### Vehicle Pipeline (CityFlowV2)

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Current Reproducible Best MTMC IDF1** | **77.96% observed / 77.94% headline** | 14i F2 (`min_length=3`, `min_avg_confidence=0.35`) observed 0.77964 but is not promoted because the lift is only +0.03pp over the confirmed 14e/14f/14h/14i F0 plateau. Headline remains 14e B1 / 14f A20 / 14h M0 / 14i F0 / 14j W0 / 14k K0 at 0.77936 with `id_switches=154`. 14j/14k 4-way score fusion (R50-IBN as quaternary stream) reached a non-promoted MARGINAL plateau below the WIN bar; all CPU-only axes are now saturated. |
| **VeRi-776 reproducible best (single-cam)** | **mAP=0.9330 / R1=0.9845** | 14t score-fusion, `w_clipsenet=0.7, w_transreid=0.3`, transreid_768 global token, AQE k=3 + rerank. 09v v17 remains the best *single-model* result at **0.8997 mAP / 0.9833 R1**. |
| **10c v8 (3-way ensemble sweep — 2026-04-22)** | 71.28% (biased; add ~5pp) | 10a v5 regression fix COMPLETE (929 tracklets, 49.4 min); 10c v8 19-config ensemble sweep COMPLETE but biased by MTMC_ONLY=True bug (~5pp penalty); best: w2=0.05, w3=0.30 → 71.28% biased → ~76.28% est. true. Fixed in commit `69e67a0`; 10c v9 RUNNING for unbiased results. |
| **10c v9 (MTMC_ONLY fix — 2026-04-22)** | MTMC_IDF1=76.625% (baseline), 76.817% (best ensemble) | COMPLETE. Baseline 76.625% is ~0.74pp below the old v80-era expectation, but the later OSNet investigation showed that the larger gap was driven by the missing `vehicle_osnet_veri776.pth` checkpoint rather than an unresolved code drift. Best 3-way: w2=0.05, w3=0.30 → 76.817% (+0.192pp). Ensemble marginal; dead end confirmed. |
| **Historical Best MTMC IDF1** | **78.4%** | v80 / v44, achieved with a specific OSNet checkpoint that is no longer available. That checkpoint lived in `mrkdagods/mtmc-weights` and was dropped on **2026-03-30**, so this result is not reproducible now. |
| **SOTA Target** | 84.86% | AIC22 1st place |
| **Gap to SOTA** | 6.93pp | Relative to new reproducible best **0.77936** |
| **Best Vehicle ReID Model (09s v1 DINOv2 ViT-L/14)** | mAP=86.79%, R1=96.15% | New best CityFlowV2 vehicle ReID model; **+6.65pp mAP / +3.88pp R1** vs the prior deployed **ViT-B/16 CLIP** baseline |
| **DINOv2 ViT-L/14 MTMC IDF1 (10c DINOv2 v2, best with AFLink)** | **74.4%** | Full pipeline complete 2026-04-25; **-3.1pp** vs ViT-B/16 CLIP (77.5%) despite +6.65pp mAP. AFLink +5.6pp for DINOv2 (unlike CLIP where AFLink always hurts). Per-camera IDF1=79.4%. Training methodology > model capacity for cross-camera invariance. |
| **Previous Deployed Best (ViT-B/16 CLIP 256px)** | mAP=80.14%, R1=92.27% | Prior strongest production-ready vehicle ReID model before the DINOv2 breakthrough |
| **ViT-Large AugReg without CLIP/DINOv2 pretraining (09r v7)** | mAP=60.38%, R1=76.57% | Large backbone alone failed badly; **-19.76pp mAP** vs the previous deployed **ViT-B/16 CLIP** baseline |
| **Experiment B (CircleLoss ablation, 09 v4)** | mAP=18.45%, R1=48.84% | Catastrophic failure with baseline augmentations; training loss was `inf` at every epoch |
| **LAION-2B CLIP CircleLoss run (09l v1)** | mAP=20.36%, R1=53.03%, mAP_rr=27.16% | Same catastrophic fp16 overflow failure as 09 v4; tells us nothing about LAION-2B backbone quality |
| **LAION-2B CLIP extended Triplet run (09l v3)** | mAP=78.61%, R1=90.43%, mAP_rr=81.09%, R1_rr=90.98% | Strong standalone alternative, but not a viable fusion secondary after 10c v56 showed CLIP ViT score fusion hurts |
| **10c v48 (09 v2 augoverhaul @ 256px)** | MTMC IDF1=0.722 | Best result after an 11-sweep association re-optimization; single-cam IDF1 only 0.752 |
| **10c v49 (09 v3 augoverhaul-EMA @ 256px)** | MTMC IDF1=0.722 | Best result after a broader association sweep; AFLink recovered 0.675 -> 0.722 but could not break the augoverhaul ceiling |
| **Secondary Model (ResNet101-IBN-a)** | mAP=52.77% | On CityFlowV2 eval split, ImageNet→CityFlowV2 only |
| **Secondary Model ResNeXt101-IBN-a ArcFace (09j v2)** | mAP=36.88%, R1=62.69% | Catastrophic failure after partial/mismatched pretrained weight loading left large parts of the backbone randomly initialized |
| **Secondary Model ViT-Small/16 IN-21k (09k v1)** | mAP=48.66%, R1=62.01% | Confirms the non-CLIP ceiling extends to ViT architectures too |
| **Secondary Model EVA02 ViT-B/16 CLIP (09o v1)** | mAP=48.17%, R1=65.90% | Much weaker than the primary ViT-B/16 CLIP baseline and even the fine-tuned R50-IBN secondary; current recipe does not transfer well to vehicle ReID |
| **Secondary Model CLIP RN50x4 CNN (09m v2)** | mAP=1.55%, R1=4.18% | Catastrophic failure; CE loss converged but retrieval features were unusable, closing out the CNN-based CLIP secondary path |
| **Secondary Model VeRi-776 pretrain (09e v2)** | mAP=62.52% | On VeRi-776 test set, ready for CityFlowV2 fine-tuning |
| **384px ViT (09b v2)** | DEAD END | Higher single-camera ReID accuracy did not transfer; MTMC IDF1 only 0.7585-0.7562 in v43-v44, -2.8pp vs 256px baseline |
| **09f CityFlowV2 fine-tune** | mAP=42.7% | v3 peaked at epoch 104/120 and still underperformed direct ImageNet→CityFlowV2 (09d v18: 52.77%) |
| **CLIP-SENet CityFlow fine-tune (13f v1)** | standalone IDF1=0.7099, fusion peak 0.7691 @ `w_cs_ft=0.30` | 12-epoch fine-tune of v6 on 666 CityFlow IDs lifted standalone +2.44pp over 13d (0.6855), but fusion peak is **−0.12pp** below production 0.7703. Train loss still decreasing at epoch 12 (10.21→7.65); longer training extrapolates to fusion ≤0.7703. **MARGINAL / DEAD END.** |
| **Association configs tested** | 225+ | All within 0.3pp of optimal |

**Updated 2026-05-07**: the new reproducible best is **0.77936** (14e B1 v1) on multi-crop TTA Stage-2 features with `aqe_k=2` instead of the long-standing production `aqe_k=3`. The 0.7703 figure below remains the previous deployed baseline. **Current reproducible vehicle MTMC IDF1 is 0.7703** from **10c v15 / 10a v7** using **CLIP+DINOv2 score-level fusion** with `w_tertiary=0.60`. The earlier **0.775 / 0.784** CLIP+OSNet-era results depended on `vehicle_osnet_veri776.pth`, a CityFlowV2-adapted OSNet checkpoint that is no longer present in the weights datasets after the **2026-03-30** regeneration of `mrkdagods/mtmc-weights`. Vehicle association remains exhausted, so future gains will need materially better features or priors rather than more stage-4 tuning.

The April 25 feature-side results materially changed the outlook. **09r v7** cleanly failed at only **60.38% mAP / 76.57% R1** despite using a larger **ViT-Large** backbone, which falsifies the idea that model size alone can rescue vehicle ReID. In contrast, **09s v1** delivered a genuine breakthrough: **DINOv2 ViT-L/14** reached **86.79% mAP / 96.15% R1** at epoch **115/120**, beating the previous deployed **ViT-B/16 CLIP** best by **+6.65pp mAP / +3.88pp R1**. The training also converged much better than 09r (**loss 1.54 vs 2.19**, **train_acc 0.9855 vs 0.9165**), which strongly suggests the gain is coming from the pretrained representation rather than incidental optimization noise.

This is the first result on the current codebase that plausibly changes the MTMC ceiling rather than merely shifting single-camera metrics. The working hypothesis is now stronger and more concrete: **feature quality is still the bottleneck, but DINOv2 may finally be strong enough to break it**. If the Stage 2 embeddings preserve this gain through the full pipeline, the project now has a realistic path to **reach or exceed the AIC22 SOTA target of 84.86% MTMC IDF1**.

The latest structural association follow-up, **10c v53 network flow solver**, confirms that conclusion. Against the same controlled **10c v52** baseline, network flow reached only **MTMC IDF1 = 0.769** versus **0.7714** for the CC baseline (**-0.24pp**). It slightly improved **MOTA** (**0.689 -> 0.694**) and **HOTA** (**0.5747 -> 0.577**) with **2 fewer ID switches** (**199 -> 197**), but it **increased conflation from 27 to 30 predicted IDs** instead of reducing it. The current **conflict_free_cc** pipeline remains the preferred solver for this problem.

The corrected **10c v48** evaluation of the **09 v2 augoverhaul** model at the intended **256px** resolution still regressed sharply: the best full sweep reached only **MTMC IDF1 = 0.722** with `sim_thresh=0.45`, `appearance_weight=0.60`, `fic_reg=1.00`, `aqe_k=3`, `aflink_gap=150`, and `aflink_dir=0.85`. The follow-up **10c v49** sweep on the **09 v3 augoverhaul-EMA** training run reproduced the same **0.722** ceiling with a broader parameter search, confirming that the regression persists across augoverhaul variants and is driven by the model family rather than missed association tuning. Single-camera **IDF1 = 0.752** in **10c v48** was also materially below the baseline (~0.82), confirming that the earlier **10c v47** collapse was partly a deployment bug, but the underlying augoverhaul recipe is itself a vehicle-MTMC regression.

The new **Experiment B** CircleLoss-only ablation on the primary vehicle ReID path failed catastrophically. Kernel **`gumfreddy/09-vehicle-reid-cityflowv2-circleloss-ablation` v1** used the original baseline augmentation stack with **CE+LS(eps=0.05) + CircleLoss(m=0.25, gamma=128) + CenterLoss** for **120 epochs**, but the training loss was **`inf` at every epoch** and the run collapsed to only **mAP = 18.45%** and **R1 = 48.84%**. This independently confirms that **CircleLoss is a dead end** on this CityFlowV2 TransReID recipe. It also sharpens the interpretation of the augoverhaul regression: either the augoverhaul augmentations themselves caused the **81.59% mAP -> 0.722 MTMC IDF1** failure, or **CircleLoss was not actually active** in that training run due to a config/path mismatch. In either case, **CircleLoss is not a viable explanation for a healthy high-mAP regime** because when it is definitely active, it destroys training entirely.

That conclusion is now reinforced by the full **09l** sequence. The original **09l v1** LAION-2B CLIP attempt reused the broken **Experiment B** recipe, kept the loss at **`inf` throughout all 120 epochs**, and collapsed to only **mAP = 20.36%**, **R1 = 53.03%**, and **mAP_rr = 27.16%**. The follow-up **09l v2** rerun replaced **CircleLoss** with **TripletLoss**, re-enabled **EMA** with **decay=0.9999**, and trained for **160 epochs**, recovering to **mAP = 61.51%**, **R1 = 81.41%**, and **mAP_rr = 67.20%**. The final **09l v3** continuation then resumed from the **v2 EMA checkpoint** and extended training to **300 total epochs**, reaching **mAP = 78.61%**, **R1 = 90.43%**, **mAP_rr = 81.09%**, and **R1_rr = 90.98%**. This closes the loop: the earlier collapse was a **recipe instability**, not a backbone failure.

The **09l v3** continuation confirmed that **v2 was schedule-limited, not architecture-limited**. The resumed training phase kept improving across **epoch 180/200/220/240/260/280/300 = 65.93/68.84/71.49/73.68/75.68/77.26/78.61 mAP**, and the finished model sits only **1.53pp** behind the deployed **OpenAI CLIP 09b v2** baseline at **80.14% mAP**. But the follow-up **10c v56** score-fusion test still regressed, showing that a strong secondary alone is not enough when the feature families are too correlated.

The new **09m v2 CLIP RN50x4 CNN** experiment closes the other obvious escape hatch. It reached only **mAP = 1.55%** and **R1 = 4.18%** despite the cross-entropy loss converging from **6.57 -> 0.99** over **200 epochs**, which means optimization did not fail in the usual sense but the learned features were still useless for retrieval. The most likely root causes are **(1)** a **QuickGELU mismatch** between `open_clip` model construction (`quick_gelu=False`) and the OpenAI pretrained weights (`quick_gelu=True`), which corrupts the pretrained feature geometry, **(2)** the **CNN attention-pooling CLIP architecture** not fitting the standard ReID projection-head recipe used elsewhere in the codebase, and **(3)** **640D CNN features** being fundamentally harder to adapt for fine-grained vehicle ReID than the **768D ViT** features. Taken together with the failed **63.64% R50-IBN**, **52.77% ResNet101-IBN-a**, **48.17% EVA02 ViT-B/16 CLIP**, **48.66% ViT-Small**, **36.88% ResNeXt**, and **10c v56** correlated-CLIP fusion regression, the **score-level ensemble path is now fully exhausted**: there is **no viable secondary model** on the current codebase.

**⚠️ Metric Disambiguation (Vehicle Pipeline):**
- **MTMC IDF1 = 77.5%** — Current reproducible best on the current codebase from 10c v52. This remains the official metric and the only number that should be compared to AIC22 SOTA.
- **Historical v80 reference** — MTMC IDF1 = 78.4%, IDF1 = 79.8%, GLOBAL IDF1 = 80.5%. These numbers are useful for historical comparison, but they depended on the now-unavailable `vehicle_osnet_veri776.pth` checkpoint and are not currently reproducible.
- All current best numbers use GT-assisted metrics (`gt_frame_clip=true`, `gt_zone_filter=true`) which inflate scores by 1-3pp vs clean evaluation.

### Person Pipeline (WILDTRACK) — NEW

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Ground-plane MODA** | **90.0%** | Best converged operating point reproduced across the final 12b tracking runs |
| **Ground-plane IDF1** | **94.7%** | Confirmed independently by 12b v1, v2, and v3; Precision=94.5%, Recall=96.1%, IDSW=5 |
| **Best Detector (MVDeTr 12a v3)** | MODA=92.1% | Epoch 20/25, Precision=95.7%, Recall=96.6%; best WILDTRACK detector yet |
| **12b v2 extended Kalman sweep** | MODA=90.0%, IDF1=94.7% | Wider interpolation/max_age/conf sweeps plus velocity-aware quadratic interpolation still converged to the same 5-IDSW operating point |
| **12b v3 global optimal tracker** | MODA=88.2%, IDF1=91.2% | Sliding-window Hungarian assignment underperformed badly; immediate frame-level costs overrode the motion-prediction advantage of Kalman and produced 15 ID switches |
| **12b v1 on 12a v3 detections** | MODA=90.0%, IDF1=94.7% | Better detections did not improve tracking; Kalman sweep converged to the same effective operating point |
| **Previous Best Tracking Baseline** | IDF1=92.8% | 12b v9 naive tracker; new tuned Kalman run is +1.9pp |
| **ReID Features (12b v8)** | mean cosine=0.720 | Fixed ViT backbone mismatch; range widened to [0.215, 1.000] |
| **ReID Merge State** | No gain over baseline | Tuned Kalman tracks are already clean; merge sweep did not improve IDF1 |
| **Gap to SOTA** | 0.6pp | Best IDF1=94.7% vs target 95.3% |
| **Status** | FULLY CONVERGED | Kalman, naive, and global-optimal trackers tested across 59 configs; remaining path is better person appearance features or graph-based approaches |

**Current person best remains 94.7% IDF1 and is now fully converged**: 12b v14 first reached **IDF1=94.7%** with a tuned Kalman tracker, and that operating point has now been independently matched by **12b v1**, **12b v2**, and **12b v3**-era validation runs around the same final configuration. The stronger **12a v3** detector did not improve tracking, broader Kalman sweeps across interpolation/max_age/confidence thresholds did not improve tracking, and the new sliding-window **GlobalOptimalTracker** regressed sharply to **IDF1=91.17%**, **MODA=88.2%**, and **15 ID switches**. Across **59 tracker configs** spanning **Kalman**, **naive**, and **global optimal** trackers, the best Kalman solutions all clustered at **IDF1=0.9467 ± 0.0004**, with confidence thresholds from **0.15-0.35** making no meaningful difference. The root cause is that global optimal assignment over-commits to immediate frame-level costs, while BoT-SORT's predictive Kalman model handles occlusions and missed detections better over time. The current WILDTRACK person pipeline is therefore **fully tracker-converged at 94.7% IDF1**. The remaining **0.6pp** gap to SOTA is most likely due to rare occlusion and re-entry failures that need better person appearance features or graph-based multi-view association, not more tracker tuning.

**14z verifier update (2026-05-16)**: PR #33 fixed the shared WILDTRACK ground-plane GT loader and Stage-5 prediction-frame range handling. The local GT diagnostic on `data/raw/wildtrack/annotations_positions` showed no raw density inflation (`raw_json_len == unique_pids`, 17-38 people on frames 0/200/360/399), but the defensive loader patch now dedupes repeated `personID`s and filters to calibrated in-camera projections when calibrations are available. Kaggle 14z v6 on master `bdb3fbf` moved detector-only precision/recall to healthy values (**P=94.75%, R=96.64%, FP=51, misses=32**), confirming the old 10% recall denominator failure is fixed. PR #35 then corrected shared ground-plane MODA semantics to **exclude ID switches** (`MODA = 1 - (FN + FP) / GT`). Kaggle 14z v7 on master `6025df8` reported **MODA=91.28%**, **P=94.75%**, **R=96.64%**, **FP=51**, **misses=32**, and **IDSW=881**. This still fails the verifier gate by **0.82pp** versus the 92.1% target (outside the ±0.5pp tolerance), so the residual 14z gap is now real detector/export mismatch or target provenance drift, not MODA-vs-MOTA accounting. Detector-only fresh prediction IDs remain the source of high IDSW, but IDSW no longer affects MODA.

## Gap Decomposition

The remaining gap to SOTA now decomposes into:

| Deficiency | Impact | Status |
|------------|:------:|--------|
| Downstream MTMC validation of the 09s DINOv2 checkpoint | **COMPLETED — DEAD END** | 10c DINOv2 v2 reached only MTMC IDF1=0.744 (best, with AFLink), **-3.1pp** vs ViT-B/16 CLIP (0.775). DINOv2's higher mAP did not transfer to MTMC. |
| Lack of a viable complementary secondary model | Still unresolved, but no longer the only path | **FULLY EXHAUSTED** for now: weak alternative secondaries all failed, including **09o v1 EVA02 at 48.17% mAP** and **09m v2** at **1.55% mAP**, and even strong **09l v3** CLIP-ViT fusion regressed in **10c v56** |
| Camera-aware single-model training (DMT) | -1.4pp | Tested and harmful (v46) |
| Multi-query track representation | -0.1pp | Tested and neutral/harmful (v51) |
| Higher-dimensional concat-patch features | -0.3pp | Tested and harmful (v48-v49) |
| Association tuning / structural association changes | Exhausted | 225+ configs plus structural variants already tried |

**Historically, higher single-camera ReID mAP did not translate into better MTMC IDF1.** Multiple earlier experiments confirmed that pattern: the augmentation-overhaul plus CircleLoss recipe, 384px deployment, DMT camera-aware training, and both tested score-level fusion paths all improved or preserved some aspect of single-model quality while hurting downstream MTMC. **09s v1 is the first serious exception candidate** because it is not a marginal recipe tweak; it is a step-change from a materially stronger pretrained representation. The project should still treat MTMC gains as unproven until the full **10a -> 10b -> 10c** run completes, but the prior **77-78%** ceiling is no longer a safe assumption.

## Critical Discovery: 384px Is a Dead End for MTMC

The earlier checkpoint mismatch was real, but it is no longer the blocker. The correct 09b v2 384px checkpoint was evaluated and still lost decisively to the 256px baseline in MTMC.

**Definitive result (2026-03-30)**:
- v43 (384px, tuned thresholds, min_hits=3): MTMC IDF1 = 0.7585
- v44 (384px, exact v43 config but min_hits=2): MTMC IDF1 = 0.7562
- v80 baseline (256px, min_hits=2): MTMC IDF1 = 0.7840

Higher input resolution improved or preserved single-camera discriminative detail, but it made cross-camera association worse by emphasizing viewpoint-specific textures that do not transfer well across cameras.

## Critical Discovery: ResNet101-IBN-a 52.77% Is Expected

The ViT achieves high mAP because of 3-stage progressive specialization:
- CLIP (400M image-text pairs) → VeRi-776 (576 IDs, 37K images) → CityFlowV2 (128 IDs, 7.5K images)

The ResNet skips the critical VeRi-776 middle step:
- ImageNet (1.3M generic) → CityFlowV2 (128 IDs, 7.5K images) directly

Published 75-80% mAP baselines for ResNet101-IBN-a are evaluated on **VeRi-776** (576 IDs), NOT on CityFlowV2 (128 IDs). These numbers were never comparable. The 52.77% is reasonable given the massive pretraining disadvantage.

**Fix**: Train ResNet101-IBN-a on VeRi-776 FIRST, then fine-tune on CityFlowV2 (same pattern as the ViT).

**Status update (2026-03-29)**: VeRi-776 pretraining completed in 09e with mAP=62.52% on the VeRi-776 test set, but the follow-up CityFlowV2 fine-tune in 09f v3 only reached mAP=42.7%. This underperforms the direct ImageNet→CityFlowV2 baseline (09d v18: 52.77%), so VeRi-776 pretraining appears to hurt rather than help for this ResNet101-IBN-a path.

**Status update (2026-04-01)**: Extending the direct ImageNet→CityFlowV2 run by resuming from the **09d v18** 52.77% checkpoint at a lower learning rate (**3e-4**) peaked at only **50.61% mAP** in **09d gumfreddy v3**. This confirms that **52.77% is effectively the ceiling** for the current ImageNet→CityFlowV2 ResNet101-IBN-a recipe rather than an undertrained checkpoint.

**Status update (2026-04-17)**: The new **09i v1 ArcFace** follow-up on **gumfreddy** also failed to break that ceiling. With **ArcFace (s=30, m=0.35) + Triplet (m=0.3) + Center loss** and a **warm-start from 09d**, the run peaked at only **50.80% mAP**, **73.46% R1**, and **54.65% mAP_rerank** at **epoch 100/160**, then declined through the rest of training. The most likely root cause is **geometry mismatch**: the checkpoint was warm-started from a **cross-entropy-optimized** solution, then forced into an **ArcFace angular-margin** regime while also keeping triplet and center objectives active. On a dataset with only **128 train IDs**, that creates **four competing losses/geometries** and pushes the model into overfitting rather than better discrimination.

**Status update (2026-04-17)**: The new **09j v2 ResNeXt101-IBN-a ArcFace** run failed catastrophically at only **36.88% mAP**, **62.69% R1**, and **40.49% mAP_rerank** after **160 epochs**. The root cause is not just poor optimization: the original **IBN-Net ResNeXt** checkpoint path was incompatible because the published weights use **32x32d grouped convolutions** while the training model here was instantiated as **32x8d**. The v2 workaround filtered state-dict loading with `strict=False`, which avoided the shape-mismatch crash but left many layers unmatched and therefore randomly initialized. That crippled the run from the start and makes the current **ResNeXt101-IBN-a** path a dead end unless a truly compatible pretrained checkpoint is found.

**Status update (2026-04-17)**: The new **09k v1 ViT-Small/16** run reached only **48.66% mAP** and **62.01% R1** after **120 epochs** despite using a ViT backbone. Taken together with **09d/09i** on ResNet101-IBN-a and **09j v2** on ResNeXt101-IBN-a, this confirmed that the secondary-model ceiling is a **non-CLIP pretraining and initialization problem across both CNN and ViT families**, not just an architecture-selection problem.

**Status update (2026-04-19)**: The new **09m v2 CLIP RN50x4 CNN** follow-up also failed catastrophically at only **1.55% mAP** and **4.18% R1** even though the **cross-entropy loss converged from 6.57 -> 0.99** over **200 epochs**. This is a worse failure mode than the non-CLIP baselines because it indicates the training loop can optimize classification while still destroying retrieval geometry. The most likely causes are a **QuickGELU mismatch** in the loaded CLIP CNN weights, poor compatibility between the **attention-pooling CNN backbone** and the standard ReID projection-head recipe, and an inherently weaker adaptation path from **640D CNN features** than from **768D ViT features** for fine-grained vehicle identity. In practice, this means the attempted **alternative CLIP-family CNN** rescue also failed. Combined with **10c v56** showing that the strong **09l v3** CLIP-ViT secondary is too correlated with the primary to help in fusion, the **secondary-model / score-level ensemble path is now fully exhausted** on the current codebase.

## What SOTA Does Differently

| Pattern | AIC22 1st | AIC22 2nd | AIC21 1st | We have? |
|---------|:-:|:-:|:-:|:-:|
| 3+ ReID backbone ensemble | 5 models | 3 models | 3 models | **NO** (1 strong primary, no viable complementary secondary) |
| 384×384 input | ✅ | ✅ | ✅ | **Tested, but harmful in our pipeline** |
| IBN-a backbones | ✅ | ✅ | ✅ | ViT only |
| Camera-pair bias (CID_BIAS) | ROI masks | NPY | NPY | **Tested twice and harmful: GT-learned -3.3pp, topology bias -1.0 to -1.2pp (DEAD END)** |
| Reranking | Box-grained | k-reciprocal | k-reciprocal | **Disabled** |
| Camera-aware training (DMT) | ✅ | ✅ | ✅ | **NO** |
| Multiple loss functions | ID+tri+circle+cam | ID+tri+cam | ID+tri+cam | ID+tri |

### AIC Winning Methods Analysis and Ensemble Implementation

- **AIC22 1st place (IDF1=0.8486)** used a **5-model ReID ensemble** plus **Box-Grained Matching**.
- **AIC22 2nd place (IDF1=0.8437)** used a **3-model ensemble** (**ResNet101-IBN-a x2 + ResNeXt101-IBN-a**) with **DMT** and **CID_BIAS**.
- **Universal pattern**: every top AIC method relies on a **3-5 model ensemble**. The consistent win is not a single stronger backbone, but diversity across multiple ReID models.
- **Implication for our results**: our single-model upgrades such as **384px**, **DMT**, and **reranking** fail in isolation because they are ensemble-dependent techniques. They can remove noise, but with only one model they also remove too much discriminative signal.

**Ensemble implementation status**

- **09g**: ResNet101-IBN-a DMT training is currently running on **gumfreddy** with **150 epochs**, a **camera-adversarial head**, and **no circle loss**.
- **09j v2**: ResNeXt101-IBN-a ArcFace finished at only **36.88% mAP** after the checkpoint compatibility workaround still left large parts of the backbone randomly initialized. This path is now a **dead end** rather than a live ensemble candidate.
- **09l v3**: LAION-2B CLIP resumed from the **v2 EMA checkpoint** to **78.61% mAP / 90.43% R1 / 81.09% mAP_rr** at **300 total epochs**. This is only **1.53pp** behind the deployed **OpenAI CLIP 09b v2** baseline, but **10c v56** showed that fusing the two CLIP ViT-B/16 models still hurts because they are too correlated.
- **09m v2**: CLIP **RN50x4** CNN training failed catastrophically at **1.55% mAP / 4.18% R1** even though the CE loss converged from **6.57 -> 0.99**. The likely causes are the **QuickGELU mismatch**, weak compatibility between the **attention-pooling CNN** backbone and the standard ReID head recipe, and poorer transfer from **640D CNN** features than from **768D ViT** features.
- **Infrastructure**: **Stage 2** and **Stage 4** have already been updated for **3-model score-level fusion**.
- **CID_BIAS**: both tested variants are dead ends. The original GT-learned CID_BIAS dropped MTMC IDF1 from **0.784 -> 0.751** (**-3.3pp**), and the later topology-bias sweep in **10c v55** reached only **0.764-0.762-0.763** versus a **0.774** control (**-1.0 to -1.2pp**). FIC whitening already handles the useful camera calibration, and additive CID_BIAS terms distort those calibrated similarities.
- **Net result**: The **score-level ensemble path is fully exhausted** on the current codebase. There is no viable secondary model that is both individually strong enough and diverse enough to improve vehicle MTMC.

**Why the previous improvements failed**

- With **1 model**, techniques like **384px**, **DMT**, and **reranking** strip away unstable but still useful signal, so MTMC gets worse.
- With **3-5 models**, those same techniques can suppress noise while identity signal survives through model diversity.
- The key blocker is therefore not that these methods are intrinsically bad, but that they were evaluated in a **single-model regime** where AIC winners never operated.

## Prioritized Action Plan

| Priority | Action | Expected Impact | Status |
|:--------:|--------|:---------------:|--------|
| **1** | ~~Run the full **10a -> 10b -> 10c** pipeline with the **09s v1 DINOv2** checkpoint~~ | **COMPLETE — DEAD END** (2026-04-25). DINOv2 reached only **0.744 MTMC IDF1** (best, with AFLink), **-3.1pp** vs ViT-B/16 CLIP. Higher mAP did not help. | **CLOSED** |
| **1** | Investigate SAM2-assisted or camera-aware fine-tuning to improve cross-camera invariance for DINOv2 embeddings, OR pursue GNN edge classification for association | DINOv2 mAP is a genuine breakthrough at the single-camera level; the gap is now confirmed to be cross-camera training methodology, not backbone quality | NEXT STEP |
| **2** | Paper writing: "mAP vs MTMC IDF1 — Training Methodology Matters More Than Model Capacity" | The DINOv2 result adds a powerful new data point to the ablation story. Four experiments now show mAP gains that hurt MTMC. | RECOMMENDED |

Association tuning remains exhausted (**225+** configs tested), and the secondary-model / score-level fusion route is now exhausted as well. But the project no longer needs to speculate about another representation jump: **09s v1 already delivered one**. The full **10a → 10b → 10c** DINOv2 pipeline completed on **2026-04-25** and produced a definitive answer: **DINOv2 ViT-L/14 achieves MTMC IDF1 = 0.744** (best, with AFLink `gap=150`, `dir_cos=0.85`), which is **-3.1pp below** the best available-weight fusion result of **0.7703**. Despite +6.65pp mAP, the stronger pretrained representation did not translate into better cross-camera association. The key finding is that **ReID mAP does not predict MTMC IDF1** — even for a model with a major mAP breakthrough. The decisive variable is training methodology for cross-camera invariance (TransReID recipe + CLIP pretraining), not raw model capacity. AFLink showed model-specific behavior: it gained +5.6pp for DINOv2 (0.688→0.744), unlike ViT-B/16 CLIP where it always hurts (-3.82pp to -13.2pp). Even with this boost, DINOv2 could not match the best verified available-weight result. **The best reproducible MTMC IDF1 is now 0.7703 (CLIP+DINOv2 score fusion, 10c v15 / 10a v7)**.

## Active Experiments (2026-04-25)

### Large-Backbone Feature Quality Follow-Up + DINOv2 MTMC Results (April 25, 2026)

- **09r v7 - ViT-L TransReID**: **FAILED** at **mAP=60.38% / R1=76.57%** (best epoch **108/120**) using `vit_large_patch16_224.augreg_in21k_ft_in1k`. This is **-19.76pp mAP** versus the prior deployed **ViT-B/16 CLIP** baseline and confirms that large backbone size without CLIP/DINOv2-quality pretraining is not enough.
- **09s v1 - DINOv2 ViT-L/14**: **BREAKTHROUGH** at **mAP=86.79% / R1=96.15%** (best epoch **115/120**) using `vit_large_patch14_dinov2.lvd142m`. This is **+6.65pp mAP / +3.88pp R1** over the previous deployed **ViT-B/16 CLIP** baseline.
- **Interpretation**: CLIP-style pretraining is important, but **DINOv2 LVD-142M pretraining matters even more** for cross-camera vehicle ReID in this project. Model size alone failed; pretrained representation quality delivered the first genuine breakthrough.
- **COMPLETE (2026-04-25)**: full pipeline run `10a (mtmc-10a-dinov2) -> 10b (mtmc-10b-dinov2-stage-3-faiss-indexing) -> 10c (mtmc-10c-dinov2-stages-4-5-association-eval) v2`. Results: baseline (no AFLink) **MTMC IDF1=0.688**, per-cam IDF1=0.794. Best (AFLink gap=150, dir_cos=0.85): **MTMC IDF1=0.744**, IDF1=0.755, MOTA=0.624, HOTA=0.547. DINOv2's +6.65pp mAP did NOT improve MTMC IDF1 — the result is **-3.1pp below ViT-B/16 CLIP (0.775)**. Training methodology for cross-camera invariance (TransReID+CLIP recipe) is the deciding factor, not raw model capacity.

### 3-Way Ensemble Attempt (April 22, 2026)

- **What we're trying**: 3-way score fusion of primary ViT-B/16 CLIP (80.14% mAP) + secondary R50-IBN (52.77% mAP) + tertiary LAION-2B CLIP (78.61% mAP) using w2 and w3 weights tuned in a 19-point sweep.
- **Motivation**: Both 2-way fusion paths (R50-IBN secondary, LAION CLIP secondary) gave only marginal gains. A 3-way combination has not been tested on a correct feature baseline.
- **Status**: 10a v5 **COMPLETE** (929 tracklets, 49.4 min); 10b v3 **COMPLETE** (12.6 MB FAISS); 10c v8 **COMPLETE** (MTMC_ONLY=True bug — biased); 10c v9 **COMPLETE** (unbiased results confirmed).
- **10c v9 unbiased results (final)**: Baseline = **76.625%** (IDF1=78.419%, MOTA=66.910%, HOTA=57.031%). Best: **w2=0.05, w3=0.30 → 76.817% (+0.192pp)**. R50-IBN alone: −0.064pp. LAION tertiary alone at w3=0.30: +0.154pp.
- **Baseline note**: 76.625% is ~0.74pp below expected 77.36%. The clean Phase C retest on `fix/baseline-drift` (`7e242f6`) with `yahiaakhalafallah/mtmc-10a-stages-0-2` v8 -> `yahiaakhalafallah/mtmc-10b-stage-3-faiss-indexing` v6 -> `yahiaakhalafallah/mtmc-10c-stages-4-5-association-eval` v17 changed only `camera_bn.enabled: true -> false` and recovered **76.66%** (**+0.03pp**), so `camera_bn` is not the drift cause.
- **Conclusion**: 3-way ensemble **CONFIRMED DEAD END** — +0.192pp within noise. The later OSNet checkpoint audit closed the broader drift investigation: the historical v80 target is not reproducible because its secondary checkpoint is no longer available.

#### Broken Baseline Run (10a v4 yahiaakhalafallah — INVALIDATED)
- 10a v4 had wrong overrides: concat_patch=true (→ 1536D, PCA expects 768D) and camera_bn.enabled=false (disables cross-camera BN).
- Result: **-3.79pp regression** — measured baseline 73.57% vs expected ~77.36%.
- Best fusion on broken features: w2=0.05, w3=0.15 → 73.73% (+0.16pp), meaningless.
- **All 10a v4 / 10c (yahia) v5 results are INVALIDATED.** Do not use these numbers.

#### Infrastructure Fix: Cross-Account Kaggle Dataset Mounting
- Public Kaggle datasets do **not** auto-mount cross-account at runtime.
- Fix: add a download cell using the kernel's own KAGGLE_KEY env var to retrieve another account's outputs.
- Applied to 10b kernel (retargeted to consume yahia's 10a output).

## Latest Experiment Results (2026-04-20)

### Completed

#### 09o v1 — EVA02 ViT-B/16 CLIP on CityFlowV2 Dead End (2026-04-20)
- **Kernel**: `gumfreddy/09o-eva02-vit-cityflowv2`
- **Task**: Test whether **EVA02 ViT-B/16 CLIP** can provide a stronger, more complementary secondary vehicle ReID backbone for ensemble use on CityFlowV2
- **Training**: **120 epochs**, **AdamW**, **backbone_lr=1e-5**, **head_lr=5e-4**, **CE + Triplet + Center**, **cosine schedule**
- **Result**: **mAP = 48.17%**, **R1 = 65.90%**, **R5 = 77.17%**, **R10 = 82.83%**
- **Comparison**: This is far below the primary **ViT-B/16 CLIP** baseline at **80.14% mAP** and even below the fine-tuned **FastReID SBS R50-IBN** secondary at **63.64% mAP**
- **Interpretation**: The current **EVA02** transfer recipe does not adapt well to fine-grained vehicle ReID on CityFlowV2. The backbone retains some ranking signal, but it is nowhere near the **>=65% mAP** practical floor for a useful ensemble secondary
- **Conclusion**: **EVA02 ViT-B/16 CLIP is a confirmed dead end with the current recipe**. At **48.17% mAP**, it is too weak for ensemble use and does not justify further score-fusion follow-up in its current form.

#### 09m v2 — CLIP RN50x4 CNN Secondary Model Dead End (2026-04-19)
- **Kernel**: `gumfreddy/09m-clip-rn50x4-vehicle-reid-cityflowv2`
- **Task**: Test whether a **CLIP RN50x4 CNN** backbone can provide an architecturally diverse secondary vehicle ReID model for score-level fusion
- **Result**: **Catastrophic failure** with **best mAP = 1.55%** and **best R1 = 4.18%**
- **Training dynamics**: Cross-entropy loss still converged from **6.57 -> 0.99** over **200 epochs**, so the run did not fail through obvious divergence or undertraining
- **Interpretation**: The backbone learned something for closed-set classification but produced retrieval features that were effectively useless for cross-camera ReID
- **Root causes**: **(1)** `open_clip` constructed the model with **`quick_gelu=False`** while the OpenAI pretrained weights expect **`quick_gelu=True`**, corrupting the transferred feature geometry; **(2)** the **attention-pooling CNN CLIP** architecture appears poorly matched to the standard ReID projection-head recipe used for the ViT-based runs; **(3)** **640D CNN features** appear fundamentally harder to adapt for fine-grained vehicle ReID than the current **768D ViT** features
- **Conclusion**: **CLIP RN50x4 CNN is a confirmed dead end** for CityFlowV2 vehicle ReID. Combined with the failed non-CLIP secondaries and the **10c v56** correlated-CLIP fusion regression, this means the **score-level ensemble path is now fully exhausted** and there is **no viable secondary model** on the current codebase.

#### 10c v56 — LAION-2B CLIP Score-Level Fusion Dead End (2026-04-18)
- **Kernel**: `gumfreddy/10c-stages45-cityflowv2-association-eval` **v56**
- **Task**: Test whether adding the new **TransReID ViT-B/16 LAION-2B CLIP 256px** secondary model improves vehicle MTMC through simple score-level fusion
- **Secondary model**: **TransReID ViT-B/16 LAION-2B CLIP 256px** from **09l v3** with **mAP = 78.61%** and **R1 = 90.43%**
- **Fusion method**: score-level fusion with **`sim = 0.70 * primary_sim + 0.30 * secondary_sim`**
- **Result**: **MTMC IDF1 = 0.769** with fusion versus **0.774** without fusion, a **-0.5pp** regression
- **Metric deltas**: all key metrics regressed together, with **IDF1 -0.2pp**, **MOTA -0.3pp**, and **HOTA -0.2pp**
- **Interpretation**: This confirms the earlier **52.77%** secondary-model fusion failure was not just caused by a weak second model. Even with a much stronger **78.61% mAP** secondary, score-level fusion still hurts.
- **Root cause**: The primary and secondary are both **CLIP ViT-B/16** models, differing mainly in backbone pretraining (**OpenAI CLIP** vs **LAION-2B CLIP**). Their features are therefore too highly correlated, so the ensemble adds noise rather than complementary identity signal.
- **Lesson**: A useful ensemble here likely requires **architecturally diverse** models with meaningfully different feature biases, such as different backbones, input sizes, or training objectives. Two **CLIP ViT-B/16** variants are too similar.
- **Conclusion**: **LAION-2B CLIP score-level fusion is a confirmed dead end** for the current vehicle MTMC pipeline despite the strong standalone **09l v3** model.

#### 10c v60 — Fine-Tuned R50-IBN Fusion Sweep (10a v37) (2026-04-20)
- **Kernel**: `gumfreddy/mtmc-10c-stages-4-5-association-eval` **v60**
- **Task**: Test whether deploying the fine-tuned **FastReID SBS R50-IBN** secondary model improves vehicle MTMC through score-level fusion on top of the restored **10a v37 -> 10b v22 -> 10c v60** pipeline
- **Secondary model**: **FastReID SBS R50-IBN** from **09n** with **mAP = 63.64%** and **R1 = 78.69%** on CityFlowV2
- **Fusion sweep**: evaluated weights **[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]**
- **Best result**: **w = 0.10 -> MTMC IDF1 = 0.7736**, only **+0.06pp** over the **w = 0.00** baseline at **0.7730**
- **Higher weights hurt**: **w = 0.15 -> 0.7713** (**-0.18pp**) and **w = 0.50 -> 0.7625** (**-1.05pp**)
- **Interpretation**: Fine-tuning improved the R50-IBN secondary substantially over the earlier zero-shot ResNet baseline (**63.64% vs 52.77% mAP**), but the downstream MTMC gain remains negligible.
- **Conclusion**: Even a fine-tuned **63.64% mAP** R50-IBN secondary is still too weak for meaningful ensemble gain. A useful secondary likely needs **>=70% mAP** on CityFlowV2 and genuinely complementary feature biases. This confirms the broader dead end for **ResNet-IBN** score-level fusion secondaries, including the already exhausted **ResNet101-IBN-a** path.

#### 10c v61 — Improved 09p R50-IBN Fusion Sweep Still Near Prior Ceiling (10a `run_kaggle_20260420_201401`, 10b v23) (2026-04-20)
- **Kernel**: `gumfreddy/mtmc-10c-stages-4-5-association-eval` **v61**
- **Task**: Re-run the score-level fusion sweep using the improved **09p FastReID SBS R50-IBN** secondary embeddings from the newer **10a** chain on top of **10a `run_kaggle_20260420_201401` -> 10b v23 -> 10c v61**
- **Secondary model**: improved **09p** R50-IBN path deployed through the updated **10a** extraction chain
- **Fusion sweep**: evaluated weights **[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]**
- **Results**:
	- **w = 0.00 -> MTMC IDF1 = 0.773021**
	- **w = 0.05 -> MTMC IDF1 = 0.773255**
	- **w = 0.10 -> MTMC IDF1 = 0.773595** (**BEST**)
	- **w = 0.15 -> MTMC IDF1 = 0.772622**
	- **w = 0.20 -> MTMC IDF1 = 0.771648**
	- **w = 0.25 -> MTMC IDF1 = 0.771440**
	- **w = 0.30 -> MTMC IDF1 = 0.771440**
	- **w = 0.40 -> MTMC IDF1 = 0.770557**
	- **w = 0.50 -> MTMC IDF1 = 0.761920**
- **Canonical result for this run**: use the **fusion-sweep best** at **0.773595 MTMC IDF1** rather than the one-pass Stage-5 log line that reported **[MTMC] IDF1=77.1%, MOTA=68.9%, HOTA=57.5%, IDSW=198**
- **Interpretation**: Even with improved **09p** secondary training and more robust ingestion, the gain remains only **+0.000574** over the **w = 0.00** baseline. That is effectively flat, still below the current reproducible **0.775** result, and far below the historical **0.784** / SOTA regime.
- **Conclusion**: This further confirms that the bottleneck remains **primary feature quality / architecture**, not a missed association sweep. The vehicle pipeline is still pinned near the same **~0.773-0.775** ceiling.

#### 09l v1 — LAION-2B CLIP CircleLoss Failure (2026-04-17)
- **Kernel**: `gumfreddy/09l-transreid-laion-2b-training` **v1**
- **Task**: Test **TransReID ViT-B/16 LAION-2B CLIP 256px** as an alternative **CLIP-family** secondary vehicle ReID backbone for future ensemble use
- **Config**: **Experiment B: CircleLoss ablation** recipe with **CE+LS(eps=0.05) + CircleLoss(m=0.25, gamma=128) + CenterLoss**, **backbone lr=1e-4**, **head lr=1e-3**, **LLRD=0.75**, **120 epochs**, **EMA disabled**, **CLIP-only init**
- **Result**: **Catastrophic failure** with **mAP = 20.36%**, **R1 = 53.03%**, **mAP_rr = 27.16%**
- **Training stability**: Loss was **`inf` throughout all epochs**, matching the earlier **09 v4 / Experiment B** failure pattern exactly
- **Interpretation**: This does **not** indicate that **LAION-2B CLIP** is weak. The training recipe itself was broken: **CircleLoss(gamma=128)** overflows in **fp16 autocast**, so the run never had a chance to evaluate the backbone fairly
- **Follow-up**: **09l v2** completed and confirmed that the backbone is viable once the broken CircleLoss recipe is removed
- **Conclusion**: Treat **09l v1** as further evidence that the current **CircleLoss** recipe is a dead end, not as evidence against the **LAION-2B** backbone

#### 09l v2 — LAION-2B CLIP TripletLoss Rerun (2026-04-18)
- **Kernel**: `gumfreddy/09l-transreid-laion-2b-training` **v2**
- **Task**: Re-evaluate **TransReID ViT-B/16 LAION-2B CLIP 256px** as a secondary vehicle ReID backbone using the stable training recipe instead of the broken CircleLoss ablation
- **Config**: **CE+LS(eps=0.05) + TripletLoss(m=0.3) + CenterLoss(weight=5e-4, delayed until epoch 15)**, **LLRD=0.75**, **EMA(decay=0.9999)**, **160 epochs**, **CLIP-only init**
- **Result**: **mAP = 61.51%**, **R1 = 81.41%**, **mAP_rerank = 67.20%**, **R1_rerank = 82.95%**
- **Training time**: **~3.7 hours on Kaggle T4** for the full **160-epoch** run
- **Training dynamics**: The model was still improving strongly at the end: **20e 12.69/35.86 -> 40e 18.47/45.32 -> 60e 25.99/53.58 -> 80e 34.23/61.06 -> 100e 42.25/66.45 -> 120e 49.73/73.05 -> 140e 55.98/77.78 -> 160e 61.51/81.41** (**mAP/R1**)
- **Key finding**: The run is **not converged**. It gained **+5.53pp mAP** in the final **20 epochs**, and the cosine LR reached **0.00** exactly at epoch 160, so optimization was cut off by the schedule rather than by a plateau
- **Interpretation**: **LAION-2B CLIP** is a legitimate live candidate for a secondary ensemble backbone. It already cleared the prior non-CLIP ceiling, but it has not yet reached its own ceiling
- **Follow-up**: **09l v3** completed the extension to **300 total epochs** and converted this backbone from a live candidate into an **ensemble-ready** secondary model

#### 09l v3 — LAION-2B CLIP Extended Training Success (2026-04-18)
- **Kernel**: `gumfreddy/09l-transreid-laion-2b-training` **v3**
- **Task**: Resume from the **09l v2 EMA checkpoint** and determine whether **LAION-2B CLIP** can clear the practical ensemble threshold on CityFlowV2
- **Config**: same stable **CE+LS(eps=0.05) + TripletLoss(m=0.3) + CenterLoss(weight=5e-4, delayed until epoch 15)** recipe, resumed from **epoch 160** with **EMA(decay=0.9999)** and extended to **300 total epochs**
- **Result**: **mAP = 78.61%**, **R1 = 90.43%**, **mAP_rerank = 81.09%**, **R1_rerank = 90.98%**
- **Training time**: **~3.3 hours on Kaggle T4** for the resumed **160 -> 300** phase
- **Training dynamics**: **180e 65.93 -> 200e 68.84 -> 220e 71.49 -> 240e 73.68 -> 260e 75.68 -> 280e 77.26 -> 300e 78.61** (**mAP**); **R1 = 90.43%** at epoch **300**
- **Key finding**: **LAION-2B CLIP** is now a **strong secondary model**, finishing only **1.53pp** behind the deployed **OpenAI CLIP 09b v2** baseline at **80.14% mAP** and well above the **65%** ensemble threshold
- **Conclusion**: The backbone is **ready for score-level fusion deployment**. The next question is now ensemble gain, not backbone viability

#### 09i v1 — ResNet101-IBN-a ArcFace Follow-Up (2026-04-17)
- **Task**: Test whether an **ArcFace-based** ResNet101-IBN-a recipe can lift the secondary vehicle model above the long-standing **52.77% mAP** ceiling and make it ensemble-worthy
- **Recipe**: **ArcFace (s=30, m=0.35) + Triplet (margin=0.3) + Center loss**, warm-started from the best **09d** checkpoint
- **Result**: Best **mAP = 50.80%** at **epoch 100/160**, **R1 = 73.46%**, **mAP_rerank = 54.65%**
- **Training dynamics**: Performance **declined after epoch 100**, indicating classic overfitting on the small **128-ID** CityFlowV2 train split
- **Comparison**: Still **worse than the 09d baseline** at **52.77% mAP**
- **Root cause**: Warm-starting from a **CE-optimized** checkpoint into an **ArcFace angular-margin** objective creates a geometry mismatch, and combining that with **triplet + center** on only **128 classes** produces too many competing objectives for this backbone/data regime
- **Conclusion**: **ArcFace is a dead end for the current ResNet101-IBN-a path**, and the secondary-model rescue attempt did not materialize

#### 10c v55 — CID_BIAS Topology Bias Sweep (2026-04-17)
- **Task**: Test whether a lightweight **topology-derived CID_BIAS** improves Stage-4 association by adding camera-pair-specific similarity offsets on top of the restored baseline recipe
- **Baseline**: Fresh baseline features from **10a v30** with a control run at **MTMC IDF1 = 0.774** and no additive bias
- **Conservative bias**: **+0.02 / -0.10** reached **MTMC IDF1 = 0.764** (**-1.0pp**)
- **Default bias**: **+0.04 / -0.15** reached **MTMC IDF1 = 0.762** (**-1.2pp**)
- **Aggressive bias**: **+0.06 / -0.20** reached **MTMC IDF1 = 0.763** (**-1.2pp**)
- **Interpretation**: Every tested additive bias degraded MTMC, including the most conservative setting. This indicates the current pipeline is already getting the useful camera-pair calibration from **FIC whitening**, and extra CID_BIAS offsets only warp the calibrated similarity geometry.
- **Conclusion**: **Topology CID_BIAS is a confirmed dead end** for the current CityFlowV2 single-model pipeline. Together with the earlier **GT-learned CID_BIAS** regression (**-3.3pp**), this now rules out both learned and hand-shaped additive CID_BIAS variants.

#### 10c v53 — Network Flow Solver vs CC Baseline (2026-04-17)
- **Task**: Test a **network flow / Hungarian-based structural association solver** as a replacement for the existing connected-components merge logic, with merge verification intended to reduce cross-camera conflation
- **Baseline**: Controlled **10c v52** CC run at **MTMC IDF1 = 0.7714**, **MOTA = 0.689**, **HOTA = 0.5747**, **ID switches = 199**, **45 fragmented GT IDs**, **27 conflated predicted IDs**
- **Network flow result**: **MTMC IDF1 = 0.769**, **MOTA = 0.694**, **HOTA = 0.577**, **ID switches = 197**, **46 fragmented GT IDs**, **30 conflated predicted IDs**
- **Delta vs baseline**: **-0.24pp MTMC IDF1**, **+0.5pp MOTA**, **+0.2pp HOTA**, **-2 ID switches**, **+1 fragmented GT ID**, **+3 conflated predicted IDs**
- **Interpretation**: The solver improved some frame-level metrics slightly, but it failed its main design goal. The Hungarian assignment plus merge verification did **not** prevent false merges and instead created **more conflation** than the CC baseline.
- **Conclusion**: **Network flow is neutral to slightly negative** for the current CityFlowV2 vehicle pipeline. It is **not a hard dead end** on the scale of AFLink or CSLS, but it is **not helpful** and does not justify replacing the existing **conflict_free_cc** approach.

#### 10a v29 / 10c v50 — SAM2 Foreground Masking Before Vehicle ReID (2026-04-16)
- **Task**: Test inference-time **SAM2 foreground masking** on vehicle crops before Stage-2 ReID feature extraction using **`facebook/sam2.1-hiera-tiny`**
- **Masking config**: **center-point prompt**, **per-crop mean fill**, **`min_crop_size=48`**
- **10a v29 runtime / scale**: **929 tracklets**, **105.2 min**, versus roughly **~65 min** for the same pipeline without SAM2 masking
- **10c v50 evaluation**: Full **60-config** Stage-4 parameter sweep with **FEATURE_TEST enabled** across **11 parameter dimensions**, including AFLink and camera-pair normalization variants
- **Best result**: **MTMC IDF1 = 0.688**
- **Per-camera 2D metrics**: **MOTA = 0.540**, **IDF1 = 0.704**, **HOTA = 0.515**
- **Baseline comparison**: Current reproducible non-SAM2 baseline is **MTMC IDF1 = 0.775** from **10c v52**, so SAM2 masking is **-8.7pp** worse despite exhaustive association retuning
- **Interpretation**: Foreground masking removes background context such as road surface and nearby scene cues that appear to help cross-camera vehicle re-identification. The masks also likely clip vehicle edges and suppress boundary features that carry identity signal.
- **Conclusion**: **SAM2 foreground masking is a confirmed dead end** for the current CityFlowV2 vehicle pipeline. It adds major runtime cost while degrading both single-camera and MTMC quality, and even aggressive downstream retuning could not recover baseline performance.

#### 09 v4 / Experiment B — CircleLoss Ablation on Baseline Augmentations (2026-04-16)
- **Kernel**: `gumfreddy/09-vehicle-reid-cityflowv2-circleloss-ablation` **v1**
- **Purpose**: Isolate whether **CircleLoss** caused the augoverhaul regression by keeping the original baseline augmentation stack and disabling EMA
- **Config**: **TransReID ViT-B/16 CLIP 256px**, **CE+LS(eps=0.05) + CircleLoss(m=0.25, gamma=128) + CenterLoss**, **120 epochs**, baseline augmentations only: `RandomHorizontalFlip`, `Pad+RandomCrop`, `ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.0)`, `Normalize`, `RandomErasing`
- **Result**: **Catastrophic failure** with **best mAP = 18.45%** and **best R1 = 48.84%**
- **Training stability**: Loss was **`inf` at every epoch**, indicating complete numerical instability rather than a weak-but-valid training regime
- **Interpretation**: This matches the pre-existing dead-end pattern that **circle-style metric learning can catastrophically destabilize training** in this codebase. It also means the prior augoverhaul regression cannot be explained as “CircleLoss helped ReID but hurt MTMC.” When CircleLoss is definitely active on the primary ViT recipe, it destroys training outright.
- **Conclusion**: **CircleLoss is a confirmed dead end** for the primary CityFlowV2 TransReID path. The augoverhaul regression is therefore most likely attributable to the augoverhaul augmentations themselves, unless the earlier augoverhaul training never actually activated CircleLoss because of a config mismatch.

#### 09 v2 — Augmentation Overhaul + EMA (2026-04-13)
- **Task**: Test a stronger augmentation recipe plus model EMA on the primary CityFlowV2 TransReID ViT-B/16 CLIP training path
- **Augmentation overhaul**: Added `RandomGrayscale(p=0.1)`, stronger `ColorJitter(0.3,0.25,0.2,0.05)`, `GaussianBlur(k=5,p=0.2)`, `RandomPerspective(0.1,p=0.2)`, and a wider `RandomErasing` scale range
- **EMA config**: Model Exponential Moving Average with `decay=0.9999`
- **Base model result**: **mAP = 81.59%**, **+1.45pp** over the prior **80.14%** baseline
- **Base model with reranking**: **mAP = 83.12%**
- **EMA model result**: **mAP = 39.09%**
- **EMA model with reranking**: **mAP = 47.76%**
- **Analysis**: The augmentation overhaul clearly works and improves generalization. The EMA branch failed because `decay=0.9999` is too high for a 120-epoch schedule; the averaged weights converge too slowly and were still improving at epoch 120.
- **Downstream outcome**: The corrected end-to-end follow-up in **10c v48** reached only **0.722 MTMC IDF1**, so the higher validation mAP did not transfer to vehicle MTMC.

#### 09 v3 — EMA Training Results (2026-04-14)
- **Task**: Re-test EMA on the primary CityFlowV2 TransReID ViT-B/16 CLIP path using the augoverhaul augmentation stack with the standard **TripletLoss + CenterLoss** recipe and a lower **EMA decay = 0.999**
- **Base model result**: **mAP = 81.53%**, **R1 = 92.41%**
- **EMA model result**: **mAP = 81.44%**, **R1 = 92.74%**
- **EMA delta vs base**: **-0.09pp mAP**, **+0.33pp R1**
- **Baseline comparison**: Relative to the prior **80.14% mAP / 92.27% R1** baseline, the augoverhaul + EMA training run still gains **+1.39pp mAP** overall
- **Interpretation**: EMA converges to essentially the same solution as the base model. The tiny validation difference is not meaningful enough to justify carrying a second checkpoint or treating EMA as a distinct improvement path.
- **Downstream implication**: Combined with **10c v48 = 0.722** and **10c v49 = 0.722 MTMC IDF1** (**-5.3pp** vs baseline in both cases), this further confirms that higher single-camera **mAP** is **not** the MTMC bottleneck in the current vehicle pipeline.
- **Conclusion**: **EMA is a dead end** for this recipe. It produces nearly identical validation quality to the base model and does not change the core finding that feature-space changes with better mAP are failing to improve MTMC.

#### 10c v47 — Augmentation Overhaul Model with 384px Deployment Bug (2026-04-14)
- **Task**: Evaluate the new **09 v2 augmentation-overhaul** primary model in the downstream **10a -> 10b -> 10c** CityFlowV2 pipeline
- **Result**: **MTMC IDF1 = 0.702**, a **-7.3pp** regression versus the current reproducible **0.775** baseline
- **Root cause**: **BUG**. The **10a** notebook deployed the augoverhaul model at **384x384** input even though it was trained for **256x256**. This happened because `configs/datasets/cityflowv2.yaml` still had `input_size: [384, 384]` from the earlier 384px dead-end work, and the notebook did not explicitly override it for the 256px model.
- **Why this matters**: This result does **not** show that the augmentation-overhaul model is worse. The model itself improved single-camera quality to **mAP = 81.59%** versus the previous **80.14%** baseline, so it should be expected to perform **better**, not worse, when deployed at the correct **256x256** resolution.
- **Fix applied**: Added an explicit `stage2.reid.vehicle.input_size=[256,256]` override to the **10a** notebook and fixed the default `cityflowv2.yaml` input size so 256px deployment is now the safe default for the primary vehicle model.
- **Status**: The fix was verified in **10a v20**, and the corrected **256px** deployment was evaluated end-to-end in **10c v48**.
- **Conclusion**: Treat **10c v47** as a configuration failure, not as evidence against the augmentation overhaul. The apparent regression was caused by a silent deployment mismatch, not by weaker learned features.

#### 10c v48 — Corrected 256px Evaluation of the 09 v2 Augoverhaul Model (2026-04-14)
- **Task**: Re-run the full downstream **10a -> 10b -> 10c** CityFlowV2 pipeline with the **09 v2 augoverhaul** model deployed at the correct **256x256** input size
- **Best association config**: `sim_thresh=0.45`, `appearance_weight=0.60`, `fic_reg=1.00`, `aqe_k=3`, `aflink_gap=150`, `aflink_dir=0.85`
- **Sweep size**: **11 association configs**
- **Result**: **MTMC IDF1 = 0.722**, which is **-5.3pp** versus the current reproducible **0.775** baseline
- **Single-camera quality**: **IDF1 = 0.752**, also materially below the baseline (~0.82)
- **Interpretation**: The 384px deployment bug in **10c v47** was real, but fixing it did **not** rescue the model. The underlying **09 v2** recipe still regresses badly for MTMC even at the correct deployment resolution.
- **Root cause**: The experiment was confounded because it changed both the augmentation stack and the loss at the same time: it added `RandomGrayscale`, stronger `ColorJitter`, `GaussianBlur`, and `RandomPerspective`, while also replacing **TripletLoss** with **CircleLoss**. The stronger augmentations likely suppress color and texture cues that are still needed to distinguish same-model vehicles across cameras, and the simultaneous loss change prevents precise attribution.
- **Conclusion**: The **09 v2 augoverhaul + CircleLoss** recipe is a confirmed **DEAD END** for CityFlowV2 vehicle MTMC despite its higher validation **mAP = 81.59%**.

#### 10c v49 — Association Sweep on the 09 v3 Augoverhaul-EMA Model (2026-04-14)
- **Task**: Re-run the downstream **10c** association sweep using the **09 v3 augoverhaul-EMA** training run (`mAP = 81.53%`, `R1 = 92.41%`) to test whether the EMA-trained augoverhaul recipe changes the MTMC operating point
- **Best association config**: `sim_thresh=0.45`, `appearance_weight=0.60`, `spatiotemporal_weight=0.40`, `fic_reg=1.00`, `aqe_k=3`, `gallery_thresh=0.45`, `orphan_match=0.35`
- **Best result**: **MTMC IDF1 = 0.722**, exactly matching **10c v48** and therefore confirming a persistent augoverhaul ceiling
- **AFLink effect**: With the best non-AFLink operating point, MTMC IDF1 peaked at **0.675**; enabling AFLink with `max_spatial_gap_px=150` and `min_direction_cos=0.85` lifted it to **0.722**, but still did not recover the baseline
- **Reranking**: **Off** was best throughout the sweep
- **Camera-pair normalization**: **Off** was best throughout the sweep
- **Intra-camera merge**: Negligible effect at `thresh=0.80` and `gap=60`
- **Interpretation**: Even after swapping to the **09 v3** augoverhaul checkpoint and expanding the association search, the ceiling stayed fixed at **0.722**. That makes the failure mode robust to association variants and points back to the augoverhaul feature space itself rather than missed stage-4 tuning.
- **Conclusion**: The augoverhaul regression is now confirmed across both **10c v48** and **10c v49**. Different sweeps and slightly different augoverhaul checkpoints converge to the same ceiling, so the model family itself is the issue, not the association parameters.

#### 10c v46 — AFLink Motion-Based Post-Association (2026-04-13)
- **Task**: Test AFLink as a motion-based post-association stage after the standard CityFlowV2 appearance-driven association
- **Merge effect**: AFLink produced **57 motion-based merges**, reducing trajectories from **239 -> 182**
- **Best AFLink params**: `spatial_gap=150px`, `direction_cos=0.85`, `camera_pair_norm=on`
- **Best result**: **MTMC IDF1 = 0.7355**, which is **down from the 0.775 baseline by -3.95pp**
- **Within-sweep behavior**: AFLink gained roughly **+6.3pp** inside its own sweep (**0.669 -> 0.732**) when tuning `spatial_gap` and `direction_cos`
- **Retest status**: This result is now **confirmed by a clean controlled retest** on **10c v52** using the exact restored baseline association recipe (`sim_thresh=0.50`, `appearance_weight=0.70`, `fic_reg=0.50`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`) and fresh baseline features from **10a v30**.
- **Controlled retest**: Control without AFLink reached **MTMC IDF1 = 0.7714**, **IDF1 = 0.7921**, **HOTA = 0.5747**. AFLink then degraded to **0.7332** at `gap=100`, `dir_cos=0.90` (**-3.82pp**), **0.7183** at `gap=150`, `dir_cos=0.85` (**-5.31pp**), and **0.6394** at `gap=200`, `dir_cos=0.70` (**-13.20pp**).
- **Root cause**: The original v46 result was initially suspected to be partly confounded by joint sweep interactions, but the clean retest removes that explanation. Even with a tight **100px** gap and strict **0.90** direction cosine, motion consistency across non-overlapping CityFlowV2 cameras is not reliable enough for identity linking. AFLink creates false cross-camera merges that fragment the identity space, and wider motion windows make the damage worse.
- **Conclusion**: AFLink is **confirmed harmful** on **CityFlowV2** and should be treated as a **DEAD END** for this dataset. The true penalty is now established at roughly **-3.8pp to -13.2pp MTMC IDF1**, not just the original **-3.95pp** point estimate from v46.
- **Key insight**: The fact that even `gap=100` with `dir_cos=0.90` still loses **-3.82pp** shows the problem is structural, not a loose-threshold artifact.

#### 12b v2 — Extended Kalman Sweep (2026-04-13)
- **Task**: Extend the WILDTRACK tracker sweep beyond the original tuned-Kalman search to test whether broader interpolation, longer track persistence, looser detection thresholds, or better interpolation dynamics can reduce the remaining 5 ID switches
- **Sweep extensions**: interpolation **[1,2,3] -> [1,2,3,4,5]**, `max_age` **[2,3,4,5] -> [2,3,4,5,6,8]**, detection confidence **[0.15, 0.20, 0.25, 0.30, 0.35]**, plus velocity-aware quadratic interpolation
- **Best config**: `max_age=2`, `min_hits=2`, `distance_gate=25.0`, `q_std=5.0`, `r_std=10.0`, `interpolation_max_gap=1`, `conf_threshold=0.15`
- **Result**: **IDF1 = 94.7%**, **MODA = 90.0%**, **IDSW = 5**
- **Comparison**: Identical to the prior best operating point; no sweep extension reduced ID switches below **5** or improved IDF1 beyond **0.9467**
- **Conclusion**: Kalman tracker parameters are fully exhausted for the current person pipeline. The remaining gap is tracker-limited, not parameter-limited.

#### 12b v3 — Global Optimal Tracker (2026-04-13)
- **Task**: Implement and test a new `GlobalOptimalTracker` with sliding-window globally optimal assignment (Hungarian algorithm with a configurable window) as a possible replacement for the current Kalman tracker on WILDTRACK
- **Global-optimal config**: `window_size=10`, `birth_death_cost=100`
- **Sweep size**: **59 tracker configs** across **Kalman**, **naive**, and **global optimal** trackers
- **Best Kalman result**: **IDF1 = 94.7%**, **MODA = 90.0%**, **IDSW = 5**
- **Best naive result**: **IDF1 = 93.25%**, **MODA = 88.6%**, **IDSW = 12**
- **Best global-optimal result**: **IDF1 = 91.17%**, **MODA = 88.2%**, **IDSW = 15**
- **Comparison**: Global optimal was the **worst** of all three tracker families, trailing Kalman by **-3.5pp IDF1** with **3x more ID switches**
- **Additional finding**: All tested Kalman configs clustered at **IDF1 = 0.9467 ± 0.0004**, and confidence threshold sweeps across **0.15-0.35** made no meaningful difference
- **Root cause**: Global optimal assignment over-commits to immediate frame-level costs and loses the motion-prediction advantage of Kalman filtering. The predictive model in BoT-SORT handles occlusions and missed detections better than pure assignment optimization.
- **Conclusion**: The person pipeline is **fully converged at IDF1=0.947**. Kalman is the optimal tracker for the current system, and further tracker architecture changes are unlikely to close the remaining **0.6pp** gap to SOTA. That gap is more likely caused by rare occlusion/re-entry cases that need stronger person appearance features or graph-based multi-view fusion, not more tracker improvements.

#### 10c v52 — Vehicle v80-Restored Reproduction
- **Task**: Re-run the vehicle pipeline with the restored v80-style association recipe on the current codebase
- **Config**: `sim_thresh=0.50`, `appearance_weight=0.70`, `fic_reg=0.50`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`
- **Result**: **MTMC IDF1 = 0.775**, IDF1 = 0.801, MOTA = 0.671, HOTA = 0.581
- **Comparison**: Improves over v50 at **0.772**, but still trails historical v80 at **0.784**
- **Conclusion**: Vehicle association remains exhausted. The latest recovery attempt squeezed out only +0.3pp, so future gains need materially better feature quality or stronger priors rather than more stage-4 sweeps.

#### 10a/10c — Score-Level Fusion Ensemble Test (TransReID + ResNet101-IBN-a at 0.30 weight)
- **Task**: Test 2-model score-level fusion with primary TransReID ViT-B/16 CLIP (256px, ~80% mAP) and secondary ResNet101-IBN-a (52.77% mAP) at 30% fusion weight
- **Config**: `fus0.3_ter0.0` — 30% secondary weight, 0% tertiary, on current v52 association recipe
- **Result**: **MTMC IDF1 = 0.774**, IDF1 = 0.803, MOTA = 0.676, HOTA = 0.581
- **Baseline**: v52 single-model: **MTMC IDF1 = 0.775**
- **Delta**: **-0.1pp** (statistically identical)
- **Root cause**: The 28pp mAP gap between primary (80%) and secondary (52.77%) means the secondary model adds noise rather than complementary signal. At 0.30 weight, 30% of the similarity score comes from a model that is essentially random for cross-camera matching. AIC22 winners used ensembles where ALL models exceeded 70% mAP on VeRi-776.
- **Conclusion**: Ensemble with a drastically weaker secondary model is a dead end. The secondary must reach **≥65% mAP** (ideally **>70%**) on CityFlowV2 before ensemble fusion can help. **09l v3** now satisfies that requirement at **78.61% mAP**, so this negative result should be interpreted as a failure of the **52.77%** ResNet secondary specifically, not of score-level fusion in general.

#### 09d v3 (gumfreddy) — Extended ResNet101-IBN-a Fine-Tuning Failure
- **Task**: Resume the best direct ImageNet→CityFlowV2 ResNet101-IBN-a checkpoint (**09d v18**, 52.77% mAP) and continue training with a lower learning rate
- **Config**: Resume from 09d v18 `best_model.pth`, lower `lr=3e-4`
- **Result**: Best **mAP = 50.61%**, which is **worse** than the original **52.77%** checkpoint
- **Conclusion**: Additional fine-tuning from the 52.77% checkpoint degrades the model instead of improving it. The direct ImageNet→CityFlowV2 path appears saturated, and **52.77% is effectively the ceiling** for this ResNet101-IBN-a recipe.

#### 12a v26 — MVDeTr WILDTRACK Detector for Best Tracking Run
- **Task**: Train the detector used by the current best 12b v14 tracking pipeline
- **Config**: MVDeTr `resnet18` backbone, `deform_trans`, 10 epochs
- **Result**: **MODA = 90.9%**, MODP = 81.9%, Precision = 95.8%, Recall = 95.1%
- **Conclusion**: Detection quality is now strong enough that the remaining IDF1 gap appears dominated by missed or imperfect detections rather than association collapse.

#### 12a v3 (gumfreddy) — Extended MVDeTr WILDTRACK Detector Training
- **Task**: Extend WILDTRACK detector training from 10 to 25 epochs on the same MVDeTr `resnet18` setup
- **Config**: MVDeTr `resnet18`, 25 epochs
- **Result**: Best **MODA = 92.1%** at **epoch 20**, with recall in the **96.4-96.6%** range and best-epoch precision **95.7%**
- **Comparison**: Previous best detector was **MODA = 90.9%**, so this is a **+1.2pp** improvement in MODA and roughly **+1.5pp** in recall
- **Trend**: Performance peaked around epoch 20 and declined slightly by epoch 25
- **Conclusion**: This is the **best WILDTRACK detection result yet**, but the follow-up **12b v1** tracking rerun stayed flat, so detector gains alone are no longer enough to move person tracking.

#### 12b v1 (gumfreddy) — Tracking Rerun on 12a v3 Detections
- **Task**: Re-run the WILDTRACK tracking sweep using the improved **12a v3** detector outputs
- **Best tracker**: Kalman `max_age=2`, `min_hits=2`, `distance_gate=25.0`, `q_std=5.0`, `r_std=10.0`
- **Result**: **MODA = 90.0%**, **IDF1 = 94.7%**, Precision = 94.5%, Recall = 96.1%, IDSW = 5
- **Comparison**: Detector quality improved from **90.9%** to **92.1% MODA**, but tracking stayed essentially flat versus **12b v14** (**90.3% MODA**, **94.7% IDF1**)
- **Conclusion**: The tracking sweep converged to the same effective operating point. The current WILDTRACK person pipeline is bottlenecked by the tracker, not the detector.

### In Progress

#### 14r Primary CLIP-ReID VeRi-776 — ABORTED by walltime guard, incomplete (2026-05-10)
- **Spec**: `docs/subagent-specs/14r-primary.md`
- **Kernel**: `mrkdagods/14r-clip-reid-veri-776-train`
- **URL**: https://www.kaggle.com/code/mrkdagods/14r-clip-reid-veri-776-train
- **Frame**: first architecturally orthogonal lever after 14p3/14q closed scale-only CLIP TransReID. Uses the CLIP text tower with learned ID-specific prompts, which the prior TransReID runs never touched.
- **Recipe**: Stage 1 freezes OpenAI CLIP ViT-B/16 image+text towers and learns 4 shared 512-d context tokens plus per-class 512-d ID token via image-text contrastive (`i2tce`), Adam lr 3.5e-4, wd 1e-4, 120ep, `P=8/K=4`, batch 32. Stage 2 unfreezes timm `vit_base_patch16_clip_224.openai` with CE + triplet + i2tce + JPM-CE, AdamW backbone 3.5e-4 / head 3.5e-3, LLRD 0.65, 120ep, 10ep warmup + cosine, AMP fp16, SIE, JPM 4-group, BNNeck.
- **Outcome**: Stage 1 completed 120 epochs in **5.66h**, loss **2.22 -> 1.53**, and saved finite prompts (`text_features [576,512]`, `ctx [4,512]`, `id_tokens [576,512]`). Stage 2 completed epoch 1 (loss **9.85**) before the kernel's own walltime guard raised `RuntimeError`: Stage 2 projected **12.4h** by itself (~6.2 min/epoch x 120 epochs), on top of 5.66h already spent, for a projected total around **18h > 14h cutoff**.
- **Diagnosis**: runtime failure, **not a methodology failure**. The published CLIP-ReID approach has not been evaluated yet. Slowness came from batch 32 (`P=8/K=4`) versus 14q's batch 64 (`P=16/K=4`), giving roughly 2x more steps, plus the `i2tce` auxiliary loss per step.
- **Bug found**: the abort `RuntimeError` fired before writing `train_log.json`, so the structured log was lost and only stdout preserved the metrics.
- **Quota cost**: about **9h MRKDaGods**, leaving roughly **6h**.

#### 14r Probe DINOv2 ViT-B/14 VeRi-776 — FAIL (2026-05-10)
- **Spec**: `docs/subagent-specs/14r-probe.md`
- **Kernel**: `gumfreddy/14r-probe-dinov2-veri-776-train`
- **URL**: https://www.kaggle.com/code/gumfreddy/14r-probe-dinov2-veri-776-train
- **Frame**: standalone SSL-pretrained backbone probe, swapping only the 09v-style backbone to DINOv2 ViT-B/14. DINOv2 has never been measured standalone on VeRi-776 in this repo.
- **Recipe**: timm `vit_base_patch14_dinov2.lvd142m`, 224², `P=8/K=4`, batch 32, `BACKBONE_LR=3.5e-4`, head LR `3.5e-3`, LLRD 0.65, 100ep, 10ep warmup + cosine, AMP fp16, JPM 4-group, BNNeck, SIE, CE label smoothing 0.1 + triplet 0.3 + JPM CE.
- **Result source**: `tmp_14r_probe_outputs/eval_results.json`
- **Results**: `single_flip_cls_base` **81.43% mAP / 97.44% R1**; `single_flip_cls_aqe2_rerank` **88.92% / 97.97%**; `concat_patch_flip_aqe2_rerank` **89.24% / 98.15%**; `concat_patch_flip_aqe3_rerank` **89.27% / 98.15%**.
- **Verdict**: **FAIL**. Best **89.27% mAP / 98.15% R1** is below WIN (**91.54% / 98.33%**) by about **2.3pp mAP / 0.2pp R1** and below the MARGINAL mAP band (**90.5% / 98.0%**) by about **1.2pp**.
- **Significance**: DINOv2 SSL pretraining alone underperforms CLIP pretraining for VeRi-776 (**89.27% vs CLIP's 91.54% post-rerank**). This confirms **CLIP pretraining is necessary, not just any SSL**. R1=98.15% is close to 09v v17's 98.33%, so DINOv2 features may still be useful as a diverse ensemble stream.

#### 14r Recovery CLIP-ReID Stage 2 Resume — FAIL (2026-05-11)
- **Spec**: `docs/subagent-specs/14r-recovery.md` (LOCKED PLAN — 2026-05-10 section)
- **Kernel**: `gumfreddy/14r-recovery-clip-reid-stage-2`
- **URL**: https://www.kaggle.com/code/gumfreddy/14r-recovery-clip-reid-stage-2
- **Strategy**: Stage-2-only resume from the saved Stage 1 prompts uploaded as Kaggle dataset `gumfreddy/14r-clip-reid-stage1-prompts`.
- **Recipe deltas vs original 14r Stage 2**: batch `P=8/K=4 -> P=16/K=4` (size **32 -> 64**), epochs **120 -> 60**, LR sqrt-scaled backbone **3.5e-4 -> 4.95e-4** and head **3.5e-3 -> 4.95e-3**, warmup **10ep -> 5ep**, periodic eval at **[20, 40, 50, 55, 60]**, Stage-2-only walltime guard **4.5h** that writes `train_log.json` before raising.
- **Result source**: `tmp_14r_recovery_outputs/14r_recovery_summary.json`
- **Results**: best concat AQE row `concat_patch_flip_aqe3_rerank_k1_80_k2_15_lambda_0_2` reached **80.55% mAP / 93.68% R1** (R5 **95.53%**, R10 **96.54%**).
- **Verdict**: **FAIL**. This is a **-9.4pp mAP regression** versus 09v v17 ViT-B/16 CLIP (**89.97% mAP / 98.33% R1**) and far below the MARGINAL gate (**90.5% mAP / 98.0% R1**).
- **Conclusion**: CLIP-ReID Stage1 prompts cannot be cleanly continued in a separate Stage2-only kernel. Stage 2 likely needs the full Stage1->Stage2 coupled trajectory, not just final prompt vectors. Single-kernel walltime on T4 (~9h practical budget) is insufficient for the full CLIP-ReID Stage1+Stage2 chain on VeRi-776 at ViT-B/16.

#### Kaggle T4 Dataset Mount Structure Discovery
- **Discovery**: On Kaggle **T4 GPU** kernels, datasets mounted under **`/kaggle/input/datasets/<owner>/<slug>/`** rather than **`/kaggle/input/<slug>/`**
- **Impact**: This caused multiple 09d failures when kernels assumed the legacy flat mount path and could not locate the dataset
- **Conclusion**: When debugging missing-dataset errors on Kaggle T4 kernels, check the nested `datasets/<owner>/` mount structure first.

## Bugs Found / Configuration Pitfalls

### CityFlowV2 `input_size` default leaked a dead-end 384px setting (2026-04-14)
- **Bug**: `configs/datasets/cityflowv2.yaml` still contained `input_size: [384, 384]`, even though **384px deployment is a confirmed dead end** for the main vehicle MTMC pipeline.
- **Failure mode**: The **10a** notebook silently inherited that config and exported the new **09 v2 augmentation-overhaul** model at **384x384** instead of its intended **256x256** input size.
- **Observed impact**: This caused the catastrophic **10c v47 = 0.702 MTMC IDF1** result and could easily have been misread as a model-quality regression.
- **Verification**: The fix was verified in **10a v20**, and the corrected **256px** deployment was then evaluated in **10c v48**.
- **Lesson**: Leaving dead-end experimental defaults in shared config files is dangerous because later notebooks can silently inherit them.
- **Fix**: Set the safe default back to **256x256** and add explicit notebook-level overrides for `stage2.reid.vehicle.input_size` whenever testing alternate deployment resolutions.

#### 12b v4/v5/v6 — Ground-plane Tracking Validation
- **Kernel**: `ali369/12b-wildtrack-mvdetr-tracking-reid` v4-v6, T4 GPU
- **Task**: World-coordinate tracking on MVDeTr detections plus person ReID feature extraction
- **Tracking result**: **MODA = 89.8%, IDF1 = 92.2%, Precision = 97.1%, Recall = 93.9%, IDSW = 12**
- **Outcome**: Ground-plane tracking is working correctly; WILDTRACK tracking quality is strong even before ReID is used for evaluation
- **Issue discovered in v6**: ReID embeddings were mode-collapsed with mean off-diagonal cosine similarity **0.874**
- **Root cause**: Pipeline instantiated CLIP ViT (`vit_base_patch16_clip_224.openai`) while 09p was trained with standard ViT (`vit_base_patch16_224`); CLIP adds `norm_pre`, which was randomly initialized and corrupted features

#### 12b v8 — ViT Backbone Fix
- **Kernel**: `ali369/12b-wildtrack-mvdetr-tracking-reid` v8, T4 GPU
- **Fix**: Matched inference backbone to the 09p training architecture (`vit_base_patch16_224`)
- **Feature quality improvement**:
	- Mean off-diagonal cosine similarity: **0.874 -> 0.720**
	- Similarity range: **[0.675, 1.000] -> [0.215, 1.000]**
	- Merge candidates: **844 -> 382**
	- Weight loading: only 4 minor missing keys (`bn_jpm.*`) instead of all backbone keys missing in v5/v6
- **Tracking metrics**: Unchanged at MODA=89.8%, IDF1=92.2% because ground-plane evaluation does not consume ReID features directly

#### 10a v44 — Local Stage 3-5 Evaluation on Correct 384px + `min_hits=2` Features
- **Input**: `data/kaggle_outputs/10a_v44/run_kaggle_20260329_192418/` with 941 tracklets, 384D primary embeddings, 384D secondary embeddings, 192D HSV
- **Task**: Run local CPU-only stages 3-5 on the exported Kaggle stage-2 artifacts using two association configs
- **Run A (384px-tuned config)**:
	- Overrides: sim=0.54, secondary_weight=0.17, appearance=0.70, gallery=0.50, FIC=0.10, cross_id_nms_iou=0.40, min_traj_frames=40, min_traj_conf=0.30, `gt_frame_clip=true`, `gt_zone_filter=true`, `mtmc_only_submission=false`
	- Result: **MTMC IDF1 = 74.62%**, IDF1 = 78.62%, MOTA = 67.99%
	- Per-camera IDF1: S01_c001 0.926, S01_c002 0.900, S01_c003 0.914, S02_c006 0.637, S02_c007 0.760, S02_c008 0.580
- **Run B (v80-style thresholds on same v44 features)**:
	- Overrides: sim=0.53, secondary_weight=0.10, appearance=0.70, gallery=0.50, FIC=0.10 with the same stage-5 filters as Run A
	- Result: **MTMC IDF1 = 74.74%**, IDF1 = 78.67%, MOTA = 68.07%
	- Per-camera IDF1: S01_c001 0.925, S01_c002 0.900, S01_c003 0.914, S02_c006 0.638, S02_c007 0.762, S02_c008 0.582
- **Outcome**: Both local evaluations underperform the current 78.4% MTMC benchmark by ~3.7pp. The v80-style thresholds are only **+0.12pp** MTMC IDF1 over the 384px-tuned config, so the issue is not primarily the chosen stage-4 thresholds.
- **Important note**: These runs used the current local stage-5 defaults unless overridden. Track smoothing remained enabled, and the requested legacy override aliases had to be translated onto the live config tree (`weights.vehicle.appearance`, `stage5.min_trajectory_*`, `stage5.cross_id_nms_iou`, `stage5.gt_frame_clip`).

#### 10a v43/v44 — Definitive 384px Verdict
- **v43**: 384px features, tuned thresholds, min_hits=3 -> **MTMC IDF1 = 0.7585**
- **v44**: 384px features, exact v43 association recipe, min_hits=2 -> **MTMC IDF1 = 0.7562**
- **v44 + CID_BIAS**: exact v44 config plus a locally-computed 6x6 CID_BIAS matrix from **464/941 GT-matched tracklets** -> **MTMC IDF1 = 0.7510**
- **Baseline for comparison**: v80 256px pipeline, min_hits=2 -> **MTMC IDF1 = 0.7840**
- **Conclusion**: 384px is a confirmed dead end for MTMC in this pipeline, despite strong single-camera mAP. The failure mode is reduced cross-camera invariance, not insufficient stage-4 tuning. CID_BIAS also made the 384px run **worse by -0.52pp**, so camera-pair calibration is not enough to rescue this feature set.

#### 10c v45-v51 — Final Vehicle Ablation Verdict
- **v45 baseline**: late 10c baseline remained in the **0.772-0.775 MTMC IDF1** range depending on exact export and evaluation path
- **v46 (DMT camera-aware training, 87.3% mAP)**: **MTMC IDF1 = 0.758**, which is **-1.4pp** vs the paired v45 baseline at **0.772**
- **v48 (`concat_patch=true`, 1536D features)**: **MTMC IDF1 = 0.773**, which is **-0.3pp** vs v45 at **0.775**
- **v49 (`concat_patch=true` + vehicle2 ensemble)**: **MTMC IDF1 = 0.769**, which is **-0.3pp** vs v50 at **0.772**
- **v50/v51 (current reproducible best / multi-query test)**: **MTMC IDF1 = 0.772**; the multi-query track representation in **v51** was **-0.1pp** vs **v50** and did not improve association quality
- **Conclusion**: Every final feature-side ablation failed to beat the current reproducible baseline. Better single-camera discrimination did not solve the MTMC bottleneck.

## Person Pipeline (WILDTRACK) — New Initiative

### Baseline Performance

| Run | Config Changes | Tracklets | MTMC IDF1 | IDF1 | MOTA |
|-----|---------------|-----------|-----------|------|------|
| Baseline (run_20260327_211115) | Default wildtrack.yaml | 1,339 | 0.176 | 0.316 | -0.281 |
| Exp 1 (run_20260327_224721) | conf=0.55, min_len=8, min_hits=3, match=0.75, merge_gap=40 | 819 | 0.233 | 0.368 | 0.118 |
| Kaggle Baseline (wildtrack_20260328_000939) | 11a→11b→11c chain on Kaggle T4; conf=0.55, min_hits=3, match=0.75, min_tracklet_length=8; ReID 768D→PCA 256D (EV=0.988), HSV 192D, flip_aug=on, camera_bn=on; sim=0.30, louvain=1.5, app=0.80, hsv=0.10, spatio=0.10 | 911 | 0.140 | 0.280 | -0.463 |
| Exp 2 (run_20260327_231511) | conf=0.65, min_len=12, fresh PCA, rerank=off, sim=0.40, louvain=2.0, app=0.90 | INCOMPLETE (interrupted) | — | — | — |

Kaggle baseline details: 911 tracklets across 7 cameras (C1:126, C2:169, C3:154, C4:88, C5:146, C6:121, C7:107), per-camera 2D metrics IDF1=0.280 / MOTA=-0.463 / HOTA=0.000 / IDSW=573, MTMC IDF1=0.140 / MOTA=-0.276 / IDSW=1006, error analysis: 164 fragmented GT IDs, 141 conflated pred IDs, 46 unmatched GT IDs, 141 unmatched pred IDs. Per-camera IDF1/MOTA/IDSW: C1 0.261/-0.012/126, C2 0.191/-0.669/107, C3 0.253/-0.113/127, C4 0.233/-1.468/23, C5 0.316/-0.687/80, C6 0.254/0.004/86, C7 0.450/-0.296/24.

Kaggle underperformed the local Exp 1 baseline (MTMC IDF1 0.140 vs 0.233; per-camera IDF1 0.280 vs 0.368). The likely causes are suboptimal fixed association parameters in 11c and GPU/framework-dependent detection differences. The 911 vs 819 tracklet count gap indicates slightly different detection/tracking behavior on Kaggle hardware/runtime, which plausibly explains part of the per-camera IDF1 regression.

### Key Discoveries (Person Pipeline)
1. **Frame ID off-by-one bug (FIXED)**: WILDTRACK GT was being written with 0-based frame IDs, but predictions use 1-based. Fixed in `scripts/prepare_dataset.py`. This single fix contributed +39.9pp MOTA improvement.
2. **Extreme tracklet fragmentation**: 1,339 tracklets for ~20 people in baseline. Increasing min_hits and min_tracklet_length helped reduce to 819.
3. **Over-detection**: Person detector has low precision (~0.33, 13K FP). Need higher confidence threshold.
4. **PCA model potentially wrong distribution**: Person PCA was trained on vehicle features. Moved to .bak to force refit on WILDTRACK data.
5. **ReID model**: Using TransReID ViT-Base/16 CLIP pretrained on Market1501 (person-specific). Not fine-tuned on WILDTRACK.
6. **All GPU-intensive pipeline stages must run on Kaggle, not locally** (local GTX 1050 Ti too slow).
7. **Kaggle chain currently regresses vs local best**: Same stage-1 thresholds but fixed 11c association settings and possible runtime differences produced 911 tracklets and lower MTMC/per-camera IDF1 than local Exp 1.

### Person Pipeline Next Steps
- Treat person tracking as fully converged at **94.7% IDF1** under the current Kalman pipeline; **12b v1**, **12b v2**, and **12b v3** all confirmed the same ceiling
- Further person gains likely require materially stronger person appearance features or graph-based multi-view association rather than more tracker tuning
- Revisit merge only if person ReID quality improves materially; current Kalman tracks are already clean and merges did not help
- Keep GPU-intensive person stages on Kaggle only

## What Actually Worked

- **Augmentation overhaul**: **+1.45pp mAP** on the primary vehicle ReID model via `RandomGrayscale`, `GaussianBlur`, `RandomPerspective`, stronger `ColorJitter`, and a wider `RandomErasing` range
- **DINOv2 ViT-L/14 pretraining**: **86.79% mAP / 96.15% R1** on CityFlowV2 in **09s v1**, a **+6.65pp mAP / +3.88pp R1** jump over the previous deployed **ViT-B/16 CLIP** baseline and the clearest evidence yet that stronger pretrained representations can move the vehicle MTMC ceiling
- **Deployment-bug correction**: The **10c v47** collapse to **0.702 MTMC IDF1** was correctly traced to a **384px deployment mismatch**; the **10a v20** fix and **10c v48** rerun verified the bug was real, even though the underlying **09 v2** recipe still regressed for MTMC.
- **VeRi-776 single-camera ReID**: 14t score-level fusion establishes the new reproducible high-water mark at **mAP=0.9330 / R1=0.9845** (`w_clipsenet=0.7, w_transreid=0.3`, transreid_768 global token, AQE k=3 + rerank). The best *single-model* result remains **09v v17** at **mAP=0.8997 / R1=0.9833** for the TransReID ViT-B/16 CLIP checkpoint; the earlier v17 eval also records **Best R1=98.33% / mAP=85.14%** (**single_flip rerank** `k1=24,k2=8,λ=0.2`), **Best mAP=89.97% / R1=97.80%** (**concat_patch_flip AQE(k=3)+rerank** `k1=80,k2=15,λ=0.2`), and a **joint optimum=98.15% / 89.71%** (**concat_patch_flip AQE(k=2)+rerank** `k1=80,k2=15,λ=0.2`) in `outputs/09v_veri_v9/`.
- **VeRi-776 ceiling analysis**: The **R1 ceiling on `vehicle_transreid_vit_base_veri776.pth` is 98.33%**. Eval-time techniques are now exhausted: **224x224** remains the winner while **256x256 costs -0.12pp R1**; **flip** helps but **ten-crop is catastrophic at about -2.4pp R1**; **CLS 768D** remains the R1 winner while **CLS+GeM patch 1536D** is only the mAP winner; the rerank grid over **k1∈{20,22,24,25,26,28,30,50,80}**, **k2∈{6,7,8,9,10,15}**, and **λ∈{0.15,0.18,0.2,0.22,0.25,0.3}** peaks at **k1=24,k2=8,λ=0.2**; **AQE k>=2** helps mAP at R1's expense; **SIE enabled** is worth about **+1.5pp R1 / +2.3pp mAP** versus disabled at the 08-baseline rerank row. Multi-scale **224+256** averaging is still blocked by timm's `strict_img_size=True`, so closing the remaining **0.12pp** to the historical **98.45%** claim would require retraining, score-level fusion with a second VeRi model, or a different checkpoint rather than more eval-time tuning.
- Historical claim **`R1=0.984505`** at **AQE(k=3) + rerank `k1=30,k2=10,λ=0.2`** is still resolved as an **R5** value, not an R1 value: the exact reproduced metric is **R5=98.4505%**, while the true **R1=97.68%** and **mAP=86.91%** at that config. The closest current headline match is the new best-R1 row at **R1=98.33% / mAP=85.14% / R5=99.05%**.
- **VeRi paper recreation chain (14p3/14q/14r/14t)**: **CLOSED WITH WIN via 14t fusion**. 14p3 ViT-L/14 CLIP failed at **87.95% mAP** post-rerank; 14q ViT-B/16 CLIP 256² failed at **89.15% mAP**; 14r-probe DINOv2 failed at **89.27% mAP / 98.15% R1**; 14r primary CLIP-ReID aborted on walltime after Stage 1; and 14r Stage2-only recovery regressed to **80.55% mAP / 93.68% R1**. The fusion route succeeded: 14t CLIP-SENet v6 × TransReID 09v v17 score-level fusion reached **mAP=93.30% / R1=98.45%**, the first true Rank-1 reproduction of the historical 0.9845 number on this checkpoint family.

## Conclusive Dead Ends (DO NOT RETRY)

- **14r CLIP-ReID Stage2-only recovery on VeRi-776**: **FAIL** (-9.4pp mAP regression). After 14r primary CLIP-ReID was walltime-aborted post-Stage1, attempting to recover by running Stage2 alone from saved prompts on a fresh notebook regressed to **80.55% mAP / 93.68% R1**. Conclusion: CLIP-ReID Stage1 prompts cannot be cleanly continued in a separate Stage2-only kernel — Stage2 needs the full Stage1->Stage2 coupled trajectory, not just the final prompt vectors. Single-kernel walltime budget on T4 (~9h) is insufficient for the full CLIP-ReID Stage1+Stage2 chain on VeRi-776 at ViT-B/16.

| Approach | Result | Evidence |
|----------|--------|---------|
| Association parameter tuning | Exhausted (225+ configs, all within 0.3pp) | Experiment log |
| CSLS distance | -34.7pp catastrophic | v74 |
| Hierarchical clustering | -1.0 to -5.1pp | v54-56, v62 |
| FAC (Feature Augmented Clustering) | -2.5pp | v26 |
| Feature concatenation (vs score fusion) | -1.6pp | Experiment log |
| CamTTA (Camera Test-Time Adaptation) | Helps global, hurts MTMC | v28-30 |
| Multi-scale TTA | Neutral/harmful | Multiple runs |
| Track smoothing / edge trim | Always harmful | Experiment log |
| Denoise preprocessing | -2.7pp | v46, v82 |
| mtmc_only submission flag | -5pp | Documented |
| Auto-generated zone polygons | -0.4pp | v54-57 |
| PCA dimension search | 384D optimal, others worse | Experiment log |
| Ensemble with 52% secondary at high weight | Dilutes signal | Current state |
| OSNet-IBN-x1.0 from-scratch CityFlowV2 (14m) | 23.80% mAP / 43.89% R1 final; gate FAIL | `data/outputs/14m_final_metrics.json` |
| TransReID ViT-L/14 CLIP @ 224² on VeRi-776 (14p3) | FAIL: base **80.90% mAP / 96.90% R1**; best post-rerank **87.95% mAP / 97.32% R1**, about **2pp mAP / 1pp R1 below** 09v v17 ViT-B/16 at base and about **3.6pp below** post-rerank. ViT-L/14's **304M params** overfit VeRi-776's 576 train IDs / 37k images; pivot to 14q tests **256² resolution on ViT-B/16**, not more capacity. | `tmp_14p3_outputs/eval_results.json`, 2026-05-10 |
| TransReID ViT-B/16 CLIP @ 256² on VeRi-776 (14q) | FAIL: base **79.68% mAP / 96.84% R1**; best post-rerank **89.15% mAP / 97.20% R1**, below 09v v17 (**89.97% mAP / 98.33% R1 base; ~91.54% post-rerank**). Resolution bump 224->256 plus 160 epochs did not beat the 224² recipe; triplet loss saturated to **0.005** by epoch 160. Scale-only CLIP TransReID axis is exhausted under standard CE+triplet supervision. | `tmp_14q_outputs/eval_results.json`, 2026-05-10 |
| DINOv2 ViT-B/14 standalone on VeRi-776 (14r-probe) | FAIL: best post-rerank **89.27% mAP / 98.15% R1** (`concat_patch_flip_aqe3_rerank`), below WIN **91.54% / 98.33%** by about **2.3pp mAP / 0.2pp R1** and below MARGINAL mAP **90.5%** by about **1.2pp**. SSL pretraining alone underperforms CLIP pretraining for VeRi-776; CLIP pretraining is necessary, not just any SSL. | `tmp_14r_probe_outputs/eval_results.json`, 2026-05-10 |
| 14r CLIP-ReID Stage2-only recovery on VeRi-776 | FAIL: best concat AQE row **80.55% mAP / 93.68% R1**, a **-9.4pp mAP regression** versus 09v v17 ViT-B/16 CLIP (**89.97% mAP / 98.33% R1**). After 14r primary CLIP-ReID was walltime-aborted post-Stage1, attempting to recover by running Stage2 alone from saved prompts on a fresh notebook did not recover the baseline. CLIP-ReID Stage1 prompts cannot be cleanly continued in a separate Stage2-only kernel; Stage2 needs the full Stage1->Stage2 coupled trajectory, not just final prompt vectors. Single-kernel walltime budget on T4 (~9h) is insufficient for the full CLIP-ReID Stage1+Stage2 chain on VeRi-776 at ViT-B/16. | `tmp_14r_recovery_outputs/14r_recovery_summary.json`, 2026-05-11 |
| Score-level ensemble with 52.77% mAP secondary at 0.30 weight | -0.1pp MTMC IDF1; noise dilutes primary signal | 10a/10c fusion test, fus0.3_ter0.0 |
| Score-level fusion with CLIP-SENet v6 (91.54% VeRi-776 mAP) on CityFlowV2 | Monotonic degradation; control 0.7679 → −0.13pp at w_cs=0.2, −1.77pp at 0.6, −3.68pp at 0.8, −8.24pp standalone (0.6855). A strong in-domain (VeRi-776) secondary still fails on CityFlowV2 because domain gap dominates secondary-model strength. | 13d v2, 2026-05-07; `outputs/13d_v2/` |
| R50-IBN 4-way score-fusion | MARGINAL plateau at 0.78048-0.78079, +0.0014 vs 14e B1, below WIN bar 0.7810. K13 sanity confirms real ensemble lift, but the gain is too small to promote and closes the CPU-only fusion axis. | 14j v1, 14k v1, 2026-05-08 |
| CLIP-SENet retrain at image_size=256, P=16 (v7) | mAP=81.36% / R1=95.71% on VeRi-776 — **−0.98pp mAP, −0.83pp R1** vs v6 (82.34 / 96.54). 13e-v7 rerank+AQE sweep at image_size=320 reached only **88.98% mAP / 96.31% R1** (rerank k1=50,k2=10,λ=0.1), **−2.56pp mAP** vs v6 post-rerank **91.54%**. Smaller crops lose fine-grained vehicle texture that SENet/AFEM relies on. Paper claim of 92.9% mAP not reachable via crop-size or sampler tuning; likely requires unavailable ingredients (original TinyCLIP weights, no-accum batch-128 BN). v6 (320²) remains canonical. | 13 v7 / 13e-v7, 2026-05-07 |
| Fine-tuned R50-IBN fusion (63.64% mAP) | Only **+0.06pp** MTMC IDF1 at best (`w=0.10`); even with an **11pp** mAP gain over the zero-shot **52.77%** baseline, the secondary is still far too weak for meaningful ensemble benefit | 10c v60 |
| Improved 09p R50-IBN fusion via newer 10a/10b chain | Only **+0.0006** MTMC IDF1 at best (`w=0.10`, **0.773595** vs **0.773021** baseline); improved secondary training plus robust ingestion still leaves the pipeline near the same ceiling | 10c v61 |
| EVA02 ViT-B/16 CLIP | **48.17% mAP** (too weak for ensemble, backbone doesn't transfer well for vehicle ReID) | 09o v1 |
| Score-level ensemble with 78.61% mAP LAION-2B CLIP secondary at 0.30 weight | -0.5pp MTMC IDF1 and all key metrics worse; two CLIP ViT-B/16 variants are too correlated to provide complementary signal | 10c v56 |
| OSNet VeRi-776 secondary (score-level or concat) | torchreid public VeRi-776 OSNet hurts in both tested forms: **76.7%** as score-level fusion (`save_separate=True`, `w=0.10`) and **76.4%** as concat (`save_separate=False`, `1280D -> 384D PCA`). The original `vehicle_osnet_veri776.pth` checkpoint used by the historical v80 path is lost from the weights datasets. | 10a v10 / 10b v7 / 10c v18 and 10a v12 / 10b v8 / 10c v22, 2026-04-25 |
| DINOv2 ViT-L/14 for MTMC (despite +6.65pp mAP) | **MTMC IDF1=0.744** (best, with AFLink gap=150/dir_cos=0.85) vs **0.775** for ViT-B/16 CLIP — **-3.1pp** despite much stronger single-camera mAP (86.79% vs 80.14%). AFLink gains +5.6pp for DINOv2 but cannot bridge the cross-camera invariance gap. ReID mAP does not predict MTMC IDF1; training methodology (TransReID cross-camera recipe) is the decisive variable. | 10c DINOv2 v2, 2026-04-25 |
| ViT-L without CLIP/DINOv2 pretraining | **60.38% mAP / 76.57% R1**, which is **-19.76pp mAP** versus the deployed **ViT-B/16 CLIP** baseline (**80.14%**); CLIP/DINOv2-quality pretraining is essential and model size alone does not help | 09r v7 |
| SGD optimizer for ResNet101-IBN-a | 30.27% mAP catastrophic | v18 mrkdagods |
| Circle loss for ResNet101-IBN-a with triplet loss | Catastrophic gradient conflict; NEVER combine on the same feature tensor | 09d v17 (29.6%), 09f v2 (16.2%) vs 09d v18 (52.77%), 09e (62.52%) with circle_weight=0 |
| ArcFace on ResNet101-IBN-a warm-started from 09d | Best **50.80% mAP**, **73.46% R1**, **54.65% mAP_rerank** at epoch 100/160, then overfit; warm-starting a CE-shaped solution into ArcFace created an angular-geometry mismatch and still failed to beat the **52.77%** baseline | 09i v1 |
| ResNeXt101-IBN-a ArcFace with partially compatible pretrained weights | Catastrophic **36.88% mAP**, **62.69% R1**, **40.49% mAP_rerank**; original weights targeted **32x32d** grouped convolutions while the model here used **32x8d**, and the `strict=False` workaround left many layers randomly initialized | 09j v2 |
| CLIP RN50x4 CNN with OpenAI weights | Catastrophic **1.55% mAP**, **4.18% R1** despite CE loss converging from **6.57 -> 0.99** over **200 epochs**; likely broken by the **QuickGELU mismatch**, poor compatibility between the **attention-pooling CNN** backbone and the standard ReID head recipe, and a fundamentally weaker **640D CNN** transfer path for vehicle ReID | 09m v2 |
| CircleLoss on TransReID ViT-B/16 CLIP-family backbones | Catastrophic numerical instability; the original OpenAI CLIP ablation collapsed to **18.45% mAP / 48.84% R1**, and the LAION-2B follow-up collapsed to **20.36% mAP / 53.03% R1 / 27.16% mAP_rr**. In both cases the training loss stayed `inf` throughout, indicating the recipe itself is broken rather than any specific CLIP backbone | `gumfreddy/09-vehicle-reid-cityflowv2-circleloss-ablation` v1, `gumfreddy/09l-transreid-laion-2b-training` v1 |
| ResNet101-IBN-a VeRi-776→CityFlowV2 fine-tuning | mAP=42.7% (09f v3), worse than direct ImageNet→CityFlowV2 (52.77% mAP, 09d v18); secondary-model ensemble path not viable without better pretraining or broader hyperparameter search | 09e v2, 09f v3, 09d v18 |
| Extended fine-tuning from the 09d v18 ResNet101-IBN-a checkpoint | mAP degraded from 52.77% to 50.61% after resuming with lower LR; confirms the direct ImageNet→CityFlowV2 path is already at its ceiling | 09d gumfreddy v3 |
| ResNet101-IBN-a path FULLY EXHAUSTED | Six variants are now closed out: **09d original**, **09d extended**, **09f VeRi-776 transfer**, **09d CircleLoss**, **09d SGD**, and **09i ArcFace** all finished at or below the **52.77%** ceiling. That is far below the **~65%** minimum needed for a useful ensemble, so this backbone path should be **abandoned** for CityFlowV2 vehicle MTMC | 09d, 09f, 09i |
| CLIP ViT backbone when checkpoint uses standard ViT | `norm_pre` randomly initialized, causing mode collapse (cosine sim 0.874) | 12b v5/v6 vs v8 |
| Default Kalman tracker with chi-squared gating on WILDTRACK | Worse than naive baseline; IDF1 fell to 88.9% | 12b v14 sweep |
| ReID merge on top of tuned WILDTRACK Kalman tracks | No improvement over baseline; tracks already clean and only 44 features matched | 12b v14 merge sweep |
| Extended Kalman parameter sweeps (person) | No improvement; 59 tracker configs clustered within +-0.0004 IDF1 around the same 0.947 Kalman operating point, so the tracker parameter space is fully exhausted | 12b v1-v3 |
| Global optimal tracker (person) | -3.5pp IDF1 vs Kalman; immediate assignment costs lost the motion-prediction advantage and produced 15 ID switches | 12b v3 |
| K-reciprocal reranking (with current features) | Always worse | v25, v35 |
| Camera-pair similarity normalization | Zero effect (FIC handles it) | v36 |
| CID_BIAS (camera-pair bias matrix) | -3.3pp MTMC IDF1 on 256px features (0.751 vs 0.784) | v44 + CID_BIAS test |
| confidence_threshold=0.20 | -2.8pp | v45 |
| max_iou_distance=0.5 | -1.6pp | v47 |
| AFLink motion linking | Confirmed harmful in controlled retest: **-3.82pp** even at `gap=100`, `dir_cos=0.90`; broader gaps degrade to **-5.31pp** (`150/0.85`) and **-13.20pp** (`200/0.70`). Motion consistency across non-overlapping cameras is unreliable and false merges dominate. | 10c v46, 10c v52 |
| SAM2 foreground masking | -8.7pp MTMC IDF1 (0.688 vs 0.775 baseline); removes useful background context and clips vehicle boundary features, while increasing runtime from ~65 min to 105.2 min | 10a v29, 10c v50 |
| Multi-crop TTA at Stage 2 + fusion sweep (14c v2 + 14d v1) | **MARGINAL POSITIVE**: best 0.77155 MTMC IDF1 at `w_t=0.50, thr=0.50` on TTA features (+0.13pp vs production 0.7703, +0.07pp vs 14c control). Consistent across `w_t∈[0.50,0.70]` at `thr=0.50`; `thr=0.40` family universally −1.4pp worse. Optimum shifted from production `w_t=0.60` to `w_t=0.50`, indicating TTA genuinely changed primary-embedding distribution. Within ~0.24pp run-to-run noise; new reproducible floor but not promoted to headline. Next: 14e tighter sweep + AQE/FIC re-tune. | 14c v2 + 14d v1, 2026-05-07 |
| 384px TransReID deployment | -2.8pp MTMC IDF1 despite +10pp mAP; v43=0.7585, v44=0.7562 vs v80=0.784 | 10a v43, 10a v44, v80 baseline |
| Augmentation overhaul + CircleLoss (09 v2 augoverhaul model) | -5.3pp MTMC IDF1 (0.722 vs 0.775 baseline) despite +1.45pp mAP; confounded recipe changed both augmentations and loss. Follow-up CircleLoss-only ablation collapsed to **18.45% mAP** with `inf` loss every epoch, so CircleLoss is independently a dead end and the augoverhaul augmentations are the most likely regression source unless the original run had a CircleLoss config mismatch | 10c v48, 09 v2, 09 v4 Experiment B |
| EMA model averaging | Model **mAP=81.53%** vs EMA **mAP=81.44%** with **R1 92.41% vs 92.74%**; EMA converges to essentially the same place and is not a meaningful improvement path | 09 v3 |
| DMT camera-aware training (87.3% mAP) | -1.4pp MTMC IDF1; v46=0.758 vs v45=0.772 | 10c v45-v46 |
| EMA (`decay=0.9999`, 120 epochs) | mAP=39.09%; too slow to converge under the current training schedule and needs a much longer run to be viable | 09 v2 |
| Multi-query track representation | -0.1pp; v51=0.771 vs v50=0.772 | 10c v50-v51 |
| `concat_patch=true` (1536D features) | -0.3pp; v48=0.773 vs v45=0.775 | 10c v45, 10c v48 |
| `concat_patch=true` + vehicle2 ensemble | -0.3pp; v49=0.769 vs v50=0.772 | 10c v49-v50 |

| **concat_patch=true deployment (PCA mismatch)** | **-3.79pp MTMC IDF1** when PCA was trained on 768D; the flag changes ViT embedding from 768D → 1536D, corrupting downstream PCA and FIC whitening. Always verify concat_patch=false for 256px deployment. | 10a v4 (yahia) 2026-04-22 |
| **camera_bn.enabled=false as a baseline-drift fix** | **HYPOTHESIS FALSIFIED**. Clean retest on `fix/baseline-drift` (commit `7e242f6`) with `yahiaakhalafallah/mtmc-10a-stages-0-2` v8, `yahiaakhalafallah/mtmc-10b-stage-3-faiss-indexing` v6, and `yahiaakhalafallah/mtmc-10c-stages-4-5-association-eval` v17 recovered only **0.7666** vs the **0.7663** control (**+0.03pp**) and remained **-0.84pp** below the **0.7750** target. The earlier 10a v4 regression was confounded with `concat_patch=true` and is not valid evidence that this setting explains the drift. | 10a v8, 10b v6, 10c v17; `fix/baseline-drift` @ `7e242f6` |
| **3-way score-level ensemble (primary ViT + R50-IBN secondary + LAION tertiary)** | **CONFIRMED DEAD END** (10c v9 unbiased). Baseline 76.625%; best w2=0.05, w3=0.30 → 76.817% (+0.192pp — within noise). R50-IBN alone: −0.064pp. LAION tertiary alone: +0.154pp at w3=0.30 (marginal). | 10c v8, 10c v9, 2026-04-22 |

- **DINOv2 score-level fusion (10c v15)**: Best `w_tertiary=0.60` gave **MTMC IDF1 = 0.7703**, the previous deployed best before the 14e TTA + AQE k=2 breakthrough. It still did not recover the historical **0.784** v80 peak, which depended on a now-unavailable OSNet checkpoint.

### 384px TransReID Input Resolution (2026-03-30)
- **Result**: -2.8pp MTMC IDF1 (0.7562 vs 0.784 baseline)
- **Details**: 384px ViT-B/16 CLIP has 80.14% single-camera mAP (same as 256px) but performs significantly worse for cross-camera association
- **v43** (min_hits=3, optimized thresholds): MTMC IDF1 = 0.7585
- **v44** (min_hits=2, exact v43 config): MTMC IDF1 = 0.7562
- **Root cause**: Higher resolution captures viewpoint-specific textures (badge positions, reflections, shadow patterns) that help within-camera ReID but hurt cross-camera matching where viewpoint/lighting change dramatically
- **Insight**: The bottleneck is cross-camera feature invariance, not raw discriminative power

## Component Health Summary

| Component | Status | Notes |
|-----------|:------:|-------|
| Detection (YOLO26m) | ✅ OK | Not the bottleneck |
| Tracking (BoT-SORT) | ✅ OK | min_hits=2 optimal |
| Feature Extraction (ViT) | ⚠️ Ceiling | 256px remains best; 384px is a verified dead end for MTMC |
| Feature Processing (PCA) | ✅ OK | 384D optimal |
| Ensemble/Fusion | ❌ Blocked | Tested at 0.30 weight: -0.1pp. Secondary (52.77% mAP) far too weak for ensemble benefit |
| Association (Stage 4) | ✅ Exhausted | 225+ configs plus network-flow v53; CC remains preferred after network flow lost -0.24pp and increased conflation |
| Evaluation | ✅ OK | Under-merging 1.69:1 ratio = feature quality issue |

## Model Training History

### TransReID ViT-B/16 CLIP (Primary)
- 09b v1: mAP=44.9% (40 epochs from 256px init, too aggressive LR)
- 09b v2: mAP=80.14%, R1=92.27% (VeRi-776 pretrained → CityFlowV2 fine-tune) ← **BEST, but wrong checkpoint on Kaggle**
- 09 v2: mAP=81.59%, rerank mAP=83.12% with augmentation overhaul; EMA branch failed at mAP=39.09% with `decay=0.9999` over 120 epochs

### ResNet101-IBN-a (Secondary)
- 09d v12: mAP=21.9% (IBN layer3 bug)
- 09d v13: mAP=11.98% at epoch 19 (timed out)
- 09d v17: mAP=29.6% (wrong recipe: lr=3.5e-4 + circle_weight=0.5)
- 09d v18 ali369: mAP=52.77% (AdamW lr=1e-3, best so far)
- 09d gumfreddy v3: mAP=50.61% after resuming from the 09d v18 52.77% checkpoint with lower `lr=3e-4`; confirmed extra fine-tuning degrades this ImageNet→CityFlowV2 path
- 09d v18 mrkdagods: mAP=30.27% (SGD lr=0.008, failed catastrophically)
- 09e v2: VeRi-776 pretraining complete, mAP=62.52% on VeRi-776 test set (384x384, 120 epochs, triplet+center, cosine, fp16)
- 09f v2: mAP=16.2% catastrophic failure (circle_weight=1.0, bs=32, label_smooth=0.1, lr=7e-5; best model at epoch 4 during warmup)
- 09f v3: mAP=42.7% at epoch 104/120 (circle_weight=0.0, bs=48, label_smoothing=0.05, AdamW lr=3.5e-4, warmup start_factor=0.1, 120 epochs); still worse than direct ImageNet→CityFlowV2 in 09d v18
- 09i v1: mAP=50.80%, R1=73.46%, mAP_rerank=54.65% at epoch 100/160 with ArcFace+Triplet+Center warm-started from 09d; declined afterward and still failed to beat the 52.77% baseline

### CLIP-Family Secondary Path
- 09l v1: mAP=20.36%, R1=53.03%, mAP_rerank=27.16%; broken CircleLoss recipe kept loss at `inf` throughout and does not reflect backbone quality
- 09l v2: mAP=61.51%, R1=81.41%, mAP_rerank=67.20%, R1_rerank=82.95% at epoch 160 with TripletLoss+EMA; strong recovery, but still schedule-limited
- 09l v3: mAP=78.61%, R1=90.43%, mAP_rerank=81.09%, R1_rerank=90.98% at 300 total epochs after resuming from the v2 EMA checkpoint; now the first ensemble-ready secondary model in the repo
- 09o v1: mAP=48.17%, R1=65.90%, R5=77.17%, R10=82.83% after 120 epochs with AdamW, `backbone_lr=1e-5`, `head_lr=5e-4`, and CE+Triplet+Center; substantially weaker than both the primary ViT baseline and the fine-tuned R50-IBN secondary, so EVA02 is a dead end with the current vehicle-ReID recipe

## Key Insight

**The system is NOT broken.** Vehicle MTMC remains capped by camera-invariant feature quality, not association logic. The current headline is **77.94% MTMC IDF1** (14e B1 / 14f A20 / 14h M0 / 14i F0), with a marginal **77.96%** observed in 14i F2 that is not promoted; the historical **78.4%** v80 result is no longer reproducible because it depended on the now-unavailable `vehicle_osnet_veri776.pth` checkpoint. Higher single-camera ReID mAP does **not** automatically translate to better MTMC IDF1 in this pipeline: augmentation overhaul plus CircleLoss (**+1.45pp mAP -> -5.3pp MTMC IDF1**), **384px** deployment (**same mAP -> -2.8pp MTMC IDF1**), and **DMT** camera-aware training (**+7pp mAP -> -1.4pp MTMC IDF1**) all made cross-camera association worse, and the CircleLoss-only Experiment B showed that when CircleLoss is definitely active on the primary ViT recipe it fails catastrophically (**18.45% mAP, `inf` loss every epoch**). That means the augoverhaul regression is most likely driven by the augmentations themselves unless the earlier training had a CircleLoss config mismatch. The key lesson remains that **mAP != MTMC IDF1** for CityFlowV2 vehicle tracking: the MTMC graph needs features with clean, thresholdable cross-camera similarity distributions, not just strong validation ranking. The **DINOv2 ViT-L/14** result (2026-04-25) adds the most decisive data point yet: despite **+6.65pp mAP** over ViT-B/16 CLIP (86.79% vs 80.14%), DINOv2 produced only **0.744 MTMC IDF1**, while the best available-weight fusion run reached only **0.7703** before the TTA/AQE breakthrough. Combined with the augoverhaul, 384px, DMT, and OSNet repro failures, the pattern is now unmistakable: **training methodology for cross-camera invariance** (TransReID + CLIP pretraining + VeRi-776 intermediate step) is the decisive variable, not raw backbone capacity or single-distribution mAP.

At the same time, **09l v3 changes the ensemble outlook materially**. The weak secondary-model paths are still dead ends: **ResNet101-IBN-a** topped out at **52.77% mAP**, the original **FastReID SBS R50-IBN** path reached only **63.64%**, the newer **10c v61** deployment of the improved **09p** R50-IBN secondary still produced only a **+0.0006** MTMC IDF1 gain over baseline, **ViT-Small/16 IN-21k** reached only **48.66%**, **EVA02 ViT-B/16 CLIP** reached only **48.17%**, and **ResNeXt101-IBN-a ArcFace** collapsed to **36.88%** because the available pretrained weights were structurally incompatible. But the **CLIP-family secondary path** is now validated in one specific form: **LAION-2B CLIP 09l v3** reached **78.61% mAP / 90.43% R1 / 81.09% mAP_rerank**, only **1.53pp** behind the deployed **OpenAI CLIP 09b v2** baseline at **80.14% mAP**. That gives the repo its first genuinely strong, ensemble-ready secondary model. On the person side, ground-plane tracking is already strong (**90.3% MODA, 94.7% IDF1**), but rerunning tracking on the stronger **12a v3** detector stayed essentially flat at **90.0% MODA, 94.7% IDF1**, indicating that the current WILDTRACK pipeline is now tracker-limited rather than detector-limited. The remaining vehicle work is no longer about more association sweeps or more single-model feature tweaks; it is now the direct evaluation of score-level fusion with a strong secondary backbone.

## Pending

- **09q v5** pending: Exp B complete at mAP=76.52% (no improvement). Exp A never ran (checkpoint path bug fixed; 09q v5 push pending).
- **Association re-tune** for camera_bn=true features: current 10a v5 baseline 76.625% vs expected 77.36% (−0.74pp). Fresh Stage-4 sweep needed.
- ~~10c v9~~ **COMPLETE** — 3-way ensemble dead end confirmed: 76.625% baseline, 76.817% best ensemble.
- ~~DINOv2 10c pipeline~~ **COMPLETE** — DINOv2 MTMC dead end confirmed: 0.744 MTMC IDF1 (best, AFLink). The best available-weight result is now 0.7703 from CLIP+DINOv2 score fusion.