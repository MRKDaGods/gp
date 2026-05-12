# Post-14k Next Experiment Spec — 14l: Genuinely New Feature Stream

**Date**: 2026-05-08
**Author**: MTMC Planner
**Status**: UPDATED — Candidate 1 failed in 14m; Candidate 3 is running as 14p under VeRi-first framing

## Decision

Run **14l: Genuinely New Feature Stream**. 14k closed the last CPU-only fusion axis as MARGINAL: K7 reached **0.78079** but stayed below the pre-registered WIN bar, K13 confirmed the lift is real ensemble signal, and the headline remains **0.77936**. The next lever must be GPU-side feature quality, not more Stage-4 weighting.

Recommendation ordering:

1. **OSNet-IBN-x1.0 trained on CityFlowV2** — **FAILED in 14m**. The v3 memory-defense notebook completed on gumfreddy, but the final checkpoint reached only **23.80% mAP / 43.89% R1** and failed the **mAP >=75% AND R1 >=90%** gate. See `docs/findings.md` and `docs/subagent-specs/14m-osnet-ibn-train.md` for the resolution. Do not retry this from-scratch path as-is.
2. **EVA-02-L/14 vehicle ReID** — second because it is the highest diversity bet, but materially more expensive.
3. **CLIP TransReID L/14** — now executing as **14p** on gumfreddy under a different frame: **VeRi-776 SOTA recreation first, CityFlowV2 port second**. This is no longer just a CityFlowV2 5-way-fusion candidate; a 14p WIN would gate a 14q port-to-CityFlow run on MRKDaGods.

## Pre-registered Verdict Bands

| Verdict | 14l MTMC IDF1 | Action |
|:-------:|:-------------:|:------|
| **WIN** | ≥ **0.7920** | Promote after a confirmation run. This would close about 1.7pp of the remaining 6.81pp gap to AIC22 SOTA 0.8486. |
| **MARGINAL** | ≥ **0.7820** and < 0.7920 | Document as a real but insufficient feature-stream lift; run targeted fusion/refinement only if the single-stream metrics clear the ensemble thresholds. |
| **FAIL** | < **0.7820** | Architecture diversity is exhausted for this path. Escalate to GNN edge classifier. |

If all three candidates underperform the single-model baseline or fail ensemble eligibility, stop the new-feature-stream branch and move to the GNN-only path.

## Decision Matrix

| Candidate | Why It Is Worth Testing | Estimated Kaggle Budget | Ensemble Eligibility Threshold | Primary Risk | Priority |
|:--|:--|:--|:--|:--|:--:|
| **OSNet-IBN-x1.0 from scratch on CityFlowV2** | Originally intended to replace the lost ali369 v80 0.784 stream with a CNN/IBN architecture-diverse branch. | Completed 14m on gumfreddy in ~6.5h T4 after v3 memory-defense patches. | Gate was ≥75% mAP and R1 ≥90%; actual final was **23.80% mAP / 43.89% R1**, best mAP checkpoint **24.27% / 43.59%**. | **FAILED — see 14m resolution in findings.md.** Too weak for fusion; do not retry from scratch unless the recipe addresses data size, batch/sampler dynamics, and OSNet-specific LR scheduling. | closed |
| **EVA-02-L/14 vehicle ReID** | Different pretraining family and large-capacity visual backbone; best diversity bet after CLIP/DINOv2/TTA/fusion saturation. | ~24–36h P100 total: training, checkpoint selection, Stage-2 extraction, and 5-way fusion sweep. | ≥78% mAP / ≥91% R1 on CityFlowV2 before fusion; ≥75% mAP minimum to enter 5-way sweep. | 09o EVA02 ViT-B/16 CLIP reached only 48.17% mAP, so recipe transfer is uncertain and may need retuning. | 2 |
| **CLIP TransReID L/14** | Same proven training family as the current primary, but now reframed as VeRi-776 SOTA recreation first. The active run is 14p with `vit_large_patch14_clip_224.openai`. | RUNNING on gumfreddy as `gumfreddy/14p-veri-vit-l-14-clip-transreid-train`; ETA ~13h T4, hard 14h cutoff. A WIN launches 14q port-to-CityFlow on MRKDaGods (~30h GPU). | 14p WIN: VeRi-776 base mAP ≥91.5%; MARGINAL: 89.97-91.5%; FAIL: <89.97% (09v v17 ViT-B ceiling). | Lowest diversity if later used in CityFlow fusion, but highest value for the user-clarified VeRi-first paper strategy. | active |

## Candidate 1 — OSNet-IBN-x1.0 Trained on VeRi-776 + CityFlowV2

### Current 14m State (resolved 2026-05-08)

- v1/v2 on `yahiaakhalafallah/14m-osnet-ibn-cityflowv2-train` exposed the TripletLoss AMP dtype issue and then Kaggle system RAM OOM; the v3 memory-defense patch fixed those runtime blockers.
- The v3 memory-defense run completed on `gumfreddy/14m-osnet-ibn-cityflowv2-train` after switching to `KAGGLE_API_TOKEN` auth from `~/.kaggle/gumfreddy_access_token`.
- Final eval at epoch 120: **mAP=23.80%, R1=43.89%, R5=53.72%, R10=60.28%**.
- Best mAP checkpoint at epoch 60: **mAP=24.27%, R1=43.59%, R5=53.91%, R10=60.65%**.
- Gate failed: required **mAP >=75% AND R1 >=90%**. Do not extract Stage-2 features, do not run 14n, and do not add OSNet-IBN-x1.0 from-scratch CityFlowV2 to fusion.

### Training Recipe Outline

- Start from ImageNet-pretrained **OSNet-IBN-x1.0** if available; otherwise use the strongest OSNet-x1.0 weights that load cleanly without shape mismatch.
- Pretrain or adapt on **VeRi-776** using the stable CE label smoothing + triplet recipe; avoid CircleLoss and ArcFace on this first pass.
- Fine-tune on **CityFlowV2** with small learning rate, EMA optional only if the base branch is already healthy, and 256px deployment-compatible crops.
- Validate every candidate checkpoint on CityFlowV2 mAP/R1 and preserve the best mAP and best joint mAP/R1 checkpoints.
- Extract Stage-2 tracklet embeddings on the 14c/14h tracklet set, L2-normalized, one row per 929 tracklets, with no feature leakage.

### Eligibility And Fusion

- Enter MTMC fusion only if CityFlowV2 mAP ≥75% and R1 ≥90%.
- Integrate as a **5-way score-fusion stream** on top of the current primary CLIP TransReID, DINOv2 tertiary, and R50-IBN quaternary infrastructure.
- Sweep `w_q5 ∈ {0.20, 0.25, 0.30, 0.35}` with `thr ∈ {0.46, 0.48, 0.50}`. Rescale the existing streams proportionally unless a sanity probe explicitly fixes balanced weights.
- Include drift control with `w_q5=0.00` reproducing 0.77936 / 154 EXACT before running the sweep.

### Quota Budget

- Training: ~8–10h P100.
- Stage-2 extraction: ~1–2h P100.
- CPU fusion sweep: ~20–30 min.
- Total expected budget: ~12h P100 plus one CPU kernel.

## Candidate 2 — EVA-02-L/14 Vehicle ReID

### Training Recipe Outline

- Build a clean EVA-02-L/14 ReID head with BNNeck and the stable CE label smoothing + triplet + delayed center loss recipe.
- Start with conservative 224px/256px crops for P100 memory safety; only test larger resolution if the baseline run clears 70% mAP.
- Use layer-wise learning-rate decay and warmup; avoid CircleLoss and aggressive augmentation changes until the base run is stable.
- Select checkpoints by CityFlowV2 mAP, R1, and rerank/AQE diagnostics, but do not assume higher mAP transfers to MTMC.
- Extract Stage-2 tracklet embeddings with the same 929-tracklet alignment check and L2 normalization.

### Eligibility And Fusion

- Preferred threshold: CityFlowV2 mAP ≥78% and R1 ≥91%.
- Minimum threshold for any 5-way fusion: mAP ≥75%.
- Sweep as a 5-way stream with `w_q5=0.20–0.35`; include a balanced sanity probe if the best point suppresses the primary below 0.10.
- Promote only via MTMC verdict bands, not single-camera metrics.

### Quota Budget

- Training: ~20–30h P100 depending on resolution and checkpoint cadence.
- Stage-2 extraction: ~2–4h P100.
- CPU fusion sweep: ~30 min.
- Total expected budget: ~24–36h P100 plus one CPU kernel.

## Candidate 3 — CLIP TransReID L/14

**Status update**: this candidate is now running as **14p** on gumfreddy with the user-clarified VeRi-first paper framing. The active slug is `gumfreddy/14p-veri-vit-l-14-clip-transreid-train`; the spec is `docs/subagent-specs/14p-veri-sota-train.md`. The target is VeRi-776 single-model SOTA recreation (WIN >=91.5% base mAP), and only a WIN should launch 14q port-to-CityFlow on MRKDaGods.

### Training Recipe Outline

- Clone the current successful TransReID ViT-B/16 CLIP recipe and change only the backbone to **CLIP TransReID L/14**.
- Keep the stable loss stack: CE label smoothing + triplet + delayed center loss; no CircleLoss, no ArcFace, no broad augmentation overhaul.
- Use 256px-compatible deployment first for direct comparison with the current primary. Consider 224px only if required by pretrained weights.
- Validate single-camera mAP/R1 and compare directly against the current primary (80.14% mAP, 92.27% R1).
- Extract Stage-2 embeddings only if the checkpoint clears the eligibility gate.

### Eligibility And Fusion

- Enter downstream MTMC only if mAP ≥82% and R1 ≥93%, or if qualitative diagnostics show clearly different failure modes despite lower mAP.
- Treat this as a likely replacement-primary or low-weight 5-way stream. Sweep `w_q5 ∈ {0.20, 0.25, 0.30, 0.35}` only after a zero-weight drift gate passes.
- Because diversity is low, require K13-style sanity if the best result mostly suppresses the existing primary.

### Quota Budget

- Training: ~14–16h P100.
- Stage-2 extraction: ~1.5–2h P100.
- CPU fusion sweep: ~20–30 min.
- Total expected budget: ~18h P100 plus one CPU kernel.

## Fusion Integration Plan

For any eligible candidate:

1. Add the new embeddings as a fifth score stream without changing the existing 14e B1 anchor.
2. Run a zero-weight drift gate first and require **0.77936 / id_switches=154 EXACT** within ±0.001 before running the sweep.
3. Sweep `w_q5=0.20–0.35` and `thr ∈ {0.46, 0.48, 0.50}`.
4. Preserve existing `aqe_k=2`, `fic_regularisation=0.5`, PCA 384D, gallery thresholds, intra-merge, temporal-overlap bonus, and `mtmc_only_submission=false`.
5. If a high `w_q5` result wins by suppressing the primary, run a balanced sanity probe before promotion.
6. Promote only if the MTMC result reaches the 14l verdict bands, not because single-camera mAP improves.

## Fallback Path

If OSNet-IBN, EVA-02-L/14, and CLIP TransReID L/14 all fail to clear ensemble eligibility or all produce MTMC IDF1 <0.7820, conclude that architecture-diverse feature streams are exhausted on the current data/recipe. Escalate directly to a **GNN edge classifier** for learned association using the existing Stage-4 candidate-pair graph, with the 0.77936 configuration as the fixed feature baseline.

## Coder Handoff Checklist

1. Treat Candidate 1 as closed: 14m completed and failed the gate. Do not spend more GPU on OSNet-IBN-x1.0 from-scratch CityFlowV2 unless a new spec addresses the documented root causes.
2. Monitor 14p on gumfreddy and record final VeRi metrics in `docs/findings.md` and `docs/experiment-log.md` when it completes.
3. If 14p reaches WIN (>=91.5% base mAP), queue 14q port-to-CityFlow on MRKDaGods; otherwise do not consume MRKDaGods budget.
4. Keep GPU jobs serial and follow Kaggle push safety rules.
5. Do not change the 0.77936 CityFlowV2 headline unless a port/fusion candidate reaches WIN and passes confirmation.