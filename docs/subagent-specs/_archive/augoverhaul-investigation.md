# Augoverhaul mAP→MTMC IDF1 Regression — Falsifiable Hypotheses & Diagnostic Plan

**Date**: 2026-04-25
**Problem**: The `augoverhaul` augmentation upgrade delivered **+1.45pp single-model mAP** (80.14% → 81.59%) but caused a **−5.3pp MTMC IDF1** drop (0.775 → 0.722). The same inverse pattern appeared for **384px deployment** (same mAP ballpark → −2.8pp MTMC IDF1) and for **DMT camera-aware training** (+7pp mAP → −1.4pp MTMC IDF1). Cracking this could unlock the gap to SOTA.

Related prior specs: `augoverhaul-regression-analysis.md`, `augoverhaul-ablation-plan.md`. This spec focuses on **falsifiable diagnostics**, not training re-runs.

---

## 1. Confirmed Facts (from experiment-log.md and findings.md)

| Run | Single-model quality | Downstream MTMC IDF1 | Notes |
|-----|---------------------|----------------------|-------|
| v80 baseline (09b v2, 256px, Triplet) | mAP 80.14%, R1 92.27% | **0.784** (historical) / **0.775** (10c v52 reproducible) | Primary ViT-B/16 CLIP; restored v80 recipe: `sim=0.50, app=0.70, fic=0.50, aqe_k=3, gallery=0.48, orphan=0.38` |
| 09 v2 augoverhaul (+CircleLoss) | **mAP 81.59%** (+1.45pp) | **0.722** (10c v48) | 11-config association re-sweep at corrected 256px deployment; single-cam IDF1 only 0.752 |
| 09 v3 augoverhaul-EMA (TripletLoss, stable) | mAP 81.53%, R1 92.41% | **0.722** (10c v49) | Broader association sweep; best `sim=0.45, app=0.60, st=0.40, fic=1.00, aqe_k=3, gallery=0.45, orphan=0.35`; AFLink recovered 0.675→0.722 but could not break ceiling |
| 384px ViT (09b v2 @ 384) | mAP ≈ baseline | **0.7585 / 0.7562** (v43/v44) | −2.8pp vs 256px baseline |
| DMT camera-aware single-model | +7pp mAP | −1.4pp MTMC IDF1 | Confirmed dead end |

**Augoverhaul augmentation delta** (09b v2 → 09 v2, findings.md §"09 v2 — Augmentation Overhaul + EMA"):
- ColorJitter: `(0.2, 0.15, 0.1, 0.0)` → `(0.3, 0.25, 0.2, 0.05)` (stronger, adds hue)
- **NEW** RandomGrayscale(p=0.1)
- **NEW** GaussianBlur(k=5, sigma=(0.1,2.0), p=0.2)
- **NEW** RandomPerspective(distortion=0.1, p=0.2)
- RandomErasing scale widened (0.33 → 0.4)
- Loss also changed: Triplet → CircleLoss in v2. **09 v3** kept the overhauled augs with **Triplet+EMA** and reproduced the same 0.722 ceiling — confirming the regression is augmentation-driven, not loss-driven.

**Crucial observation**: In v49 the optimal association thresholds **already shifted** relative to baseline: `sim_thresh 0.50 → 0.45`, `appearance_weight 0.70 → 0.60`, `fic_reg 0.50 → 1.00`. This is direct evidence that the augoverhaul feature distribution differs materially from baseline — but re-tuning inside that shifted space did **not** recover the 5pp gap.

---

## 2. Hypotheses Ranked by Probability

### H2 (TOP) — Augmentation destroyed color/viewpoint cues that cross-camera association depends on. Probability ≈ 50%.

**Mechanism**: RandomGrayscale(p=0.1) forces 10% of crops to match shape-only; stronger ColorJitter + GaussianBlur + RandomPerspective further suppress color, fine texture, and viewpoint-specific cues. On CityFlowV2, many vehicles share make/model and **color is often the only discriminator between same-model vehicles across cameras**. mAP on a single query-vs-gallery retrieval task still rises because it averages over many easy same-camera positives where shape+partial-texture is enough. MTMC association fails because the remaining features are no longer thresholdable across cameras — same-model different-vehicle pairs collapse toward each other (false merges) while same-vehicle different-camera pairs lose their discriminative edge.

**Evidence**: Mirrors the 384px dead end (same pattern: more detail preserved intrinsically, worse cross-camera transfer). Findings.md line 93: "higher input resolution ... made cross-camera association worse by emphasizing viewpoint-specific textures that do not transfer well across cameras." Single-cam IDF1 in 10c v48 dropped to **0.752** (vs ~0.82 baseline) — so it is not only a cross-camera issue, the features became less discriminative at the **decision boundary** in general, not less accurate in **ranking** (which is what mAP measures).

**Falsification test** (cheapest: no training, reuse existing v49 features):
- Compute per-query **split retrieval mAP**: same-camera queries→gallery vs cross-camera queries→gallery, on the existing augoverhaul features vs v80 baseline features. Use the CityFlowV2 query/gallery lists that already exist in the 09 eval pipeline.
- Prediction if H2 true: augoverhaul intra-camera mAP ≥ baseline intra-camera mAP (or comparable), but cross-camera mAP is **lower** by ≥ 2pp.
- If true → retrain without RandomGrayscale + revert ColorJitter to baseline; keep RandomErasing widening only. (This is a one-variable ablation — cheaper than the full augoverhaul-ablation-plan.)

### H3 — Augoverhaul shifts gains toward easy (same-camera, similar-viewpoint) positives. Probability ≈ 25%.

**Mechanism**: Same as H2 at the metric level but different locus — the augmentations increase robustness to within-camera noise (blur, grayscale, perspective wobble), which yields many new easy-positive retrievals; hard cross-camera positives were never the bottleneck the augmentations addressed. mAP sums over all queries, so the net rises.

**Evidence**: Aug families that add within-image noise (blur, grayscale, perspective) are known to help in-distribution robustness more than distribution-shift transfer. Consistent with finding that the R1 gain was small (92.27 → 92.41%) while mAP gained 1.45pp — most of the mAP gain came from ranking middle-tier positives more consistently, not from finding new top-1 matches.

**Falsification test** (same diagnostic as H2): Split mAP by camera pair.
- Prediction if H3 true: **intra-camera mAP up ≥ 3pp**, cross-camera mAP flat or slightly down. Distinguished from H2 by requiring intra-camera gain, not just cross-camera loss.
- If true → the mAP metric itself is the wrong proxy; switch to cross-camera-mAP as the ReID gating criterion going forward.

### H1 — Distribution shift silently broke threshold calibration. Probability ≈ 12%.

**Mechanism**: Stronger augmentations broaden the feature distribution; absolute cosine scores shift, so the v52-calibrated `sim_thresh=0.50, app=0.70, fic=0.50` no longer sits at the correct operating point.

**Evidence for**: v49 **already observed** the shift (`sim→0.45, app→0.60, fic→1.00`), which is direct confirmation the distribution moved. **Evidence against**: v49 swept a wide grid around the new optimum and still capped at 0.722. An 11-config sweep plus the v49 broader sweep leaves little room for a missed threshold.

**Falsification test**: Inspect the **similarity-score distributions** of matched vs non-matched pairs (using GT track IDs) for v80 vs augoverhaul features. If H1 is the sole cause, both distributions should remain well-separated, just translated. If they are overlapping/collapsed, H1 is falsified and the problem is discriminative, not calibrative.
- Script: load `features.npy` + GT cross-camera track pairs, compute cosine similarities, plot histograms for same-ID cross-camera pairs vs different-ID cross-camera pairs, compare AUC and Bhattacharyya distance for baseline vs augoverhaul.
- If AUC drops substantially (e.g., 0.95 → 0.88) → H1 falsified, problem is discriminative.
- If AUC is preserved but shifted → H1 confirmed; re-run with finer threshold resolution around v49 optimum and check for sharper peak.

### H4 — FIC whitening regularization is calibrated to the v52 feature distribution. Probability ≈ 8%.

**Mechanism**: FIC's Tikhonov regularizer scales with the per-camera covariance eigenvalue spectrum. Augoverhaul features have different per-camera variance, so fic_reg=0.50 no longer matches.

**Evidence for**: v49's best config used `fic_reg=1.00` (2× baseline), confirming FIC needed adjustment. **Evidence against**: The sweep already went out to fic=1.00 and still hit the same 0.722 ceiling; a much finer or wider sweep has not been done.

**Falsification test**: Rerun 10c on v49 features with a denser FIC sweep: `fic_reg ∈ {0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.0, 3.0, 5.0, 10.0}` plus `fic_off` control. Hold all other association params at v49 optimum.
- Prediction if H4 true: non-monotonic response with peak materially above 0.722 at some untested fic value.
- If false (peak stays ≤ 0.723 flat across range) → FIC is not the limiting factor.

### H5 — PCA whitening collapses the augoverhaul-specific diversity. Probability ≈ 5%.

**Mechanism**: Augoverhaul pushes variance into directions that distinguish augmentation-invariant identity cues; PCA whitening is fit per-run on the gallery and will retain the top-384 eigen-directions. If those top eigen-directions are now dominated by augmentation-induced intra-class variance rather than cross-camera identity variance, whitening amplifies noise.

**Evidence for**: PCA `n_components=384` was chosen on the baseline distribution. **Evidence against**: The PCA is refit per run on the current gallery, so it adapts automatically; and the explained-variance ratio is logged and has not shown a catastrophic drop.

**Falsification test**:
1. Compare explained-variance curves for baseline vs augoverhaul embeddings. Look at cumulative variance at the 384D cut.
2. Run stage 3–5 with PCA **disabled** on both feature sets (raw 768D L2-normed), compare MTMC IDF1 delta. If the augoverhaul gap shrinks materially (e.g., 5.3pp → <2pp) when PCA is off, H5 is supported. If the gap persists, H5 is falsified.
3. Also test PCA `n_components ∈ {256, 384, 512, 640, 768 (off)}` on augoverhaul features.

---

## 3. Recommended Diagnostic Experiment Plan (cheapest first)

**Stage A — Feature-space diagnostics (1 run, no training, < 10 min on existing features)**
1. Load existing 10a v48/v49 augoverhaul embeddings and v80 baseline embeddings (`features.npy`).
2. Compute per-query-camera **split mAP** (intra-camera mAP, cross-camera mAP separately) for both feature sets. This single diagnostic distinguishes H2/H3 from H1/H4/H5 in one shot.
3. On the same features, compute same-ID-cross-camera vs different-ID-cross-camera similarity histograms; report AUC, mean, std. This distinguishes H1 (shift-preserving) from H2/H3 (distribution-collapsing).
4. Log PCA explained variance at 384D for both feature sets.

**Decision gate after Stage A**:
- If intra-cam mAP ≈ or > baseline AND cross-cam mAP ↓ ≥ 2pp AND AUC ↓ ≥ 0.03 → **H2/H3 confirmed**, go to Stage B1.
- If intra-cam ≈ cross-cam deltas (both shift similarly) but AUC preserved → **H1 dominant**, go to Stage B2.
- If PCA explained-variance at 384D drops ≥ 2pp on augoverhaul → **H5 partial**, go to Stage B3.

**Stage B1 — Single-variable augmentation ablation (one Kaggle training run)**
Retrain 09 v2 with the overhauled recipe **minus RandomGrayscale and minus ColorJitter strengthening** (keep only widened RandomErasing + RandomPerspective + GaussianBlur). Re-evaluate mAP and MTMC IDF1. If MTMC recovers to ≥ 0.76 and mAP stays ≥ 80.5% → H2 confirmed, color suppression is the root cause.

**Stage B2 — Extended association calibration (no training, one 10c sweep)**
Run a denser association sweep on existing v49 features: `sim_thresh ∈ {0.35..0.55 step 0.02}`, `app ∈ {0.50..0.80 step 0.05}`, `fic_reg ∈ {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0}`. If no config exceeds 0.725 → H1 falsified as sole cause.

**Stage B3 — PCA ablation (no training)**
Rerun stage 3–5 on augoverhaul features with PCA disabled and with `n_components ∈ {256, 512, 640, 768}`. Report MTMC IDF1 per setting.

**Stage A is the gate — none of the other stages should run until Stage A has produced numbers.** Stage A is quick and will decisively rank the hypotheses.

---

## 4. Response if H_i Is Confirmed

- **H2 confirmed (color/viewpoint invariance damage)** → Retrain the primary ViT with only **widened RandomErasing** from the overhaul; drop RandomGrayscale, revert ColorJitter to baseline, keep GaussianBlur and RandomPerspective only if Stage B1 shows they are neutral. Document that CityFlowV2 vehicle ReID is **color-dependent** and that grayscale/hue augmentation is a confirmed dead end. Use **cross-camera mAP** (not pooled mAP) as the primary ReID gating metric going forward.
- **H3 confirmed (intra-camera gain only)** → Same retraining response as H2. Additionally, replace the mAP selection criterion in the 09 evaluation with **cross-camera mAP** to prevent future augmentation upgrades from being silently anti-correlated with MTMC.
- **H1 confirmed (threshold shift only)** → No retraining needed. Deploy augoverhaul features with the Stage B2 optimum and revise the `findings.md` ceiling estimate upward.
- **H4 confirmed (FIC regularization mismatch)** → Treat `fic_reg` as a per-feature-set hyperparameter and re-tune after every ReID retraining, not once per codebase generation. Add an automated FIC sweep to the standard 10c evaluation chain.
- **H5 confirmed (PCA collapse)** → Move to no-PCA deployment (raw 768D L2-normed) as the default, or replace PCA with per-camera whitening before dimensionality reduction. Update stage2 pipeline.
- **None confirmed (all falsified)** → The regression is not in any of the five proposed mechanisms; promote the **aleatoric feature-geometry** hypothesis (augmentation changed the local manifold in ways that affect clustering but not ranking) and move to GNN edge classification as the next association upgrade.

---

## 5. Key References

- Confirmed facts: `docs/findings.md` §"Current Performance", §"Critical Discovery: 384px Is a Dead End for MTMC", §"09 v2 — Augmentation Overhaul + EMA", §"10c v47/v48/v49"
- Experiment numbers: `docs/experiment-log.md` §"2.3 2026-04 Augoverhaul Downstream Follow-Ups", §"4.1 TransReID ViT-B/16 CLIP (09b)"
- Augmentation code: `notebooks/kaggle/09_vehicle_reid_cityflowv2/09_vehicle_reid_cityflowv2.ipynb` (lines 618-622 for baseline transforms)
- Association code: `src/stage4_association/similarity.py`, `src/stage4_association/fic.py`, `src/stage4_association/query_expansion.py`
- Feature post-processing: `src/stage2_features/pca_whitening.py`, `src/stage2_features/embeddings.py` (camera_aware_batch_normalize)
- Prior analyses: `docs/subagent-specs/augoverhaul-regression-analysis.md`, `docs/subagent-specs/augoverhaul-ablation-plan.md`

---

After writing the file, confirm its path and line count. Do not modify anything else. Do not update findings.md or experiment-log.md.