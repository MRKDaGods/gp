# Implementation Spec: Learned Edge Classifier / Re-ranker for Stage-4 Cross-Camera Association

**Date:** 2026-06-05
**Status:** PROPOSED (design only)
**Pipeline:** Vehicle / CityFlowV2
**Fixed feature baseline:** 14e B1 @ 0.77936 MTMC IDF1 (id_switches=154); current marginal best K7 = 0.78079 (`vehicle_mtmc_14k_v1_k7`).

## 0. TL;DR
Replace the hand-tuned cosine + `graph.similarity_threshold` edge gate in Stage 4 with a **learned per-edge `P(same-vehicle)` model** (v1: LightGBM on ~18 engineered edge features; v2: GNN). The model rescores `combined_sim[(i,j)]` between its production point (`src/stage4_association/pipeline.py:539`) and the graph solve. This is the **only remaining lever that can move `id_switches` off the 154 floor** — every score-fusion / view / pooling / threshold axis is saturated across 6 documented axes. Honest odds of a deployable WIN (>=0.7820): **~20-30%**, gated heavily by a leakage constraint (only the 6 eval cameras are local). Primary value: de-risking the data/feature pipeline for the eventual GNN, and a clean result on whether a learned gate on the *same features* can beat the algebraic threshold.

## 1. Problem analysis
- 225+ fusion configs land at the same `id_switches = 154`. Re-weighting streams shifts `combined_sim` *magnitudes* but not *which pairs cross the threshold* — hence the pinned 154. `docs/dead-ends.md` lists "GNN edge classification (not implemented)" as a viable next move. Start with a GBM (cheaper, CPU, interpretable) before the GNN.
- **Current edge gate (code path):** candidate pairs built at `pipeline.py:417-438` (`_build_all_cross_camera_pairs`, `:1134`); per-stream score fusion at `pipeline.py:491-514` (the per-stream cosines computed there are exactly the classifier features); FIC whitening `pipeline.py:289-295`; AQE `pipeline.py:383-408`; `compute_combined_similarity` (`similarity.py:97`) -> `combined_sim` dict at `pipeline.py:539` (**the single rescoring surface**); graph solve at `pipeline.py:781-874`, gate `sim >= threshold` at `graph_solver.py:69`/`:276`.
- **Why AFLink failed (dead-ends.md):** raw cross-camera motion/velocity/direction is poison on non-overlapping cameras (-4 to -13pp). The edge classifier must therefore **exclude** absolute velocity / direction / extrapolated position, and use only appearance cosines, temporal feasibility (`st_score`), camera-pair topology priors, and tracklet quality.
- **CID_BIAS / CSLS lessons:** anything that globally reshapes the score distribution breaks the conflict-free greedy's descending-similarity ordering. Integrate as a calibrated **blend** (not an additive offset, not a full replace).

## 2. Training-data generation
- **GT:** `data/raw/cityflowv2/<CAM>/gt/gt.txt`, format `frame_id(1-based), global_id, x, y, w, h, conf, -1,-1,-1`. `global_id` is consistent across cameras within a scene = the supervision signal.
- **Frame convention (CLAUDE.md rule 4):** GT is 1-based; internal tracklets 0-based. Convert `gt_frame = internal_frame + 1`. GT bbox is `(x,y,w,h)`; internal is `(x1,y1,x2,y2)`.
- **GT-id assignment to predicted tracklets:** per predicted-tracklet frame, match GT box (same converted frame) with IoU >= 0.5; majority-vote GT id across frames; require >=50% agreement else label tracklet ambiguous (excluded). Reuse `src/stage5_evaluation/` IoU utilities.
- **Pair labels:** cross-camera, same-class, scene-blocked pairs. Positive if same GT id (diff camera); negative if diff id; exclude if either is ambiguous.
- **Class balance:** keep all positives; hard-negative mine all negatives with fused cosine >= 0.3; random-subsample easy tail to ~3x positives; pass `scale_pos_weight = N_neg/N_pos`.
- **LEAKAGE (the crux):** only the 6 eval cameras (S01_c001-3, S02_c006-8) are local; CityFlow train scenes (S03/S04) are NOT. **Never train on the eval cameras.** Two protocols:
  - **Protocol A (local, recommended first):** scene-disjoint CV — train on S02, eval full pipeline on S01 with the classifier on; mirror. 2 folds, ~75-82 ids each (small, high-variance).
  - **Protocol B (proper, Kaggle):** pull real train scenes S03/S04 onto Kaggle, run Stages 1-2 with the frozen 14e recipe, build pairs there, train, eval locally on the untouched 6 cameras. Stage-2 extraction is GPU -> Kaggle; GBM training is CPU.
  - Harness MUST assert disjointness (fail-loud if any eval-fold camera appears in the training set).

## 3. Edge feature vector (~18, all in FIC+AQE space)
Appearance: `cos_primary`, `cos_dinov2`, `cos_r50ibn`, `cos_fused` (today's gate input), `cos_hsv`, `cos_min/max/std` (stream disagreement). Rank/reciprocity: `rank_i_of_j`, `rank_j_of_i`, `is_mutual_top1`, `recip_rank_harmonic`. Temporal (AFLink-safe): `time_gap`, `st_score` (`spatial_temporal.py:94`), `temporal_overlap_ratio`. Camera-pair topology: `camera_pair_id` (categorical), `pair_mean_time`, `pair_max_time`. Quality: `min_track_len`, `len_ratio`, `min_mean_conf`.
> The features a fixed weighted-sum **cannot** represent — `cos_std` (stream disagreement) and `camera_pair_id` (per-pair topology) — are where any genuine gain must come from.

## 4. Model choice
- **v1 LightGBM (CPU, recommended):** trains in seconds, interpretable (SHAP tells us if any learnable signal exists), native categorical + imbalance (`scale_pos_weight`), monotonic constraints (force P non-decreasing in `cos_primary`/`cos_fused`/`st_score`) to protect the greedy ordering. Shallow (depth<=4, high `min_child_samples`, strong L1/L2) for the ~150-300 positives/fold. Isotonic/Platt calibration on the val fold. Fallback: sklearn `HistGradientBoostingClassifier` or small MLP; log a logistic-regression baseline.
- **v2 GNN (Kaggle GPU, upgrade path):** nodes=tracklets, edges=candidate pairs; 2-3 edge-conditioned message-passing layers; edge-BCE + triangle-consistency reg. Exploits graph context (triangle consistency, neighborhood competition) a per-edge GBM can't. Build only after v1 de-risks the pipeline.

## 5. Integration into Stage 4
- **Recommended: `mode=blend`** — `score' = (1-lambda)*combined_sim + lambda*P`, optional secondary gate `P >= tau_p`. Keeps the FIC-calibrated cosine ordering dominant (protects `conflict_free_cc` + gallery/intra-merge thresholds) while P breaks ties / re-gates the borderline 154 residual. lambda=0 is a provable no-op (drift gate).
- **Injection point:** new `src/stage4_association/edge_classifier.py::rescore_edges(...)`, called immediately after `combined_sim` is produced (`pipeline.py:539`), before the Step 5 post-adjustments and the solve. **Fail-loud guard** if enabled but model missing / feature-dim mismatch.
- **Config block (default-off):**
```yaml
    edge_classifier:
      enabled: false
      model_path: "models/association/edge_clf_lgbm.pkl"
      mode: "blend"            # blend | replace | gate
      blend_lambda: 0.5
      prob_threshold: 0.0
      feature_version: 1
      calibrated: true
```
  Overrides use the full `stage4.association.edge_classifier.*` path (CLAUDE.md rule 5).
- Append per-edge `P` to `GlobalTrajectory.evidence` for the forensic trail.

## 6. Train/eval location + protocol
- v1 GBM training: **CPU, local** (tabular, not a GPU stage). v2 GNN: Kaggle GPU. MTMC eval (pipeline with classifier on): local CPU via `run_pipeline.py -s 4,5` reusing cached 14e Stage-0-3 artifacts.
- **Pre-registered protocol:** (1) reproduce baseline (K7 0.78079 / 14e 0.77936 + id_sw=154) with classifier off; (2) lambda=0 no-op must reproduce baseline exactly; (3) scene-disjoint eval (Protocol A both folds + avg, or Protocol B); (4) report `id_switches` AND IDF1 — **key signal = id_switches moving off 154**. A tie at 154 = model re-learned the threshold = no-go.
- **Verdict bands:** WIN >= 0.7820 (promote behind flag + confirmation run); MARGINAL >= 0.7810 (document, flag-off, GNN only if id_sw clearly moved); No-go < 0.7810 or IDF1 flat with id_sw==154.

## 7. Risks + rollback + odds
- **Feature-quality ceiling (load-bearing caveat):** the 154 switches are the residual hard cases where the embeddings themselves don't separate look-alikes. A learned gate on those same cosines may just re-learn the same 154 decisions and tie at 0.77936. It can only win via `cos_std` / `camera_pair_id` / rank / graph-context signals. **Most likely outcome = tie.**
- **Leakage** = biggest validity risk; assert disjointness, fail loud.
- **Overfitting** ~150-300 positives/fold: shallow trees, regularization, monotonic constraints, calibration, report train-vs-val gap.
- **Calibration/greedy interaction:** blend (don't replace), monotonic, isotonic calibration, sweep lambda.
- **Train/infer distribution match:** features computed in identical FIC+AQE space via one shared feature-builder.
- **Rollback:** `enabled=false` (default) -> bit-identical to 14e B1; lambda=0 second safety net; new `.pkl` under `models/association/`.
- **Odds:** P(lambda=0 no-op clean) ~95%; P(v1 moves id_sw off 154 at all) ~40-50%; **P(v1 deployable WIN >=0.7820 on a fair fold) ~20-30%**; P(v2 GNN beats v1 if v1 shows signal) ~40% (multi-week). EV high regardless: reusable labeled-pair pipeline + separability evidence + citable "learned gate can't beat threshold => bottleneck is the embeddings" result.

## 8. First concrete step (DE-RISK GATE)
Write `scripts/build_edge_pairs.py` (offline, read-only on the pipeline): (1) load frozen 14e Stage-1 tracklets + Stage-2 per-stream embeddings (FIC-whitened identically to `pipeline.py`), (2) assign GT `global_id` via IoU majority-vote with the 1-based->0-based frame conversion, (3) emit `edge_pairs_S01.parquet` / `edge_pairs_S02.parquet` with the section-3 features + labels, (4) print AUC of `cos_fused` alone vs label, and scene-disjoint LightGBM held-out AUC + feature importances. **If the held-out GBM AUC on hard negatives doesn't beat the `cos_fused`-threshold baseline, STOP — no learnable signal, GNN not worth building.** ~1 hour CPU, zero pipeline-integration risk.

## 9. Critical files
- `src/stage4_association/pipeline.py` — inject `rescore_edges` between :539 and the solve.
- `src/stage4_association/similarity.py` — `compute_combined_similarity` (:97), hsv (:20), temporal-overlap (:28), length (:249).
- `src/stage4_association/graph_solver.py` — `solve` (:45), `conflict_free_cc` (:243), gate (:69,:276).
- `src/stage4_association/spatial_temporal.py` — `transition_score` (:94).
- `configs/datasets/cityflowv2.yaml` — add `stage4.association.edge_classifier` block; GT `data/raw/cityflowv2` (:324); K7 stream paths (:179-181).
- *(new)* `src/stage4_association/edge_classifier.py`, `scripts/build_edge_pairs.py`.
