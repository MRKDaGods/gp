# CityTrack / Team28 (AIC22 Track-1, 85.45% IDF1) — Component Audit Evidence

> Read-only audit of the MTMC Tracker CityFlowV2 vehicle pipeline against the 7 critical components of the AIC22 Track-1 winning solution. All citations are real `file:line` references verified on disk. Where a technique is absent it is stated plainly.

## Per-Component Verdict Table

| # | Component | Verdict | Key file:line | What we do instead |
|---|-----------|---------|---------------|--------------------|
| 1 | Zone-Based Search Space Reduction | ⚠️ Partial (impl, disabled, harmful) | src/stage4_association/zone_scoring.py:1; pipeline.py:619 | Soft ±0.03 similarity bonus/penalty, NOT hard candidate pruning; `zone_model.enabled: false` (-0.4pp) |
| 2 | Spatio-Temporal Time Window | ✅ Implemented (enabled) | src/stage4_association/spatial_temporal.py:62; similarity.py:200 | Hard gate (drop) outside window + Gaussian soft score inside, instead of ×2 distance penalty |
| 3 | Stationary Sensitive Association (SSA) | ❌ Missing | (none in src/stage1_tracking) | Opposite: stage5 REMOVES stationary cars (stage5_evaluation/pipeline.py:693); no detection-freezing |
| 4 | Trajectory Re-Link (TRL) | ⚠️ Partial | pipeline.py:808 (intra-camera ReID relink, enabled); aflink.py:48 (cross-cam motion relink, disabled) | Intra-camera cosine relink enabled; cross-camera motion AFLink disabled (-3.8 to -13.2pp); no zone-boundary-aware mid-scene relink |
| 5 | Bidirectional Tracking (BT) | ❌ Missing | (none in src/stage1_tracking) | Forward-only BoT-SORT |
| 6 | Occlusion-Aware Distance Matrix | ❌ Missing | (none) | No per-box occlusion rate, no D_final = D×(1+0.1·I(occ≥0.6)) |
| 7 | Box-Grained Matching + k-reciprocal (k=7) | ⚠️ Partial (impl, disabled) | reranking.py:30 (k-recip, disabled); pipeline.py:84 (multi-query box-level, disabled) | Mean-pooled tracklet embeddings + connected-components graph; box-grained only via disabled multi-query; k-recip disabled (hurts) |

## What Stage-4 Actually Does (verified)

Pipeline order (src/stage4_association/pipeline.py, `run_stage4`):
1. Build embedding matrix from **mean/pooled per-tracklet embeddings** (`f.embedding`, stacked at pipeline.py:150). Box-level embeddings are NOT retained by default.
2. FIC per-camera feature whitening (pipeline.py:265; `per_camera_whiten` in fic.py). AIC21 technique.
3. Average Query Expansion (AQE) k=2 + DBA index rebuild (pipeline.py:330-360; query_expansion.py).
4. Exhaustive cross-camera pairwise cosine, `exhaustive_cross_camera: true` (pipeline.py:404).
5. Hard temporal pre-filter of same-camera time-overlapping pairs (pipeline.py:440).
6. Mutual nearest-neighbour filter (pipeline.py:457; similarity.py:53).
7. (Optional, OFF) k-reciprocal re-ranking (pipeline.py:469; reranking.py:30).
8. Score-level fusion with DINOv2 **tertiary** embeddings, `w_tertiary=0.525` (pipeline.py:480-505).
9. Spatio-temporal validation + class-adaptive weighted combine (pipeline.py:509-530; similarity.py:108).
10. (Optional, OFF) camera-pair norm, camera bias, **zone scoring** (pipeline.py:619), camera-pair boost.
11. (Optional, OFF) reciprocal best-match seeding, CSLS, per-pair thresholds; (ON) intra-camera ReID merge (pipeline.py:808).
12. **GraphSolver** builds similarity graph (threshold 0.48) and finds clusters via **`conflict_free_cc`** (greedy conflict-free connected components) — graph_solver.py:108-118. Also available: `connected_components`, `community_detection` (Louvain), `agglomerative`, `network_flow` (Hungarian via `scipy.linear_sum_assignment`). **NOT k-reciprocal NN, NOT Hungarian by default.**
13. Same-camera conflict resolution, optional hierarchical centroid expansion, gallery expansion.

**Tracklet embedding aggregation**: MEAN-pooled to one vector per tracklet (default). Multi-query (`multi_query.enabled: false`) can retain multiple representative embeddings and do max-of-K×K cosine (pipeline.py:84-105) — the only box-grained-style path, disabled by default.

**Matching algorithm**: similarity-graph + connected components (`conflict_free_cc`). NOT k-reciprocal, NOT box-grained, NOT Hungarian.

**Spatial/temporal/zone constraints present?** YES for temporal (enabled hard gate + Gaussian, spatial_temporal.py) and zone (implemented but disabled). No 3D/BEV geometric constraint for vehicles.

### Config keys under `cfg.stage4.association` (configs/default.yaml:185-340, overrides in configs/datasets/cityflowv2.yaml:161-240)
- `tertiary_embeddings.weight` / `w_tertiary` = 0.525
- `fic.enabled: true`, `fic.regularisation` (0.1 default / 0.5 in 14e B1)
- `query_expansion.enabled: true`, `k` (aqe_k=2), `alpha`, `dba`
- `reranking.enabled: false` (k1=30, k2=10, lambda=0.4) — k-reciprocal
- `weights` (appearance/hsv/spatiotemporal, per-class person/vehicle), `length_weight_power`
- `spatiotemporal.min_time_gap`, `max_time_gap`, `camera_transitions` (per-pair learned priors)
- `graph.similarity_threshold` (0.48), `graph.algorithm` (conflict_free_cc), `bridge_prune_margin`, `max_component_size`, `louvain_resolution`
- `gallery_expansion`, `hierarchical`, `mutual_nn`, `exhaustive_cross_camera`, `exhaustive_min_similarity`
- `zone_model.enabled: false` (zone_data_path = configs/datasets/cityflowv2_zones.json, bonus/penalty 0.03)
- `reciprocal_best_match` (off), `csls` (off), `camera_bias` (off), `camera_pair_boost` (off), `intra_camera_merge`, `temporal_overlap`, `aflink` (off), `multi_query` (off)

## Component-by-Component Evidence

### 1. Zone-Based Search Space Reduction — ⚠️ Partial (implemented, disabled, found harmful)
- EVIDENCE: `ZoneScorer` class assigns entry/exit zones by nearest centroid and validates GT transition patterns — src/stage4_association/zone_scoring.py:1-160. Wired into Stage-4 at src/stage4_association/pipeline.py:619-660 (Step 5c). Zone data file exists: configs/datasets/cityflowv2_zones.json.
- WHAT WE DO INSTEAD: It is a SOFT signal — valid transition adds `+0.03`, invalid subtracts `-0.03` (zone_scoring.py:117-160). CityTrack uses zones to PRUNE the candidate search space (only match valid exit→entry zone pairs). We never remove candidates by zone.
- CONFIG: `stage4.association.zone_model.enabled: false` (default.yaml:329-336). Dead-ended at **-0.4pp** (docs/experiment-log.md:59; docs/dead-ends.md).

### 2. Spatio-Temporal Time Window — ✅ Implemented (enabled)
- EVIDENCE: `SpatioTemporalValidator.is_valid_transition` hard-gates by per-pair min/max time and `transition_score` returns a Gaussian centred on learned mean — src/stage4_association/spatial_temporal.py:62-180. Applied in src/stage4_association/similarity.py:189-205 (`if st_score <= 0: continue` drops impossible transitions). Per-pair learned priors (min/max/mean/std) in configs/default.yaml:238-252 (`camera_transitions`).
- WHAT WE DO INSTEAD: CityTrack applies a ×2 distance penalty outside the window; we HARD-DROP pairs outside the window and apply a Gaussian plausibility score inside (stronger gating). Spatiotemporal weight = 0.30 (vehicle, cityflowv2.yaml:217).
- CONFIG: `stage4.association.spatiotemporal.{min_time_gap,max_time_gap,camera_transitions}`.

### 3. Stationary Sensitive Association (SSA) — ❌ Missing
- EVIDENCE: Searched src/stage1_tracking/{tracker.py,pipeline.py,tracklet_builder.py,detector.py}. Only hit is a BoxMOT Kalman `unfreeze` numpy patch (tracker.py:16-57) — unrelated internal KF mechanism, NOT stationary-detection freezing.
- WHAT WE DO INSTEAD (opposite purpose): Stage-5 `_filter_stationary` REMOVES parked/stationary vehicles (src/stage5_evaluation/pipeline.py:693-751), enabled `stationary_filter` disp=150 (experiment-log.md:73). We keep stopped vehicles alive via large BoT-SORT `track_buffer: 450` (~45s) (configs/datasets/cityflowv2.yaml stage1.tracker) but never freeze a high-confidence detection in place of Kalman prediction.
- NOT FOUND after searching stage1 tracker/pipeline/builder and stage4.

### 4. Trajectory Re-Link (TRL) — ⚠️ Partial
- EVIDENCE (closest analogs):
  - Intra-camera ReID relink (ENABLED): src/stage4_association/pipeline.py:808-845 reconnects same-camera non-overlapping tracklets by cosine ≥ threshold within a time gap (`intra_camera_merge`, thresh 0.80, gap 30 — experiment-log.md:48). This is ReID-cosine relink, matching CityTrack's spirit for within-camera fragments.
  - Cross-camera motion relink (DISABLED): src/stage4_association/aflink.py:48 `aflink_post_association` merges trajectory endpoints by velocity/direction-cosine/spatial-gap. This is StrongSORT-style AFLink, not CityTrack TRL.
  - Gallery/hierarchical orphan expansion (pipeline.py:880+) also reconnects orphans by centroid similarity.
- WHAT WE DO INSTEAD: We have ReID-cosine relink intra-camera (greedy by threshold), but NO zone-boundary-aware detection of "broken mid-scene trajectories" specifically. Cross-camera AFLink is dead-ended.
- CONFIG: `intra_camera_merge` (on), `aflink.enabled: false` (-3.8 to -13.2pp, dead-ends.md:12; experiment-log.md).

### 5. Bidirectional Tracking (BT) — ❌ Missing
- EVIDENCE: No `backward`/`reverse`/`bidirectional` logic in src/stage1_tracking. Tracking is forward-only single-pass BoT-SORT (src/stage1_tracking/tracker.py, pipeline.py).
- NOT FOUND after searching stage1 for backward/bidirectional/reverse/forward.

### 6. Occlusion-Aware Distance Matrix — ❌ Missing
- EVIDENCE: No per-box occlusion-rate computation and no `D_final = D_refined × (1 + 0.1 × I(occlusion≥0.6))` anywhere. The only "occlusion" hits are comments: src/stage2_features/pipeline.py:227 and src/stage4_association/pipeline.py:815.
- NOT FOUND after searching src/** for occlusion/occluded/IoU-overlap distance adjustment.

### 7. Box-Grained Matching + k-reciprocal (k=7) — ⚠️ Partial (implemented, disabled)
- EVIDENCE:
  - Tracklet embeddings are MEAN-pooled to one vector (src/stage4_association/pipeline.py:150 `np.stack([f.embedding ...])`). This is the default — NOT box-grained.
  - Box-grained-style path exists via multi-query: `_compute_multi_query_pair_similarity` does max-of-K×K box-level cosine (pipeline.py:84-105), but `multi_query.enabled: false` (default.yaml:188).
  - k-reciprocal re-ranking implemented: `k_reciprocal_rerank` (src/stage4_association/reranking.py:30) with weighted Jaccard; default `k1=30, k2=10` (NOT k=7), and `reranking.enabled: false`.
  - Matching is graph connected-components (`conflict_free_cc`), graph_solver.py:108-118 — NOT k-reciprocal NN, NOT Hungarian.
- WHAT WE DO INSTEAD: mean-pooled embeddings → exhaustive cosine → similarity graph → connected components. k-reciprocal exists only as an optional similarity refinement, not as the inter-camera matcher.
- CONFIG: `reranking.enabled: false` (k1=30,k2=10,lambda=0.4); `multi_query.enabled: false`.

## Real Metrics (CityFlowV2 Vehicle)
- **Best reproducible MTMC IDF1 = 0.77936** (14e B1 v1) — docs/performance-state.md:10, docs/what-worked.md:22.
  - Config: multi-crop TTA Stage-2 features (14c v2) + Stage-4 fusion `w_tertiary=0.525, similarity_threshold=0.48, aqe_k=2, fic_regularisation=0.5`.
- Historical best 0.784 (v80/ali369) — NOT reproducible (lost OSNet checkpoint) — performance-state.md:11.
- **SOTA target = 0.8486** (AIC22 1st place, 5-model ensemble). **Gap = 6.93pp** — performance-state.md:13-14.
- MOTA: no headline value recorded; published CityFlowV2 baseline range MOTA ≈ 60-78% (configs/datasets/cityflowv2.yaml:8).
- Per-camera/scene IDF1 breakdown (docs/BREAKTHROUGH_PLAN.md:35-36):
  - S01 average IDF1 = 91.6% (excellent)
  - S02 average IDF1 = 80.1% (11.5pp worse); S02_c006 catastrophic at IDF1 = 74.0%.
- Plateau: **5-axis confirmed, feature-quality-limited** (performance-state.md:17-25): (1) 14e Stage-4 saturation, (2) 14g tertiary view expansion, (3) 14h aggregation, (4) 14i track-quality pre-filter, (5) 14k 4-way fusion. Association is EXHAUSTED (225+ configs).

## Dead-End History for These 7 Techniques
- **Zone model (#1)**: tested, disabled — **-0.4pp** (experiment-log.md:59; dead-ends.md).
- **Temporal window (#2)**: ENABLED and beneficial (temporal_overlap +0.9pp, what-worked.md:8); spatiotemporal weight retained.
- **SSA (#3)**: never tried as detection-freezing; opposite stage-5 stationary REMOVAL filter is enabled (disp=150).
- **TRL / AFLink (#4)**: AFLink dead-ended **-3.8 to -13.2pp** (dead-ends.md:12); intra-camera ReID merge retained (on).
- **Bidirectional (#5)**: never implemented or tried.
- **Occlusion-aware matrix (#6)**: never implemented or tried.
- **k-reciprocal / box-grained (#7)**: reranking "always hurts" vehicles (dead-ends.md:16); multi-query box-grained off; network-flow/Hungarian -0.24pp (dead-ends.md). 
- Related dead ends: CSLS -34.7pp, hierarchical -1 to -5pp, FAC -2.5pp, CID_BIAS -1 to -3.3pp (dead-ends.md).

## Critical Gaps vs Already-Covered vs Already-Tried
- **CRITICAL gaps (genuinely missing, never tried)**: #5 Bidirectional Tracking, #6 Occlusion-Aware Distance Matrix, #3 SSA (Stationary Sensitive Association). These are Stage-1/tracking-level and untouched — the only components not yet explored.
- **Already covered (enabled & working)**: #2 Spatio-Temporal Time Window (stronger hard-gate variant); #4 partially (intra-camera ReID relink enabled).
- **Already tried and dead-ended (implemented but disabled/harmful)**: #1 Zone reduction (-0.4pp), #7 k-reciprocal re-ranking (hurts) + box-grained multi-query (off), #4 cross-camera AFLink relink (-3.8 to -13.2pp).

## Surprises
- We already have **zone scoring** under the name `zone_model` — but as a soft ±0.03 bonus, not hard search-space pruning, and it's disabled/harmful with current 280D-PCA features.
- We already have **k-reciprocal** (reranking.py) and a **box-grained** path (multi_query max-of-K×K) — both disabled because they hurt vehicles with current features.
- Our stationary handling is the **opposite** of SSA: CityTrack preserves stationary detections; we delete parked cars in Stage-5.
- Our temporal constraint is **stronger** than CityTrack's: hard-drop outside window + Gaussian score, vs CityTrack's ×2 soft penalty.
- The default inter-camera matcher is **conflict_free_cc connected components**, not Hungarian or k-reciprocal NN as CityTrack uses.