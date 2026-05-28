# Confirmed Dead Ends — DO NOT RETRY

**Read this file before proposing ANY experiment or parameter change.** Every entry here cost real GPU hours and is now closed.

Cross-references: [findings.md](findings.md) (strategic analysis), [experiment-log.md](experiment-log.md) (full ledger), [what-worked.md](what-worked.md) (positive results).

---

## Association / Stage 4

- **CSLS**: -34.7pp (catastrophic — penalizes genuine vehicle-type hubs)
- **AFLink motion linking**: -3.8pp to -13.2pp MTMC IDF1 in clean retests; even `gap=100, dir_cos=0.90` loses -3.82pp. Motion consistency is unreliable across non-overlapping CityFlowV2 cameras; AFLink creates false merges.
- **CID_BIAS**: GT-learned -3.3pp; topology CID_BIAS -1.0 to -1.2pp (additive bias distorts FIC-calibrated similarities)
- **Hierarchical clustering**: -1 to -5pp (centroid averaging loses discriminative signal)
- **FAC**: -2.5pp (cross-camera KNN consensus overwrites distinguishing details)
- **Reranking**: Always hurts (k-reciprocal sets contain false positives with current features)
- **Feature concatenation**: -1.6pp (mixes uncalibrated feature spaces)
- **Network flow solver**: -0.24pp MTMC IDF1, increased conflation 27→30 instead of reducing it

## Features / Stage 2

- **384px ViT deployment**: -2.8pp (captures viewpoint-specific textures that hurt cross-camera matching)
- **DMT camera-aware training**: -1.4pp single-model (also 09g: 43.8% mAP, too weak)
- **VeRi-776 → CityFlowV2 ResNet pretrain**: 42.7% mAP (worse than direct 52.77%)
- **Extended ResNet fine-tuning**: 50.61% mAP (degraded from 52.77%)
- **ArcFace on ResNet101-IBN-a**: 50.80% mAP (warm-start geometry mismatch, 6 variants exhausted at 52.77% ceiling)
- **ResNeXt101-IBN-a ArcFace**: 36.88% mAP (IBN-Net pretrained weights were for 32x32d while the model here used 32x8d; `strict=False` partial loading left many layers random and crippled training)
- **OSNet VeRi-776 as secondary** (score-level or concat): both hurt -0.8pp to -1.1pp; the v80 78.4% checkpoint (`vehicle_osnet_veri776.pth`) is lost from the weights datasets
- **CLIP-SENet × CityFlowV2 score-level fusion**: monotonic degradation (control 0.7679 → −0.13pp at w_cs=0.2, −1.77pp at 0.6, −3.68pp at 0.8, −8.24pp standalone); 91.54% VeRi-776 mAP secondary cannot bridge the cross-camera domain gap (13d v2)
- **CLIP-SENet CityFlow fine-tuned fusion (13f v1 + 13h sweep)**: peak fusion IDF1 0.7691 at `w_cs_ft=0.30` is −0.12pp below production 0.7703; standalone fine-tuned model only 0.7099 IDF1 (vs TransReID's ~0.75+); fine-tune fixes the domain gap but feature stream remains too correlated with primary CLIP+DINOv2 to add net value
- **CLIP-SENet retrain at image_size=256, P=16 (v7)**: −0.98pp mAP / −0.83pp R1 vs v6 320² (81.36 vs 82.34); smaller crops lose fine-grained vehicle texture; v6 320² remains canonical
- **SAM2 box-prompt masking (14a v8)**: -0.56pp MTMC IDF1 (0.7647 vs production 0.7703); 5px dilation, zeros background, applied in Stage 2 ReID feature extraction. SAM2 base-plus with center-point prompt removes too much vehicle context (wheels/tires/road-reflection cues). Configurable variants unlikely to recover 0.56pp gap.

## TTA + Feature Plateau (14e–14k campaign)

- **AQE k=1 on TTA features (14f Block B)**: -0.88 to -1.00pp MTMC IDF1 vs 14e B1 0.77936 (range 0.76933–0.77059 across 9 configs at `fic_reg=0.5`, varying `w_t × thr`). On TTA-smoothed features the AQE axis is concave with the discrete optimum at k=2: too little neighbour expansion (k=1) re-introduces single-query noise, too much (k=3, k=4) over-smooths. **k=2 is locked** for TTA features; do NOT re-test k=1 on this feature family. 14g S6 reproduced the k=3 regression on the new DINOv2-4view feature build (0.77149 vs 0.77936, id_switches 154 → 213).
- **Multi-crop TTA at Stage 2 + fusion sweep (14c v2 + 14d v1)**: MARGINAL POSITIVE → SUPERSEDED BY 14e WIN. 14c v2 4-view primary {original, hflip, scale_0.95, scale_1.05} + 2-view DINOv2 {original, hflip} L2-mean TTA gave 0.77085 MTMC IDF1. 14d v1 CPU sweep peaks at 0.77155 with `w_tertiary=0.50, sim_thresh=0.50`. Within ~0.24pp run-to-run noise. Optimum shifted from production `w_t=0.60` to `w_t=0.50` — a real signal TTA changed the primary embedding distribution. 14e B1 v1 (CPU sweep at A10 anchor with aqe_k=2) achieved 0.77936 MTMC IDF1 — TTA features promoted to new headline; AQE k=2 was the unlock.
- **DINOv2 4-view TTA expansion at Stage 2 (14g v1)**: NEUTRAL/SATURATED. Symmetrizing the tertiary DINOv2 ViT-L/14 stream from 2 TTA views to 4 views produced **zero change in MTMC IDF1** vs 14e B1. All 7 `aqe_k=2` configs landed at `id_switches=154` exact. **TTA expansion family (both primary 4-view and tertiary 4-view) is now fully saturated** — more views of the same models cannot lift IDF1 beyond 0.77936. Do NOT re-run further TTA-view-count sweeps on either CLIP or DINOv2.
- **Robust tracklet pooling (14h v3)**: NEUTRAL/DEAD END. Enabled `stage2.multi_query.k=24` and ran 8 robust aggregation modes (mean / median / geo_median / medoid / trimmed_mean_10 / trimmed_mean_25 / top12_to_mean / top12_to_medoid). **All 8 robust modes worse**: range 0.76881–0.77829. Medoid (M4) cut id_switches to 134 but IDF1 dropped to 0.77234 — "stable but wrong" pattern. ID-switch count is NOT a reliable proxy for IDF1 on this floor. The existing softmax-quality-weighted mean is already optimal. Do NOT re-test robust pooling, do NOT re-test multi-query K above 24, and do NOT confuse low id_switches with high IDF1.
- **Track-quality pre-filter (14i v2)**: NEUTRAL/MARGINAL. CPU-only sweep over `min_track_length L_min ∈ {3,5,8,12}` × `min_avg_detection_confidence τ_c ∈ {0.30..0.50}` (20 configs). Best filter F2 (`L_min=3, τ_c=0.35`) = 0.77964 — only +0.03pp over F0, below WIN threshold 0.781 and within run-to-run noise. The 22% ID-switch reduction (154 → 120) WITHOUT a meaningful IDF1 lift confirms residual error is feature-quality limited. Do NOT re-test track-length/confidence filtering on this feature build.
- **R50-IBN as 4-way score-fusion stream (14j v1)**: MARGINAL/closed by 14k. CPU-only 16-config sweep adding FastReID R50-IBN-a (CityFlowV2-trained) as quaternary score-fusion stream. Best W14 (`w_q=0.30, thr=0.48, w_p=0.175, w_t=0.525`) = 0.78032 (+0.00097 over W0). Below WIN threshold 0.7810.
- **14k v1 extended sweep (MARGINAL, NOT PROMOTED)**: 14-config grid with R50-IBN quaternary at `w_q∈{0.35..0.50}` × `thr∈{0.46,0.48,0.50}` peaks at K7 = 0.78079. Plateau confirmed across 5+ configs at ~0.78048. Below pre-registered WIN bar 0.7810 and below historical noise band ~0.24pp. **All CPU-only axes saturated** — feature-quality ceiling confirmed across 5 axes.

## Training Recipes

- **Score-level ensemble with 52.77% secondary**: -0.1pp (secondary too weak, adds noise)
- **Circle loss + triplet**: 16-30% mAP (conflicting gradients)
- **SGD for ResNet**: 30.27% mAP (catastrophic — AdamW essential for small datasets)

## Person Pipeline

- **Global optimal tracker (person)**: -3.5pp IDF1 vs Kalman (assignment costs lose motion prediction advantage)
- **Extended Kalman sweeps (person)**: 59 configs within ±0.0004 IDF1 — fully exhausted
- **Person: improved detector → better tracking**: MODA 90.9 → 92.1% but IDF1 unchanged at 94.7% — tracker-limited, not detector-limited

---

## Remaining Untried Approaches (NOT dead ends — viable next moves)
- GNN edge classification for association (not implemented)
- Graph-based multi-view tracking for person pipeline (not implemented)
- Genuinely new feature stream (different architecture + pretraining) — GPU required
- Pseudo-label self-training — GPU required
