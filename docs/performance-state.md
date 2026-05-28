# Performance State — MTMC Tracker

Detailed performance numbers, model checkpoints, and integration status. See [findings.md](findings.md) for narrative analysis and [experiment-log.md](experiment-log.md) for the full 225+ experiment ledger. Slim summary in [.github/copilot-instructions.md](../.github/copilot-instructions.md).

---

## Vehicle Pipeline (CityFlowV2)

### Headline numbers
- **Best Reproducible MTMC IDF1**: **0.77936** (14e B1 v1) — multi-crop TTA Stage-2 features (14c v2) + Stage-4 fusion `w_tertiary=0.525, similarity_threshold=0.48, aqe_k=2, fic_regularisation=0.5`.
- **+0.91pp** vs the prior deployed baseline 0.7703 (10c v15 / 10a v7 production CLIP+DINOv2 score-fusion at `w_tertiary=0.60, aqe_k=3`). Unlock came from dropping AQE `k` from 3 → 2 on TTA-smoothed features (ID switches 213 → 154).
- **Historical Best MTMC IDF1**: 0.784 (v80/v44, ali369; requires unavailable OSNet checkpoint, not reproducible).
- **SOTA target**: IDF1 ≈ 0.8486 (AIC22 1st place, 5-model ensemble).
- **Gap to SOTA**: 6.93pp — caused by feature quality (single model), NOT association tuning.

### Plateau confirmation (5 axes)
The 0.77936 plateau is confirmed across:
1. **14e** Stage-4 saturation (sweep `aqe_k`, `fic_reg`, `w_t × thr`)
2. **14g** tertiary DINOv2 view expansion (2 → 4 views)
3. **14h** robust tracklet aggregation (8 modes tried)
4. **14i** track-quality pre-filter (`L_min × τ_c` grid)
5. **14k** R50-IBN 4-way score fusion (`w_q × thr` sweep, peak 0.78079 K7, MARGINAL)

All cheap CPU-only axes are now exhausted. Plateau is **feature-diversity-limited**, not aggregation/filter/tuning-limited.

### Models — Vehicle
| Model | Dataset | mAP / R1 | Role | Notes |
|-------|---------|----------|------|-------|
| TransReID ViT-B/16 CLIP 256px | CityFlowV2 | 81.53% / 92.41% | **Primary** | Verified, AugOverhaul+EMA kernel `gumfreddy/09-vehicle-reid-cityflowv2-augoverhaul-ema`, see findings.md L748. Older 80.14% / 92.27% values were 09b v2 baseline, not deployed checkpoint. |
| TransReID ViT-B/16 CLIP (VeRi-only) | VeRi-776 | 89.97% / 98.33% | Single-cam expert | Joint optimum R1=98.15% / mAP=89.71% (09v v17, `outputs/09v_veri_v9`). R1 ceiling is 98.33% on this checkpoint. Historical 98.45% claim is not reachable via eval-time techniques alone — old `0.984505` value reproduces as R5=98.45% at AQE(k=3),k1=30,k2=10,λ=0.2. |
| CLIP-SENet v6 | VeRi-776 | 82.34% / 96.54% | Single-cam expert | 320² P=8/K=8 canonical; with rerank+AQE → **91.54% mAP**. v7 256² P=16 retrain regressed to 81.36% / 95.71% (DEAD END). |
| ResNet101-IBN-a | CityFlowV2 | 52.77% | Secondary | Too weak for ensemble (needs ≥65%); 6 ArcFace variants exhausted at this ceiling. |
| DINOv2 ViT-L/14 | (frozen) | — | Tertiary stream | Used in score fusion at `w_tertiary=0.525`. |

### Cross-domain fusion attempts (all DEAD ENDS)
- **CLIP-SENet × CityFlowV2 cross-domain fusion** (13d v2): monotonic IDF1 degradation across `w_cs∈{0.2..1.0}`; standalone CLIP-SENet on CityFlow → 0.6855 IDF1. Strong VeRi-776 expert does not transfer; domain gap dominates secondary-model strength.
- **CLIP-SENet CityFlow fine-tune fusion (13f → 13h)**: MARGINAL/DEAD END. 12-epoch fine-tune of CLIP-SENet v6 on 666 CityFlow IDs lifted standalone IDF1 from 0.6855 → 0.7099 (+2.44pp), confirming domain adaptation works. But fusion sweep peaked at `w_cs_ft=0.30 → 0.7691` (+0.12pp over 13h control 0.7679, but −0.12pp below production 0.7703). Fine-tune feature stream is too correlated with existing CLIP+DINOv2 pair.

### Association status
**EXHAUSTED** — 225+ configs, all within 0.3pp of optimal. See [findings.md](findings.md) and [dead-ends.md](dead-ends.md) for the full sweep history.

---

## Person Pipeline (WILDTRACK)

- **Best Ground-plane IDF1**: 0.947 (confirmed across 12b v1, v2, v3; 59+ configs tested)
- **Best Ground-plane MODA**: 0.903 (12b v14)
- **Detector**: MVDeTr ResNet18, MODA=0.921 (12a v3, epoch-20 training-time best); exported checkpoint loaded-model log line verifies MODA=0.913 (see [docs/models.md](models.md) L63)
- **SOTA target**: IDF1 ≈ 0.953
- **Gap to SOTA**: 0.6pp — tracker-limited (Kalman), NOT detector-limited
- **Status**: FULLY CONVERGED — tracker-limited and exhaustively tested; Kalman, global optimal, and naive trackers all failed to beat 0.947

---

## Integration Status

### Completed
- ✅ PR #4: 14e B1 CityFlow values promoted to `configs/datasets/cityflowv2.yaml`
- ✅ PR #5: Person pipeline routing in `scripts/run_pipeline.py`, 12b Kalman params in `configs/datasets/wildtrack.yaml`, and backend dataset switcher support
- ✅ PR #6: Canonical model/pipeline inventory docs added
- ✅ PR #7: TransReID mAP corrected from deep-hunt provenance
- ✅ PR #8: Backend model registry Phase 1 implemented
- ✅ PR #9: `model_id` wired into pipeline runs for config/model resolution
- ✅ PR #10: Frontend model registry dropdown and model cards added
- ✅ PR #11: Registry verification scripts and Kaggle cross-check added
- ✅ PR #12: Exhaustive reproduction guide and E2E smoke tests added
- ✅ PR #13: Kaggle slug recovery completed for missing kernel slugs
- ✅ PR #14: Canonical VeRi-776 reproducibility reference block added
- ✅ PR #15: 14t slug fix, P=16 typo fix, and 09v/13 empty metric blocks resolved
- ✅ Local checkpoint paths and `models/reid/README.md` provenance documented
- ✅ Backend `models/{requests,embedding}.py` + `repositories/__init__.py` scaffolding present

### Remaining TODOs
- PR #17 is pending merge for registry citation line-number fixes only; not a production metric regression.
- Frontend dataset switching incomplete: backend config/model resolution supports CityFlowV2 and WILDTRACK, and the UI displays model dataset metadata, but no clear global dataset selector was found in the frontend audit.
- Lightweight Python CI is not present yet if `.github/workflows/` is still absent; add backend/core smoke coverage before relying on PR checks as the integration gate.

---

## Paper Strategy
See [docs/paper-strategy.md](paper-strategy.md). Best angle: "One Model, 91% of SOTA" — efficiency + exhaustive ablation. Target venues: IEEE Access, Multimedia Tools & Applications, Scientific Reports. Key contribution: 225+ experiments proving feature quality (not association) is the MTMC bottleneck.
