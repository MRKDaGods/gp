# GitHub Copilot Instructions — MTMC Tracker

## Project Overview
Multi-camera multi-target tracking system (vehicles/humans) on CityFlowV2 (AI City Challenge 2022 Track 1). 7-stage offline pipeline: Ingestion → Tracking → Features → Indexing → Association → Evaluation → Visualization. Python 3.10+, PyTorch, YOLO26m, TransReID ViT-Base/16 CLIP, FAISS, SQLite, Streamlit.

## Critical Rules (NEVER violate)

### GPU Pipeline Execution
- **NEVER** run GPU-intensive pipeline stages (0, 1, 2) locally — the local machine has a GTX 1050 Ti which is too slow and starves other work
- ALL GPU-intensive work (detection, tracking, feature extraction, ReID training) MUST run on Kaggle
- Local machine is ONLY for: code editing, pushing notebooks, monitoring Kaggle kernels, and running CPU-only stages (3, 4, 5) on small datasets
- The virtual environment MUST be used for any local Python: `.venv` (Python 3.11.9), NOT system Python 3.13
- Activate with: `.\.venv\Scripts\activate`

### Notebook Editing
- **NEVER** use `replace_string_in_file` on `.ipynb` files — it edits VS Code's in-memory buffer but may NOT save to disk
- For `.ipynb` edits: use `json.load() → modify → json.dump()` via a Python script
- After any `.ipynb` edit, verify on-disk state with `python -c "import json; ..."`
- Each line in notebook `source` arrays MUST end with `\n` EXCEPT the last line
- On Windows, use `ensure_ascii=True` in `json.dump` to avoid charmap codec errors

### Frame ID Convention
- Internal pipeline (Stages 0-4): **0-based** frame IDs
- MOT submission format: **1-based** (converted via `frame_id + 1` in format_converter)
- CityFlowV2 GT: **1-based** (standard MOT format)
- Never mix these — check which context you're in

### Config Override Paths
- Stage4 reads from `cfg.stage4.association` — override must be `stage4.association.graph.similarity_threshold=X`
- NOT `stage4.graph.similarity_threshold=X` (wrong level)
- NOT `stage4.similarity_threshold=X` (wrong level)
- Similarly: `stage4.association.fic.regularisation=X`, `stage4.association.gallery_expansion.threshold=X`
- Config loading: `default.yaml` → merge `cityflowv2.yaml` → CLI overrides

### Research Findings — MUST READ
- **Always read `docs/findings.md` before proposing experiments or changes** — it contains all dead ends, current performance, and strategic analysis
- **Update `docs/findings.md`** whenever: new experiments produce results, new dead ends are discovered, performance numbers change, or new insights are gained
- The findings doc is the canonical source of truth for what has been tried and what to do next
- Before trying ANY approach, check the "Dead Ends" section to avoid repeating failed experiments

## Architecture

Post-merge application stack: `backend/` (FastAPI) and `frontend/` (Next.js ATHAR) now live alongside the offline `src/` pipeline.

```
configs/          YAML configuration (OmegaConf)
backend/          FastAPI service layer and API routers
frontend/         Next.js ATHAR dashboard and workflow UI
src/core/         Shared data models, config loader, utilities
src/stage0/       Frame extraction, preprocessing
src/stage1/       YOLO26m detection + BoxMOT tracking (BoT-SORT)
src/stage2/       ReID embeddings (TransReID ViT) + HSV + PCA whitening
src/stage3/       FAISS IndexFlatIP + SQLite metadata
src/stage4/       Cross-camera association (similarity graph + connected components)
src/stage5/       TrackEval metrics (HOTA, IDF1, MOTA)
src/stage6/       Visualization (annotated video, BEV, timeline)
src/apps/         Streamlit dashboard, NL query, 3D sim
scripts/          CLI entry points + helper scripts
notebooks/kaggle/ Kaggle training notebooks (10a/10b/10c pipeline chain)
tests/            pytest test suite
docs/findings.md  Research findings, dead ends, strategic analysis (KEEP UPDATED)
```

## Key Dependencies & Versions
- Python >=3.10, <3.14
- PyTorch >=2.1, torchvision >=0.16
- timm >=0.9 (TransReID ViT-Base/16 CLIP backbone)
- ultralytics >=8.1 (YOLO26m detection)
- boxmot >=10.0 (BoT-SORT tracker)
- faiss-cpu >=1.7 (cosine similarity indexing)
- omegaconf >=2.3 (hierarchical config)
- networkx >=3.1 (graph algorithms)
- streamlit >=1.28 (web dashboard)
- P100 GPUs on Kaggle: need PyTorch 2.4.1+cu124 (sm_60 compat)

## Code Patterns

### Naming
- Stages: `src/stageN_name/` with `pipeline.py` as entry point
- Config sections mirror stage names: `stage0`, `stage1`, ..., `stage4.association`
- Test files: `tests/test_stageN/test_component.py`

### Configuration
- All config via OmegaConf from YAML + CLI overrides
- No env vars or OmegaConf resolvers — all explicit `--override` args
- Access: `cfg.stage4.association.graph.similarity_threshold`

### Data Flow Between Stages
- Stages communicate through files in `data/outputs/`
- Each stage reads predecessor's output, writes its own
- Backend services read run-scoped artifacts under `data/outputs/<run_id>/...`
- Tracklets: list of dicts with `camera_id`, `track_id`, `frames`, `boxes`, `embeddings`

## Current Performance State

### Vehicle Pipeline (CityFlowV2)
- **Best Reproducible MTMC IDF1**: **0.77936 (14e B1 v1, NEW HEADLINE)** — multi-crop TTA Stage-2 features (14c v2) + Stage-4 fusion `w_tertiary=0.525, similarity_threshold=0.48, aqe_k=2, fic_regularisation=0.5`. **+0.91pp vs the prior deployed baseline of 0.7703** (10c v15 / 10a v7 production CLIP+DINOv2 score-fusion at `w_tertiary=0.60, aqe_k=3`). The lift comes from dropping AQE `k` from 3 → 2 on TTA-smoothed features (ID switches 213 → 154). Confirmed reproducible on 14f (`aqe_k=2` plateau, 8 ties at 0.77936 exact) and on 14g new tertiary feature build (S0 drift = 0.77902 within ±0.005; all `aqe_k=2` configs id_switches=154 exact). TTA expansion family fully saturated. 14h v3 robust tracklet pooling sweep further confirmed the plateau (M0 drift gate = 0.77936 / id_sw=154 EXACT bit-identical; all 8 robust modes worse; existing softmax-quality mean is optimal). 0.77936 is now confirmed across FIVE axes (14e Stage-4 saturation, 14g tertiary view expansion, 14h tracklet aggregation, 14i track-quality pre-filter, 14k R50-IBN 4-way score fusion) — feature-diversity-limited, not aggregation-limited, not filter-limited. All cheap CPU-only experiments are now exhausted. 14d v1 floor (0.77155) is now superseded. 14j/14k 4-way fusion (R50-IBN quaternary) observed a MARGINAL plateau up to 0.78079 at K7 — NOT promoted; headline stays 0.77936.
- **Historical Best MTMC IDF1**: 0.784 (v80/v44, ali369; requires unavailable OSNet checkpoint, not reproducible)
- **SOTA target**: IDF1≈0.8486 (AIC22 1st place, 5-model ensemble)
- **Gap to SOTA**: 6.93pp — caused by feature quality (single model), NOT association tuning. Closed 0.90pp via 14e WIN (TTA + AQE k=2).
- **Primary model**: TransReID ViT-B/16 CLIP 256px — mAP=80.14%, R1=92.27% on CityFlowV2
- **Vehicle ReID single-cam (VeRi-776, VeRi-only TransReID ViT-B/16 CLIP)**: Best R1=98.33%, best mAP=89.97%, joint optimum R1=98.15% / mAP=89.71% (09v v17, `outputs/09v_veri_v9`); R1 ceiling is 98.33% on this checkpoint, and the historical 98.45% claim is not reachable via eval-time techniques alone; the old `0.984505` value still reproduces as R5=98.45% at AQE(k=3),k1=30,k2=10,λ=0.2
- **Secondary model**: ResNet101-IBN-a — mAP=52.77% (too weak for ensemble, needs ≥65%)
- **CLIP-SENet (VeRi-776)**: v6 canonical at **mAP=82.34%, R1=96.54%** (320², P=8/K=8); with rerank+AQE → **91.54% mAP**. v7 256²/P=16 retrain regressed to 81.36% / 95.71%; post-rerank 88.98%, -2.56pp vs v6 91.54% — DEAD END.
- **CLIP-SENet × CityFlowV2 cross-domain fusion**: DEAD END (13d v2). Monotonic IDF1 degradation across `w_cs∈{0.2..1.0}`; standalone CLIP-SENet on CityFlow → 0.6855 IDF1. Strong VeRi-776 expert does not transfer; domain gap dominates secondary-model strength.
- **CLIP-SENet CityFlow fine-tune fusion (13f→13h)**: MARGINAL / DEAD END. 12-epoch fine-tune of CLIP-SENet v6 on 666 CityFlow IDs lifted standalone IDF1 from 0.6855 → 0.7099 (+2.44pp), confirming domain adaptation works. But fusion sweep peaked at `w_cs_ft=0.30 → 0.7691` (+0.12pp over 13h control 0.7679, but −0.12pp below production 0.7703). Fine-tune feature stream is too correlated with existing CLIP+DINOv2 pair.
- **Association**: EXHAUSTED (225+ configs, all within 0.3pp of optimal)
- See `docs/findings.md` for full analysis, dead ends, and action plan

### Person Pipeline (WILDTRACK)
- **Best Ground-plane IDF1**: 0.947 (confirmed across 12b v1, v2, v3; 59+ configs tested)
- **Best Ground-plane MODA**: 0.903 (12b v14)
- **Detector**: MVDeTr ResNet18, MODA=0.921 (12a v3, best achieved)
- **SOTA target**: IDF1≈0.953
- **Gap to SOTA**: 0.6pp — tracker-limited (Kalman), NOT detector-limited
- **Status**: FULLY CONVERGED — tracker-limited and exhaustively tested; Kalman, global optimal, and naive trackers all failed to beat 0.947

### Integration Status (this branch: feat/integrate-vehicle-mtmc)
- ✅ 14e B1 CityFlow values promoted to `configs/datasets/cityflowv2.yaml`
- ✅ Local checkpoint paths + `models/reid/README.md` provenance
- ⏳ Person pipeline routing → see `feat/integrate-person-mtmc`
- ⏳ Backend dataset switcher → see `feat/integrate-person-mtmc`

## Experiment History
- **Full experiment log**: See `docs/experiment-log.md` for 225+ tracked experiments
- **Research findings**: See `docs/findings.md` for dead ends, strategic analysis, and what to do next
- Before trying ANY parameter change, check BOTH documents to avoid repeating failed experiments

### Confirmed Dead Ends (DO NOT RETRY)
- **CSLS**: -34.7pp (catastrophic — penalizes genuine vehicle-type hubs)
- **384px ViT deployment**: -2.8pp (captures viewpoint-specific textures that hurt cross-camera matching)
- **AFLink motion linking**: confirmed harmful at **-3.8pp to -13.2pp MTMC IDF1** in clean retests; even `gap=100`, `dir_cos=0.90` loses **-3.82pp** because motion consistency is unreliable across non-overlapping CityFlowV2 cameras and AFLink creates false merges
- **CID_BIAS**: GT-learned version -3.3pp; topology CID_BIAS -1.0 to -1.2pp (additive bias distorts FIC-calibrated similarities)
- **DMT camera-aware training**: -1.4pp single-model (also 09g: 43.8% mAP, too weak)
- **Hierarchical clustering**: -1 to -5pp (centroid averaging loses discriminative signal)
- **FAC**: -2.5pp (cross-camera KNN consensus overwrites distinguishing details)
- **Reranking**: Always hurts (k-reciprocal sets contain false positives with current features)
- **Feature concatenation**: -1.6pp (mixes uncalibrated feature spaces)
- **Network flow solver**: -0.24pp MTMC IDF1, increased conflation from 27→30 instead of reducing it
- **VeRi-776→CityFlowV2 ResNet pretrain**: 42.7% mAP (worse than direct 52.77%)
- **Extended ResNet fine-tuning**: 50.61% mAP (degraded from 52.77%)
- **ArcFace on ResNet101-IBN-a**: 50.80% mAP (warm-start geometry mismatch, 6 variants exhausted at 52.77% ceiling)
- **ResNeXt101-IBN-a ArcFace**: 36.88% mAP (IBN-Net pretrained weights were for 32x32d while the model here used 32x8d; `strict=False` partial loading left many layers random and crippled training)
- **OSNet VeRi-776 as secondary (score-level or concat)**: both hurt (**-0.8pp to -1.1pp**); the v80 **78.4%** checkpoint (`vehicle_osnet_veri776.pth`) is lost from the weights datasets
- **CLIP-SENet × CityFlowV2 score-level fusion**: monotonic degradation (control 0.7679 → −0.13pp at w_cs=0.2, −1.77pp at 0.6, −3.68pp at 0.8, −8.24pp standalone); 91.54% VeRi-776 mAP secondary cannot bridge the cross-camera domain gap (13d v2)
- **CLIP-SENet CityFlow-fine-tuned fusion (13f v1 + 13h sweep)**: peak fusion IDF1 0.7691 at `w_cs_ft=0.30` is −0.12pp below production 0.7703; standalone fine-tuned model only 0.7099 IDF1 (vs TransReID's ~0.75+); fine-tune fixes the domain gap but feature stream remains too correlated with primary CLIP+DINOv2 to add net value
- **CLIP-SENet retrain at image_size=256, P=16 (v7)**: −0.98pp mAP / −0.83pp R1 vs v6 320² (81.36 vs 82.34); smaller crops lose fine-grained vehicle texture; v6 320² remains canonical
- **SAM2 box-prompt masking (14a v8)**: -0.56pp MTMC IDF1 (0.7647 vs production 0.7703); 5px dilation, zeros background, applied in Stage 2 ReID feature extraction. SAM2 base-plus with center-point prompt removes too much vehicle context (wheels/tires/road-reflection cues) that the cross-camera matcher relies on; trackeval_idf1=0.7866 suggests within-camera tracking improved slightly but cross-camera regressed. Configurable variants (mean fill, wider dilation) untested but unlikely to recover 0.56pp gap.
- **AQE k=1 on TTA features (14f Block B)**: -0.88 to -1.00pp MTMC IDF1 vs 14e B1 0.77936 (range 0.76933–0.77059 across 9 configs at `fic_reg=0.5`, varying `w_t × thr`). On TTA-smoothed features the AQE axis is concave with the discrete optimum at k=2: too little neighbour expansion (k=1) re-introduces single-query noise, too much (k=3, k=4) over-smooths. **k=2 is locked** for TTA features; do NOT re-test k=1 on this feature family. 14g S6 reproduced the k=3 regression on the new DINOv2-4view feature build (0.77149 vs 0.77936, id_switches 154 → 213), confirming the AQE k=2 unlock is invariant to DINOv2 view count.
- **Multi-crop TTA at Stage 2 + fusion sweep (14c v2 + 14d v1)**: MARGINAL POSITIVE → SUPERSEDED BY 14e WIN. 14c v2 4-view primary {original, hflip, scale_0.95, scale_1.05} + 2-view DINOv2 {original, hflip} L2-mean TTA gave 0.77085 MTMC IDF1 (+0.05pp vs production 0.7703 at production fusion). 14d v1 CPU sweep on the same features peaks at 0.77155 with `w_tertiary=0.50, sim_thresh=0.50` (+0.13pp vs production, +0.07pp vs 14c control). Consistent +0.03 to +0.07pp lift across `w_t∈[0.50,0.70]` at `thr=0.50`; `thr=0.40` universally -1.4pp worse. Optimum shifted from production `w_t=0.60` to `w_t=0.50` — a real signal TTA changed the primary embedding distribution. Within ~0.24pp run-to-run noise so not promoted to headline. Next: 14e tighter `w_t × thr` grid + AQE/FIC sweep (CPU only, ~10-15 min). If 14e <0.7720 → TTA family closed → escalate to GNN edge classifier. Pending decision; do NOT retreat to 14c-only or production fusion on TTA features. 14e B1 v1 (CPU sweep at A10 anchor with aqe_k=2) achieved 0.77936 MTMC IDF1 — clearing WIN threshold by +0.74pp. AQE k=2 (vs production k=3) was the unlock; TTA features are now PROMOTED to the new reproducible headline.
- **14e expanded TTA + AQE/FIC sweep (B1 v1)**: **WIN, NEW HEADLINE 0.77936**. CPU-only 16-config sweep on 14c v2 TTA features. Block A (fine w_t × thr at production aqe_k=3) is flat at 0.7707–0.7717. Block B at A10 anchor (w_t=0.525, thr=0.48) tested aqe_k ∈ {2, 4} and fic_reg ∈ {0.3, 0.7}; **aqe_k=2 unlocked +0.77pp** (0.77171 → 0.77936) and dropped ID switches 213 → 154 (-28%). aqe_k=4 worsens (-1.19pp), confirming TTA pre-smooths features so production aqe_k=3 was over-smoothing. FIC sensitivity small (±0.001). +0.91pp vs deployed 0.7703 baseline. Production-deployed config unchanged for now — pending 14f confirmation sweep around aqe_k=2 plus aqe_k=1 probes (CPU, ~25 min). See `docs/findings.md` and `docs/subagent-specs/post-14e-next.md`.
- **DINOv2 4-view TTA expansion at Stage 2 (14g v1)**: NEUTRAL / SATURATED. Symmetrizing the tertiary DINOv2 ViT-L/14 stream from 2 TTA views `{original, hflip}` to 4 views `{original, hflip, scale_0.95, scale_1.05}` produced **zero change in MTMC IDF1** vs 14e B1. S0 anchor = 0.77902 (drift −0.00034 within ±0.005 gate); best of 8 configs S2 = 0.77926; all 7 `aqe_k=2` configs landed at `id_switches=154` *exact* — identical to 14e B1 on the 2-view tertiary build. With `w_t=0.525` the primary CLIP TransReID stream dominates; tertiary embedding noise is no longer the residual error source. **The TTA expansion family (both primary 4-view and tertiary 4-view) is now fully saturated** — more views of the same models cannot lift IDF1 beyond 0.77936. The 0.77936 plateau is **feature-diversity limited and tracklet-aggregation limited**, not view-coverage limited. Next axis to probe: robust tracklet pooling (14h), then either a genuinely-new architecture stream (different pretraining + architecture) or a GNN edge classifier. Do NOT re-run further TTA-view-count sweeps on either CLIP or DINOv2.
- **Robust tracklet pooling (14h v3)**: NEUTRAL / DEAD END. Enabled `stage2.multi_query.k=24` on the 14c v2 TTA build and ran 8 robust aggregation modes (mean / median / geo_median / medoid / trimmed_mean_10 / trimmed_mean_25 / top12_to_mean / top12_to_medoid) at the 14e B1 anchor. M0 drift gate reproduced 0.77936 / id_sw=154 EXACT. **All 8 robust modes worse**: range 0.76881–0.77829 (−0.11pp to −1.06pp); plain mean (M1) closest at 0.77829 / id_sw=163. Medoid (M4) cut id_switches to 134 (lowest in sweep, −13%) but IDF1 dropped to 0.77234 (−0.70pp) — "stable but wrong" pattern, ID-switch count is NOT a reliable proxy for IDF1 on this floor. The existing softmax-quality-weighted mean is already optimal; TTA pre-smoothing removes the per-frame outliers robust statistics would otherwise clip. **The 154 ID-switch floor is feature-quality limited, NOT aggregation/TTA/Stage-4-tuning limited.** Plateau is now confirmed across THREE independent feature-side axes (14e Stage-4 saturation, 14g tertiary view expansion, 14h tracklet aggregation). Do NOT re-test robust pooling, do NOT re-test multi-query K above 24, and do NOT confuse low id_switches with high IDF1.
- **Track-quality pre-filter (14i v2)**: NEUTRAL / MARGINAL. CPU-only sweep over `min_track_length L_min ∈ {3,5,8,12}` × `min_avg_detection_confidence τ_c ∈ {0.30,0.35,0.40,0.45,0.50}` (20 configs) plus F0 no-filter drift gate, all at the 14e B1 anchor on the 14h v3 Stage-2 outputs. F0 reproduced 0.77936 / id_switches=154 EXACT. Best filter F2 (`L_min=3, τ_c=0.35`, kept 818/929) = **0.77964 / id_switches=120** — only **+0.00028 IDF1 (+0.03pp)** over F0, below the 0.781 WIN threshold and within run-to-run noise; not promoted. The 22% ID-switch reduction (154 → 120) WITHOUT a meaningful IDF1 lift is direct evidence that the residual error is genuinely feature-quality limited at this floor — the dropped low-confidence/short tracklets were already harmless to IDF1. More aggressive filters (e.g. F9 `L=5, τ=0.45`) cut IDS to 97 but dropped IDF1 to 0.77604 (the same "stable but wrong" pattern as 14h medoid). The 0.77936 plateau is now confirmed across **FOUR axes**: Stage-4 tuning (14e/14f), tertiary view expansion (14g), tracklet aggregation (14h), and track-quality pre-filter (14i). **All cheap CPU-only experiments are now exhausted.** Do NOT re-test track-length/confidence filtering on this feature build. Remaining viable levers all require GPU work: (a) genuinely new feature stream (different architecture + pretraining), (b) learned association (GNN edge classifier), (c) pseudo-label self-training.
- **R50-IBN as 4-way score-fusion stream (14j v1)**: MARGINAL / closed by 14k. CPU-only 16-config sweep adding FastReID R50-IBN-a (CityFlowV2-trained) as a quaternary score-fusion stream on top of primary CLIP TransReID + tertiary DINOv2. W0 drift gate reproduced 0.77936 / id_switches=154 EXACT. Best W14 (`w_q=0.30, thr=0.48, w_p=0.175, w_t=0.525`) = **0.78032 / id_switches=207** (+0.00097 over W0, +0.0010 over previous deployed baseline 0.7703). Verdict MARGINAL per 14j spec bands (WIN ≥0.7810 not reached). W14 sat on the upper boundary of the `w_q` grid, motivating 14k.
- **14k v1 extended sweep (MARGINAL, NOT PROMOTED)**: 14-config grid with R50-IBN quaternary at `w_q∈{0.35..0.50}` × `thr∈{0.46,0.48,0.50}` peaks at K7 = **0.78079** (`w_p=0.10, w_t=0.45, w_q=0.45, thr=0.46`), +0.0014 vs 14e B1. Plateau confirmed across 5+ configs at ~0.78048; turnover at `w_q=0.50`; K13 literal sanity (0.30/0.30/0.40 = 0.78048) confirms real ensemble lift, not primary suppression. Below pre-registered WIN bar 0.7810 and below historical noise band ~0.24pp. **All CPU-only axes saturated** — feature-quality ceiling confirmed across 5 axes (Stage-4 tuning, tertiary view expansion, tracklet aggregation, track-quality filter, 4-way score fusion).
- **Score-level ensemble with 52.77% secondary**: -0.1pp (secondary too weak, adds noise)
- **Circle loss + triplet**: 16-30% mAP (conflicting gradients)
- **SGD for ResNet**: 30.27% mAP (catastrophic — AdamW essential for small datasets)
- **Global optimal tracker (person)**: -3.5pp IDF1 vs Kalman (assignment costs lose motion prediction advantage)
- **Extended Kalman sweeps (person)**: 59 configs within +-0.0004 IDF1 — fully exhausted
- **Person: improved detector→better tracking**: MODA 90.9→92.1% but IDF1 unchanged at 94.7%

### What Actually Worked
- Conflict-free CC (+0.21pp), intra-merge (+0.28pp), temporal overlap bonus (+0.9pp)
- FIC whitening (+1-2pp), power normalization (+0.5pp), PCA 384D, AQE K=3
- min_hits=2 (+0.2pp), Kalman tuning for person (+1.9pp IDF1)
- **14f confirmation (NEUTRAL but valuable)**: 14e B1 = 0.77936 is now a *confirmed reproducible plateau* (A20 drift check reproduced 0.77936 exact with id_switches=154 exact; 8 Block A configs at `aqe_k=2, w_t=0.525` tied at 0.77936). TTA × Stage-4-tuning family is **EXHAUSTED at 0.77936** (not a dead end — confirmed-win plateau). All 9 `aqe_k=1` probes universally worse (0.7693–0.7706). k=2 is the discrete AQE optimum for TTA features.

### Remaining Untried Approaches
- GNN edge classification for association (not implemented)
- Graph-based multi-view tracking for person pipeline (not implemented)

## Kaggle Workflow
- Pipeline chain: 10a (stages 0-2, GPU) → 10b (stage 3, CPU) → 10c (stages 4-5, CPU)
- Backend/frontend integration is local orchestration only; GPU-heavy stages still run on Kaggle
- Push: `kaggle kernels push -p notebooks/kaggle/10X_stagesNN/`
- Logs: `python scripts/kaggle_logs.py <kernel_slug> --tail N`
- Auth tokens in `~/.kaggle/`: abdo (gumfreddy), mrk (mrkdagods), ali369 (lolo)
- Current active account: gumfreddy (gumfreddy_access_token) — ali369/mrkdagods tokens may be missing from ~/.kaggle/

### Disk Hygiene (CRITICAL — disk is tight)
- After every `kaggle kernels output` download, immediately delete useless artifacts:
  - DELETE `last.pth` (never useful — only `best_mAP.pth` / `best_R1.pth` are kept)
  - For FAILed runs, DELETE all `.pth` files; keep only `eval_results.json`, `recipe.json`, `train_log.json`, summary JSONs
  - DELETE empty/0-byte log files
  - Keep `best_mAP.pth` only if the run has plausible ensemble value (close to baseline R1 even on FAIL)
- Always run `Get-ChildItem | Measure-Object -Sum Length` before/after cleanup and report GB reclaimed
- Old `tmp_*_outputs/` directories from completed verdict runs should be pruned, not accumulated

### Session Lifecycle (CRITICAL — never exit a turn waiting)
- NEVER end a turn without queueing the next action — every turn ends with a tool call (sleep/poll/subagent) or `vscode_askQuestions` if blocked
- NEVER use `mode=async` for `Start-Sleep` waits. Always use `mode=sync` with a generous `timeout` so the sleep + poll completes within the turn
- If user says "monitor", "check back in N hours", "wait Xh" — execute `Start-Sleep -Seconds <N>` synchronously in `mode=sync`. Do NOT exit the turn waiting for an async system notification.
- Between subagent invocations, immediately start the next sleep/poll or ask the user via `vscode_askQuestions` — never end the conversation in an idle state

### Kaggle Push Safety Rules (CRITICAL)
- **NEVER push a kernel more than once without confirming the previous version is fully running or complete** — rapid re-pushes create duplicate GPU sessions that consume both slots and block all other work
- After every push, check for warning lines like `The following are not valid dataset sources` — these indicate the run started but with missing inputs; **immediately cancel** the bad run via `kaggle kernels cancel <owner/slug>` before attempting a fix-and-repush
- If `kaggle kernels cancel` CLI command fails or is unavailable (e.g., older CLI versions lack the subcommand), **post the kernel URL to the user, then keep polling `kaggle kernels status <slug>` every ~60s in a loop** until the status reaches `cancelled`/`error`/`complete`. Do NOT stop the workflow — once cancellation is confirmed, resume the planned work automatically. Only halt entirely if polling shows the run is still `running` after the user has been notified and a reasonable wait has passed.
- Kaggle allows a maximum of 2 concurrent GPU sessions per account — always check active sessions before pushing a GPU-enabled notebook
- When iterating on kernel-metadata.json fixes, validate metadata locally first, then push **once**

## Paper Strategy
- A full publishability analysis and paper strategy is documented in `docs/paper-strategy.md`
- Best paper angle: "One Model, 91% of SOTA" — efficiency + exhaustive ablation study
- Target venues: IEEE Access, Multimedia Tools & Applications, Scientific Reports
- Key contribution: 225+ experiments proving feature quality (not association) is the MTMC bottleneck
- Before any paper-related work, read `docs/paper-strategy.md` for the full analysis

## Testing
- Framework: pytest
- Run: `pytest tests/ -v`
- Smoke test: `python scripts/run_pipeline.py --config configs/default.yaml --smoke-test`

## What NOT to Do
- Don't add `mtmc_only=True` for submission — it drops single-cam tracks and hurts IDF1 by ~5pp
- Don't enable track smoothing or edge trim — neutral to harmful
- Don't use text find/replace on raw JSON strings for Unicode — breaks JSON structure
- Don't guess config override paths — always trace from `cfg.stageN` in the pipeline code
- Don't repeat dead-end experiments — check `docs/findings.md` first
- Don't compare ResNet101-IBN-a mAP to VeRi-776 baselines — our eval is on CityFlowV2 (different dataset, 128 vs 576 IDs)
