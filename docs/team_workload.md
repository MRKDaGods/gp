# Team Workload Division

## Overview
4-person team building a multi-camera tracking system. The project is divided into 6 phases (A-F) with clear module ownership.

## Developer Assignments

### Developer 1: Project Lead / Infrastructure
**Primary**: Project skeleton, data pipeline, integration, CI

| Phase | Task | Key Files |
|---|---|---|
| A | Project skeleton, config system | pyproject.toml, Makefile, configs/ |
| A | Core data models & I/O | src/core/* |
| A | Stage 0: Ingestion pipeline | src/stage0_ingestion/* |
| A | CLI scripts | scripts/run_pipeline.py, run_stage.py |
| D | Integration testing | tests/test_integration/ |
| F | Export & deployment | src/stage6_visualization/export.py |

**Deliverables**: Working skeleton, reproducible builds, end-to-end smoke tests.

### Developer 2: Detection & Tracking
**Primary**: Stage 1 (detection + single-camera tracking), video annotation

| Phase | Task | Key Files |
|---|---|---|
| B | YOLO detector wrapper | src/stage1_tracking/detector.py |
| B | BoxMOT tracker wrapper | src/stage1_tracking/tracker.py |
| B | Tracklet builder | src/stage1_tracking/tracklet_builder.py |
| B | Tracker benchmarking (BoT-SORT vs Deep-OCSORT vs ByteTrack) | configs/experiments/ |
| F | Video annotator | src/stage6_visualization/video_annotator.py |
| F | BEV mapper | src/stage6_visualization/bev_mapper.py |

**Deliverables**: Per-camera tracklets, tracker comparison report, annotated videos.

### Developer 3: ReID & Feature Extraction
**Primary**: Kaggle training, Stage 2 (feature extraction), re-ranking

| Phase | Task | Key Files |
|---|---|---|
| C | Kaggle notebook 01: Dataset prep | notebooks/kaggle/01_*.ipynb |
| C | Kaggle notebook 02: Person ReID | notebooks/kaggle/02_*.ipynb |
| C | Kaggle notebook 03: Vehicle ReID | notebooks/kaggle/03_*.ipynb |
| C | Crop extractor | src/stage2_features/crop_extractor.py |
| C | ReID model wrapper | src/stage2_features/reid_model.py |
| C | HSV + PCA + embeddings | src/stage2_features/hsv_extractor.py, pca_whitening.py |
| D | k-reciprocal re-ranking | src/stage4_association/reranking.py |

**Deliverables**: Trained ReID weights, feature extraction pipeline, ReID quality report.

### Developer 4: Association & Applications
**Primary**: Stages 3-4, evaluation, dashboard, NL query, 3D simulation

| Phase | Task | Key Files |
|---|---|---|
| D | FAISS index | src/stage3_indexing/faiss_index.py |
| D | Metadata store | src/stage3_indexing/metadata_store.py |
| D | Similarity + spatial-temporal | src/stage4_association/similarity.py, spatial_temporal.py |
| D | Graph solver | src/stage4_association/graph_solver.py |
| E | Evaluation (TrackEval) | src/stage5_evaluation/* |
| F | Streamlit dashboard | src/apps/web_dashboard.py |
| F | NL query engine | src/apps/nl_query.py |
| F | 3D simulation | src/apps/simulation_3d.py |

**Deliverables**: Cross-camera association, evaluation reports, web dashboard with all features.

## Phase Dependencies

```
Phase A (Foundation) ─────────────────────────────────────────┐
    │                                                          │
    ├──→ Phase B (Detection & Tracking) ──┐                    │
    │                                      │                    │
    └──→ Phase C (ReID Training) ─────────┼──→ Phase D (Association)
                                           │         │
                                           │         ├──→ Phase E (Evaluation)
                                           │         │
                                           └─────────┴──→ Phase F (Visualization & Apps)
```

- **Phase A**: Must complete first (skeleton, config, core models)
- **Phase B & C**: Can run in parallel after A
- **Phase D**: Needs B (tracklets) and C (features)
- **Phase E & F**: Can start partially during D, complete after

## Communication Protocol
- All inter-stage data formats defined in `docs/data_contracts.md`
- Each developer writes unit tests for their modules
- Integration tests verify stage-to-stage handoffs
- Config changes go through `configs/default.yaml` (reviewed by Dev 1)

## Code Review Checklist
- [ ] Follows data contract specifications
- [ ] Unit tests pass (`pytest tests/test_stageN/`)
- [ ] No hardcoded paths (use config system)
- [ ] Loguru logging for key operations
- [ ] Type hints on public functions
- [ ] Smoke test works (`--smoke-test` flag)
