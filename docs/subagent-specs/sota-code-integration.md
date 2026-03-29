# SOTA Code Integration Plan — Vehicle & Person MTMC

## Status: RESEARCH COMPLETE, PENDING IMPLEMENTATION

---

## Part A: Vehicle Pipeline (CityFlowV2)

### Current: MTMC IDF1 = 78.4% | Target: 84.86% (AIC22 1st) | Gap: 6.46pp

### Key SOTA Repos

| Repo | URL Pattern | Techniques | Metrics |
|------|------------|------------|---------|
| DMT (AIC21 1st) | `michuanhaohao/AICITY2021_Track2_DMT` | Camera-aware + viewpoint-aware training, pseudo-label DBSCAN, FIC/CID_BIAS computation | IDF1=80.95% |
| AIC22 solutions | Various AIC22 Track 1 repos | 3-5 model ensemble, IBN backbones, 384px, box-grained matching | IDF1=84.86% |

### Gap Decomposition → Integration Map

| Gap Component | Impact | SOTA Technique | Our Status | Integration Target |
|--------------|:------:|---------------|------------|-------------------|
| Single model | -2.5-3.5pp | 3-5 model ensemble @ 384px, score-level fusion | ResNet101-IBN exists (52.77% mAP, needs VeRi-776 pretrain) | stage2 + stage4 |
| 256px input | -1.0-1.5pp | 384×384 crops with proper training | 384px ViT trained (80.14% mAP) but wrong checkpoint on Kaggle | 10a notebook |
| No camera-aware training | -1.0-1.5pp | DMT: adversarial camera loss + viewpoint aux | NOT STARTED | Training pipeline (09b/09d) |
| No CID_BIAS | -0.5-1.0pp | Per-camera-pair distance bias from GT matches | Code EXISTS in stage4 (L310-350), no .npy matrix | scripts/ + stage4 |
| Reranking disabled | -0.5-1.0pp | k-reciprocal reranking (helps with strong features) | Disabled (hurts with current weak features) | stage4 |

### Phased Integration Plan

#### Phase 0: Quick Wins (CPU-only, days)
**0a: Deploy correct 384px checkpoint**
- Upload 09b v2 (80.14% mAP) to Kaggle dataset `mrkdagods/mtmc-weights`
- Update 10a notebook to load 384px model
- Refit PCA on 384px embeddings (384D vs 256D)
- **Expected**: +1.0-2.5pp MTMC IDF1
- **Status**: 384px checkpoint exists locally, was never properly deployed

**0b: Generate CID_BIAS matrix**
- Compute per-camera-pair bias from CityFlowV2 GT training matches
- Code path: `src/stage4_association/pipeline.py` L310-350 reads `cid_bias.npy`
- Script needed: `scripts/compute_cid_bias.py`
- **Expected**: +0.5-1.0pp
- **Status**: Code exists, just needs the .npy file

#### Phase 1: Feature Quality (Kaggle GPU, 1-2 weeks)
**1a: VeRi-776 pretraining for ResNet101-IBN-a**
- Current ResNet: ImageNet → CityFlowV2 = 52.77% mAP (expected, missing middle step)
- SOTA ResNet: ImageNet → VeRi-776 → CityFlowV2 = 75-80% mAP
- Train on VeRi-776 dataset (576 IDs, 50k images) first
- Then fine-tune on CityFlowV2
- **Expected**: ResNet mAP 70%+ → enables meaningful ensemble

**1b: Score-level ensemble**
- ViT 256px (80.14% mAP) + ResNet101-IBN-a (70%+ after VeRi pretrain)
- Score-level fusion: `similarity_fused = α × sim_vit + (1-α) × sim_resnet`
- Infrastructure exists in stage4, just needs working secondary model
- **Expected**: +1.5-2.5pp over single model

**1c: Re-enable reranking with stronger features**
- k-reciprocal reranking hurts with weak features but helps with strong ones
- After ensemble features, sweep k1={10,20}, k2={3,6}, lambda={0.1,0.3}
- **Expected**: +0.5-1.0pp

#### Phase 2: Camera-Aware Training (Kaggle GPU, 2-3 weeks)
**DMT Training Pipeline** (from `michuanhaohao/AICITY2021_Track2_DMT`):

**2a: Camera-aware adversarial loss**
- Key idea: Gradient reversal layer makes features camera-invariant
- Files in DMT: `train_cam.py`, `modeling/make_cam_model.py`
- Add `camera_id` prediction head with gradient reversal
- Loss: `L_total = L_triplet + L_crossentropy + λ_cam × L_camera_adversarial`
- Modify: our training notebook to add camera-aware branch

**2b: Viewpoint-aware auxiliary loss**  
- Key idea: Add viewpoint classifier to learn orientation-invariant features
- Files in DMT: `train_view.py`
- Requires viewpoint annotations or pseudo-labels from orientation estimation

**2c: Pseudo-label DBSCAN fine-tuning**
- Key idea: Cluster unlabeled target domain features, assign pseudo-labels, retrain
- Files in DMT: `train_stage2_v1.py`, `cluster_ids.py`
- 2-stage: supervised pretrain → unsupervised fine-tune with DBSCAN clusters
- **Expected combined Phase 2**: +1.0-2.0pp

#### Phase 3: Association Refinements (CPU, 1 week)
- Deploy CID_BIAS from Phase 0b
- Re-sweep association params with new features
- Optional: box-grained matching (per-detection features within tracklet)

### Cumulative Expected Impact
| After Phase | MTMC IDF1 (est.) |
|:-----------:|:----------------:|
| Current | 78.4% |
| Phase 0 | 79.5-80.5% |
| Phase 1 | 82.0-83.5% |
| Phase 2 | 83.0-85.0% |
| Phase 3 | 84.0-86.0% |

---

## Part B: Person Pipeline (WILDTRACK)

### Current: IDF1 = 31.5% | Architecture is fundamentally wrong

### Critical Finding: Our Architecture is Wrong for WILDTRACK

WILDTRACK has **heavily overlapping FOVs** across 7 cameras covering the same area. SOTA methods work completely differently:

| Method | Approach | MODA |
|--------|---------|:----:|
| **MVDeTr** (ECCV 2022) | Deformable transformer, multi-view to ground-plane | **91.5%** |
| **MVDet** (ECCV 2020) | CNN feature projection to ground plane | 88.2% |
| **MVFlow** (WACV 2024) | Ground-plane detection + optical flow tracking | — |
| **Our pipeline** | Per-camera detect→track→associate | ~31% IDF1 |

**ALL SOTA methods**:
1. Project camera features onto a shared **ground plane** (BEV)
2. Detect **once** on the ground plane (not per-camera)
3. Track on the ground plane with 3D positions

**Our pipeline** detects per-camera, tracks per-camera, then tries to match — creating 800+ tracklets for ~20 people because each person is detected independently in each of 7 cameras.

### Key SOTA Repos for Person MTMC

| Repo | URL Pattern | Approach |
|------|------------|---------|
| MVDeTr | `hou-yz/MVDeTr` | Multi-view deformable transformer → ground plane detection |
| MVDet | `hou-yz/MVDet` | Feature perspective transform → ground plane CNN |
| MVFlow | `cvlab-epfl/MVFlow` | Ground-plane tracking with MuSSP (min-cost flow) |
| SHOT | Various | Self-training / pseudo-label domain adaptation for person ReID |

### Recommendation

**Option A (Recommended): Port MVDeTr as separate person pipeline**
- Replace stages 0-4 entirely for overlapping-camera datasets
- MVDeTr gives 91.5% MODA on WILDTRACK
- Requires: camera calibration matrices (WILDTRACK provides these)
- Implementation: new `src/stage_wildtrack/` module with ground-plane detection + tracking
- Effort: 2-3 weeks for training + integration

**Option B: Improve current pipeline (diminishing returns)**  
- Continue tweaking detect→track→associate
- Ceiling likely around 40-50% IDF1 at best
- Fundamental problem: per-camera detection creates N×7 tracklets for N people

**Option C: Hybrid — use MVDeTr for detection, our pipeline for tracking**
- Use MVDeTr to get ground-plane detections
- Project back to camera views for appearance features
- Use our existing ReID + association
- Less architectural change but still significant

### Person Pipeline Action Items
1. Clone MVDeTr repo, study architecture
2. Check if WILDTRACK calibration matrices are available in our dataset
3. Decide: full MVDeTr port vs hybrid approach
4. Train MVDeTr on WILDTRACK (requires GPU, ~6-12h on P100)

---

## Integration Priority Order (Both Pipelines)

| Priority | Task | Pipeline | GPU? | Expected Impact |
|:--------:|------|----------|:----:|:---------------:|
| **1** | Deploy correct 384px ViT checkpoint | Vehicle | No | +1.0-2.5pp |
| **2** | Generate CID_BIAS .npy matrix | Vehicle | No | +0.5-1.0pp |
| **3** | VeRi-776 pretrain ResNet101-IBN-a | Vehicle | Yes (12h) | enables ensemble |
| **4** | Score-level ensemble | Vehicle | No | +1.5-2.5pp |
| **5** | Re-enable reranking sweep | Vehicle | No | +0.5-1.0pp |
| **6** | Clone + study MVDeTr for person pipeline | Person | No | research |
| **7** | Camera-aware DMT training | Vehicle | Yes (12h) | +1.0-2.0pp |
| **8** | Train MVDeTr on WILDTRACK | Person | Yes (12h) | step change |
| **9** | Pseudo-label DBSCAN fine-tuning | Vehicle | Yes (6h) | +0.5-1.0pp |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| 384px PCA refit changes similarity distribution | Re-sweep association params after deployment |
| ResNet still weak after VeRi-776 | Use 0.0 fusion weight (single model) as fallback |
| DMT requires camera labels in training data | CityFlowV2 has camera_id in annotations |
| MVDeTr requires calibration matrices | WILDTRACK provides intrinsic + extrinsic matrices |
| Kaggle session timeout (12h) | Split training into checkpointed stages |