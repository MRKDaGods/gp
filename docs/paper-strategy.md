# Paper Strategy — MTMC Tracker

## Publishability Assessment

### Where We Stand (as of April 2026)

**Vehicle Pipeline (CityFlowV2)**
| Metric | Value | SOTA (AIC22 1st) | Gap |
|---|---|---|---|
| MTMC IDF1 | **77.5%** | 84.86% | 7.4pp |
| Primary model | TransReID ViT-B/16 CLIP 256px, mAP=80.14% | 5-model ensemble | Single model |
| Association configs tested | **225+** | — | All within 0.3pp of optimal |
| Dead ends explored | 12+ major approaches | — | Exhaustive |

**Person Pipeline (WILDTRACK)**
| Metric | Value | SOTA | Gap |
|---|---|---|---|
| Ground-plane IDF1 | **94.7%** | ~95.3% | 0.6pp |
| Ground-plane MODA | **90.3%** | — | — |
| Detector (MVDeTr 12a v3) | MODA=92.1% | — | Best achieved |
| Status | **Effectively converged** | — | Tracker-limited |

### The Two Paper Archetypes

1. **"We beat everyone"** — Needs SOTA numbers. We're at 0.775 vs 0.849. Not viable.
2. **"We match them with less"** — Needs a clear efficiency/simplicity angle. **This is our best shot.**

### What's Publishing in This Space (2024-2026)

| Paper | Venue | IDF1 | What Got It Published |
|---|---|---|---|
| Score-based matching (Yang et al.) | Multimedia Tools & Apps, 2025 | ~0.80 | Novel matching formulation |
| Hierarchical clustering (Sz\u0171cs et al.) | Multimedia Tools & Apps, 2024 | ~0.72 | Novel clustering method (cited 10x!) |
| Spatial-temporal multi-cuts (Herzog et al.) | arXiv/under review, 2024 | Online SOTA | Novel graph formulation (no training) |
| Dynamic Global Tracker (Chen et al.) | Scientific Reports, 2026 | Competitive | Online tracking framework |
| Real-time MTMC (Al\u00e0s et al.) | IEEE Access, 2026 | Lower | Real-time deployment angle |
| Lightweight MTMC (Liu et al.) | SPIE ICAIP, 2026 | Lower | Lightweight neural networks |
| GMT (Zhen et al.) | arXiv, 2024 | +21pp CVMA | New dataset + unified framework |

**Key observation**: Papers with IDF1 in the 0.72-0.80 range ARE getting published at respectable venues (Springer, IEEE Access, Scientific Reports), as long as they bring a novel contribution or angle.

### AIC22 Track 1 Leaderboard (Reference)

| Rank | Team | IDF1 | Method |
|---|---|---|---|
| 1 | Team28 (matcher) | 0.8486 | 5-model ensemble + Box-Grained Reranking (ConvNeXt, R50, Res2Net200, HR48, ResNeXt101) |
| 2 | Team59 (BOE) | 0.8437 | 3-model ensemble |
| 3 | Team37 (TAG) | 0.8371 | Multi-model |
| 4 | Team50 (FraunhoferIOSB) | 0.8348 | — |
| 10 | Team94 (SKKU) | 0.8129 | — |
| 18 | Team4 (HCMIU) | 0.7255 | — |
| — | **Ours** | **0.7750** | Single ViT-B/16 CLIP + FIC + conflict-free CC |

## Recommended Paper Angle

### Primary: "One Model, 91% of SOTA"

**Title idea**: *"One Model, 91% of SOTA: A Systematic Study of Feature Quality vs. Association Tuning Bottlenecks in Multi-Camera Vehicle Tracking"*

**Core argument**: Single CLIP-pretrained ViT-B/16 with FIC whitening achieves 77.5% MTMC IDF1 (91% of the 5-model-ensemble SOTA at 84.86%) using ~5x less compute. 225+ exhaustive experiments definitively prove that association algorithm tuning is NOT the bottleneck — cross-camera feature invariance is.

**Novel contributions**:
1. **Empirical proof that feature quality, not association, is the MTMC bottleneck** — 225+ configs, all within 0.3pp
2. **CLIP pretraining for vehicle ReID** — first systematic evaluation showing CLIP outperforms ImageNet for city-scale ReID
3. **Comprehensive dead-end catalog** — 12+ failed approaches documented (CSLS, hierarchical clustering, FAC, 384px, DMT, CID_BIAS, etc.) saving future researchers time
4. **Dual-domain evaluation** — same pipeline handles vehicles (CityFlowV2, non-overlapping cameras) AND pedestrians (WILDTRACK, overlapping cameras)
5. **FIC + modern backbone** — systematic ablation of AIC21 FIC technique with CLIP ViT features
6. **Conflict-free connected components** — same-camera conflict resolution preventing cascading merge errors
7. **Quality-aware crop selection** — Laplacian-scored crops preventing blurry detections from polluting embeddings

### Publishable Angles (Backup)

| Angle | Pitch | Target Venue | Viability |
|---|---|---|---|
| **Efficiency** | "Single model, 5x less compute, 91% accuracy" | IEEE Access, MTA | **HIGH** |
| **Ablation study** | "225+ configs prove feature > association" | CVPRW Reproducibility, Sensors | MEDIUM |
| **Dual-domain** | "Unified pipeline for vehicles + pedestrians" | IEEE T-ITS, PRL | MEDIUM |
| **System/tool paper** | "Modular 7-stage pipeline with dashboard" | SoftwareX, JOSS | LOW-MEDIUM |

## Target Venues

| Venue | Realistic? | Why |
|---|---|---|
| **CVPR/ICCV/ECCV main** | No | Need SOTA or strong novelty |
| **CVPRW/ECCVW AI City** | Maybe | If framed as efficiency study with solid ablation |
| **IEEE T-ITS** | Possible | Applied focus, IDF1=0.775 is respectable |
| **IEEE Access** | **Yes** | Papers with lower results + novel angle published regularly (IF ~3.4) |
| **Multimedia Tools & Apps** | **Yes** | Sz\u0171cs et al. published with IDF1=0.72 |
| **Scientific Reports** | **Yes** | Active MTMC papers publishing right now |

## What's Needed Before Submission

### Must-Have
1. **Compute efficiency comparison** — GPU hours: ours (1x T4, ~3h pipeline) vs AIC22 top (5 models on A100s, ~50h+)
2. **Clean ablation table** — baseline → +CLIP pretrain → +FIC → +quality crops → +PCA → final
3. **Error analysis** — fragmentation (87) vs conflation (35) decomposition, per-camera heatmap
4. **Feature similarity distributions** — same-ID vs different-ID across cameras (the "invariance gap" visualization)

### Nice-to-Have
5. **Synthehicle dataset evaluation** — third dataset strengthens multi-dataset story
6. **Inference latency** — per-stage timing breakdown
7. **Qualitative examples** — correct matches, failure cases, cross-camera identity strips

## What to Skip
- Don't waste more GPU hours on ensemble attempts — 09g (43.8% mAP) confirms weak secondary
- Don't try to close the SOTA gap — paper is stronger as honest analysis of *why* the gap exists
- Don't add features just for the paper — the system is mature enough as-is