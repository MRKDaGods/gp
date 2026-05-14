# Spec: Recreate `docs/system-comparative-analysis.md`

**Goal**: Replace the current hallucinated `docs/system-comparative-analysis.md` with the verified content below.

**Action for coder**: Overwrite `docs/system-comparative-analysis.md` with the contents of the fenced block below verbatim.

**Why**: The previous version contained fabricated VeRi-776 numbers (87.33 mAP / 98.45 R1 with no eval evidence in repo), wrong WildTrack MODA (0.903 instead of 0.900), wrong vehicle headline (0.775 instead of 0.7703), and wrong gap math (7.4pp / 91.3% instead of 7.83pp / 90.8%).

**Citations verified against**:
- `docs/findings.md` — "Final Result — CLIP+DINOv2 Score-Level Ensemble" (2026-04-25), "Current Performance" tables for both pipelines
- `docs/experiment-log.md` — header (current verified best), Section 1 (current best config), Section 2.1 (10c version history)
- `docs/paper-strategy.md` — "AIC22 Track 1 Leaderboard"
- `.github/copilot-instructions.md` — "Current Performance State", "Confirmed Dead Ends", "What Actually Worked"

---

## New content for `docs/system-comparative-analysis.md`

```markdown
# System Comparative Analysis

*Conservative note: any value marked with an asterisk (*) is a literature claim, not measured by us. Values without an asterisk were measured directly in this repository and cite a section in `docs/findings.md`, `docs/experiment-log.md`, `docs/paper-strategy.md`, or `.github/copilot-instructions.md`.*

## 1. Abstract

This document compares the MTMC Tracker system against published state-of-the-art on two MTMC benchmarks: CityFlowV2 (vehicles, AI City Challenge 2022 Track 1) and WildTrack (people, overlapping cameras). The headline result is that a single CLIP-pretrained ViT-B/16 backbone, augmented at association time with DINOv2 score-level fusion, reaches **MTMC IDF1 = 0.7703** on CityFlowV2 — **90.8%** of AIC22 1st-place performance (0.8486) — using one ReID model rather than the 5-model ensemble of the leaderboard winner. On WildTrack, the same Stage 1–4 pipeline shell (with MVDeTr replacing the vehicle detector) reaches **Ground-plane IDF1 = 0.947**, within **0.6pp** of the literature reference of ~0.953*. The remaining **7.83pp** CityFlowV2 gap is attributable to cross-camera feature invariance, not association tuning: 225+ association configs have been exhausted, and a comprehensive dead-end catalog rules out the obvious structural alternatives.

## 2. Headline Performance

### 2.1 Vehicle Pipeline (CityFlowV2)

| Metric | Value | Source |
|---|---:|---|
| MTMC IDF1 (best, fusion) | **0.7703** | `findings.md` § "Final Result — CLIP+DINOv2 Score-Level Ensemble"; `experiment-log.md` header |
| MTMC IDF1 (no-fusion control, single CLIP) | **0.7663** | `findings.md` § "Final Result …" (row `no_fusion_control`) |
| MTMC IDF1 (DINOv2 standalone) | **0.744** | `findings.md` § "Current Performance" — DINOv2 ViT-L/14 row |
| Single-camera ReID mAP (CLIP ViT-B/16) | **80.14%** | `findings.md` § "Current Performance"; `.github/copilot-instructions.md` § "Current Performance State" |
| Single-camera ReID R1 (CLIP ViT-B/16) | **92.27%** | same as above |
| Single-camera ReID mAP (DINOv2 ViT-L/14, 09s v1) | **86.79%** | `findings.md` § "Current Performance" |
| Single-camera ReID R1 (DINOv2 ViT-L/14, 09s v1) | **96.15%** | same as above |
| Secondary ResNet101-IBN-a mAP | **52.77%** | `findings.md` § "Current Performance" — Secondary Model row; `.github/copilot-instructions.md` |

**Best operating point**: 10c v15 / 10a v7, CLIP+DINOv2 score-level fusion, `w_secondary=0.00`, `w_tertiary=0.60`. Fusion delivers **+0.40pp** over the single-CLIP no-fusion control (0.7663 → 0.7703). DINOv2 alone regresses by **−3.1pp** vs CLIP alone (0.744 vs 0.7663+) despite +6.65pp single-camera mAP — the canonical "mAP-vs-MTMC paradox" documented in `findings.md`.

### 2.2 Person Pipeline (WildTrack)

| Metric | Value | Source |
|---|---:|---|
| Ground-plane IDF1 | **0.947** | `findings.md` § "Person Pipeline (WILDTRACK) — NEW"; `.github/copilot-instructions.md` § "Person Pipeline (WILDTRACK)" |
| Ground-plane MODA | **0.900** | `findings.md` § "Person Pipeline (WILDTRACK)" — best converged operating point |
| Detector MODA (MVDeTr ResNet18, 12a v3) | **0.921** | `findings.md` § "Person Pipeline (WILDTRACK)"; `.github/copilot-instructions.md` |
| Tracker configs tested | **59+** | `findings.md` § "Person Pipeline (WILDTRACK)" — Status row |
| Status | **FULLY CONVERGED** | same |

The IDF1=0.947 figure is reproduced independently across **12b v1, v2, and v3**. Kalman, naive, and global-optimal trackers were all evaluated; the best Kalman runs cluster at **IDF1 = 0.9467 ± 0.0004** across confidence thresholds 0.15–0.35. The pipeline is tracker-limited, not detector-limited.

## 3. Comparison to State of the Art

### 3.1 CityFlowV2 (AI City Challenge 2022 Track 1)

| Rank | System | MTMC IDF1 | Models | Source |
|---|---|---:|:---:|---|
| 1 | Team28 (matcher) | 0.8486 | 5 | `paper-strategy.md` § "AIC22 Track 1 Leaderboard" |
| 2 | Team59 (BOE) | 0.8437 | 3 | same |
| 3 | Team37 (TAG) | 0.8371 | — | same |
| 4 | Team50 (FraunhoferIOSB) | 0.8348 | — | same |
| 10 | Team94 (SKKU) | 0.8129 | — | same |
| 18 | Team4 (HCMIU) | 0.7255 | — | same |
| — | **Ours (single CLIP + DINOv2 fusion)** | **0.7703** | 1 (+1 score stream) | `findings.md` § "Final Result …" |

**Gap to SOTA**: 0.8486 − 0.7703 = **7.83pp** (`.github/copilot-instructions.md` § "Current Performance State").
**Relative efficiency**: 0.7703 / 0.8486 = **90.8%** of 1st-place IDF1 with **1 ReID model** vs 5.

### 3.2 WildTrack

| System | GP IDF1 | GP MODA | Source |
|---|---:|---:|---|
| Literature reference (best published) | 0.953* | 0.915* | [CITE_NEEDED — literature claim, not measured by us] |
| **Ours (12b v1/v2/v3)** | **0.947** | **0.900** | `findings.md` § "Person Pipeline (WILDTRACK)" |

**Gap to SOTA**: ~0.6pp IDF1 (`.github/copilot-instructions.md` § "Person Pipeline (WILDTRACK)").

## 4. Single-Camera ReID Component Results

### 4.1 CityFlowV2 (deployed and candidate backbones)

| Model | mAP | R1 | Source |
|---|---:|---:|---|
| TransReID ViT-B/16 CLIP @ 256px (deployed primary) | **80.14%** | **92.27%** | `findings.md` § "Current Performance" |
| DINOv2 ViT-L/14 (09s v1, fusion partner only) | **86.79%** | **96.15%** | same |
| ResNet101-IBN-a ImageNet→CityFlowV2 (secondary, 09d v18) | **52.77%** | — | same; see also `findings.md` § "Critical Discovery: ResNet101-IBN-a 52.77% Is Expected" |
| ViT-Large AugReg without CLIP/DINOv2 pretraining (09r v7) | 60.38% | 76.57% | `findings.md` § "Current Performance" |
| ResNeXt101-IBN-a ArcFace (09j v2) | 36.88% | 62.69% | same |
| ViT-Small/16 IN-21k (09k v1) | 48.66% | 62.01% | same |
| EVA02 ViT-B/16 CLIP (09o v1) | 48.17% | 65.90% | same |
| CLIP RN50x4 CNN (09m v2) | 1.55% | 4.18% | same |
| LAION-2B CLIP Triplet (09l v3) | 78.61% | 90.43% | same |

### 4.2 VeRi-776 (Cross-Dataset)

A direct VeRi-776 evaluation of the deployed TransReID ViT-B/16 (CLIP) checkpoint is currently running on Kaggle. Results will be filled in once the run completes.

| Config | mAP | R1 | R5 | R10 |
|---|---:|---:|---:|---:|
| Baseline (no AQE, no rerank) | TBD | TBD | TBD | TBD |
| AQE only (k=3) | TBD | TBD | TBD | TBD |
| Rerank only (k1=30, k2=10, λ=0.2) | TBD | TBD | TBD | TBD |
| AQE + Rerank | TBD | TBD | TBD | TBD |

> **Note**: The previous version of this document reported VeRi-776 mAP=87.33% / R1=98.45% as a measured result. **That claim is hallucinated** — no such evaluation artifact exists in the repository. It has been removed and replaced with the placeholder above pending the live Kaggle run.

Literature reference for context: VeRi-776 SOTA mAP / R1 ≈ 87.0* / 97.7* [CITE_NEEDED — literature claim, not measured by us].

## 5. What Worked (Cumulative Gains)

| Change | Magnitude | Source |
|---|---:|---|
| Conflict-free CC (vs vanilla CC) | **+0.21pp** | `.github/copilot-instructions.md` § "What Actually Worked"; `experiment-log.md` § 2.1 v23–v26 |
| Intra-merge (thresh=0.80, gap=30) | **+0.28pp** | `.github/copilot-instructions.md` § "What Actually Worked"; `experiment-log.md` § 2.1 v34 |
| Temporal overlap bonus | **+0.9pp** | `.github/copilot-instructions.md` § "What Actually Worked" |
| FIC whitening | **+1 to +2pp** | same |
| Power normalization | **+0.5pp** | same |
| AQE K=3 | small positive | `experiment-log.md` § 2.1 v14–v22, v39 |
| min_hits=2 (Stage 1 tracker) | **+0.2pp** | `.github/copilot-instructions.md` § "What Actually Worked" |
| Kalman tuning (person pipeline) | **+1.9pp IDF1** | same; cf. `findings.md` § "Person Pipeline" — vs 12b v9 naive 92.8% |
| CLIP+DINOv2 score-level fusion (`w_tertiary=0.60`) | **+0.40pp** over single CLIP | `findings.md` § "Final Result — CLIP+DINOv2 Score-Level Ensemble" |

## 6. Dead Ends (with Magnitudes)

| Approach | Impact | Source |
|---|---:|---|
| CSLS | **−34.7pp** (catastrophic; penalizes vehicle-type hubs) | `.github/copilot-instructions.md` § "Confirmed Dead Ends"; `findings.md` |
| AFLink motion linking (CityFlowV2) | **−3.82pp** typical, **−13.2pp** worst | `.github/copilot-instructions.md` § "Confirmed Dead Ends" |
| 384px ViT deployment | **−2.8pp** | same; `findings.md` § "Critical Discovery: 384px Is a Dead End for MTMC" |
| FAC (cross-camera KNN consensus) | **−2.5pp** | `.github/copilot-instructions.md` § "Confirmed Dead Ends" |
| Feature concatenation (vs score fusion) | **−1.6pp** | same |
| DMT camera-aware training | **−1.4pp** single-model | same |
| CID_BIAS (topology variant) | **−1.0 to −1.2pp** | `findings.md` § "AIC Winning Methods Analysis…"; `.github/copilot-instructions.md` |
| CID_BIAS (GT-learned variant) | **−3.3pp** | same |
| Hierarchical clustering | **−1 to −5pp** | `.github/copilot-instructions.md` § "Confirmed Dead Ends" |
| OSNet secondary (current weights, score-level or concat) | **−0.8 to −1.1pp** | same |
| DINOv2 standalone (vs CLIP standalone) | **−3.1pp** | `findings.md` § "Final Result — CLIP+DINOv2 Score-Level Ensemble" — paragraph on mAP-vs-MTMC paradox |
| Network flow solver | **−0.24pp** (and conflation 27→30) | `.github/copilot-instructions.md` § "Confirmed Dead Ends"; `findings.md` § "Final Result …" closing paragraph |
| Reranking on vehicle pipeline | always hurts | `.github/copilot-instructions.md` § "Confirmed Dead Ends" |
| Score-level ensemble with 52.77% secondary | **−0.1pp** | same |
| VeRi-776→CityFlowV2 ResNet pretrain (09f v3) | 42.7% mAP (worse than direct 52.77%) | same; `findings.md` § "Critical Discovery: ResNet101-IBN-a 52.77%…" |
| Extended ResNet fine-tuning (09d gumfreddy v3) | 50.61% mAP (degraded) | same |
| ArcFace on ResNet101-IBN-a (09i v1) | 50.80% mAP | same |
| ResNeXt101-IBN-a ArcFace (09j v2) | 36.88% mAP (weight loading mismatch) | same |
| Circle loss + triplet (Experiment B; 09l v1 LAION-2B) | 16–30% mAP (loss `inf`) | same |
| SGD for ResNet | 30.27% mAP (catastrophic) | same |
| Global optimal tracker (person, 12b v3) | **−3.5pp IDF1** vs Kalman | same; `findings.md` § "Person Pipeline" |
| Extended Kalman sweeps (person) | 59 configs within ±0.0004 IDF1 | same |
| Better person detector (MODA 90.9 → 92.1%) | IDF1 unchanged at 0.947 | `.github/copilot-instructions.md` § "Confirmed Dead Ends" |

## 7. Conclusion: The Bottleneck Is Feature Quality, Not Association

Three independent lines of evidence converge on the same conclusion:

1. **Association is exhausted.** 225+ Stage-4 configs have been tested across `experiment-log.md` Sections 2.1 and 3, and all cluster within **0.3pp** of the optimal. Structural variants — network flow solver (−0.24pp), hierarchical clustering (−1 to −5pp), CSLS (−34.7pp), CID_BIAS (−1.0 to −3.3pp), AFLink (−3.82pp typical), and FAC (−2.5pp) — uniformly regress.

2. **Stronger single-camera features do not transfer.** DINOv2 ViT-L/14 improves single-camera mAP from 80.14% → 86.79% (+6.65pp) and R1 from 92.27% → 96.15% (+3.88pp), yet its standalone MTMC IDF1 of 0.744 is **−3.1pp** below the CLIP standalone control. Score-level fusion of CLIP and DINOv2 recovers some of that loss for a net **+0.40pp** gain, but this is a small refinement, not a step change. Higher input resolution (384px), camera-aware training (DMT), and reranking — all techniques that help SOTA ensembles — actively hurt this single-model pipeline.

3. **The remaining 7.83pp gap to AIC22 1st place tracks model count, not algorithmic complexity.** SOTA systems use 3–5 ReID backbones; we use one strong primary plus one fusion stream. Every attempted secondary on the current codebase (R50-IBN, R101-IBN-a, ResNeXt-IBN-a, ViT-Small, EVA02, CLIP RN50x4, LAION-2B CLIP-ViT) was either too weak or too correlated with the primary to deliver ensemble diversity, and OSNet — which historically did contribute (v80 era, IDF1 ≈ 0.784) — depended on the now-unavailable `vehicle_osnet_veri776.pth` checkpoint dropped from `mrkdagods/mtmc-weights` on 2026-03-30.

The actionable implication is that further MTMC gains require **better cross-camera invariance training** (e.g., camera-aware fine-tuning of DINOv2, SAM2 foreground masking, or GNN-based association edge classification), not more Stage-4 tuning and not more uncorrelated-secondary search on the current weight set.

---

### Footnotes

- (*) Literature value, not re-measured in this repository. See [CITE_NEEDED] markers in the relevant sections.
```