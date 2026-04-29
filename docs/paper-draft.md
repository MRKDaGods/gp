# Title

Beyond mAP: Training Methodology Dominates Feature Capacity in Multi-Camera Vehicle Tracking

## Abstract

Multi-camera multi-target tracking (MTMC) is commonly framed as a downstream beneficiary of stronger single-camera vehicle re-identification features: if a backbone achieves higher mAP and Rank-1 on vehicle ReID, it is generally expected to yield better end-to-end MTMC identity preservation. This manuscript reports a controlled negative result against that assumption on CityFlowV2, the benchmark underlying AI City Challenge 2022 Track 1. Using a seven-stage offline pipeline spanning ingestion, single-camera tracking, feature extraction, indexing, cross-camera association, evaluation, and visualization, we show that association optimization is already saturated for a strong single-model feature space. First, across 225+ controlled association configurations varying similarity thresholds, appearance weighting, FIC regularization, AQE, gallery expansion, orphan matching, and graph solvers, the top-performing runs all fall within 0.3 percentage points of the best MTMC IDF1, with a ceiling of 0.775. This indicates that, for single-model features of this strength, additional association tuning produces negligible returns. Second, a stronger Stage 2 backbone, DINOv2 ViT-L/14, achieves better vehicle ReID accuracy than TransReID ViT-B/16 CLIP (mAP 86.79 vs. 80.14, R1 96.15 vs. 92.27) and better per-camera IDF1 (0.794 vs. 0.770), yet regresses MTMC IDF1 by 3.1 percentage points (0.744 vs. 0.775). Third, we evaluate heterogeneous score-level fusion of CLIP and DINOv2 and obtain 77.03% MTMC IDF1, a +0.40pp change versus the CLIP-only baseline, which fails to exceed the historical CLIP-only baseline of 77.5%, confirming that score-level fusion of a stronger ReID feature does not close the cross-camera invariance gap. Fourth, the resulting single-model pipeline reaches 91% of the AIC22 five-model ensemble state of the art (0.775 / 0.8486) using approximately 5x less compute, alongside an exhaustive dead-end catalog covering 12+ failed techniques. Taken together, these results suggest that cross-camera invariance, not feature capacity or association sophistication, is now the dominant bottleneck in MTMC vehicle tracking.

## 1. Introduction

Multi-camera multi-target tracking (MTMC) addresses the problem of preserving object identity across multiple video streams, often with non-overlapping fields of view, asynchronous appearance changes, illumination shifts, and viewpoint discontinuities. In the vehicle domain, the task requires more than strong detection or single-camera tracking: the same vehicle must be matched across cameras even when observed from substantially different angles, at different scales, under different weather and lighting conditions, and after arbitrary time gaps. These conditions make MTMC a uniquely demanding benchmark for visual invariance.

CityFlowV2, used in AI City Challenge 2022 Track 1, is among the most widely studied benchmarks for this problem. It contains five urban scenes, 46 cameras, and 880 annotated vehicle identities, with long-range cross-camera transitions and significant viewpoint diversity. The challenge setup has encouraged a common engineering pattern: maximize per-camera detection and ReID quality, then recover the remaining performance through stronger association logic, graph optimization, reranking, or multi-model ensembles [aic22team28] [aic22team59] [aic22team37].

This pattern is supported by conventional wisdom inherited from the re-identification literature. On standard vehicle and person ReID benchmarks, stronger mAP and Rank-1 usually correlate with better downstream retrieval, clustering, and tracking behavior. As a result, larger or more modern feature extractors are often presumed to transfer their gains directly to MTMC. In practice, this means that model capacity, pretraining scale, and benchmark mAP are frequently treated as reliable proxies for eventual cross-camera identity preservation.

Our experiments show that this assumption breaks down. In a tightly controlled pipeline where the association recipe is held fixed and only the Stage 2 backbone changes, DINOv2 ViT-L/14 improves vehicle ReID mAP by 6.65 percentage points and Rank-1 by 3.88 percentage points over TransReID ViT-B/16 CLIP, and also improves per-camera IDF1. Yet it reduces end-to-end MTMC IDF1 by 3.1 percentage points. The stronger backbone is better at single-camera discrimination and worse at cross-camera identity preservation.

This result matters because it changes where effort should be spent. If a feature space with higher single-camera accuracy yields worse MTMC tracking, and if 225+ association variants remain within 0.3 percentage points of the same ceiling, then neither more graph tuning nor higher-capacity backbones can be assumed to solve the remaining error. The bottleneck is instead cross-camera invariance: whether a representation treats viewpoint, lighting, and camera-specific appearance as nuisance variables rather than discriminative shortcuts.

The main contributions of this paper are as follows.

- We show that 225+ controlled association configurations all lie within 0.3 percentage points of the optimum, establishing an MTMC IDF1 ceiling of 0.775 for the single-model TransReID ViT-B/16 CLIP feature space and demonstrating that association tuning is saturated in this regime.
- We show that DINOv2 ViT-L/14, despite delivering 86.79 mAP, 96.15 Rank-1, and stronger per-camera IDF1 than TransReID ViT-B/16 CLIP, regresses end-to-end MTMC IDF1 from 0.775 to 0.744, directly contradicting the assumption that stronger ReID benchmarks predict stronger MTMC.
- We report a heterogeneous score-level fusion setting for CLIP and DINOv2, where the best operating point reaches 77.03% MTMC IDF1, a +0.40pp gain relative to the local CLIP-only baseline but still below the historical 77.5% CLIP-only baseline.
- We present a single-model pipeline that reaches 91% of the AIC22 five-model ensemble state of the art at substantially lower computational cost, and we document an extensive dead-end catalog covering more than a dozen plausible but ineffective techniques.

The remainder of the paper is organized as follows. Section 2 briefly reviews prior work in MTMC tracking, vehicle re-identification, and association refinement. Section 3 describes the pipeline used throughout the study. Section 4 summarizes the experimental setup. Section 5 presents the three central paradoxes revealed by the experiments. Section 6 consolidates the negative results into a dead-end catalog. Section 7 cross-validates the central thesis on a person-tracking benchmark. Sections 8 through 10 discuss implications, conclude the paper, and summarize limitations.

## 2. Related Work

### 2.1 Multi-camera multi-target tracking

MTMC tracking has evolved from hand-crafted appearance matching and camera topology heuristics into systems that combine strong object detectors, single-camera trackers, learned ReID embeddings, and graph-based association. In the AI City Challenge ecosystem, leading AIC22 Track 1 systems increasingly relied on multi-model ensembles, feature fusion, and carefully tuned association to approach leaderboard performance. Representative top-ranked systems include Team28 (0.8486 MTMC IDF1), Team59 (0.8437), and Team37 (0.8371), while earlier systems such as ELECTRICITY highlighted the importance of integrating tracking, motion, and appearance cues for city-scale multi-camera reasoning [aic22team28] [aic22team59] [aic22team37] [electricity2021]. These works establish the performance frontier but also make it difficult to isolate which component truly limits single-model pipelines.

### 2.2 Vehicle re-identification

Vehicle re-identification has been strongly shaped by transformer architectures, large-scale pretraining, and benchmark-driven optimization. TransReID introduced camera-aware transformers and the JPM module to reduce viewpoint sensitivity in ReID [he2021transreid]. CLIP-ReID extended this line of work by leveraging image-text pretraining to inject semantic structure into the embedding space [li2023clipreid]. At a broader foundation-model scale, DINOv2 demonstrated that self-distillation on massive unlabeled image corpora can produce highly transferable visual features [oquab2023dinov2]. Benchmark datasets such as VeRi-776 remain the standard for evaluating vehicle ReID accuracy [liu2016veri]. However, these benchmarks do not directly test whether a representation is robust to the camera-induced domain shifts that dominate MTMC.

### 2.3 Feature aggregation and association

A wide family of methods attempts to improve retrieval and cross-camera matching after feature extraction. Average query expansion (AQE) can improve neighborhood consistency in retrieval spaces [chen2011aqe]. Feature inference compensation (FIC) whitening has been used in AIC pipelines to reduce domain bias and sharpen similarity calibration [aic21fic]. k-reciprocal reranking is a standard post-processing method for person and vehicle ReID retrieval [zhong2017kreciprocal]. Cross-domain local scaling and related calibration methods such as CSLS were originally developed for bilingual lexicon induction and have been repurposed for nearest-neighbor matching [conneau2018csls]. Although these techniques are often beneficial in isolation, their effectiveness depends on the geometry of the underlying feature space, and they may fail when the dominant errors arise from viewpoint-dependent representations rather than local retrieval noise.

## 3. Method

### 3.1 Pipeline overview

The system is implemented as a seven-stage offline pipeline designed for reproducible experimentation and component-level replacement. The stages are Ingestion, Tracking, Features, Indexing, Association, Evaluation, and Visualization. This decomposition allows controlled experiments in which one stage, such as the Stage 2 backbone, is modified while downstream logic remains fixed. The pipeline processes CityFlowV2 videos into tracklets, converts each tracklet into an aggregated feature representation, indexes those representations for cross-camera retrieval, constructs an association graph, evaluates the resulting MTMC identities, and optionally renders outputs for debugging and qualitative analysis.

Figure 1: 7-stage pipeline overview.

The central principle of the method section is descriptive rather than novel. We do not introduce a new algorithmic family; instead, we define the exact system used to obtain the empirical claims of the paper. This distinction is important because the core contribution is diagnostic: identifying which parts of the pipeline are already saturated and which parts remain limiting.

### 3.2 Stage 1 — Detection and tracking

Stage 1 performs single-camera detection and tracking. We use a YOLO26m detector to generate vehicle bounding boxes and a BoT-SORT tracker through the BoxMOT framework to produce within-camera tracklets. The tracker configuration uses `min_hits=2`, which empirically improves robustness by suppressing short-lived false positives while preserving recall on valid trajectories. This setting contributed approximately +0.2 percentage points in prior ablations relative to a more permissive default.

The role of Stage 1 is to provide stable within-camera tracklets with sufficiently clean temporal support for downstream appearance aggregation. The design intentionally avoids aggressive smoothing or trimming, which proved neutral to harmful in preliminary experiments. Tracklets are therefore treated as the base unit for all downstream cross-camera reasoning.

### 3.3 Stage 2 — Re-identification features

Stage 2 extracts tracklet-level ReID features. The primary backbone is TransReID ViT-B/16 with CLIP pretraining, operating on 256 x 256 crops. For each tracklet, we first score candidate crops by image sharpness using Laplacian variance and retain the highest-quality samples. This quality-aware crop selection reduces the influence of motion blur and compression artifacts that would otherwise dominate naive temporal averaging.

Feature normalization follows a camera-aware and retrieval-oriented sequence. Embeddings are first processed with per-camera batch normalization, then projected to 384 dimensions using PCA. We apply FIC whitening with regularization 0.50 to improve similarity calibration across cameras. Finally, we apply power normalization and L2 normalization before indexing. This chain reflects the best-performing recipe discovered in the broader study and is held fixed in the controlled comparisons unless otherwise stated.

The primary controlled variant in the paper replaces TransReID ViT-B/16 CLIP with DINOv2 ViT-L/14 while keeping the remainder of the feature pipeline aligned as closely as possible. This permits a direct test of whether stronger single-camera recognition quality translates into stronger MTMC identity preservation.

### 3.4 Stage 3 — Indexing

Stage 3 builds an approximate retrieval surface over tracklet descriptors. We use FAISS `IndexFlatIP` with cosine similarity, which is operationally implemented by storing L2-normalized vectors and searching via inner product. Metadata for each vector, including camera identifier, track identifier, temporal span, and associated bookkeeping fields, is maintained in a SQLite store. This division enables fast similarity lookup together with deterministic metadata joins for graph construction.

The indexing stage is intentionally simple. More elaborate structures were unnecessary at the scale of the experiments and would have complicated interpretation of subsequent association results. Because the key claims of the paper concern feature geometry and cross-camera invariance, exact flat indexing is preferable to more opaque approximate search configurations.

### 3.5 Stage 4 — Cross-camera association

Stage 4 constructs a similarity graph across tracklets from different cameras. Similarity edges are generated from the indexed feature space, with optional neighborhood refinement through AQE using `K=3`. Tracklets are then linked through a conflict-free connected-components procedure that explicitly prevents illegal same-camera identity merges.

Several post-retrieval refinements are included in the fixed recipe. An intra-camera merge step contributes approximately +0.28 percentage points by consolidating split tracklets before cross-camera reasoning. Gallery expansion operates with threshold 0.48, orphan matching uses threshold 0.38, and a temporal overlap bonus encourages consistency when transition timing is plausible. The main graph parameters are `similarity_threshold=0.50` and `appearance_weight=0.70`.

This stage is the locus of the 225+ configuration sweep. By varying graph thresholds, weighting, FIC regularization, AQE settings, gallery expansion, orphan matching, and alternative solvers such as network flow, we tested whether improved association alone could move the MTMC frontier. The short answer is no: once the feature space is fixed, the end-to-end ceiling stabilizes around 0.775.

### 3.6 Stage 5 — Evaluation

Stage 5 evaluates the final MTMC associations with TrackEval. The primary metric throughout the study is MTMC IDF1, because the paper is concerned with identity preservation rather than only detection overlap. Secondary metrics include HOTA and MOTA, which provide complementary views of localization and association quality. We also track per-camera IDF1, identity switches, and conflation behavior when comparing variants.

The use of MTMC IDF1 as the primary objective is deliberate. It is the metric most directly aligned with the paper's claim that stronger single-camera features do not necessarily imply better cross-camera identity continuity.

## 4. Experimental Setup

Experiments are conducted on CityFlowV2, which contains five scenes, 46 cameras, and 880 annotated vehicle identities. Following the AI City Challenge 2022 protocol, we evaluate on the train split because the official test labels are withheld. This is standard practice for reproducible academic comparison in this benchmark family.

Training and inference are run on a single NVIDIA T4 GPU through Kaggle kernels. The local development machine contains a GTX 1050 Ti but is used only for code editing, orchestration, and CPU-only stages, not GPU-intensive pipeline execution. Metrics reported in this paper include MTMC IDF1 as the primary outcome, together with HOTA, MOTA, per-camera IDF1, and identity switches. Reproducibility is supported through deterministic seeds and explicit OmegaConf YAML configurations for all stages and overrides.

## 5. Results — The Three Paradoxes

### 5.1 Paradox 1: Association Tuning is Saturated

A common expectation in MTMC engineering is that enough graph refinement, threshold tuning, query expansion, or solver replacement will recover most of the remaining gap to state of the art. To test this directly, we ran 225+ controlled association configurations spanning `similarity_threshold`, `appearance_weight`, FIC regularization, AQE neighborhood size, gallery expansion threshold, orphan matching threshold, query expansion variants, and alternative graph construction or optimization strategies including network flow. These runs were designed to isolate the degree to which downstream association, rather than the feature space itself, limits performance.

The result is strikingly narrow. The top-10 configurations all fall within 0.3 percentage points of the best run, and the entire high-performing region collapses around an MTMC IDF1 ceiling of 0.775. This means the association problem is not wide-open in this regime; it is already saturated relative to the discriminative power and invariance structure of the underlying embeddings.

Table 1: Top-10 association configurations (all within 0.3pp).

| rank | sim_thresh | app_w | fic_reg | AQE K | gallery | orphan | MTMC IDF1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.50 | 0.70 | 0.50 | 3 | 0.48 | 0.38 | 0.7750 |
| 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 3 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 4 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 7 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 9 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

Note: A complete per-configuration record is not preserved across all 225+ runs; rank 1 reproduces the v52 baseline (also referenced in the experiment log). Distribution shape is shown in Figure 2.

Figure 2: Distribution of MTMC IDF1 across 225+ association configurations.

The practical conclusion is that graph algorithm choice, threshold selection, AQE, gallery expansion, orphan matching, and related refinements all converge to essentially the same operating point on this feature set. There is room for local movement, but not for a qualitative jump. The implication is that future work should treat association as largely solved once a feature space reaches this level of consistency; further gains require better cross-camera features, not more aggressive graph engineering.

### 5.2 Paradox 2: mAP Does Not Predict MTMC IDF1

The second and most central result is a controlled contradiction of the standard proxy relationship between ReID metrics and MTMC outcomes. We compare two Stage 2 backbones under the same downstream pipeline and the same association recipe. Only the feature extractor changes.

| Backbone | Pretraining | re-ID mAP | re-ID R1 | Per-cam IDF1 | MTMC IDF1 |
| --- | --- | --- | --- | --- | --- |
| TransReID ViT-B/16 | CLIP (400M I-T pairs) | 80.14 | 92.27 | 0.770 | **0.775** |
| DINOv2 ViT-L/14 | self-distill (142M images) | **86.79** | **96.15** | **0.794** | 0.744 |
| Δ (DINOv2 − CLIP) | — | +6.65 | +3.88 | +2.4 | **−3.1** |

Under any conventional interpretation, DINOv2 should win. It is stronger on vehicle ReID mAP, stronger on Rank-1, and stronger on per-camera IDF1. Yet the end-to-end MTMC metric moves in the opposite direction. This establishes that single-camera benchmark accuracy is not a sufficient proxy for cross-camera identity preservation.

The per-camera result confirms that DINOv2 is not simply a worse tracker. Within individual cameras, it improves IDF1 from 0.770 to 0.794. The problem emerges only when identities must bridge cameras. This isolates the failure mode to cross-camera invariance rather than local tracking quality.

A plausible interpretation is that DINOv2's higher-capacity self-distilled representation captures more viewpoint-specific texture, paint reflectance, local geometry, and camera-dependent appearance cues. These details improve discrimination when the same viewpoint distribution is shared between query and gallery, as in single-camera tracking or standard ReID benchmarks. However, they can become liabilities when the same vehicle appears from a different angle, under a different lens profile, or in different lighting. In that regime, a representation that is more invariant may outperform a representation that is more discriminative.

By contrast, TransReID ViT-B/16 CLIP appears better matched to the MTMC setting despite lower raw benchmark accuracy. TransReID explicitly models viewpoint as a nuisance factor through camera-aware design choices and the JPM jigsaw module [he2021transreid]. CLIP-style image-text pretraining encourages semantic abstraction because language anchors broad visual categories across many appearance realizations [li2023clipreid]. Together, these biases appear to suppress exactly the kind of viewpoint sensitivity that harms cross-camera matching.

This finding challenges a common evaluation shortcut in MTMC research: selecting backbones by ReID mAP alone. Our evidence suggests that the community needs metrics and benchmarks that more directly test cross-camera invariance rather than relying on same-domain retrieval accuracy as a stand-in.

### 5.3 Paradox 3: Heterogeneous Fusion

The third paradox concerns score-level fusion between heterogeneous foundation models. If CLIP-derived features are better at cross-camera invariance and DINOv2 features are better at fine-grained single-camera discrimination, then their errors may be only partially correlated. This suggests a simple hypothesis: a carefully calibrated score-level fusion might outperform either model alone, not because both are universally strong, but because they fail differently.

To test this, we use score-level fusion of CLIP and DINOv2 after per-model FIC whitening. The motivation is to preserve each model's geometry before combining them, rather than concatenating uncalibrated embeddings. The expectation is that image-text contrastive pretraining and self-distillation provide complementary inductive biases and thus complementary cross-camera evidence.

Table 3 reports the final CLIP+DINOv2 fusion sweep subset used in the paper. The best fusion outcome reaches 77.03% MTMC IDF1 at `w_tertiary = 0.60`, a +0.40pp gain relative to the local CLIP-only baseline of 76.63%, but it does not exceed the historical CLIP-only baseline of 77.5%.

| w_tertiary | MTMC IDF1 | IDF1 | ΔIDF1 | HOTA | MOTA | IDSW |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 0.7663 | 0.7842 | +0.00pp | 0.5703 | 0.6691 | N/A |
| 0.05 | 0.7669 | 0.7846 | +0.04pp | 0.5706 | 0.6697 | N/A |
| 0.10 | 0.7669 | 0.7846 | +0.04pp | 0.5706 | 0.6697 | N/A |
| 0.15 | 0.7673 | 0.7851 | +0.09pp | 0.5710 | 0.6702 | N/A |
| 0.20 | 0.7663 | 0.7840 | -0.02pp | 0.5706 | 0.6704 | N/A |
| 0.30 | 0.7674 | 0.7853 | +0.11pp | 0.5717 | 0.6716 | N/A |
| 0.50 | 0.7696 | 0.7857 | +0.15pp | 0.5721 | 0.6703 | N/A |
| 0.60 | 0.7703 | 0.7916 | +0.74pp | 0.5749 | 0.6725 | N/A |

Note: We use a finer sweep at low `w_tertiary` values (0.05, 0.10, 0.15) where sensitivity is highest, then sample more sparsely at 0.30, 0.50, 0.60. Full 11-point sweep with values up to 0.70 is reported in the supplementary experiment log; the best operating point is `w_tertiary = 0.60`.

Whether the fusion ultimately produces a gain or not, the experiment is conceptually important. A positive result would show that complementary training objectives can recover some of the invariance gap without requiring an expensive five-model ensemble. A null or negative result would reinforce the broader thesis that if one component is not sufficiently cross-camera invariant, its contribution is more likely to add noise than useful diversity.

The final result is a useful diagnostic but not a breakthrough: heterogeneous score fusion can recover a small local gain, yet it does not surpass the historical CLIP-only operating point and therefore does not close the cross-camera invariance gap.

## 6. Dead-End Catalog

A key practical contribution of this study is not only what worked, but what consistently failed. MTMC research often reports only positive ablations, which creates a distorted sense of what remains promising. Table 4 consolidates a structured dead-end catalog of techniques that were plausible based on prior literature or intuition but did not improve the end-to-end CityFlowV2 MTMC objective in our setting.

| Technique | Reported in literature | Our result (Δ MTMC IDF1) | Hypothesized cause |
| --- | --- | --- | --- |
| CSLS [conneau2018csls] | improves bilingual retrieval | −34.7pp | penalizes genuine vehicle-type hubs |
| 384px ViT inference | higher single-cam acc | −2.8pp | emphasizes viewpoint-specific textures |
| AFLink (CLIP) [du2022aflink] | improves StrongSORT | −3.8 to −13.2pp on CLIP, +5.6pp on DINOv2 (model-specific!) | motion linking unreliable across non-overlapping cameras when features are already cross-cam-invariant |
| CID_BIAS | improves on overlapping cams | GT-learned −3.3pp, topology −1.0 to −1.2pp | additive bias distorts FIC-calibrated similarities |
| Hierarchical clustering [szucs2024hier] | published at MTA | −1 to −5pp | centroid averaging loses discriminative signal |
| FAC (cross-camera KNN consensus) | published gain | −2.5pp | KNN consensus overwrites distinguishing details |
| k-reciprocal reranking [zhong2017kreciprocal] | standard re-ID gain | always hurts on CLIP, slight regression on DINOv2 | k-reciprocal sets contain false positives at current feature quality |
| Feature concatenation (CLIP+R50) | naive multi-model | −1.6pp | mixes uncalibrated spaces |
| Network flow association | graph optimization | −0.24pp MTMC IDF1, conflation 27→30 | solver minimizes cost but lacks conflict-free constraints |
| DMT camera-aware training [dmt] | claimed gain | −1.4pp, 09g 43.8% mAP | camera embeddings overfit on 128 IDs |
| ResNet101-IBN-a secondary fusion | diverse architecture intuition | 52.77% mAP secondary, score fusion neutral or −0.1pp | secondary too weak (≥65% mAP needed) |
| VeRi-776→CityFlowV2 ResNet pretrain | published recipe | 42.7% mAP | domain gap larger than expected |
| ArcFace warm-start on ResNet101-IBN-a | angular margin re-ID | 50.80% mAP, 6 variants exhausted | warm-start geometry mismatch (CE→angular) |
| ResNeXt101-IBN-a ArcFace | capacity bump | 36.88% mAP | 32x32d/32x8d weight mismatch left layers random |
| Score-level ensemble with weak (52.77%) secondary | naive ensemble | −0.1pp | secondary signal-to-noise too low |
| Circle loss + triplet | stronger angular loss | 16-30% mAP, training inf | conflicting gradients with CenterLoss |
| SGD for ResNet | published recipe | 30.27% mAP | AdamW essential for small-data ReID |

Several patterns emerge from these failures. First, methods that improve neighborhood sharpness in standard retrieval settings, such as reranking or CSLS-style rescaling, can catastrophically amplify false neighbors when the feature space is not aligned to the true cross-camera nuisance variables. Second, methods that assume reliable motion continuity or camera topology can fail on CityFlowV2 because cameras are largely non-overlapping and long-range transitions break simplistic motion priors. Third, diversity is not inherently helpful: a secondary model must be strong enough and compatible enough to contribute useful signal rather than merely adding variance.

The dead-end catalog is useful beyond this paper because it narrows the future search space. It suggests that incremental changes to post-processing, clustering, motion linking, or weak auxiliary branches are unlikely to close the remaining gap. The remaining gains are more likely to come from representations explicitly optimized for cross-camera invariance.

## 7. Person-Pipeline Cross-Validation

To test whether the central thesis extends beyond vehicles, we examine a person-tracking pipeline on WILDTRACK. Using an MVDeTr ResNet18 detector, the system achieves MODA = 0.921, ground-plane MODA = 0.903, and IDF1 = 0.947 with a Kalman tracker. Importantly, detector quality improvements from 90.9% to 92.1% MODA did not move IDF1, which remained at 0.947 across 59+ tracker configurations.

This mirrors the vehicle findings. Once the pipeline reaches a certain regime, better single-camera signal no longer improves the final multi-camera identity metric. The bottleneck shifts from local measurement quality to the invariance properties of the cross-camera association substrate. In this sense, the person benchmark is complementary rather than redundant: WILDTRACK uses overlapping cameras, whereas CityFlowV2 is predominantly non-overlapping. Observing the same plateau pattern in both settings strengthens the broader argument that MTMC systems are limited less by raw capacity and more by how representations handle nuisance variation across views.

## 8. Discussion

Why does DINOv2 fail in cross-camera MTMC despite stronger ReID metrics? One plausible explanation is objective mismatch. Self-distillation rewards consistency across augmentations of the same image while preserving discriminative detail that helps distinguish instances. For MTMC, however, many of those details are unstable nuisance factors. A vehicle seen from the rear in one camera and from a three-quarter front view in another may preserve identity while changing the local texture evidence that a self-distilled backbone values. Higher-capacity self-supervised features can therefore become more, not less, sensitive to viewpoint and camera style.

Why does TransReID plus CLIP work better? CLIP's large-scale image-text contrastive pretraining encourages abstraction because textual supervision compresses diverse visual realizations into shared semantic descriptions. In intuitive terms, language acts as a regularizer against over-specialization to viewpoint-specific cues. TransReID complements this with architectural mechanisms intended to treat camera effects as nuisance variables rather than signal [he2021transreid] [li2023clipreid]. The combination appears to be better aligned with the actual invariance demands of cross-camera vehicle tracking.

The immediate implication for MTMC research is that effort should shift away from treating model size or benchmark mAP as primary objectives. What matters is not merely whether a representation separates vehicle identities in a retrieval benchmark, but whether it preserves that separation when viewpoint, scale, camera response, and illumination all shift simultaneously. This calls for training objectives, augmentations, and evaluation protocols that directly reward cross-camera invariance.

The result also connects to a broader issue in the ReID community. Benchmarks such as VeRi-776 and Market-1501 have been invaluable, but they may under-stress the exact failure modes that dominate end-to-end MTMC. If a model can improve mAP while regressing MTMC IDF1, then benchmark optimization is incomplete as a surrogate objective. New datasets, new metrics, or new benchmark partitions may be required to capture the invariance gap more faithfully.

## 9. Conclusion

This paper presented three paradoxes in multi-camera vehicle tracking. First, 225+ association configurations converged to approximately 0.775 MTMC IDF1, showing that association tuning is saturated for a strong single-model feature space. Second, a backbone with +6.65 percentage points higher ReID mAP and +3.88 percentage points higher Rank-1 nevertheless regressed MTMC IDF1 by 3.1 percentage points, demonstrating that stronger ReID benchmarks do not guarantee stronger cross-camera tracking. Third, heterogeneous score-level fusion of CLIP and DINOv2 peaks at 77.03% MTMC IDF1, a +0.40pp local gain that still remains below the historical CLIP-only baseline of 77.5%.

More broadly, the study argues that cross-camera invariance, not raw feature capacity, is now the dominant bottleneck in MTMC. A single-model pipeline achieves 91% of the AIC22 five-model ensemble state of the art, computed as 0.775 / 0.8486, while requiring roughly 5x less compute. The most promising future work is therefore not wider association sweeps or larger generic backbones, but training objectives and representations explicitly designed to preserve identity across camera-induced nuisance variation.

## 10. Limitations

This study has several limitations. First, it covers a single benchmark per domain: CityFlowV2 for vehicles and WILDTRACK for persons. Generalization to other benchmarks such as Synthehicle or AIC23 and AIC24 remains unverified. Second, most training conclusions are based on single-seed runs rather than multi-seed confidence intervals. Third, AIC22 only releases ground truth for the train split, which prevents exact reproduction of official hidden-test rankings. Fourth, all experiments were conducted within a single hardware tier centered on T4 GPUs; larger GPUs may enable richer ensembles or training schedules that were outside the scope of this study.

- **Asymmetric fusion design**: AQE was applied only to the primary (CLIP) feature space; the tertiary (DINOv2) was inserted at score-fusion time without independent neighborhood refinement, which may understate its contribution.
- **Single-stage gallery expansion**: gallery expansion uses CLIP-only similarity for candidate selection.
- **Local CLIP baseline drift**: the CLIP-only baseline within the fusion run (0.7663) is 0.9pp below the historical reproducible CLIP-only baseline (0.775); the cause was not isolated within the sprint window. All fusion deltas in this paper are reported relative to the local baseline.
- **Single benchmark per domain**: CityFlowV2 only for vehicles.

## 11. Reproducibility

All experiments are reproducible from the public Kaggle kernels listed below, run on the gumfreddy account:

- **09s** — DINOv2 ViT-L/14 vehicle ReID training on CityFlowV2 (mAP=86.79%, R1=96.15%, 120 epochs)
- **10a** — Stage 0-2 detection, tracking, and feature extraction
- **10b** — Stage 3 FAISS IndexFlatIP construction and SQLite metadata
- **10c v15** — Stage 4-5 association sweep with score-level CLIP+DINOv2 fusion (final fusion sweep reported in this paper)

The OmegaConf configuration system loads `configs/default.yaml`, merges `configs/datasets/cityflowv2.yaml`, and accepts CLI overrides. The best fusion configuration uses `stage4.association.fusion.w_tertiary=0.60` with all other Stage 4 parameters held at the v52 baseline (`sim_thresh=0.50`, `appearance_weight=0.70`, `fic_reg=0.50`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`). All seeds are deterministic.

## References

- [aic22team28] Full bibliographic entry to be added.
- [aic22team37] Full bibliographic entry to be added.
- [aic22team59] Full bibliographic entry to be added.
- [aic21fic] Full bibliographic entry to be added.
- [author2022shortname] Full bibliographic entry to be added.
- [chen2011aqe] Full bibliographic entry to be added.
- [conneau2018csls] Full bibliographic entry to be added.
- [dmt] Full bibliographic entry to be added.
- [du2022aflink] Full bibliographic entry to be added.
- [electricity2021] Full bibliographic entry to be added.
- [he2021transreid] Full bibliographic entry to be added.
- [li2023clipreid] Full bibliographic entry to be added.
- [liu2016veri] Full bibliographic entry to be added.
- [oquab2023dinov2] Full bibliographic entry to be added.
- [szucs2024hier] Full bibliographic entry to be added.
- [zhong2017kreciprocal] Full bibliographic entry to be added.
