# MTMC Tracker — Research Findings & Strategic Analysis

> **IMPORTANT**: This is a living document. Update it whenever new experiments are run, new dead ends are discovered, or performance numbers change. Keep the "Current Performance" and "Dead Ends" sections current.

## Current Performance (Last Updated: 2026-04-22)

### Vehicle Pipeline (CityFlowV2)

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Current Reproducible Best MTMC IDF1** | **77.5%** | 10c v52, v80-restored association recipe on the current codebase (10a v30 features). Current 10a v5 baseline (camera_bn=true): 76.625% — needs fresh association re-tune. |
| **10c v8 (3-way ensemble sweep — 2026-04-22)** | 71.28% (biased; add ~5pp) | 10a v5 regression fix COMPLETE (929 tracklets, 49.4 min); 10c v8 19-config ensemble sweep COMPLETE but biased by MTMC_ONLY=True bug (~5pp penalty); best: w2=0.05, w3=0.30 → 71.28% biased → ~76.28% est. true. Fixed in commit `69e67a0`; 10c v9 RUNNING for unbiased results. |
| **10c v9 (MTMC_ONLY fix — 2026-04-22)** | MTMC_IDF1=76.625% (baseline), 76.817% (best ensemble) | COMPLETE. Baseline 76.625% is ~0.74pp below expected 77.36% due to feature distribution shift (camera_bn=true). Best 3-way: w2=0.05, w3=0.30 → 76.817% (+0.192pp). Ensemble marginal; dead end confirmed. |
| **Historical Best MTMC IDF1** | **78.4%** | v80, but not reproducible with the current codebase (~1pp drift) |
| **SOTA Target** | 84.86% | AIC22 1st place |
| **Gap to SOTA** | 7.36pp | Relative to the current reproducible best |
| **Primary Model (ViT-B/16 CLIP 256px, 09 v2 aug overhaul)** | mAP=81.59% | On CityFlowV2 eval split; +1.45pp vs the prior 80.14% baseline |
| **Experiment B (CircleLoss ablation, 09 v4)** | mAP=18.45%, R1=48.84% | Catastrophic failure with baseline augmentations; training loss was `inf` at every epoch |
| **LAION-2B CLIP CircleLoss run (09l v1)** | mAP=20.36%, R1=53.03%, mAP_rr=27.16% | Same catastrophic fp16 overflow failure as 09 v4; tells us nothing about LAION-2B backbone quality |
| **LAION-2B CLIP extended Triplet run (09l v3)** | mAP=78.61%, R1=90.43%, mAP_rr=81.09%, R1_rr=90.98% | Strong standalone alternative, but not a viable fusion secondary after 10c v56 showed CLIP ViT score fusion hurts |
| **10c v48 (09 v2 augoverhaul @ 256px)** | MTMC IDF1=0.722 | Best result after an 11-sweep association re-optimization; single-cam IDF1 only 0.752 |
| **10c v49 (09 v3 augoverhaul-EMA @ 256px)** | MTMC IDF1=0.722 | Best result after a broader association sweep; AFLink recovered 0.675 -> 0.722 but could not break the augoverhaul ceiling |
| **Secondary Model (ResNet101-IBN-a)** | mAP=52.77% | On CityFlowV2 eval split, ImageNet→CityFlowV2 only |
| **Secondary Model ResNeXt101-IBN-a ArcFace (09j v2)** | mAP=36.88%, R1=62.69% | Catastrophic failure after partial/mismatched pretrained weight loading left large parts of the backbone randomly initialized |
| **Secondary Model ViT-Small/16 IN-21k (09k v1)** | mAP=48.66%, R1=62.01% | Confirms the non-CLIP ceiling extends to ViT architectures too |
| **Secondary Model EVA02 ViT-B/16 CLIP (09o v1)** | mAP=48.17%, R1=65.90% | Much weaker than the primary ViT-B/16 CLIP baseline and even the fine-tuned R50-IBN secondary; current recipe does not transfer well to vehicle ReID |
| **Secondary Model CLIP RN50x4 CNN (09m v2)** | mAP=1.55%, R1=4.18% | Catastrophic failure; CE loss converged but retrieval features were unusable, closing out the CNN-based CLIP secondary path |
| **Secondary Model VeRi-776 pretrain (09e v2)** | mAP=62.52% | On VeRi-776 test set, ready for CityFlowV2 fine-tuning |
| **384px ViT (09b v2)** | DEAD END | Higher single-camera ReID accuracy did not transfer; MTMC IDF1 only 0.7585-0.7562 in v43-v44, -2.8pp vs 256px baseline |
| **09f CityFlowV2 fine-tune** | mAP=42.7% | v3 peaked at epoch 104/120 and still underperformed direct ImageNet→CityFlowV2 (09d v18: 52.77%) |
| **Association configs tested** | 225+ | All within 0.3pp of optimal |

**Current reproducible vehicle MTMC IDF1 is 0.775** with 10c v52 using the v80-restored recipe (`sim_thresh=0.50`, `appearance_weight=0.70`, `fic_reg=0.50`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`). This is a small improvement over v50/v51 at 0.772, but it still trails the historical v80 result of 0.784. Vehicle association remains exhausted, so future gains will need materially better features or priors rather than more stage-4 tuning.

The latest structural association follow-up, **10c v53 network flow solver**, confirms that conclusion. Against the same controlled **10c v52** baseline, network flow reached only **MTMC IDF1 = 0.769** versus **0.7714** for the CC baseline (**-0.24pp**). It slightly improved **MOTA** (**0.689 -> 0.694**) and **HOTA** (**0.5747 -> 0.577**) with **2 fewer ID switches** (**199 -> 197**), but it **increased conflation from 27 to 30 predicted IDs** instead of reducing it. The current **conflict_free_cc** pipeline remains the preferred solver for this problem.

The corrected **10c v48** evaluation of the **09 v2 augoverhaul** model at the intended **256px** resolution still regressed sharply: the best full sweep reached only **MTMC IDF1 = 0.722** with `sim_thresh=0.45`, `appearance_weight=0.60`, `fic_reg=1.00`, `aqe_k=3`, `aflink_gap=150`, and `aflink_dir=0.85`. The follow-up **10c v49** sweep on the **09 v3 augoverhaul-EMA** training run reproduced the same **0.722** ceiling with a broader parameter search, confirming that the regression persists across augoverhaul variants and is driven by the model family rather than missed association tuning. Single-camera **IDF1 = 0.752** in **10c v48** was also materially below the baseline (~0.82), confirming that the earlier **10c v47** collapse was partly a deployment bug, but the underlying augoverhaul recipe is itself a vehicle-MTMC regression.

The new **Experiment B** CircleLoss-only ablation on the primary vehicle ReID path failed catastrophically. Kernel **`gumfreddy/09-vehicle-reid-cityflowv2-circleloss-ablation` v1** used the original baseline augmentation stack with **CE+LS(eps=0.05) + CircleLoss(m=0.25, gamma=128) + CenterLoss** for **120 epochs**, but the training loss was **`inf` at every epoch** and the run collapsed to only **mAP = 18.45%** and **R1 = 48.84%**. This independently confirms that **CircleLoss is a dead end** on this CityFlowV2 TransReID recipe. It also sharpens the interpretation of the augoverhaul regression: either the augoverhaul augmentations themselves caused the **81.59% mAP -> 0.722 MTMC IDF1** failure, or **CircleLoss was not actually active** in that training run due to a config/path mismatch. In either case, **CircleLoss is not a viable explanation for a healthy high-mAP regime** because when it is definitely active, it destroys training entirely.

That conclusion is now reinforced by the full **09l** sequence. The original **09l v1** LAION-2B CLIP attempt reused the broken **Experiment B** recipe, kept the loss at **`inf` throughout all 120 epochs**, and collapsed to only **mAP = 20.36%**, **R1 = 53.03%**, and **mAP_rr = 27.16%**. The follow-up **09l v2** rerun replaced **CircleLoss** with **TripletLoss**, re-enabled **EMA** with **decay=0.9999**, and trained for **160 epochs**, recovering to **mAP = 61.51%**, **R1 = 81.41%**, and **mAP_rr = 67.20%**. The final **09l v3** continuation then resumed from the **v2 EMA checkpoint** and extended training to **300 total epochs**, reaching **mAP = 78.61%**, **R1 = 90.43%**, **mAP_rr = 81.09%**, and **R1_rr = 90.98%**. This closes the loop: the earlier collapse was a **recipe instability**, not a backbone failure.

The **09l v3** continuation confirmed that **v2 was schedule-limited, not architecture-limited**. The resumed training phase kept improving across **epoch 180/200/220/240/260/280/300 = 65.93/68.84/71.49/73.68/75.68/77.26/78.61 mAP**, and the finished model sits only **1.53pp** behind the deployed **OpenAI CLIP 09b v2** baseline at **80.14% mAP**. But the follow-up **10c v56** score-fusion test still regressed, showing that a strong secondary alone is not enough when the feature families are too correlated.

The new **09m v2 CLIP RN50x4 CNN** experiment closes the other obvious escape hatch. It reached only **mAP = 1.55%** and **R1 = 4.18%** despite the cross-entropy loss converging from **6.57 -> 0.99** over **200 epochs**, which means optimization did not fail in the usual sense but the learned features were still useless for retrieval. The most likely root causes are **(1)** a **QuickGELU mismatch** between `open_clip` model construction (`quick_gelu=False`) and the OpenAI pretrained weights (`quick_gelu=True`), which corrupts the pretrained feature geometry, **(2)** the **CNN attention-pooling CLIP architecture** not fitting the standard ReID projection-head recipe used elsewhere in the codebase, and **(3)** **640D CNN features** being fundamentally harder to adapt for fine-grained vehicle ReID than the **768D ViT** features. Taken together with the failed **63.64% R50-IBN**, **52.77% ResNet101-IBN-a**, **48.17% EVA02 ViT-B/16 CLIP**, **48.66% ViT-Small**, **36.88% ResNeXt**, and **10c v56** correlated-CLIP fusion regression, the **score-level ensemble path is now fully exhausted**: there is **no viable secondary model** on the current codebase.

**⚠️ Metric Disambiguation (Vehicle Pipeline):**
- **MTMC IDF1 = 77.5%** — Current reproducible best on the current codebase from 10c v52. This remains the official metric and the only number that should be compared to AIC22 SOTA.
- **Historical v80 reference** — MTMC IDF1 = 78.4%, IDF1 = 79.8%, GLOBAL IDF1 = 80.5%. These numbers are useful for historical comparison, but they are not currently reproducible.
- All current best numbers use GT-assisted metrics (`gt_frame_clip=true`, `gt_zone_filter=true`) which inflate scores by 1-3pp vs clean evaluation.

### Person Pipeline (WILDTRACK) — NEW

| Metric | Value | Notes |
|--------|:-----:|-------|
| **Ground-plane MODA** | **90.0%** | Best converged operating point reproduced across the final 12b tracking runs |
| **Ground-plane IDF1** | **94.7%** | Confirmed independently by 12b v1, v2, and v3; Precision=94.5%, Recall=96.1%, IDSW=5 |
| **Best Detector (MVDeTr 12a v3)** | MODA=92.1% | Epoch 20/25, Precision=95.7%, Recall=96.6%; best WILDTRACK detector yet |
| **12b v2 extended Kalman sweep** | MODA=90.0%, IDF1=94.7% | Wider interpolation/max_age/conf sweeps plus velocity-aware quadratic interpolation still converged to the same 5-IDSW operating point |
| **12b v3 global optimal tracker** | MODA=88.2%, IDF1=91.2% | Sliding-window Hungarian assignment underperformed badly; immediate frame-level costs overrode the motion-prediction advantage of Kalman and produced 15 ID switches |
| **12b v1 on 12a v3 detections** | MODA=90.0%, IDF1=94.7% | Better detections did not improve tracking; Kalman sweep converged to the same effective operating point |
| **Previous Best Tracking Baseline** | IDF1=92.8% | 12b v9 naive tracker; new tuned Kalman run is +1.9pp |
| **ReID Features (12b v8)** | mean cosine=0.720 | Fixed ViT backbone mismatch; range widened to [0.215, 1.000] |
| **ReID Merge State** | No gain over baseline | Tuned Kalman tracks are already clean; merge sweep did not improve IDF1 |
| **Gap to SOTA** | 0.6pp | Best IDF1=94.7% vs target 95.3% |
| **Status** | FULLY CONVERGED | Kalman, naive, and global-optimal trackers tested across 59 configs; remaining path is better person appearance features or graph-based approaches |

**Current person best remains 94.7% IDF1 and is now fully converged**: 12b v14 first reached **IDF1=94.7%** with a tuned Kalman tracker, and that operating point has now been independently matched by **12b v1**, **12b v2**, and **12b v3**-era validation runs around the same final configuration. The stronger **12a v3** detector did not improve tracking, broader Kalman sweeps across interpolation/max_age/confidence thresholds did not improve tracking, and the new sliding-window **GlobalOptimalTracker** regressed sharply to **IDF1=91.17%**, **MODA=88.2%**, and **15 ID switches**. Across **59 tracker configs** spanning **Kalman**, **naive**, and **global optimal** trackers, the best Kalman solutions all clustered at **IDF1=0.9467 ± 0.0004**, with confidence thresholds from **0.15-0.35** making no meaningful difference. The root cause is that global optimal assignment over-commits to immediate frame-level costs, while BoT-SORT's predictive Kalman model handles occlusions and missed detections better over time. The current WILDTRACK person pipeline is therefore **fully tracker-converged at 94.7% IDF1**. The remaining **0.6pp** gap to SOTA is most likely due to rare occlusion and re-entry failures that need better person appearance features or graph-based multi-view association, not more tracker tuning.

## Gap Decomposition

The remaining gap to SOTA now decomposes into:

| Deficiency | Impact | Status |
|------------|:------:|--------|
| Lack of a viable complementary secondary model | Largest remaining gap | **FULLY EXHAUSTED** for now: weak alternative secondaries all failed, including **09o v1 EVA02 at 48.17% mAP** and **09m v2** at **1.55% mAP**, and even strong **09l v3** CLIP-ViT fusion regressed in **10c v56** |
| Camera-aware single-model training (DMT) | -1.4pp | Tested and harmful (v46) |
| Multi-query track representation | -0.1pp | Tested and neutral/harmful (v51) |
| Higher-dimensional concat-patch features | -0.3pp | Tested and harmful (v48-v49) |
| Association tuning / structural association changes | Exhausted | 225+ configs plus structural variants already tried |

**Higher single-camera ReID mAP is not translating into better MTMC IDF1.** Multiple experiments now confirm this: the augmentation-overhaul plus CircleLoss recipe, 384px deployment, DMT camera-aware training, and both tested score-level fusion paths all improved or preserved some aspect of single-model quality while hurting downstream MTMC. With the ensemble/secondary-model rescue path now exhausted on the current codebase, the realistic ceiling still appears to be roughly **77-78% MTMC IDF1** unless the project adds a materially new association model or finds another genuine jump in primary-feature quality.

## Critical Discovery: 384px Is a Dead End for MTMC

The earlier checkpoint mismatch was real, but it is no longer the blocker. The correct 09b v2 384px checkpoint was evaluated and still lost decisively to the 256px baseline in MTMC.

**Definitive result (2026-03-30)**:
- v43 (384px, tuned thresholds, min_hits=3): MTMC IDF1 = 0.7585
- v44 (384px, exact v43 config but min_hits=2): MTMC IDF1 = 0.7562
- v80 baseline (256px, min_hits=2): MTMC IDF1 = 0.7840

Higher input resolution improved or preserved single-camera discriminative detail, but it made cross-camera association worse by emphasizing viewpoint-specific textures that do not transfer well across cameras.

## Critical Discovery: ResNet101-IBN-a 52.77% Is Expected

The ViT achieves high mAP because of 3-stage progressive specialization:
- CLIP (400M image-text pairs) → VeRi-776 (576 IDs, 37K images) → CityFlowV2 (128 IDs, 7.5K images)

The ResNet skips the critical VeRi-776 middle step:
- ImageNet (1.3M generic) → CityFlowV2 (128 IDs, 7.5K images) directly

Published 75-80% mAP baselines for ResNet101-IBN-a are evaluated on **VeRi-776** (576 IDs), NOT on CityFlowV2 (128 IDs). These numbers were never comparable. The 52.77% is reasonable given the massive pretraining disadvantage.

**Fix**: Train ResNet101-IBN-a on VeRi-776 FIRST, then fine-tune on CityFlowV2 (same pattern as the ViT).

**Status update (2026-03-29)**: VeRi-776 pretraining completed in 09e with mAP=62.52% on the VeRi-776 test set, but the follow-up CityFlowV2 fine-tune in 09f v3 only reached mAP=42.7%. This underperforms the direct ImageNet→CityFlowV2 baseline (09d v18: 52.77%), so VeRi-776 pretraining appears to hurt rather than help for this ResNet101-IBN-a path.

**Status update (2026-04-01)**: Extending the direct ImageNet→CityFlowV2 run by resuming from the **09d v18** 52.77% checkpoint at a lower learning rate (**3e-4**) peaked at only **50.61% mAP** in **09d gumfreddy v3**. This confirms that **52.77% is effectively the ceiling** for the current ImageNet→CityFlowV2 ResNet101-IBN-a recipe rather than an undertrained checkpoint.

**Status update (2026-04-17)**: The new **09i v1 ArcFace** follow-up on **gumfreddy** also failed to break that ceiling. With **ArcFace (s=30, m=0.35) + Triplet (m=0.3) + Center loss** and a **warm-start from 09d**, the run peaked at only **50.80% mAP**, **73.46% R1**, and **54.65% mAP_rerank** at **epoch 100/160**, then declined through the rest of training. The most likely root cause is **geometry mismatch**: the checkpoint was warm-started from a **cross-entropy-optimized** solution, then forced into an **ArcFace angular-margin** regime while also keeping triplet and center objectives active. On a dataset with only **128 train IDs**, that creates **four competing losses/geometries** and pushes the model into overfitting rather than better discrimination.

**Status update (2026-04-17)**: The new **09j v2 ResNeXt101-IBN-a ArcFace** run failed catastrophically at only **36.88% mAP**, **62.69% R1**, and **40.49% mAP_rerank** after **160 epochs**. The root cause is not just poor optimization: the original **IBN-Net ResNeXt** checkpoint path was incompatible because the published weights use **32x32d grouped convolutions** while the training model here was instantiated as **32x8d**. The v2 workaround filtered state-dict loading with `strict=False`, which avoided the shape-mismatch crash but left many layers unmatched and therefore randomly initialized. That crippled the run from the start and makes the current **ResNeXt101-IBN-a** path a dead end unless a truly compatible pretrained checkpoint is found.

**Status update (2026-04-17)**: The new **09k v1 ViT-Small/16** run reached only **48.66% mAP** and **62.01% R1** after **120 epochs** despite using a ViT backbone. Taken together with **09d/09i** on ResNet101-IBN-a and **09j v2** on ResNeXt101-IBN-a, this confirmed that the secondary-model ceiling is a **non-CLIP pretraining and initialization problem across both CNN and ViT families**, not just an architecture-selection problem.

**Status update (2026-04-19)**: The new **09m v2 CLIP RN50x4 CNN** follow-up also failed catastrophically at only **1.55% mAP** and **4.18% R1** even though the **cross-entropy loss converged from 6.57 -> 0.99** over **200 epochs**. This is a worse failure mode than the non-CLIP baselines because it indicates the training loop can optimize classification while still destroying retrieval geometry. The most likely causes are a **QuickGELU mismatch** in the loaded CLIP CNN weights, poor compatibility between the **attention-pooling CNN backbone** and the standard ReID projection-head recipe, and an inherently weaker adaptation path from **640D CNN features** than from **768D ViT features** for fine-grained vehicle identity. In practice, this means the attempted **alternative CLIP-family CNN** rescue also failed. Combined with **10c v56** showing that the strong **09l v3** CLIP-ViT secondary is too correlated with the primary to help in fusion, the **secondary-model / score-level ensemble path is now fully exhausted** on the current codebase.

## What SOTA Does Differently

| Pattern | AIC22 1st | AIC22 2nd | AIC21 1st | We have? |
|---------|:-:|:-:|:-:|:-:|
| 3+ ReID backbone ensemble | 5 models | 3 models | 3 models | **NO** (1 strong primary, no viable complementary secondary) |
| 384×384 input | ✅ | ✅ | ✅ | **Tested, but harmful in our pipeline** |
| IBN-a backbones | ✅ | ✅ | ✅ | ViT only |
| Camera-pair bias (CID_BIAS) | ROI masks | NPY | NPY | **Tested twice and harmful: GT-learned -3.3pp, topology bias -1.0 to -1.2pp (DEAD END)** |
| Reranking | Box-grained | k-reciprocal | k-reciprocal | **Disabled** |
| Camera-aware training (DMT) | ✅ | ✅ | ✅ | **NO** |
| Multiple loss functions | ID+tri+circle+cam | ID+tri+cam | ID+tri+cam | ID+tri |

### AIC Winning Methods Analysis and Ensemble Implementation

- **AIC22 1st place (IDF1=0.8486)** used a **5-model ReID ensemble** plus **Box-Grained Matching**.
- **AIC22 2nd place (IDF1=0.8437)** used a **3-model ensemble** (**ResNet101-IBN-a x2 + ResNeXt101-IBN-a**) with **DMT** and **CID_BIAS**.
- **Universal pattern**: every top AIC method relies on a **3-5 model ensemble**. The consistent win is not a single stronger backbone, but diversity across multiple ReID models.
- **Implication for our results**: our single-model upgrades such as **384px**, **DMT**, and **reranking** fail in isolation because they are ensemble-dependent techniques. They can remove noise, but with only one model they also remove too much discriminative signal.

**Ensemble implementation status**

- **09g**: ResNet101-IBN-a DMT training is currently running on **gumfreddy** with **150 epochs**, a **camera-adversarial head**, and **no circle loss**.
- **09j v2**: ResNeXt101-IBN-a ArcFace finished at only **36.88% mAP** after the checkpoint compatibility workaround still left large parts of the backbone randomly initialized. This path is now a **dead end** rather than a live ensemble candidate.
- **09l v3**: LAION-2B CLIP resumed from the **v2 EMA checkpoint** to **78.61% mAP / 90.43% R1 / 81.09% mAP_rr** at **300 total epochs**. This is only **1.53pp** behind the deployed **OpenAI CLIP 09b v2** baseline, but **10c v56** showed that fusing the two CLIP ViT-B/16 models still hurts because they are too correlated.
- **09m v2**: CLIP **RN50x4** CNN training failed catastrophically at **1.55% mAP / 4.18% R1** even though the CE loss converged from **6.57 -> 0.99**. The likely causes are the **QuickGELU mismatch**, weak compatibility between the **attention-pooling CNN** backbone and the standard ReID head recipe, and poorer transfer from **640D CNN** features than from **768D ViT** features.
- **Infrastructure**: **Stage 2** and **Stage 4** have already been updated for **3-model score-level fusion**.
- **CID_BIAS**: both tested variants are dead ends. The original GT-learned CID_BIAS dropped MTMC IDF1 from **0.784 -> 0.751** (**-3.3pp**), and the later topology-bias sweep in **10c v55** reached only **0.764-0.762-0.763** versus a **0.774** control (**-1.0 to -1.2pp**). FIC whitening already handles the useful camera calibration, and additive CID_BIAS terms distort those calibrated similarities.
- **Net result**: The **score-level ensemble path is fully exhausted** on the current codebase. There is no viable secondary model that is both individually strong enough and diverse enough to improve vehicle MTMC.

**Why the previous improvements failed**

- With **1 model**, techniques like **384px**, **DMT**, and **reranking** strip away unstable but still useful signal, so MTMC gets worse.
- With **3-5 models**, those same techniques can suppress noise while identity signal survives through model diversity.
- The key blocker is therefore not that these methods are intrinsically bad, but that they were evaluated in a **single-model regime** where AIC winners never operated.

## Prioritized Action Plan

| Priority | Action | Expected Impact | Status |
|:--------:|--------|:---------------:|--------|
| **1** | Paper writing: document the single-model pipeline, the exhaustive **225+** experiment campaign, and the finding that feature quality rather than association tuning is the MTMC bottleneck | Highest practical value from the current codebase state | RECOMMENDED |
| **2** | Implement a **GNN edge-classification** association model | Only remaining clearly distinct technical path beyond the exhausted heuristics and score-fusion variants | NOT IMPLEMENTED; requires significant new code |
| **3** | Push the primary **ViT-B/16 CLIP** training further | Possible but lower-confidence upside because **81.59% mAP** is already very strong for a single model | HIGH EFFORT / UNCERTAIN RETURN |

Association tuning remains exhausted (**225+** configs tested), and the secondary-model / score-level fusion route is now exhausted as well. The remaining realistic vehicle options are to **write the paper**, build a materially new **graph-based association model**, or attempt another step-change in the **primary ViT** representation despite the already strong **81.59% mAP** baseline.

## Active Experiments (2026-04-22)

### 3-Way Ensemble Attempt (April 22, 2026)

- **What we're trying**: 3-way score fusion of primary ViT-B/16 CLIP (80.14% mAP) + secondary R50-IBN (52.77% mAP) + tertiary LAION-2B CLIP (78.61% mAP) using w2 and w3 weights tuned in a 19-point sweep.
- **Motivation**: Both 2-way fusion paths (R50-IBN secondary, LAION CLIP secondary) gave only marginal gains. A 3-way combination has not been tested on a correct feature baseline.
- **Status**: 10a v5 **COMPLETE** (929 tracklets, 49.4 min); 10b v3 **COMPLETE** (12.6 MB FAISS); 10c v8 **COMPLETE** (MTMC_ONLY=True bug — biased); 10c v9 **COMPLETE** (unbiased results confirmed).
- **10c v9 unbiased results (final)**: Baseline = **76.625%** (IDF1=78.419%, MOTA=66.910%, HOTA=57.031%). Best: **w2=0.05, w3=0.30 → 76.817% (+0.192pp)**. R50-IBN alone: −0.064pp. LAION tertiary alone at w3=0.30: +0.154pp.
- **Baseline note**: 76.625% is ~0.74pp below expected 77.36% — camera_bn=true shifted feature distribution. V52 association params need re-tuning.
- **Conclusion**: 3-way ensemble **CONFIRMED DEAD END** — +0.192pp within noise. Priorities: (1) association re-tune for camera_bn=true, (2) 09q v5 Exp A.

#### Broken Baseline Run (10a v4 yahiaakhalafallah — INVALIDATED)
- 10a v4 had wrong overrides: concat_patch=true (→ 1536D, PCA expects 768D) and camera_bn.enabled=false (disables cross-camera BN).
- Result: **-3.79pp regression** — measured baseline 73.57% vs expected ~77.36%.
- Best fusion on broken features: w2=0.05, w3=0.15 → 73.73% (+0.16pp), meaningless.
- **All 10a v4 / 10c (yahia) v5 results are INVALIDATED.** Do not use these numbers.

#### Infrastructure Fix: Cross-Account Kaggle Dataset Mounting
- Public Kaggle datasets do **not** auto-mount cross-account at runtime.
- Fix: add a download cell using the kernel's own KAGGLE_KEY env var to retrieve another account's outputs.
- Applied to 10b kernel (retargeted to consume yahia's 10a output).

## Latest Experiment Results (2026-04-20)

### Completed

#### 09o v1 — EVA02 ViT-B/16 CLIP on CityFlowV2 Dead End (2026-04-20)
- **Kernel**: `gumfreddy/09o-eva02-vit-cityflowv2`
- **Task**: Test whether **EVA02 ViT-B/16 CLIP** can provide a stronger, more complementary secondary vehicle ReID backbone for ensemble use on CityFlowV2
- **Training**: **120 epochs**, **AdamW**, **backbone_lr=1e-5**, **head_lr=5e-4**, **CE + Triplet + Center**, **cosine schedule**
- **Result**: **mAP = 48.17%**, **R1 = 65.90%**, **R5 = 77.17%**, **R10 = 82.83%**
- **Comparison**: This is far below the primary **ViT-B/16 CLIP** baseline at **80.14% mAP** and even below the fine-tuned **FastReID SBS R50-IBN** secondary at **63.64% mAP**
- **Interpretation**: The current **EVA02** transfer recipe does not adapt well to fine-grained vehicle ReID on CityFlowV2. The backbone retains some ranking signal, but it is nowhere near the **>=65% mAP** practical floor for a useful ensemble secondary
- **Conclusion**: **EVA02 ViT-B/16 CLIP is a confirmed dead end with the current recipe**. At **48.17% mAP**, it is too weak for ensemble use and does not justify further score-fusion follow-up in its current form.

#### 09m v2 — CLIP RN50x4 CNN Secondary Model Dead End (2026-04-19)
- **Kernel**: `gumfreddy/09m-clip-rn50x4-vehicle-reid-cityflowv2`
- **Task**: Test whether a **CLIP RN50x4 CNN** backbone can provide an architecturally diverse secondary vehicle ReID model for score-level fusion
- **Result**: **Catastrophic failure** with **best mAP = 1.55%** and **best R1 = 4.18%**
- **Training dynamics**: Cross-entropy loss still converged from **6.57 -> 0.99** over **200 epochs**, so the run did not fail through obvious divergence or undertraining
- **Interpretation**: The backbone learned something for closed-set classification but produced retrieval features that were effectively useless for cross-camera ReID
- **Root causes**: **(1)** `open_clip` constructed the model with **`quick_gelu=False`** while the OpenAI pretrained weights expect **`quick_gelu=True`**, corrupting the transferred feature geometry; **(2)** the **attention-pooling CNN CLIP** architecture appears poorly matched to the standard ReID projection-head recipe used for the ViT-based runs; **(3)** **640D CNN features** appear fundamentally harder to adapt for fine-grained vehicle ReID than the current **768D ViT** features
- **Conclusion**: **CLIP RN50x4 CNN is a confirmed dead end** for CityFlowV2 vehicle ReID. Combined with the failed non-CLIP secondaries and the **10c v56** correlated-CLIP fusion regression, this means the **score-level ensemble path is now fully exhausted** and there is **no viable secondary model** on the current codebase.

#### 10c v56 — LAION-2B CLIP Score-Level Fusion Dead End (2026-04-18)
- **Kernel**: `gumfreddy/10c-stages45-cityflowv2-association-eval` **v56**
- **Task**: Test whether adding the new **TransReID ViT-B/16 LAION-2B CLIP 256px** secondary model improves vehicle MTMC through simple score-level fusion
- **Secondary model**: **TransReID ViT-B/16 LAION-2B CLIP 256px** from **09l v3** with **mAP = 78.61%** and **R1 = 90.43%**
- **Fusion method**: score-level fusion with **`sim = 0.70 * primary_sim + 0.30 * secondary_sim`**
- **Result**: **MTMC IDF1 = 0.769** with fusion versus **0.774** without fusion, a **-0.5pp** regression
- **Metric deltas**: all key metrics regressed together, with **IDF1 -0.2pp**, **MOTA -0.3pp**, and **HOTA -0.2pp**
- **Interpretation**: This confirms the earlier **52.77%** secondary-model fusion failure was not just caused by a weak second model. Even with a much stronger **78.61% mAP** secondary, score-level fusion still hurts.
- **Root cause**: The primary and secondary are both **CLIP ViT-B/16** models, differing mainly in backbone pretraining (**OpenAI CLIP** vs **LAION-2B CLIP**). Their features are therefore too highly correlated, so the ensemble adds noise rather than complementary identity signal.
- **Lesson**: A useful ensemble here likely requires **architecturally diverse** models with meaningfully different feature biases, such as different backbones, input sizes, or training objectives. Two **CLIP ViT-B/16** variants are too similar.
- **Conclusion**: **LAION-2B CLIP score-level fusion is a confirmed dead end** for the current vehicle MTMC pipeline despite the strong standalone **09l v3** model.

#### 10c v60 — Fine-Tuned R50-IBN Fusion Sweep (10a v37) (2026-04-20)
- **Kernel**: `gumfreddy/mtmc-10c-stages-4-5-association-eval` **v60**
- **Task**: Test whether deploying the fine-tuned **FastReID SBS R50-IBN** secondary model improves vehicle MTMC through score-level fusion on top of the restored **10a v37 -> 10b v22 -> 10c v60** pipeline
- **Secondary model**: **FastReID SBS R50-IBN** from **09n** with **mAP = 63.64%** and **R1 = 78.69%** on CityFlowV2
- **Fusion sweep**: evaluated weights **[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]**
- **Best result**: **w = 0.10 -> MTMC IDF1 = 0.7736**, only **+0.06pp** over the **w = 0.00** baseline at **0.7730**
- **Higher weights hurt**: **w = 0.15 -> 0.7713** (**-0.18pp**) and **w = 0.50 -> 0.7625** (**-1.05pp**)
- **Interpretation**: Fine-tuning improved the R50-IBN secondary substantially over the earlier zero-shot ResNet baseline (**63.64% vs 52.77% mAP**), but the downstream MTMC gain remains negligible.
- **Conclusion**: Even a fine-tuned **63.64% mAP** R50-IBN secondary is still too weak for meaningful ensemble gain. A useful secondary likely needs **>=70% mAP** on CityFlowV2 and genuinely complementary feature biases. This confirms the broader dead end for **ResNet-IBN** score-level fusion secondaries, including the already exhausted **ResNet101-IBN-a** path.

#### 10c v61 — Improved 09p R50-IBN Fusion Sweep Still Near Prior Ceiling (10a `run_kaggle_20260420_201401`, 10b v23) (2026-04-20)
- **Kernel**: `gumfreddy/mtmc-10c-stages-4-5-association-eval` **v61**
- **Task**: Re-run the score-level fusion sweep using the improved **09p FastReID SBS R50-IBN** secondary embeddings from the newer **10a** chain on top of **10a `run_kaggle_20260420_201401` -> 10b v23 -> 10c v61**
- **Secondary model**: improved **09p** R50-IBN path deployed through the updated **10a** extraction chain
- **Fusion sweep**: evaluated weights **[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]**
- **Results**:
	- **w = 0.00 -> MTMC IDF1 = 0.773021**
	- **w = 0.05 -> MTMC IDF1 = 0.773255**
	- **w = 0.10 -> MTMC IDF1 = 0.773595** (**BEST**)
	- **w = 0.15 -> MTMC IDF1 = 0.772622**
	- **w = 0.20 -> MTMC IDF1 = 0.771648**
	- **w = 0.25 -> MTMC IDF1 = 0.771440**
	- **w = 0.30 -> MTMC IDF1 = 0.771440**
	- **w = 0.40 -> MTMC IDF1 = 0.770557**
	- **w = 0.50 -> MTMC IDF1 = 0.761920**
- **Canonical result for this run**: use the **fusion-sweep best** at **0.773595 MTMC IDF1** rather than the one-pass Stage-5 log line that reported **[MTMC] IDF1=77.1%, MOTA=68.9%, HOTA=57.5%, IDSW=198**
- **Interpretation**: Even with improved **09p** secondary training and more robust ingestion, the gain remains only **+0.000574** over the **w = 0.00** baseline. That is effectively flat, still below the current reproducible **0.775** result, and far below the historical **0.784** / SOTA regime.
- **Conclusion**: This further confirms that the bottleneck remains **primary feature quality / architecture**, not a missed association sweep. The vehicle pipeline is still pinned near the same **~0.773-0.775** ceiling.

#### 09l v1 — LAION-2B CLIP CircleLoss Failure (2026-04-17)
- **Kernel**: `gumfreddy/09l-transreid-laion-2b-training` **v1**
- **Task**: Test **TransReID ViT-B/16 LAION-2B CLIP 256px** as an alternative **CLIP-family** secondary vehicle ReID backbone for future ensemble use
- **Config**: **Experiment B: CircleLoss ablation** recipe with **CE+LS(eps=0.05) + CircleLoss(m=0.25, gamma=128) + CenterLoss**, **backbone lr=1e-4**, **head lr=1e-3**, **LLRD=0.75**, **120 epochs**, **EMA disabled**, **CLIP-only init**
- **Result**: **Catastrophic failure** with **mAP = 20.36%**, **R1 = 53.03%**, **mAP_rr = 27.16%**
- **Training stability**: Loss was **`inf` throughout all epochs**, matching the earlier **09 v4 / Experiment B** failure pattern exactly
- **Interpretation**: This does **not** indicate that **LAION-2B CLIP** is weak. The training recipe itself was broken: **CircleLoss(gamma=128)** overflows in **fp16 autocast**, so the run never had a chance to evaluate the backbone fairly
- **Follow-up**: **09l v2** completed and confirmed that the backbone is viable once the broken CircleLoss recipe is removed
- **Conclusion**: Treat **09l v1** as further evidence that the current **CircleLoss** recipe is a dead end, not as evidence against the **LAION-2B** backbone

#### 09l v2 — LAION-2B CLIP TripletLoss Rerun (2026-04-18)
- **Kernel**: `gumfreddy/09l-transreid-laion-2b-training` **v2**
- **Task**: Re-evaluate **TransReID ViT-B/16 LAION-2B CLIP 256px** as a secondary vehicle ReID backbone using the stable training recipe instead of the broken CircleLoss ablation
- **Config**: **CE+LS(eps=0.05) + TripletLoss(m=0.3) + CenterLoss(weight=5e-4, delayed until epoch 15)**, **LLRD=0.75**, **EMA(decay=0.9999)**, **160 epochs**, **CLIP-only init**
- **Result**: **mAP = 61.51%**, **R1 = 81.41%**, **mAP_rerank = 67.20%**, **R1_rerank = 82.95%**
- **Training time**: **~3.7 hours on Kaggle T4** for the full **160-epoch** run
- **Training dynamics**: The model was still improving strongly at the end: **20e 12.69/35.86 -> 40e 18.47/45.32 -> 60e 25.99/53.58 -> 80e 34.23/61.06 -> 100e 42.25/66.45 -> 120e 49.73/73.05 -> 140e 55.98/77.78 -> 160e 61.51/81.41** (**mAP/R1**)
- **Key finding**: The run is **not converged**. It gained **+5.53pp mAP** in the final **20 epochs**, and the cosine LR reached **0.00** exactly at epoch 160, so optimization was cut off by the schedule rather than by a plateau
- **Interpretation**: **LAION-2B CLIP** is a legitimate live candidate for a secondary ensemble backbone. It already cleared the prior non-CLIP ceiling, but it has not yet reached its own ceiling
- **Follow-up**: **09l v3** completed the extension to **300 total epochs** and converted this backbone from a live candidate into an **ensemble-ready** secondary model

#### 09l v3 — LAION-2B CLIP Extended Training Success (2026-04-18)
- **Kernel**: `gumfreddy/09l-transreid-laion-2b-training` **v3**
- **Task**: Resume from the **09l v2 EMA checkpoint** and determine whether **LAION-2B CLIP** can clear the practical ensemble threshold on CityFlowV2
- **Config**: same stable **CE+LS(eps=0.05) + TripletLoss(m=0.3) + CenterLoss(weight=5e-4, delayed until epoch 15)** recipe, resumed from **epoch 160** with **EMA(decay=0.9999)** and extended to **300 total epochs**
- **Result**: **mAP = 78.61%**, **R1 = 90.43%**, **mAP_rerank = 81.09%**, **R1_rerank = 90.98%**
- **Training time**: **~3.3 hours on Kaggle T4** for the resumed **160 -> 300** phase
- **Training dynamics**: **180e 65.93 -> 200e 68.84 -> 220e 71.49 -> 240e 73.68 -> 260e 75.68 -> 280e 77.26 -> 300e 78.61** (**mAP**); **R1 = 90.43%** at epoch **300**
- **Key finding**: **LAION-2B CLIP** is now a **strong secondary model**, finishing only **1.53pp** behind the deployed **OpenAI CLIP 09b v2** baseline at **80.14% mAP** and well above the **65%** ensemble threshold
- **Conclusion**: The backbone is **ready for score-level fusion deployment**. The next question is now ensemble gain, not backbone viability

#### 09i v1 — ResNet101-IBN-a ArcFace Follow-Up (2026-04-17)
- **Task**: Test whether an **ArcFace-based** ResNet101-IBN-a recipe can lift the secondary vehicle model above the long-standing **52.77% mAP** ceiling and make it ensemble-worthy
- **Recipe**: **ArcFace (s=30, m=0.35) + Triplet (margin=0.3) + Center loss**, warm-started from the best **09d** checkpoint
- **Result**: Best **mAP = 50.80%** at **epoch 100/160**, **R1 = 73.46%**, **mAP_rerank = 54.65%**
- **Training dynamics**: Performance **declined after epoch 100**, indicating classic overfitting on the small **128-ID** CityFlowV2 train split
- **Comparison**: Still **worse than the 09d baseline** at **52.77% mAP**
- **Root cause**: Warm-starting from a **CE-optimized** checkpoint into an **ArcFace angular-margin** objective creates a geometry mismatch, and combining that with **triplet + center** on only **128 classes** produces too many competing objectives for this backbone/data regime
- **Conclusion**: **ArcFace is a dead end for the current ResNet101-IBN-a path**, and the secondary-model rescue attempt did not materialize

#### 10c v55 — CID_BIAS Topology Bias Sweep (2026-04-17)
- **Task**: Test whether a lightweight **topology-derived CID_BIAS** improves Stage-4 association by adding camera-pair-specific similarity offsets on top of the restored baseline recipe
- **Baseline**: Fresh baseline features from **10a v30** with a control run at **MTMC IDF1 = 0.774** and no additive bias
- **Conservative bias**: **+0.02 / -0.10** reached **MTMC IDF1 = 0.764** (**-1.0pp**)
- **Default bias**: **+0.04 / -0.15** reached **MTMC IDF1 = 0.762** (**-1.2pp**)
- **Aggressive bias**: **+0.06 / -0.20** reached **MTMC IDF1 = 0.763** (**-1.2pp**)
- **Interpretation**: Every tested additive bias degraded MTMC, including the most conservative setting. This indicates the current pipeline is already getting the useful camera-pair calibration from **FIC whitening**, and extra CID_BIAS offsets only warp the calibrated similarity geometry.
- **Conclusion**: **Topology CID_BIAS is a confirmed dead end** for the current CityFlowV2 single-model pipeline. Together with the earlier **GT-learned CID_BIAS** regression (**-3.3pp**), this now rules out both learned and hand-shaped additive CID_BIAS variants.

#### 10c v53 — Network Flow Solver vs CC Baseline (2026-04-17)
- **Task**: Test a **network flow / Hungarian-based structural association solver** as a replacement for the existing connected-components merge logic, with merge verification intended to reduce cross-camera conflation
- **Baseline**: Controlled **10c v52** CC run at **MTMC IDF1 = 0.7714**, **MOTA = 0.689**, **HOTA = 0.5747**, **ID switches = 199**, **45 fragmented GT IDs**, **27 conflated predicted IDs**
- **Network flow result**: **MTMC IDF1 = 0.769**, **MOTA = 0.694**, **HOTA = 0.577**, **ID switches = 197**, **46 fragmented GT IDs**, **30 conflated predicted IDs**
- **Delta vs baseline**: **-0.24pp MTMC IDF1**, **+0.5pp MOTA**, **+0.2pp HOTA**, **-2 ID switches**, **+1 fragmented GT ID**, **+3 conflated predicted IDs**
- **Interpretation**: The solver improved some frame-level metrics slightly, but it failed its main design goal. The Hungarian assignment plus merge verification did **not** prevent false merges and instead created **more conflation** than the CC baseline.
- **Conclusion**: **Network flow is neutral to slightly negative** for the current CityFlowV2 vehicle pipeline. It is **not a hard dead end** on the scale of AFLink or CSLS, but it is **not helpful** and does not justify replacing the existing **conflict_free_cc** approach.

#### 10a v29 / 10c v50 — SAM2 Foreground Masking Before Vehicle ReID (2026-04-16)
- **Task**: Test inference-time **SAM2 foreground masking** on vehicle crops before Stage-2 ReID feature extraction using **`facebook/sam2.1-hiera-tiny`**
- **Masking config**: **center-point prompt**, **per-crop mean fill**, **`min_crop_size=48`**
- **10a v29 runtime / scale**: **929 tracklets**, **105.2 min**, versus roughly **~65 min** for the same pipeline without SAM2 masking
- **10c v50 evaluation**: Full **60-config** Stage-4 parameter sweep with **FEATURE_TEST enabled** across **11 parameter dimensions**, including AFLink and camera-pair normalization variants
- **Best result**: **MTMC IDF1 = 0.688**
- **Per-camera 2D metrics**: **MOTA = 0.540**, **IDF1 = 0.704**, **HOTA = 0.515**
- **Baseline comparison**: Current reproducible non-SAM2 baseline is **MTMC IDF1 = 0.775** from **10c v52**, so SAM2 masking is **-8.7pp** worse despite exhaustive association retuning
- **Interpretation**: Foreground masking removes background context such as road surface and nearby scene cues that appear to help cross-camera vehicle re-identification. The masks also likely clip vehicle edges and suppress boundary features that carry identity signal.
- **Conclusion**: **SAM2 foreground masking is a confirmed dead end** for the current CityFlowV2 vehicle pipeline. It adds major runtime cost while degrading both single-camera and MTMC quality, and even aggressive downstream retuning could not recover baseline performance.

#### 09 v4 / Experiment B — CircleLoss Ablation on Baseline Augmentations (2026-04-16)
- **Kernel**: `gumfreddy/09-vehicle-reid-cityflowv2-circleloss-ablation` **v1**
- **Purpose**: Isolate whether **CircleLoss** caused the augoverhaul regression by keeping the original baseline augmentation stack and disabling EMA
- **Config**: **TransReID ViT-B/16 CLIP 256px**, **CE+LS(eps=0.05) + CircleLoss(m=0.25, gamma=128) + CenterLoss**, **120 epochs**, baseline augmentations only: `RandomHorizontalFlip`, `Pad+RandomCrop`, `ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.0)`, `Normalize`, `RandomErasing`
- **Result**: **Catastrophic failure** with **best mAP = 18.45%** and **best R1 = 48.84%**
- **Training stability**: Loss was **`inf` at every epoch**, indicating complete numerical instability rather than a weak-but-valid training regime
- **Interpretation**: This matches the pre-existing dead-end pattern that **circle-style metric learning can catastrophically destabilize training** in this codebase. It also means the prior augoverhaul regression cannot be explained as “CircleLoss helped ReID but hurt MTMC.” When CircleLoss is definitely active on the primary ViT recipe, it destroys training outright.
- **Conclusion**: **CircleLoss is a confirmed dead end** for the primary CityFlowV2 TransReID path. The augoverhaul regression is therefore most likely attributable to the augoverhaul augmentations themselves, unless the earlier augoverhaul training never actually activated CircleLoss because of a config mismatch.

#### 09 v2 — Augmentation Overhaul + EMA (2026-04-13)
- **Task**: Test a stronger augmentation recipe plus model EMA on the primary CityFlowV2 TransReID ViT-B/16 CLIP training path
- **Augmentation overhaul**: Added `RandomGrayscale(p=0.1)`, stronger `ColorJitter(0.3,0.25,0.2,0.05)`, `GaussianBlur(k=5,p=0.2)`, `RandomPerspective(0.1,p=0.2)`, and a wider `RandomErasing` scale range
- **EMA config**: Model Exponential Moving Average with `decay=0.9999`
- **Base model result**: **mAP = 81.59%**, **+1.45pp** over the prior **80.14%** baseline
- **Base model with reranking**: **mAP = 83.12%**
- **EMA model result**: **mAP = 39.09%**
- **EMA model with reranking**: **mAP = 47.76%**
- **Analysis**: The augmentation overhaul clearly works and improves generalization. The EMA branch failed because `decay=0.9999` is too high for a 120-epoch schedule; the averaged weights converge too slowly and were still improving at epoch 120.
- **Downstream outcome**: The corrected end-to-end follow-up in **10c v48** reached only **0.722 MTMC IDF1**, so the higher validation mAP did not transfer to vehicle MTMC.

#### 09 v3 — EMA Training Results (2026-04-14)
- **Task**: Re-test EMA on the primary CityFlowV2 TransReID ViT-B/16 CLIP path using the augoverhaul augmentation stack with the standard **TripletLoss + CenterLoss** recipe and a lower **EMA decay = 0.999**
- **Base model result**: **mAP = 81.53%**, **R1 = 92.41%**
- **EMA model result**: **mAP = 81.44%**, **R1 = 92.74%**
- **EMA delta vs base**: **-0.09pp mAP**, **+0.33pp R1**
- **Baseline comparison**: Relative to the prior **80.14% mAP / 92.27% R1** baseline, the augoverhaul + EMA training run still gains **+1.39pp mAP** overall
- **Interpretation**: EMA converges to essentially the same solution as the base model. The tiny validation difference is not meaningful enough to justify carrying a second checkpoint or treating EMA as a distinct improvement path.
- **Downstream implication**: Combined with **10c v48 = 0.722** and **10c v49 = 0.722 MTMC IDF1** (**-5.3pp** vs baseline in both cases), this further confirms that higher single-camera **mAP** is **not** the MTMC bottleneck in the current vehicle pipeline.
- **Conclusion**: **EMA is a dead end** for this recipe. It produces nearly identical validation quality to the base model and does not change the core finding that feature-space changes with better mAP are failing to improve MTMC.

#### 10c v47 — Augmentation Overhaul Model with 384px Deployment Bug (2026-04-14)
- **Task**: Evaluate the new **09 v2 augmentation-overhaul** primary model in the downstream **10a -> 10b -> 10c** CityFlowV2 pipeline
- **Result**: **MTMC IDF1 = 0.702**, a **-7.3pp** regression versus the current reproducible **0.775** baseline
- **Root cause**: **BUG**. The **10a** notebook deployed the augoverhaul model at **384x384** input even though it was trained for **256x256**. This happened because `configs/datasets/cityflowv2.yaml` still had `input_size: [384, 384]` from the earlier 384px dead-end work, and the notebook did not explicitly override it for the 256px model.
- **Why this matters**: This result does **not** show that the augmentation-overhaul model is worse. The model itself improved single-camera quality to **mAP = 81.59%** versus the previous **80.14%** baseline, so it should be expected to perform **better**, not worse, when deployed at the correct **256x256** resolution.
- **Fix applied**: Added an explicit `stage2.reid.vehicle.input_size=[256,256]` override to the **10a** notebook and fixed the default `cityflowv2.yaml` input size so 256px deployment is now the safe default for the primary vehicle model.
- **Status**: The fix was verified in **10a v20**, and the corrected **256px** deployment was evaluated end-to-end in **10c v48**.
- **Conclusion**: Treat **10c v47** as a configuration failure, not as evidence against the augmentation overhaul. The apparent regression was caused by a silent deployment mismatch, not by weaker learned features.

#### 10c v48 — Corrected 256px Evaluation of the 09 v2 Augoverhaul Model (2026-04-14)
- **Task**: Re-run the full downstream **10a -> 10b -> 10c** CityFlowV2 pipeline with the **09 v2 augoverhaul** model deployed at the correct **256x256** input size
- **Best association config**: `sim_thresh=0.45`, `appearance_weight=0.60`, `fic_reg=1.00`, `aqe_k=3`, `aflink_gap=150`, `aflink_dir=0.85`
- **Sweep size**: **11 association configs**
- **Result**: **MTMC IDF1 = 0.722**, which is **-5.3pp** versus the current reproducible **0.775** baseline
- **Single-camera quality**: **IDF1 = 0.752**, also materially below the baseline (~0.82)
- **Interpretation**: The 384px deployment bug in **10c v47** was real, but fixing it did **not** rescue the model. The underlying **09 v2** recipe still regresses badly for MTMC even at the correct deployment resolution.
- **Root cause**: The experiment was confounded because it changed both the augmentation stack and the loss at the same time: it added `RandomGrayscale`, stronger `ColorJitter`, `GaussianBlur`, and `RandomPerspective`, while also replacing **TripletLoss** with **CircleLoss**. The stronger augmentations likely suppress color and texture cues that are still needed to distinguish same-model vehicles across cameras, and the simultaneous loss change prevents precise attribution.
- **Conclusion**: The **09 v2 augoverhaul + CircleLoss** recipe is a confirmed **DEAD END** for CityFlowV2 vehicle MTMC despite its higher validation **mAP = 81.59%**.

#### 10c v49 — Association Sweep on the 09 v3 Augoverhaul-EMA Model (2026-04-14)
- **Task**: Re-run the downstream **10c** association sweep using the **09 v3 augoverhaul-EMA** training run (`mAP = 81.53%`, `R1 = 92.41%`) to test whether the EMA-trained augoverhaul recipe changes the MTMC operating point
- **Best association config**: `sim_thresh=0.45`, `appearance_weight=0.60`, `spatiotemporal_weight=0.40`, `fic_reg=1.00`, `aqe_k=3`, `gallery_thresh=0.45`, `orphan_match=0.35`
- **Best result**: **MTMC IDF1 = 0.722**, exactly matching **10c v48** and therefore confirming a persistent augoverhaul ceiling
- **AFLink effect**: With the best non-AFLink operating point, MTMC IDF1 peaked at **0.675**; enabling AFLink with `max_spatial_gap_px=150` and `min_direction_cos=0.85` lifted it to **0.722**, but still did not recover the baseline
- **Reranking**: **Off** was best throughout the sweep
- **Camera-pair normalization**: **Off** was best throughout the sweep
- **Intra-camera merge**: Negligible effect at `thresh=0.80` and `gap=60`
- **Interpretation**: Even after swapping to the **09 v3** augoverhaul checkpoint and expanding the association search, the ceiling stayed fixed at **0.722**. That makes the failure mode robust to association variants and points back to the augoverhaul feature space itself rather than missed stage-4 tuning.
- **Conclusion**: The augoverhaul regression is now confirmed across both **10c v48** and **10c v49**. Different sweeps and slightly different augoverhaul checkpoints converge to the same ceiling, so the model family itself is the issue, not the association parameters.

#### 10c v46 — AFLink Motion-Based Post-Association (2026-04-13)
- **Task**: Test AFLink as a motion-based post-association stage after the standard CityFlowV2 appearance-driven association
- **Merge effect**: AFLink produced **57 motion-based merges**, reducing trajectories from **239 -> 182**
- **Best AFLink params**: `spatial_gap=150px`, `direction_cos=0.85`, `camera_pair_norm=on`
- **Best result**: **MTMC IDF1 = 0.7355**, which is **down from the 0.775 baseline by -3.95pp**
- **Within-sweep behavior**: AFLink gained roughly **+6.3pp** inside its own sweep (**0.669 -> 0.732**) when tuning `spatial_gap` and `direction_cos`
- **Retest status**: This result is now **confirmed by a clean controlled retest** on **10c v52** using the exact restored baseline association recipe (`sim_thresh=0.50`, `appearance_weight=0.70`, `fic_reg=0.50`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`) and fresh baseline features from **10a v30**.
- **Controlled retest**: Control without AFLink reached **MTMC IDF1 = 0.7714**, **IDF1 = 0.7921**, **HOTA = 0.5747**. AFLink then degraded to **0.7332** at `gap=100`, `dir_cos=0.90` (**-3.82pp**), **0.7183** at `gap=150`, `dir_cos=0.85` (**-5.31pp**), and **0.6394** at `gap=200`, `dir_cos=0.70` (**-13.20pp**).
- **Root cause**: The original v46 result was initially suspected to be partly confounded by joint sweep interactions, but the clean retest removes that explanation. Even with a tight **100px** gap and strict **0.90** direction cosine, motion consistency across non-overlapping CityFlowV2 cameras is not reliable enough for identity linking. AFLink creates false cross-camera merges that fragment the identity space, and wider motion windows make the damage worse.
- **Conclusion**: AFLink is **confirmed harmful** on **CityFlowV2** and should be treated as a **DEAD END** for this dataset. The true penalty is now established at roughly **-3.8pp to -13.2pp MTMC IDF1**, not just the original **-3.95pp** point estimate from v46.
- **Key insight**: The fact that even `gap=100` with `dir_cos=0.90` still loses **-3.82pp** shows the problem is structural, not a loose-threshold artifact.

#### 12b v2 — Extended Kalman Sweep (2026-04-13)
- **Task**: Extend the WILDTRACK tracker sweep beyond the original tuned-Kalman search to test whether broader interpolation, longer track persistence, looser detection thresholds, or better interpolation dynamics can reduce the remaining 5 ID switches
- **Sweep extensions**: interpolation **[1,2,3] -> [1,2,3,4,5]**, `max_age` **[2,3,4,5] -> [2,3,4,5,6,8]**, detection confidence **[0.15, 0.20, 0.25, 0.30, 0.35]**, plus velocity-aware quadratic interpolation
- **Best config**: `max_age=2`, `min_hits=2`, `distance_gate=25.0`, `q_std=5.0`, `r_std=10.0`, `interpolation_max_gap=1`, `conf_threshold=0.15`
- **Result**: **IDF1 = 94.7%**, **MODA = 90.0%**, **IDSW = 5**
- **Comparison**: Identical to the prior best operating point; no sweep extension reduced ID switches below **5** or improved IDF1 beyond **0.9467**
- **Conclusion**: Kalman tracker parameters are fully exhausted for the current person pipeline. The remaining gap is tracker-limited, not parameter-limited.

#### 12b v3 — Global Optimal Tracker (2026-04-13)
- **Task**: Implement and test a new `GlobalOptimalTracker` with sliding-window globally optimal assignment (Hungarian algorithm with a configurable window) as a possible replacement for the current Kalman tracker on WILDTRACK
- **Global-optimal config**: `window_size=10`, `birth_death_cost=100`
- **Sweep size**: **59 tracker configs** across **Kalman**, **naive**, and **global optimal** trackers
- **Best Kalman result**: **IDF1 = 94.7%**, **MODA = 90.0%**, **IDSW = 5**
- **Best naive result**: **IDF1 = 93.25%**, **MODA = 88.6%**, **IDSW = 12**
- **Best global-optimal result**: **IDF1 = 91.17%**, **MODA = 88.2%**, **IDSW = 15**
- **Comparison**: Global optimal was the **worst** of all three tracker families, trailing Kalman by **-3.5pp IDF1** with **3x more ID switches**
- **Additional finding**: All tested Kalman configs clustered at **IDF1 = 0.9467 ± 0.0004**, and confidence threshold sweeps across **0.15-0.35** made no meaningful difference
- **Root cause**: Global optimal assignment over-commits to immediate frame-level costs and loses the motion-prediction advantage of Kalman filtering. The predictive model in BoT-SORT handles occlusions and missed detections better than pure assignment optimization.
- **Conclusion**: The person pipeline is **fully converged at IDF1=0.947**. Kalman is the optimal tracker for the current system, and further tracker architecture changes are unlikely to close the remaining **0.6pp** gap to SOTA. That gap is more likely caused by rare occlusion/re-entry cases that need stronger person appearance features or graph-based multi-view fusion, not more tracker improvements.

#### 10c v52 — Vehicle v80-Restored Reproduction
- **Task**: Re-run the vehicle pipeline with the restored v80-style association recipe on the current codebase
- **Config**: `sim_thresh=0.50`, `appearance_weight=0.70`, `fic_reg=0.50`, `aqe_k=3`, `gallery_thresh=0.48`, `orphan_match=0.38`
- **Result**: **MTMC IDF1 = 0.775**, IDF1 = 0.801, MOTA = 0.671, HOTA = 0.581
- **Comparison**: Improves over v50 at **0.772**, but still trails historical v80 at **0.784**
- **Conclusion**: Vehicle association remains exhausted. The latest recovery attempt squeezed out only +0.3pp, so future gains need materially better feature quality or stronger priors rather than more stage-4 sweeps.

#### 10a/10c — Score-Level Fusion Ensemble Test (TransReID + ResNet101-IBN-a at 0.30 weight)
- **Task**: Test 2-model score-level fusion with primary TransReID ViT-B/16 CLIP (256px, ~80% mAP) and secondary ResNet101-IBN-a (52.77% mAP) at 30% fusion weight
- **Config**: `fus0.3_ter0.0` — 30% secondary weight, 0% tertiary, on current v52 association recipe
- **Result**: **MTMC IDF1 = 0.774**, IDF1 = 0.803, MOTA = 0.676, HOTA = 0.581
- **Baseline**: v52 single-model: **MTMC IDF1 = 0.775**
- **Delta**: **-0.1pp** (statistically identical)
- **Root cause**: The 28pp mAP gap between primary (80%) and secondary (52.77%) means the secondary model adds noise rather than complementary signal. At 0.30 weight, 30% of the similarity score comes from a model that is essentially random for cross-camera matching. AIC22 winners used ensembles where ALL models exceeded 70% mAP on VeRi-776.
- **Conclusion**: Ensemble with a drastically weaker secondary model is a dead end. The secondary must reach **≥65% mAP** (ideally **>70%**) on CityFlowV2 before ensemble fusion can help. **09l v3** now satisfies that requirement at **78.61% mAP**, so this negative result should be interpreted as a failure of the **52.77%** ResNet secondary specifically, not of score-level fusion in general.

#### 09d v3 (gumfreddy) — Extended ResNet101-IBN-a Fine-Tuning Failure
- **Task**: Resume the best direct ImageNet→CityFlowV2 ResNet101-IBN-a checkpoint (**09d v18**, 52.77% mAP) and continue training with a lower learning rate
- **Config**: Resume from 09d v18 `best_model.pth`, lower `lr=3e-4`
- **Result**: Best **mAP = 50.61%**, which is **worse** than the original **52.77%** checkpoint
- **Conclusion**: Additional fine-tuning from the 52.77% checkpoint degrades the model instead of improving it. The direct ImageNet→CityFlowV2 path appears saturated, and **52.77% is effectively the ceiling** for this ResNet101-IBN-a recipe.

#### 12a v26 — MVDeTr WILDTRACK Detector for Best Tracking Run
- **Task**: Train the detector used by the current best 12b v14 tracking pipeline
- **Config**: MVDeTr `resnet18` backbone, `deform_trans`, 10 epochs
- **Result**: **MODA = 90.9%**, MODP = 81.9%, Precision = 95.8%, Recall = 95.1%
- **Conclusion**: Detection quality is now strong enough that the remaining IDF1 gap appears dominated by missed or imperfect detections rather than association collapse.

#### 12a v3 (gumfreddy) — Extended MVDeTr WILDTRACK Detector Training
- **Task**: Extend WILDTRACK detector training from 10 to 25 epochs on the same MVDeTr `resnet18` setup
- **Config**: MVDeTr `resnet18`, 25 epochs
- **Result**: Best **MODA = 92.1%** at **epoch 20**, with recall in the **96.4-96.6%** range and best-epoch precision **95.7%**
- **Comparison**: Previous best detector was **MODA = 90.9%**, so this is a **+1.2pp** improvement in MODA and roughly **+1.5pp** in recall
- **Trend**: Performance peaked around epoch 20 and declined slightly by epoch 25
- **Conclusion**: This is the **best WILDTRACK detection result yet**, but the follow-up **12b v1** tracking rerun stayed flat, so detector gains alone are no longer enough to move person tracking.

#### 12b v1 (gumfreddy) — Tracking Rerun on 12a v3 Detections
- **Task**: Re-run the WILDTRACK tracking sweep using the improved **12a v3** detector outputs
- **Best tracker**: Kalman `max_age=2`, `min_hits=2`, `distance_gate=25.0`, `q_std=5.0`, `r_std=10.0`
- **Result**: **MODA = 90.0%**, **IDF1 = 94.7%**, Precision = 94.5%, Recall = 96.1%, IDSW = 5
- **Comparison**: Detector quality improved from **90.9%** to **92.1% MODA**, but tracking stayed essentially flat versus **12b v14** (**90.3% MODA**, **94.7% IDF1**)
- **Conclusion**: The tracking sweep converged to the same effective operating point. The current WILDTRACK person pipeline is bottlenecked by the tracker, not the detector.

### In Progress

#### Kaggle T4 Dataset Mount Structure Discovery
- **Discovery**: On Kaggle **T4 GPU** kernels, datasets mounted under **`/kaggle/input/datasets/<owner>/<slug>/`** rather than **`/kaggle/input/<slug>/`**
- **Impact**: This caused multiple 09d failures when kernels assumed the legacy flat mount path and could not locate the dataset
- **Conclusion**: When debugging missing-dataset errors on Kaggle T4 kernels, check the nested `datasets/<owner>/` mount structure first.

## Bugs Found / Configuration Pitfalls

### CityFlowV2 `input_size` default leaked a dead-end 384px setting (2026-04-14)
- **Bug**: `configs/datasets/cityflowv2.yaml` still contained `input_size: [384, 384]`, even though **384px deployment is a confirmed dead end** for the main vehicle MTMC pipeline.
- **Failure mode**: The **10a** notebook silently inherited that config and exported the new **09 v2 augmentation-overhaul** model at **384x384** instead of its intended **256x256** input size.
- **Observed impact**: This caused the catastrophic **10c v47 = 0.702 MTMC IDF1** result and could easily have been misread as a model-quality regression.
- **Verification**: The fix was verified in **10a v20**, and the corrected **256px** deployment was then evaluated in **10c v48**.
- **Lesson**: Leaving dead-end experimental defaults in shared config files is dangerous because later notebooks can silently inherit them.
- **Fix**: Set the safe default back to **256x256** and add explicit notebook-level overrides for `stage2.reid.vehicle.input_size` whenever testing alternate deployment resolutions.

#### 12b v4/v5/v6 — Ground-plane Tracking Validation
- **Kernel**: `ali369/12b-wildtrack-mvdetr-tracking-reid` v4-v6, T4 GPU
- **Task**: World-coordinate tracking on MVDeTr detections plus person ReID feature extraction
- **Tracking result**: **MODA = 89.8%, IDF1 = 92.2%, Precision = 97.1%, Recall = 93.9%, IDSW = 12**
- **Outcome**: Ground-plane tracking is working correctly; WILDTRACK tracking quality is strong even before ReID is used for evaluation
- **Issue discovered in v6**: ReID embeddings were mode-collapsed with mean off-diagonal cosine similarity **0.874**
- **Root cause**: Pipeline instantiated CLIP ViT (`vit_base_patch16_clip_224.openai`) while 09p was trained with standard ViT (`vit_base_patch16_224`); CLIP adds `norm_pre`, which was randomly initialized and corrupted features

#### 12b v8 — ViT Backbone Fix
- **Kernel**: `ali369/12b-wildtrack-mvdetr-tracking-reid` v8, T4 GPU
- **Fix**: Matched inference backbone to the 09p training architecture (`vit_base_patch16_224`)
- **Feature quality improvement**:
	- Mean off-diagonal cosine similarity: **0.874 -> 0.720**
	- Similarity range: **[0.675, 1.000] -> [0.215, 1.000]**
	- Merge candidates: **844 -> 382**
	- Weight loading: only 4 minor missing keys (`bn_jpm.*`) instead of all backbone keys missing in v5/v6
- **Tracking metrics**: Unchanged at MODA=89.8%, IDF1=92.2% because ground-plane evaluation does not consume ReID features directly

#### 10a v44 — Local Stage 3-5 Evaluation on Correct 384px + `min_hits=2` Features
- **Input**: `data/kaggle_outputs/10a_v44/run_kaggle_20260329_192418/` with 941 tracklets, 384D primary embeddings, 384D secondary embeddings, 192D HSV
- **Task**: Run local CPU-only stages 3-5 on the exported Kaggle stage-2 artifacts using two association configs
- **Run A (384px-tuned config)**:
	- Overrides: sim=0.54, secondary_weight=0.17, appearance=0.70, gallery=0.50, FIC=0.10, cross_id_nms_iou=0.40, min_traj_frames=40, min_traj_conf=0.30, `gt_frame_clip=true`, `gt_zone_filter=true`, `mtmc_only_submission=false`
	- Result: **MTMC IDF1 = 74.62%**, IDF1 = 78.62%, MOTA = 67.99%
	- Per-camera IDF1: S01_c001 0.926, S01_c002 0.900, S01_c003 0.914, S02_c006 0.637, S02_c007 0.760, S02_c008 0.580
- **Run B (v80-style thresholds on same v44 features)**:
	- Overrides: sim=0.53, secondary_weight=0.10, appearance=0.70, gallery=0.50, FIC=0.10 with the same stage-5 filters as Run A
	- Result: **MTMC IDF1 = 74.74%**, IDF1 = 78.67%, MOTA = 68.07%
	- Per-camera IDF1: S01_c001 0.925, S01_c002 0.900, S01_c003 0.914, S02_c006 0.638, S02_c007 0.762, S02_c008 0.582
- **Outcome**: Both local evaluations underperform the current 78.4% MTMC benchmark by ~3.7pp. The v80-style thresholds are only **+0.12pp** MTMC IDF1 over the 384px-tuned config, so the issue is not primarily the chosen stage-4 thresholds.
- **Important note**: These runs used the current local stage-5 defaults unless overridden. Track smoothing remained enabled, and the requested legacy override aliases had to be translated onto the live config tree (`weights.vehicle.appearance`, `stage5.min_trajectory_*`, `stage5.cross_id_nms_iou`, `stage5.gt_frame_clip`).

#### 10a v43/v44 — Definitive 384px Verdict
- **v43**: 384px features, tuned thresholds, min_hits=3 -> **MTMC IDF1 = 0.7585**
- **v44**: 384px features, exact v43 association recipe, min_hits=2 -> **MTMC IDF1 = 0.7562**
- **v44 + CID_BIAS**: exact v44 config plus a locally-computed 6x6 CID_BIAS matrix from **464/941 GT-matched tracklets** -> **MTMC IDF1 = 0.7510**
- **Baseline for comparison**: v80 256px pipeline, min_hits=2 -> **MTMC IDF1 = 0.7840**
- **Conclusion**: 384px is a confirmed dead end for MTMC in this pipeline, despite strong single-camera mAP. The failure mode is reduced cross-camera invariance, not insufficient stage-4 tuning. CID_BIAS also made the 384px run **worse by -0.52pp**, so camera-pair calibration is not enough to rescue this feature set.

#### 10c v45-v51 — Final Vehicle Ablation Verdict
- **v45 baseline**: late 10c baseline remained in the **0.772-0.775 MTMC IDF1** range depending on exact export and evaluation path
- **v46 (DMT camera-aware training, 87.3% mAP)**: **MTMC IDF1 = 0.758**, which is **-1.4pp** vs the paired v45 baseline at **0.772**
- **v48 (`concat_patch=true`, 1536D features)**: **MTMC IDF1 = 0.773**, which is **-0.3pp** vs v45 at **0.775**
- **v49 (`concat_patch=true` + vehicle2 ensemble)**: **MTMC IDF1 = 0.769**, which is **-0.3pp** vs v50 at **0.772**
- **v50/v51 (current reproducible best / multi-query test)**: **MTMC IDF1 = 0.772**; the multi-query track representation in **v51** was **-0.1pp** vs **v50** and did not improve association quality
- **Conclusion**: Every final feature-side ablation failed to beat the current reproducible baseline. Better single-camera discrimination did not solve the MTMC bottleneck.

## Person Pipeline (WILDTRACK) — New Initiative

### Baseline Performance

| Run | Config Changes | Tracklets | MTMC IDF1 | IDF1 | MOTA |
|-----|---------------|-----------|-----------|------|------|
| Baseline (run_20260327_211115) | Default wildtrack.yaml | 1,339 | 0.176 | 0.316 | -0.281 |
| Exp 1 (run_20260327_224721) | conf=0.55, min_len=8, min_hits=3, match=0.75, merge_gap=40 | 819 | 0.233 | 0.368 | 0.118 |
| Kaggle Baseline (wildtrack_20260328_000939) | 11a→11b→11c chain on Kaggle T4; conf=0.55, min_hits=3, match=0.75, min_tracklet_length=8; ReID 768D→PCA 256D (EV=0.988), HSV 192D, flip_aug=on, camera_bn=on; sim=0.30, louvain=1.5, app=0.80, hsv=0.10, spatio=0.10 | 911 | 0.140 | 0.280 | -0.463 |
| Exp 2 (run_20260327_231511) | conf=0.65, min_len=12, fresh PCA, rerank=off, sim=0.40, louvain=2.0, app=0.90 | INCOMPLETE (interrupted) | — | — | — |

Kaggle baseline details: 911 tracklets across 7 cameras (C1:126, C2:169, C3:154, C4:88, C5:146, C6:121, C7:107), per-camera 2D metrics IDF1=0.280 / MOTA=-0.463 / HOTA=0.000 / IDSW=573, MTMC IDF1=0.140 / MOTA=-0.276 / IDSW=1006, error analysis: 164 fragmented GT IDs, 141 conflated pred IDs, 46 unmatched GT IDs, 141 unmatched pred IDs. Per-camera IDF1/MOTA/IDSW: C1 0.261/-0.012/126, C2 0.191/-0.669/107, C3 0.253/-0.113/127, C4 0.233/-1.468/23, C5 0.316/-0.687/80, C6 0.254/0.004/86, C7 0.450/-0.296/24.

Kaggle underperformed the local Exp 1 baseline (MTMC IDF1 0.140 vs 0.233; per-camera IDF1 0.280 vs 0.368). The likely causes are suboptimal fixed association parameters in 11c and GPU/framework-dependent detection differences. The 911 vs 819 tracklet count gap indicates slightly different detection/tracking behavior on Kaggle hardware/runtime, which plausibly explains part of the per-camera IDF1 regression.

### Key Discoveries (Person Pipeline)
1. **Frame ID off-by-one bug (FIXED)**: WILDTRACK GT was being written with 0-based frame IDs, but predictions use 1-based. Fixed in `scripts/prepare_dataset.py`. This single fix contributed +39.9pp MOTA improvement.
2. **Extreme tracklet fragmentation**: 1,339 tracklets for ~20 people in baseline. Increasing min_hits and min_tracklet_length helped reduce to 819.
3. **Over-detection**: Person detector has low precision (~0.33, 13K FP). Need higher confidence threshold.
4. **PCA model potentially wrong distribution**: Person PCA was trained on vehicle features. Moved to .bak to force refit on WILDTRACK data.
5. **ReID model**: Using TransReID ViT-Base/16 CLIP pretrained on Market1501 (person-specific). Not fine-tuned on WILDTRACK.
6. **All GPU-intensive pipeline stages must run on Kaggle, not locally** (local GTX 1050 Ti too slow).
7. **Kaggle chain currently regresses vs local best**: Same stage-1 thresholds but fixed 11c association settings and possible runtime differences produced 911 tracklets and lower MTMC/per-camera IDF1 than local Exp 1.

### Person Pipeline Next Steps
- Treat person tracking as fully converged at **94.7% IDF1** under the current Kalman pipeline; **12b v1**, **12b v2**, and **12b v3** all confirmed the same ceiling
- Further person gains likely require materially stronger person appearance features or graph-based multi-view association rather than more tracker tuning
- Revisit merge only if person ReID quality improves materially; current Kalman tracks are already clean and merges did not help
- Keep GPU-intensive person stages on Kaggle only

## What Actually Worked

- **Augmentation overhaul**: **+1.45pp mAP** on the primary vehicle ReID model via `RandomGrayscale`, `GaussianBlur`, `RandomPerspective`, stronger `ColorJitter`, and a wider `RandomErasing` range
- **Deployment-bug correction**: The **10c v47** collapse to **0.702 MTMC IDF1** was correctly traced to a **384px deployment mismatch**; the **10a v20** fix and **10c v48** rerun verified the bug was real, even though the underlying **09 v2** recipe still regressed for MTMC.

## Conclusive Dead Ends (DO NOT RETRY)

| Approach | Result | Evidence |
|----------|--------|---------|
| Association parameter tuning | Exhausted (225+ configs, all within 0.3pp) | Experiment log |
| CSLS distance | -34.7pp catastrophic | v74 |
| Hierarchical clustering | -1.0 to -5.1pp | v54-56, v62 |
| FAC (Feature Augmented Clustering) | -2.5pp | v26 |
| Feature concatenation (vs score fusion) | -1.6pp | Experiment log |
| CamTTA (Camera Test-Time Adaptation) | Helps global, hurts MTMC | v28-30 |
| Multi-scale TTA | Neutral/harmful | Multiple runs |
| Track smoothing / edge trim | Always harmful | Experiment log |
| Denoise preprocessing | -2.7pp | v46, v82 |
| mtmc_only submission flag | -5pp | Documented |
| Auto-generated zone polygons | -0.4pp | v54-57 |
| PCA dimension search | 384D optimal, others worse | Experiment log |
| Ensemble with 52% secondary at high weight | Dilutes signal | Current state |
| Score-level ensemble with 52.77% mAP secondary at 0.30 weight | -0.1pp MTMC IDF1; noise dilutes primary signal | 10a/10c fusion test, fus0.3_ter0.0 |
| Fine-tuned R50-IBN fusion (63.64% mAP) | Only **+0.06pp** MTMC IDF1 at best (`w=0.10`); even with an **11pp** mAP gain over the zero-shot **52.77%** baseline, the secondary is still far too weak for meaningful ensemble benefit | 10c v60 |
| Improved 09p R50-IBN fusion via newer 10a/10b chain | Only **+0.0006** MTMC IDF1 at best (`w=0.10`, **0.773595** vs **0.773021** baseline); improved secondary training plus robust ingestion still leaves the pipeline near the same ceiling | 10c v61 |
| EVA02 ViT-B/16 CLIP | **48.17% mAP** (too weak for ensemble, backbone doesn't transfer well for vehicle ReID) | 09o v1 |
| Score-level ensemble with 78.61% mAP LAION-2B CLIP secondary at 0.30 weight | -0.5pp MTMC IDF1 and all key metrics worse; two CLIP ViT-B/16 variants are too correlated to provide complementary signal | 10c v56 |
| SGD optimizer for ResNet101-IBN-a | 30.27% mAP catastrophic | v18 mrkdagods |
| Circle loss for ResNet101-IBN-a with triplet loss | Catastrophic gradient conflict; NEVER combine on the same feature tensor | 09d v17 (29.6%), 09f v2 (16.2%) vs 09d v18 (52.77%), 09e (62.52%) with circle_weight=0 |
| ArcFace on ResNet101-IBN-a warm-started from 09d | Best **50.80% mAP**, **73.46% R1**, **54.65% mAP_rerank** at epoch 100/160, then overfit; warm-starting a CE-shaped solution into ArcFace created an angular-geometry mismatch and still failed to beat the **52.77%** baseline | 09i v1 |
| ResNeXt101-IBN-a ArcFace with partially compatible pretrained weights | Catastrophic **36.88% mAP**, **62.69% R1**, **40.49% mAP_rerank**; original weights targeted **32x32d** grouped convolutions while the model here used **32x8d**, and the `strict=False` workaround left many layers randomly initialized | 09j v2 |
| CLIP RN50x4 CNN with OpenAI weights | Catastrophic **1.55% mAP**, **4.18% R1** despite CE loss converging from **6.57 -> 0.99** over **200 epochs**; likely broken by the **QuickGELU mismatch**, poor compatibility between the **attention-pooling CNN** backbone and the standard ReID head recipe, and a fundamentally weaker **640D CNN** transfer path for vehicle ReID | 09m v2 |
| CircleLoss on TransReID ViT-B/16 CLIP-family backbones | Catastrophic numerical instability; the original OpenAI CLIP ablation collapsed to **18.45% mAP / 48.84% R1**, and the LAION-2B follow-up collapsed to **20.36% mAP / 53.03% R1 / 27.16% mAP_rr**. In both cases the training loss stayed `inf` throughout, indicating the recipe itself is broken rather than any specific CLIP backbone | `gumfreddy/09-vehicle-reid-cityflowv2-circleloss-ablation` v1, `gumfreddy/09l-transreid-laion-2b-training` v1 |
| ResNet101-IBN-a VeRi-776→CityFlowV2 fine-tuning | mAP=42.7% (09f v3), worse than direct ImageNet→CityFlowV2 (52.77% mAP, 09d v18); secondary-model ensemble path not viable without better pretraining or broader hyperparameter search | 09e v2, 09f v3, 09d v18 |
| Extended fine-tuning from the 09d v18 ResNet101-IBN-a checkpoint | mAP degraded from 52.77% to 50.61% after resuming with lower LR; confirms the direct ImageNet→CityFlowV2 path is already at its ceiling | 09d gumfreddy v3 |
| ResNet101-IBN-a path FULLY EXHAUSTED | Six variants are now closed out: **09d original**, **09d extended**, **09f VeRi-776 transfer**, **09d CircleLoss**, **09d SGD**, and **09i ArcFace** all finished at or below the **52.77%** ceiling. That is far below the **~65%** minimum needed for a useful ensemble, so this backbone path should be **abandoned** for CityFlowV2 vehicle MTMC | 09d, 09f, 09i |
| CLIP ViT backbone when checkpoint uses standard ViT | `norm_pre` randomly initialized, causing mode collapse (cosine sim 0.874) | 12b v5/v6 vs v8 |
| Default Kalman tracker with chi-squared gating on WILDTRACK | Worse than naive baseline; IDF1 fell to 88.9% | 12b v14 sweep |
| ReID merge on top of tuned WILDTRACK Kalman tracks | No improvement over baseline; tracks already clean and only 44 features matched | 12b v14 merge sweep |
| Extended Kalman parameter sweeps (person) | No improvement; 59 tracker configs clustered within +-0.0004 IDF1 around the same 0.947 Kalman operating point, so the tracker parameter space is fully exhausted | 12b v1-v3 |
| Global optimal tracker (person) | -3.5pp IDF1 vs Kalman; immediate assignment costs lost the motion-prediction advantage and produced 15 ID switches | 12b v3 |
| K-reciprocal reranking (with current features) | Always worse | v25, v35 |
| Camera-pair similarity normalization | Zero effect (FIC handles it) | v36 |
| CID_BIAS (camera-pair bias matrix) | -3.3pp MTMC IDF1 on 256px features (0.751 vs 0.784) | v44 + CID_BIAS test |
| confidence_threshold=0.20 | -2.8pp | v45 |
| max_iou_distance=0.5 | -1.6pp | v47 |
| AFLink motion linking | Confirmed harmful in controlled retest: **-3.82pp** even at `gap=100`, `dir_cos=0.90`; broader gaps degrade to **-5.31pp** (`150/0.85`) and **-13.20pp** (`200/0.70`). Motion consistency across non-overlapping cameras is unreliable and false merges dominate. | 10c v46, 10c v52 |
| SAM2 foreground masking | -8.7pp MTMC IDF1 (0.688 vs 0.775 baseline); removes useful background context and clips vehicle boundary features, while increasing runtime from ~65 min to 105.2 min | 10a v29, 10c v50 |
| 384px TransReID deployment | -2.8pp MTMC IDF1 despite +10pp mAP; v43=0.7585, v44=0.7562 vs v80=0.784 | 10a v43, 10a v44, v80 baseline |
| Augmentation overhaul + CircleLoss (09 v2 augoverhaul model) | -5.3pp MTMC IDF1 (0.722 vs 0.775 baseline) despite +1.45pp mAP; confounded recipe changed both augmentations and loss. Follow-up CircleLoss-only ablation collapsed to **18.45% mAP** with `inf` loss every epoch, so CircleLoss is independently a dead end and the augoverhaul augmentations are the most likely regression source unless the original run had a CircleLoss config mismatch | 10c v48, 09 v2, 09 v4 Experiment B |
| EMA model averaging | Model **mAP=81.53%** vs EMA **mAP=81.44%** with **R1 92.41% vs 92.74%**; EMA converges to essentially the same place and is not a meaningful improvement path | 09 v3 |
| DMT camera-aware training (87.3% mAP) | -1.4pp MTMC IDF1; v46=0.758 vs v45=0.772 | 10c v45-v46 |
| EMA (`decay=0.9999`, 120 epochs) | mAP=39.09%; too slow to converge under the current training schedule and needs a much longer run to be viable | 09 v2 |
| Multi-query track representation | -0.1pp; v51=0.771 vs v50=0.772 | 10c v50-v51 |
| `concat_patch=true` (1536D features) | -0.3pp; v48=0.773 vs v45=0.775 | 10c v45, 10c v48 |
| `concat_patch=true` + vehicle2 ensemble | -0.3pp; v49=0.769 vs v50=0.772 | 10c v49-v50 |

| **concat_patch=true deployment (PCA mismatch)** | **-3.79pp MTMC IDF1** when PCA was trained on 768D; the flag changes ViT embedding from 768D → 1536D, corrupting downstream PCA and FIC whitening. Always verify concat_patch=false for 256px deployment. | 10a v4 (yahia) 2026-04-22 |
| **camera_bn.enabled=false** | ~**-2pp MTMC IDF1**; cross-camera batch normalisation is critical for feature calibration. Never disable. | 10a v4 (yahia) 2026-04-22 |
| **3-way score-level ensemble (primary ViT + R50-IBN secondary + LAION tertiary)** | **CONFIRMED DEAD END** (10c v9 unbiased). Baseline 76.625%; best w2=0.05, w3=0.30 → 76.817% (+0.192pp — within noise). R50-IBN alone: −0.064pp. LAION tertiary alone: +0.154pp at w3=0.30 (marginal). | 10c v8, 10c v9, 2026-04-22 |

### 384px TransReID Input Resolution (2026-03-30)
- **Result**: -2.8pp MTMC IDF1 (0.7562 vs 0.784 baseline)
- **Details**: 384px ViT-B/16 CLIP has 80.14% single-camera mAP (same as 256px) but performs significantly worse for cross-camera association
- **v43** (min_hits=3, optimized thresholds): MTMC IDF1 = 0.7585
- **v44** (min_hits=2, exact v43 config): MTMC IDF1 = 0.7562
- **Root cause**: Higher resolution captures viewpoint-specific textures (badge positions, reflections, shadow patterns) that help within-camera ReID but hurt cross-camera matching where viewpoint/lighting change dramatically
- **Insight**: The bottleneck is cross-camera feature invariance, not raw discriminative power

## Component Health Summary

| Component | Status | Notes |
|-----------|:------:|-------|
| Detection (YOLO26m) | ✅ OK | Not the bottleneck |
| Tracking (BoT-SORT) | ✅ OK | min_hits=2 optimal |
| Feature Extraction (ViT) | ⚠️ Ceiling | 256px remains best; 384px is a verified dead end for MTMC |
| Feature Processing (PCA) | ✅ OK | 384D optimal |
| Ensemble/Fusion | ❌ Blocked | Tested at 0.30 weight: -0.1pp. Secondary (52.77% mAP) far too weak for ensemble benefit |
| Association (Stage 4) | ✅ Exhausted | 225+ configs plus network-flow v53; CC remains preferred after network flow lost -0.24pp and increased conflation |
| Evaluation | ✅ OK | Under-merging 1.69:1 ratio = feature quality issue |

## Model Training History

### TransReID ViT-B/16 CLIP (Primary)
- 09b v1: mAP=44.9% (40 epochs from 256px init, too aggressive LR)
- 09b v2: mAP=80.14%, R1=92.27% (VeRi-776 pretrained → CityFlowV2 fine-tune) ← **BEST, but wrong checkpoint on Kaggle**
- 09 v2: mAP=81.59%, rerank mAP=83.12% with augmentation overhaul; EMA branch failed at mAP=39.09% with `decay=0.9999` over 120 epochs

### ResNet101-IBN-a (Secondary)
- 09d v12: mAP=21.9% (IBN layer3 bug)
- 09d v13: mAP=11.98% at epoch 19 (timed out)
- 09d v17: mAP=29.6% (wrong recipe: lr=3.5e-4 + circle_weight=0.5)
- 09d v18 ali369: mAP=52.77% (AdamW lr=1e-3, best so far)
- 09d gumfreddy v3: mAP=50.61% after resuming from the 09d v18 52.77% checkpoint with lower `lr=3e-4`; confirmed extra fine-tuning degrades this ImageNet→CityFlowV2 path
- 09d v18 mrkdagods: mAP=30.27% (SGD lr=0.008, failed catastrophically)
- 09e v2: VeRi-776 pretraining complete, mAP=62.52% on VeRi-776 test set (384x384, 120 epochs, triplet+center, cosine, fp16)
- 09f v2: mAP=16.2% catastrophic failure (circle_weight=1.0, bs=32, label_smooth=0.1, lr=7e-5; best model at epoch 4 during warmup)
- 09f v3: mAP=42.7% at epoch 104/120 (circle_weight=0.0, bs=48, label_smoothing=0.05, AdamW lr=3.5e-4, warmup start_factor=0.1, 120 epochs); still worse than direct ImageNet→CityFlowV2 in 09d v18
- 09i v1: mAP=50.80%, R1=73.46%, mAP_rerank=54.65% at epoch 100/160 with ArcFace+Triplet+Center warm-started from 09d; declined afterward and still failed to beat the 52.77% baseline

### CLIP-Family Secondary Path
- 09l v1: mAP=20.36%, R1=53.03%, mAP_rerank=27.16%; broken CircleLoss recipe kept loss at `inf` throughout and does not reflect backbone quality
- 09l v2: mAP=61.51%, R1=81.41%, mAP_rerank=67.20%, R1_rerank=82.95% at epoch 160 with TripletLoss+EMA; strong recovery, but still schedule-limited
- 09l v3: mAP=78.61%, R1=90.43%, mAP_rerank=81.09%, R1_rerank=90.98% at 300 total epochs after resuming from the v2 EMA checkpoint; now the first ensemble-ready secondary model in the repo
- 09o v1: mAP=48.17%, R1=65.90%, R5=77.17%, R10=82.83% after 120 epochs with AdamW, `backbone_lr=1e-5`, `head_lr=5e-4`, and CE+Triplet+Center; substantially weaker than both the primary ViT baseline and the fine-tuned R50-IBN secondary, so EVA02 is a dead end with the current vehicle-ReID recipe

## Key Insight

**The system is NOT broken.** Vehicle MTMC remains capped by camera-invariant feature quality, not association logic. The current reproducible ceiling is **77.5% MTMC IDF1**, while the historical **78.4%** v80 result is no longer reproducible on the current codebase. Higher single-camera ReID mAP does **not** automatically translate to better MTMC IDF1 in this pipeline: augmentation overhaul plus CircleLoss (**+1.45pp mAP -> -5.3pp MTMC IDF1**), **384px** deployment (**same mAP -> -2.8pp MTMC IDF1**), and **DMT** camera-aware training (**+7pp mAP -> -1.4pp MTMC IDF1**) all made cross-camera association worse, and the CircleLoss-only Experiment B showed that when CircleLoss is definitely active on the primary ViT recipe it fails catastrophically (**18.45% mAP, `inf` loss every epoch**). That means the augoverhaul regression is most likely driven by the augmentations themselves unless the earlier training had a CircleLoss config mismatch. The key lesson remains that **mAP != MTMC IDF1** for CityFlowV2 vehicle tracking: the MTMC graph needs features with clean, thresholdable similarity distributions, not just strong validation ranking.

At the same time, **09l v3 changes the ensemble outlook materially**. The weak secondary-model paths are still dead ends: **ResNet101-IBN-a** topped out at **52.77% mAP**, the original **FastReID SBS R50-IBN** path reached only **63.64%**, the newer **10c v61** deployment of the improved **09p** R50-IBN secondary still produced only a **+0.0006** MTMC IDF1 gain over baseline, **ViT-Small/16 IN-21k** reached only **48.66%**, **EVA02 ViT-B/16 CLIP** reached only **48.17%**, and **ResNeXt101-IBN-a ArcFace** collapsed to **36.88%** because the available pretrained weights were structurally incompatible. But the **CLIP-family secondary path** is now validated in one specific form: **LAION-2B CLIP 09l v3** reached **78.61% mAP / 90.43% R1 / 81.09% mAP_rerank**, only **1.53pp** behind the deployed **OpenAI CLIP 09b v2** baseline at **80.14% mAP**. That gives the repo its first genuinely strong, ensemble-ready secondary model. On the person side, ground-plane tracking is already strong (**90.3% MODA, 94.7% IDF1**), but rerunning tracking on the stronger **12a v3** detector stayed essentially flat at **90.0% MODA, 94.7% IDF1**, indicating that the current WILDTRACK pipeline is now tracker-limited rather than detector-limited. The remaining vehicle work is no longer about more association sweeps or more single-model feature tweaks; it is now the direct evaluation of score-level fusion with a strong secondary backbone.

## Pending

- **09q v5** pending: Exp B complete at mAP=76.52% (no improvement). Exp A never ran (checkpoint path bug fixed; 09q v5 push pending).
- **Association re-tune** for camera_bn=true features: current 10a v5 baseline 76.625% vs expected 77.36% (−0.74pp). Fresh Stage-4 sweep needed.
- ~~10c v9~~ **COMPLETE** — 3-way ensemble dead end confirmed: 76.625% baseline, 76.817% best ensemble.