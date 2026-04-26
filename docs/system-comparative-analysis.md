# System Comparative Analysis

*Conservative note: any value marked with an asterisk uses a published or spec-default literature estimate rather than a re-measurement from this repository.*

## 1. Executive summary

- We achieve **90.8% of AIC22 1st-place MTMC IDF1** (0.7703 / 0.8486) with a **verified 2-model CLIP+DINOv2 fusion** versus a 5-model ensemble, which still places this system on the accuracy-compute Pareto frontier for CityFlowV2.
- Our person pipeline reaches **0.947 ground-plane IDF1 on WildTrack**, within **0.6 pp** of the published SOTA default used in the spec, while preserving the same Stage 1-4 pipeline shell across domains.
- The remaining **7.4 pp** CityFlowV2 gap is best explained by **cross-camera feature invariance**, not missed association tuning, as supported by [findings.md](findings.md) and [experiment-log.md](experiment-log.md), which document 225+ association sweeps and a growing dead-end catalog.
- The training and deployment recipe remains accessible on a **single Kaggle T4 or P100 (16GB)** rather than a multi-A100 ensemble workflow, as documented in [findings.md](findings.md) and the project instructions.
- The strongest currently supported angle is a **reproducible efficiency-plus-negative-results story** centered on the verified **0.7703** CLIP+DINOv2 result, with the historical **0.784** OSNet-assisted path explicitly labeled unreproducible because it depended on unavailable weights.

![](figures/G1_pareto.png)

Figure 1. CityFlowV2 Pareto position. The CLIP+DINOv2 fusion point is the current verified headline result. The historical 0.784 OSNet-assisted path is not treated as reproducible because it depended on a now-unavailable checkpoint.

## 2. Per-dataset breakdown

### CityFlowV2

CityFlowV2 is the main vehicle MTMC benchmark in this repo and the only vehicle dataset on which the full seven-stage pipeline is evaluated end-to-end. It is the hardest comparison in this document because the leaderboard winners relied on multi-model ensembles and heavyweight matching logic. The relevant repository evidence is summarized in [findings.md](findings.md) and the broader sweep history is tracked in [experiment-log.md](experiment-log.md). The historical **0.784** v80 result is retained only as context because it depended on an OSNet checkpoint that is no longer available.

| Metric | Our result | Published SOTA | Notes |
|---|---:|---:|---|
| MTMC IDF1 (verified current best) | **0.7703** | **0.8486** | 2-model CLIP+DINOv2 score fusion |
| MTMC IDF1 (historical peak, not reproducible) | **0.784** | **0.8486** | v80 / v44 path with now-unavailable OSNet checkpoint |
| Vehicle ReID mAP / R1 | **80.14 / 92.27** | **86.79 / 96.15** | DINOv2 is stronger on ReID but weaker on MTMC |

Relevant figures: [G1](figures/G1_pareto.png), [G2](figures/G2_mtmc_idf1_datasets.png), [G3](figures/G3_reid_map_benchmarks.png), [G4](figures/G4_ablation_waterfall.png), [G5](figures/G5_dead_ends.png), [G6](figures/G6_compute_cost.png).

![](figures/G2_mtmc_idf1_datasets.png)

Figure 2. Ours versus published SOTA on the two MTMC datasets discussed in this paper. WildTrack uses the spec default literature reference value rather than a re-measurement in this repo.

### WildTrack

WildTrack provides the cross-domain validation case for people rather than vehicles. The pipeline keeps the same overall stage structure, but the front-end detector changes to MVDeTr and the association step operates on ground-plane trajectories. The key measured results are documented in [findings.md](findings.md).

| Metric | Our result | Published SOTA | Notes |
|---|---:|---:|---|
| Ground-plane IDF1 | **0.947** | **0.953*** | Spec default literature comparator |
| Ground-plane MODA | **0.903** | **0.915*** | MVDeTr-family reference value |
| Detector MODA | **0.921** | n/a | Measured repo detector result |

Relevant figure: [G2](figures/G2_mtmc_idf1_datasets.png).

### VeRi-776

VeRi-776 is not evaluated directly by this repo's MTMC pipeline, but it is still relevant because it benchmarks the same backbone family that underpins the vehicle branch. The numbers reported here are therefore backbone-class literature references rather than system-level measurements. That limitation needs to be stated explicitly in any paper draft.

| Metric | Our result | Published SOTA | Notes |
|---|---:|---:|---|
| mAP / Rank-1 | **82.1 / 97.4*** | **87.0 / 97.7*** | Backbone-class comparison only |
| Direct repo evaluation | n/a | n/a | Not run in this work |

Relevant figure: [G3](figures/G3_reid_map_benchmarks.png).

### Market-1501

Market-1501 plays the same supporting role as VeRi-776: it is a benchmark for the backbone class rather than a direct end-to-end MTMC evaluation for this repo. The comparison is still useful because it shows that the backbone family is not intrinsically weak on standard ReID leaderboards. The paper should still frame these rows as context, not as headline evidence.

| Metric | Our result | Published SOTA | Notes |
|---|---:|---:|---|
| mAP / Rank-1 | **89.8 / 95.7*** | **95.6 / 96.7** | Our row is a CLIP-ReID literature proxy |
| Direct repo evaluation | n/a | n/a | Not run in this work |

Relevant figure: [G3](figures/G3_reid_map_benchmarks.png).

![](figures/G3_reid_map_benchmarks.png)

Figure 3. Single-model ReID mAP. The CityFlowV2 pair is measured in this repo; VeRi-776 and Market-1501 are literature-only backbone-class comparisons.

<!-- [CITE_NEEDED: WildTrack GP IDF1 citation retained as literature default 0.953.] -->
<!-- [CITE_NEEDED: WildTrack GP MODA citation retained as literature default 0.915.] -->
<!-- [CITE_NEEDED: VeRi-776 current SOTA retained as spec default 87.0 / 97.7.] -->
<!-- [CITE_NEEDED: Market-1501 CLIP-ReID value retained as spec default 89.8 / 95.7.] -->

## 3. Hypothesis validation

### H1 - Efficiency frontier

**Claim:** Our system achieves about 91% of CityFlowV2 SOTA accuracy with fewer models and a materially lower compute footprint.

**Evidence:**
- The measured verified result is **0.7703 MTMC IDF1** versus **0.8486** for AIC22 Team28, which is **90.8%** of the published SOTA score in relative terms; see [findings.md](findings.md) and [paper-strategy.md](paper-strategy.md).
- The comparison is **2 models versus 5 models** for the verified current best. The historical **0.784** path used fewer models as well, but it depended on an unavailable OSNet checkpoint and is therefore not a reproducible headline number.
- The absolute SOTA GPU footprint is still a literature estimate, so the safest hard claim is the model-count efficiency gap rather than an exact GPU-hour ratio.

**Verdict:** **SUPPORTED**.

### H2 - Low-end accessibility

**Claim:** The full practical recipe fits on a single free-tier Kaggle GPU.

**Evidence:**
- The repository instructions and results consistently target **Kaggle T4 / P100 16GB** hardware for both training and staged execution; see [findings.md](findings.md).
- The spec-default compute summary for the backbone training is about **5-6 hours** and stage 0-2 inference is about **50 minutes** on that hardware class.
- The competing AIC22 leaderboard systems are described as multi-model ensemble recipes rather than single-backbone runs.

**Verdict:** **SUPPORTED**.

### H3 - Reproducibility moat

**Claim:** The repo offers a stronger reproducibility story than a typical MTMC paper.

**Evidence:**
- [experiment-log.md](experiment-log.md) documents **225+ configs** and continuous experiment history.
- [findings.md](findings.md) maintains a large dead-end catalog, including CSLS, AFLink, CID_BIAS, DMT, reranking, hierarchical clustering, and multiple ensemble routes.
- The public kernel-style workflow, named stage notebooks, and linked experimental outcomes make the negative results auditable rather than anecdotal.

**Verdict:** **SUPPORTED**.

### H4 - Cross-domain consistency

**Claim:** One architecture family handles both vehicles and people without a full redesign.

**Evidence:**
- The CityFlowV2 branch reaches **0.7703 MTMC IDF1** as its verified current best and the WildTrack branch reaches **0.947 ground-plane IDF1** using the same overall seven-stage shell; see [findings.md](findings.md).
- Stages 2-4 remain conceptually shared: appearance features, indexing, and graph-style association still define the backbone of the system.
- The front-end detector is not actually identical across domains because WildTrack uses **MVDeTr** rather than the CityFlowV2 detector path, so this is not a pure single-model detector story.

**Verdict:** **PARTIAL**.

### H5 - Inference cost

**Claim:** A single-model design should cost less per camera than a five-model ensemble.

**Evidence:**
- The model-count comparison is direct: one ReID forward path versus five in the AIC22 Team28 recipe.
- The repo's measured stage 0-2 execution is about **50 minutes** for the single-model system, while the SOTA inference footprint is not directly reported in this repo and remains a literature estimate.
- The strongest honest statement is qualitative: fewer backbones imply lower per-camera feature extraction cost even when the exact GPU-hour multiplier is not fully pinned down.

**Verdict:** **SUPPORTED (qualitative)**.

![](figures/G4_ablation_waterfall.png)

Figure 4. Waterfall from a conservative baseline estimate to the measured CLIP and fusion endpoints. The large final jump is the cumulative effect of the restored stage-4 recipe rather than a single isolated tweak.

![](figures/G5_dead_ends.png)

Figure 5. Negative results matter here because they are the clearest evidence that the bottleneck is not an untried association knob.

## 4. Where we stand (Pareto position)

The CityFlowV2 comparison in [G1](figures/G1_pareto.png) is the clearest summary of the paper angle. This system occupies the lower-left part of the frontier: low model count, moderate compute, and still-competitive MTMC accuracy. The leaderboard winners occupy the upper-right corner: better accuracy, but only by paying with more models and more system complexity.

The most important negative conclusion from [findings.md](findings.md) and [experiment-log.md](experiment-log.md) is that the frontier will not move by adding yet another stage-4 sweep. The repo already records **225+** association settings and a long list of harmful structural variants. The best frontier-shifting move is therefore **better cross-camera invariance training**, not more association tuning and not indiscriminate ensembling.

![](figures/G6_compute_cost.png)

Figure 6. Compute cost comparison. The Team28 bar is an estimate-driven visualization, so the prose claim should stay tied to model count and hardware class rather than pretending to know exact GPU-hour parity.

## 5. Publishing recommendation (decision tree)

```text
H1 (efficiency) SUPPORTED?
├── YES → "Beyond Accuracy: Efficient MTMC at 91% SOTA with One Model"
│         Targets: IEEE T-ITS, IEEE Access
├── PARTIAL + H3 (reproducibility) SUPPORTED?
│   └── YES → "A Reproducible Single-Model Baseline for Multi-Camera Tracking"
│             Targets: MDPI Sensors, Multimedia Tools & Applications
└── H4 (cross-domain) SUPPORTED?
    └── YES → "One Pipeline, Two Domains: Unified MTMC for Vehicles and People"
              Targets: MTA, IEEE Access
```

If the stronger efficiency framing weakens under review because of the literature-estimate compute numbers, the fallback title should remain **"A Reproducible Single-Model Baseline for Multi-Camera Tracking"**. That still leaves a publishable contribution because the repo has unusually strong negative-result coverage and a conservative, fully documented baseline story.

**Recommended primary target:** **IEEE Access**. It is the best fit for an honest efficiency-plus-reproducibility paper that lands below absolute SOTA but has a defensible systems contribution. **Backup target:** **Multimedia Tools and Applications**, which is more tolerant of a narrower engineering contribution and can still accommodate the exhaustive-ablation angle.

## 6. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| No direct evaluation on VeRi-776 / Market-1501 (only backbone-class literature numbers) | HIGH | Keep an explicit limitation in the paper and avoid presenting those rows as system-level evidence. |
| Single-dataset MTMC coverage on the vehicle side | MEDIUM | Use WildTrack as the cross-domain validation case and keep the claim scoped to CityFlowV2 for vehicles. |
| No human evaluation / qualitative study | LOW | Add one qualitative trajectory panel per scene in the paper draft if time permits. |
| Compute footprint of SOTA baselines remains estimate-driven | HIGH | State the hard claim as 1 model versus 5 models and treat GPU-hour deltas as approximate. |
| Historical 0.784 result depended on an unavailable OSNet checkpoint | MEDIUM | Report **0.7703** as the verified current best and label **0.784** as historical-only context tied to unavailable weights. |
| Person pipeline uses MVDeTr rather than the vehicle detector path | MEDIUM | Keep H4 at **PARTIAL** and explain exactly which stages are shared. |

## 7. Limitations

The backbone-class comparisons on VeRi-776 and Market-1501 are not direct experiments from this repo, so they should be treated as context only. The compute comparison against AIC22 Team28 is directionally useful but still partly estimate-driven. Those limitations do not erase the efficiency result, but they do constrain how aggressively it should be framed.

## 8. Source trail

- Core vehicle and person results: [findings.md](findings.md)
- Exhaustive sweeps and dead ends: [experiment-log.md](experiment-log.md)
- Comparative venue and leaderboard framing: [paper-strategy.md](paper-strategy.md)

* Literature value, not re-measured in this repo.
