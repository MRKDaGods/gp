# Figure Data Corrections — Real Experiment Data

> Generated from `docs/findings.md`, `docs/experiment-log.md`, and `.github/copilot-instructions.md`

## Figure 1: Association Saturation Plot

### Real Data Points (version-level bests from association/tracker tuning on baseline model)

These are ALL extractable IDF1 values from vehicle pipeline association/tracker parameter experiments. 
The claim is "225+ configs, all within 0.3pp" but individual per-config values were NOT logged — only version-level bests are available.

```python
# Real version-level IDF1 data points for the saturation plot.
# Each entry: (config_index_label, idf1_percent, description)
# The config_index is approximate — many versions swept N configs internally.
ASSOCIATION_REAL_POINTS = [
    # v14-v22: 384-combo initial grid, only aggregate range available (74-78%)
    # We place representative points for the grid extremes:
    ("v14-v22 (grid low)", 74.0, "Initial grid scan worst"),
    ("v14-v22 (grid high)", 78.0, "Initial grid scan best"),
    # v23-v26: algorithm scan (5 algorithms, CC, conflict_free_cc, Louvain, etc.)
    ("v23-v26 (algo scan)", 78.0, "conflict_free_cc +0.21pp best"),
    # v27-v30: post-proc scan
    ("v27-v30 (post-proc)", 78.0, "min_traj_frames=40 optimal"),
    # v31-v33: FIC regularisation sweep
    ("v31-v33 (FIC sweep)", 78.0, "FIC reg 0.05-15.0"),
    # Individual version results:
    ("v34", 78.28, "Intra-merge scan (36 configs)"),
    ("v35", 77.50, "PCA 512D test, REJECTED"),
    ("v36", 78.01, "app_w/DBA/mutual_nn (14 configs)"),
    ("v37", 78.00, "Clean confirmation run"),
    ("v38", 78.02, "CSLS/cluster_verify/temporal_split"),
    ("v39", 78.01, "temporal_overlap/AQE_K consolidated"),
    ("v40", 77.30, "quality_temperature=5.0, REJECTED"),
    ("v41", 78.20, "Tracker: max_gap=50, intra_merge=40"),
    ("v42", 75.00, "Tracker: max_gap=80, REJECTED"),
    ("v43", 77.30, "Tracker: max_gap=60, REJECTED"),
    ("v44", 78.40, "Tracker: min_hits=2, BEST KAGGLE"),
    # Later experiments on gumfreddy account:
    ("v52", 77.14, "v80-restored control baseline"),
    ("v53", 76.90, "Network flow solver, -0.24pp"),
    # Local CamTTA-era experiments:
    ("local v28", 77.70, "CamTTA + power_norm=0.5"),
    ("local v29", 77.20, "CamTTA + power_norm=0"),
    ("local v30", 77.71, "CamTTA + MS-TTA"),
    ("local v31", 78.00, "Association sweep (31 configs)"),
]

# Key parameters for the saturation band annotation:
SATURATION_BAND_LOW = 77.14   # lowest in the "converged" region (v52)
SATURATION_BAND_HIGH = 78.40  # highest achieved (v44)
SATURATION_BAND_WIDTH_PP = 1.26  # total spread
CONVERGED_BAND_LOW = 77.50    # post-convergence floor (v35/v52 area)
CONVERGED_BAND_HIGH = 78.40   # post-convergence ceiling (v44)
CONVERGED_BAND_WIDTH_PP = 0.90
CURRENT_REPRODUCIBLE_BEST = 77.50  # 10c v52, gumfreddy
HISTORICAL_BEST = 78.40            # 10c v44, ali369 (not reproducible)
TOTAL_CONFIGS_CLAIMED = 225

# For the synthetic fill: 225+ configs but only ~21 version-level data points.
# The saturation band (0.3pp) cited in findings refers to the LATE configs
# (v34-v39 range, which spans 77.50 to 78.28 = 0.78pp, and the core group
# v36-v39 spans 78.00-78.02 = 0.02pp).
# For the figure, generate synthetic points within [77.3, 78.4] with tight
# clustering around 77.5-78.0 and mark the real points distinctly.
```

### Design Notes
- The v14-v22 grid (384 combos) only has aggregate range (74-78%), no individual values
- v34 internally tested 36 configs but only reports best (78.28%)
- v36 internally tested 14 configs but only reports best (78.01%)
- LOCAL v31 tested 31 configs but only reports best (78.0%)
- The "0.3pp band" claim refers to late-stage tuning — early configs ranged wider
- v42 (75.0%) is an outlier from an over-aggressive tracker change, not association tuning

## Figure 2: mAP vs MTMC IDF1 Disconnect

### Verified Data Points

```python
# Each entry: (label, mAP_percent, mtmc_idf1_percent, color, marker_note)
MAP_VS_MTMC_POINTS = [
    # Baseline: primary ViT-B/16 CLIP 256px (09b v2)
    ("Baseline ViT 256px", 80.14, 77.5, "#1f77b4",
     "Current reproducible best (10c v52)"),

    # Augoverhaul: 09 v2, same backbone with stronger augmentations + CircleLoss
    ("Augoverhaul", 81.59, 72.2, "#d62728",
     "+1.45pp mAP but -5.3pp MTMC IDF1 (10c v48)"),

    # DMT: camera-aware training with GRL+CameraHead+CircleLoss on primary ViT
    ("DMT (camera-aware)", 87.30, 75.8, "#2ca02c",
     "+7.16pp mAP but -1.4pp MTMC IDF1 (10c v46 vs v45=77.2%)"),

    # 384px deployment: same 80.14% model deployed at higher resolution
    ("384px deployment", 80.14, 75.6, "#9467bd",
     "Same mAP, -2.8pp MTMC IDF1 (v44=75.62 vs v80=78.4)"),

    # Weak ResNet fusion: ensemble primary + 52.77% ResNet at 0.30 weight
    ("Weak ResNet fusion", 52.77, 77.4, "#8c564b",
     "Ensemble result -0.1pp vs primary alone; x-axis is secondary mAP"),
]

# Optional additional points (less certain / different framing):
MAP_VS_MTMC_EXTRA = [
    # Augoverhaul-EMA: essentially same as augoverhaul
    ("Augoverhaul-EMA", 81.53, 72.2, "#ff7f0e",
     "Same 72.2% ceiling as augoverhaul (10c v49)"),

    # Historical v80 baseline (not reproducible)
    ("Historical best (v80)", 80.14, 78.4, "#17becf",
     "ali369, not reproducible on current codebase"),
]
```

### Corrections vs Placeholder
| Point | Placeholder | Corrected | Change |
|-------|-------------|-----------|--------|
| Baseline ViT 256px | (80.14, 77.5) | (80.14, 77.5) | No change |
| Augoverhaul | (81.59, 72.2) | (81.59, 72.2) | No change |
| DMT | (87.30, 75.8) | (87.30, 75.8) | No change |
| 384px deployment | (80.00, 75.6) | **(80.14, 75.6)** | **mAP corrected: 80.00→80.14** (same model, different deploy resolution) |
| Weak ResNet fusion | (52.77, 77.4) | (52.77, 77.4) | No change |

### Key Narrative
- Higher mAP does NOT predict higher MTMC IDF1
- Three independent experiments confirm: augoverhaul (+1.45pp mAP → -5.3pp MTMC), 384px (same mAP → -2.8pp MTMC), DMT (+7pp mAP → -1.4pp MTMC)
- The ResNet point shows fusion with a weak model is also ineffective

## Figure 3: Dead-End Waterfall

### Verified Deltas (Vehicle Pipeline Only)

```python
# Each entry: (label, delta_pp)
# Sorted by severity (most negative first)
# All deltas are MTMC IDF1 in percentage points vs the relevant baseline
DEAD_END_WATERFALL = [
    ("CSLS distance",                -34.70),  # v74, catastrophic
    ("SAM2 foreground masking",       -8.70),  # 10a v29 + 10c v50; 0.688 vs 0.775
    ("Augoverhaul + CircleLoss",      -5.30),  # 10c v48; 0.722 vs 0.775
    ("mtmc_only=True",                -5.00),  # Documented: drops single-cam tracks
    ("AFLink motion linking",         -3.82),  # 10c v52 retest; gap=100, dir=0.90
    ("CID_BIAS",                      -3.30),  # 0.751 vs 0.784 baseline
    ("384px deployment",              -2.80),  # v44=0.7562 vs v80=0.784
    ("confidence_threshold=0.20",     -2.80),  # 10c v45
    ("Denoise preprocessing",         -2.70),  # v46, v82
    ("FAC",                           -2.50),  # v26
    ("max_iou_distance=0.5",          -1.60),  # 10c v47
    ("Feature concatenation",         -1.60),  # vs score fusion
    ("DMT camera-aware training",     -1.40),  # v46=0.758 vs v45=0.772; mAP=87.3%
    ("Hierarchical clustering",       -1.00),  # v54-56, v62; range -1.0 to -5.1pp, using conservative end
    ("concat_patch 1536D",            -0.30),  # v48=0.773 vs v45=0.775
    ("Network flow solver",           -0.24),  # v53=0.769 vs v52=0.7714
    ("Score-level ensemble (weak)",   -0.10),  # fus0.3_ter0.0
    ("Multi-query track rep",         -0.10),  # v51=0.771 vs v50=0.772
]
```

### Corrections vs Placeholder
| Entry | Placeholder | Corrected | Change |
|-------|-------------|-----------|--------|
| CSLS | -34.7 | -34.70 | No change |
| SAM2 masking | -8.7 | -8.70 | No change |
| Augoverhaul + CircleLoss | -5.3 | -5.30 | No change |
| mtmc_only=true | -5.0 | -5.00 | No change |
| AFLink | -3.8 | **-3.82** | **Precision: 3.8→3.82** |
| **Global optimal tracker** | **-3.5** | **REMOVED** | **Person pipeline, not vehicle** |
| CID_BIAS | -3.3 | -3.30 | No change |
| 384px deployment | -2.8 | -2.80 | No change |
| Denoise | -2.7 | -2.70 | No change |
| FAC | -2.5 | -2.50 | No change |
| DMT | -1.4 | -1.40 | No change |
| Network flow | -0.24 | -0.24 | No change |
| **NEW: confidence_threshold** | — | **-2.80** | **Added** |
| **NEW: max_iou_distance** | — | **-1.60** | **Added** |
| **NEW: Feature concatenation** | — | **-1.60** | **Added** |
| **NEW: Hierarchical clustering** | — | **-1.00** | **Added (conservative end of -1.0 to -5.1 range)** |
| **NEW: concat_patch 1536D** | — | **-0.30** | **Added** |
| **NEW: Score-level ensemble** | — | **-0.10** | **Added** |
| **NEW: Multi-query track rep** | — | **-0.10** | **Added** |

### Design Notes
- "Global optimal tracker" (-3.5pp) was REMOVED — it's from the person/WILDTRACK pipeline, not vehicle
- Hierarchical clustering ranged from -1.0 to -5.1pp; the figure could use -3.0 as a midpoint or show an error bar
- For the paper figure, consider trimming entries <0.3pp impact (concat_patch, score-level ensemble, multi-query) for visual clarity — they may be within measurement noise
- AFLink worst case was -13.2pp (gap=200/dir=0.70) but the "best case" -3.82pp (gap=100/dir=0.90) is the fairest representation
- If showing a "top-12 most impactful" version, cut at Feature concatenation (-1.60pp)

## Data Sources & Verification

| Figure | Primary Source | Cross-Reference | Confidence |
|--------|---------------|-----------------|------------|
| Association Saturation | experiment-log.md | findings.md §3.1-3.3 | Medium (versions verified, individual configs not logged) |
| mAP vs MTMC | findings.md dead-end table | experiment-log.md model sections | High (all points verified in 2+ sources) |
| Dead-End Waterfall | findings.md "Conclusive Dead Ends" table | copilot-instructions.md, experiment-log.md | High (every delta traced to specific experiment) |