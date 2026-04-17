# Paper Table Corrections Spec

> Cross-reference audit of paper tables against `docs/findings.md`, `docs/experiment-log.md`, and `.github/copilot-instructions.md`.
> Generated: 2026-04-17

---

## 1. `paper/tables/ablation.tex` — Table I: Cumulative Vehicle Ablation

### Corrections Needed

| Row | Field | Current Value | Correct Value | Source |
|-----|-------|--------------|---------------|--------|
| 6 | Source | `10c v75 overlap-off ablation` | `10c v39 (code v75) overlap-off ablation` | experiment-log: 10c v39 maps to code v75; "10c v75" conflates the two version numbers |

### TODOs That Cannot Be Filled

Rows 1–7 have `TODO` in the IDF1, MOTA, and HOTA columns. The experiment log only records MTMC IDF1 for intermediate ablation points — per-camera IDF1, MOTA, and HOTA were not logged at each cumulative step. **These TODOs can only be resolved by re-running the ablation with full metric logging**, or by dropping those columns from the table.

### Minor Arithmetic Note

Row 6: The cumulative narrative jumps from 77.6% (row 5) + Δ=+0.9pp to 78.0%, but 77.6 + 0.9 = 78.5%. The footnote correctly explains this discrepancy (temporal overlap was an ON/OFF ablation, not a clean cumulative step). No correction needed — the footnote is adequate.

---

## 2. `paper/tables/dead_ends.tex` — Table II: Dead-End Catalog

### Corrections Needed

| Row | Field | Current Value | Correct Value | Source |
|-----|-------|--------------|---------------|--------|
| 8 (Tracker max\_gap=80) | Actual IDF1 | `74.5\%` | `75.0\%` | experiment-log §2.1: 10c v42 = 75.0%, REJECTED (-3.0pp) from v41=78.2% |
| 10 (Denoise preprocessing) | Actual IDF1 | `TODO` | `75.7\%` | experiment-log §2.1: 10c v46 (code v82) = 75.7%, REJECTED (-2.7pp) |
| 15 (max\_iou\_distance=0.5) | Actual IDF1 | `TODO` | `76.8\%` | experiment-log §2.1: 10c v47 (code v83) = 76.8%, REJECTED (-1.6pp) |
| 16 (PCA 512D) | Actual IDF1 | `TODO` | `77.5\%` | experiment-log §2.1: 10c v35 = 77.5%, REJECTED (-0.78pp) |

### TODOs That Cannot Be Filled From Current Data

| Row | Field | Notes |
|-----|-------|-------|
| 4 (mtmc\_only=true) | Actual IDF1 | Only the delta (-5pp) is documented; no absolute IDF1 value in the experiment log. If the baseline is 77.5%, the implied value is ~72.5%. |
| 11 (FAC feature augmentation) | Actual IDF1 | Only the delta (-2.5pp) is documented. No absolute value in the experiment log. |
| 13 (Feature concatenation) | Actual IDF1 | Only the delta (-1.6pp) is documented from experiment-log §3.4. No absolute value. |

### Remove TODO Comment

The file header has `% TODO: verify exact numbers and final wording against docs/findings.md and the experiment log.` — this audit is now complete; remove or replace with `% Verified against docs/findings.md and experiment-log.md on 2026-04-17.`

---

## 3. `paper/tables/person_pipeline.tex` — Table IV: WILDTRACK Results

### Corrections Needed

| Subtable | Row | Field | Current Value | Correct Value | Source |
|----------|-----|-------|--------------|---------------|--------|
| (b) Tracking | Naive baseline | MODA | `TODO` | `88.6\%` | findings.md §12b v3: naive result MODA=88.6% (from the 59-config comprehensive sweep) |
| (b) Tracking | Naive baseline | IDSW | (missing) | `12` | findings.md §12b v3: naive IDSW=12 |

### Consistency Note: Naive Baseline Source

The table cites `12b v9` (IDF1=92.8%) for the naive baseline but `12b v3` (IDF1=91.2%) for the global optimal tracker. These use **different detections** (12b v9 used older 12a v26 detections; 12b v3 used newer 12a v3 detections). For a fair comparison using the same detection source (12a v3), the naive baseline from 12b v3 reached **IDF1=93.25%** with **MODA=88.6%** and **IDSW=12**.

**Recommendation**: Either (a) update the naive row to use the 12b v3 result (93.25%, same detections as the other trackers) for a fair comparison, or (b) add a footnote explaining the detection source difference.

### Remove TODO Comment

The file header has `% TODO: verify exact source references...` — remove or mark verified.

---

## 4. `paper/tables/sota_comparison.tex` — Table V: SOTA Comparison

### Corrections Needed

| Subtable | Row | Field | Current Value | Correct Value | Source |
|----------|-----|-------|--------------|---------------|--------|
| (a) CityFlowV2 | Ours | Relative | `91.1\%` | `91.3\%` | 77.5 / 84.86 = 0.9133 = 91.3% |

### Remove TODO Comment

The file header has `% TODO: verify literature values...` — remove or mark verified.

---

## 5. `paper/main.tex` — Inline Prose Numbers

### Corrections Needed

| Location | Current Text | Correct Text | Reason |
|----------|-------------|-------------|--------|
| Title | `91\% of SOTA` | `91\% of SOTA` | **No change needed** — "91%" is acceptable rounding of 91.3% |
| Abstract, line ~35 | `91.1\% of the 5-model-ensemble state-of-the-art (84.86\%)` | `91.3\% of the 5-model-ensemble state-of-the-art (84.86\%)` | 77.5 / 84.86 = 91.33% ≈ 91.3% |
| Abstract, line ~37 | `three independent experiments show improved mAP degrading cross-camera association by 1.4--5.3pp` | `two independent experiments show improved mAP degrading cross-camera association by 1.4--5.3pp, while a third shows that preserving mAP at higher resolution still degrades MTMC by 2.8pp` | Only augoverhaul (+1.45pp mAP) and DMT (+7pp mAP) actually improved mAP. The 384px deployment preserved the same mAP (80.14%) rather than improving it. |
| Introduction, ~line 60 | `91.1\% of the 2022 challenge leader` | `91.3\% of the 2022 challenge leader` | Same calculation as above |
| Conclusion, paragraph 1 | `91.1\% of the 2022 state of the art` | `91.3\% of the 2022 state of the art` | Same calculation |

### Numbers Verified as Correct

- 77.5% MTMC IDF1 ✓
- 84.86% SOTA ✓
- 225+ association configs ✓
- 0.3pp variation band ✓
- CSLS -34.7pp ✓
- SAM2 -8.7pp ✓
- AFLink -3.8pp to -13.2pp ✓
- 384px -2.8pp ✓
- augoverhaul degradation 1.4–5.3pp ✓
- 94.7% person IDF1 ✓ (99.4% of SOTA ✓)
- 7.36pp gap ✓
- 80.1% IDF1, 67.1% MOTA, 58.1% HOTA for v52 ✓
- 78.4% historical best ✓
- 92.1% person detector MODA ✓
- 59 person tracker configs ✓
- 45 fragmented GT IDs, 27 conflated pred IDs, 199 ID switches ✓
- per-camera S01: 0.925, 0.900, 0.914; S02: 0.638, 0.762, 0.582 ✓ (matches experiment-log Run B)
- network flow 76.9% vs 77.14% CC baseline ✓
- conflation 27→30 ✓
- DMT 87.3% mAP → 75.8% MTMC (Δ=-1.4pp vs v45=77.2%) ✓
- augoverhaul 81.59% mAP → 72.2% MTMC ✓
- 384px range 75.6–75.9% ✓

---

## 6. `.github/copilot-instructions.md` — Key Performance Numbers

### Corrections Needed

| Section | Current Value | Correct Value | Source |
|---------|--------------|---------------|--------|
| Person Pipeline | `Best Ground-plane MODA: 0.903 (12b v14)` | `Best Ground-plane MODA: 0.900 (12b v1, v2, v14)` | findings.md: all converged runs report MODA=90.0%, not 90.3% |

### Numbers Verified as Correct

- Best Reproducible MTMC IDF1: 0.775 (10c v52) ✓
- Historical Best: 0.784 (v80/v44) ✓
- SOTA: 0.8486 ✓
- Gap: 7.36pp ✓
- Primary model mAP=80.14%, R1=92.27% ✓ (this is the deployed model, not the augoverhaul)
- Secondary model mAP=52.77% ✓
- Person IDF1=0.947 ✓
- Person SOTA: 0.953 ✓
- Person gap: 0.6pp ✓
- Association configs: 225+ ✓

---

## 7. Cross-Document Consistency Issues

### "Primary Model" mAP ambiguity

- `copilot-instructions.md` says: "Primary model mAP=80.14%" (the deployed 09b v2 baseline)
- `findings.md` header table says: "Primary Model (ViT-B/16 CLIP 256px, 09 v2 aug overhaul): mAP=81.59%"
- The **deployed** model is the 09b v2 baseline at 80.14%. The augoverhaul model (81.59%) was never deployed because it regressed MTMC to 72.2%.
- **Recommendation**: Update findings.md to clarify the deployed vs. trained distinction, or relabel the 81.59% entry as "augoverhaul variant (not deployed)".

### Experiment Log Header Staleness

- `experiment-log.md` header says: `Current best (local, recent): MTMC IDF1 = **77.7%** (10c v28, CamTTA + power_norm=0.5)`
- This is outdated — the current reproducible best is 77.5% from 10c v52 (without CamTTA).
- **Recommendation**: Update the header to reflect the current v52 result.

---

## Summary of All Changes

| File | Changes | Severity |
|------|---------|----------|
| `paper/tables/dead_ends.tex` | Fix 1 wrong value (75.0% not 74.5%), fill 3 TODOs, remove header TODO comment | **High** — factual error |
| `paper/main.tex` | Fix 91.1%→91.3% in 3 locations, clarify "three experiments" claim in abstract | **High** — appears in title/abstract |
| `paper/tables/sota_comparison.tex` | Fix Relative 91.1%→91.3%, remove header TODO | **Medium** |
| `paper/tables/ablation.tex` | Clarify source for row 6, acknowledge unfillable TODOs | **Low** |
| `paper/tables/person_pipeline.tex` | Fill naive MODA/IDSW, address detection-source inconsistency, remove header TODO | **Medium** |
| `.github/copilot-instructions.md` | Fix MODA 0.903→0.900 | **Low** |
| `docs/experiment-log.md` | Update stale header | **Low** |
| `docs/findings.md` | Clarify "Primary Model" label | **Low** |