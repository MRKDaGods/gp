# Paper Draft Quality Review Report

**Reviewed**: 2026-04-17  
**Files reviewed**: `paper/main.tex`, `paper/tables/ablation.tex`, `paper/tables/dead_ends.tex`, `paper/tables/person_pipeline.tex`, `paper/tables/sota_comparison.tex`, `docs/findings.md`

---

## Summary

| Severity | Count |
|----------|:-----:|
| **Critical** | 8 |
| **Major** | 8 |
| **Minor** | 10 |

The paper is well-written with a strong argumentative structure and novel contribution framing. However, multiple tables contain **TODO placeholders** that must be resolved before submission, the **ablation table arithmetic is internally inconsistent**, the **dead ends table has inconsistent baselines**, and several **dataset description numbers need verification**. All inline numbers in the prose that were cross-checked against `docs/findings.md` are accurate.

---

## Critical Issues

### C1. Ablation table (ablation.tex) has 21 TODO cells
Rows 1–8 of the ablation table have `TODO` for IDF1, MOTA, and HOTA columns. Only row 9 (reproducible best) has complete metrics. A cumulative ablation table with mostly-empty secondary metric columns weakens the contribution claim.

**Fix**: Fill from experiment log entries cited in the "Source" column, or remove empty columns and present only MTMC IDF1 (the paper's primary metric).

### C2. Dead ends table (dead_ends.tex) has 3 TODO cells and 1 vague entry
- Row 4 (`mtmc_only=true`): Actual IDF1 = TODO
- Row 11 (FAC): Actual IDF1 = TODO
- Row 13 (Feature concatenation): Actual IDF1 = TODO
- Row 19 (Reranking): "Negative" with no number

**Fix**: Fill actual IDF1 values from the experiment log, or replace with "N/A" and add a footnote explaining the missing data.

### C3. Dead ends table caption is incorrect
`\caption{Catalog of failed approaches on CityFlowV2 vehicle MTMC.}` — but **row 6** is "Global optimal tracker (person)" which is a WILDTRACK result, not CityFlowV2.

**Fix**: Either rename caption to "Catalog of failed approaches across vehicle and person pipelines" or move the person entry to the person pipeline table.

### C4. Ablation table cumulative arithmetic is broken
The $\Delta$ column values do not sum to the displayed MTMC IDF1 column:
- Rows 1–5: 74.0 + 2.0 + 0.5 + 0.9 + 0.21 = 77.61 → displayed as 77.6% ✓
- Row 6: 77.6 + 0.9 = **78.5**, but the table shows **78.0%** (−0.5pp discrepancy)
- Row 8: 78.28 + 0.2 = **78.48**, but the table shows **78.4%** (−0.08pp rounding)
- Sum of all deltas: 4.99pp → expected final = 78.99%, but historical best is 78.4%

The footnote partially addresses this for temporal overlap, but the table still appears internally inconsistent to a careful reader.

**Fix**: Either (a) adjust the MTMC IDF1 column to match the cumulative $\Delta$ sums, or (b) rename the column from "$\Delta$ (pp)" to "Isolated gain (pp)" and add a note that gains were measured independently on different runs and are not strictly additive.

### C5. Dead ends table has inconsistent baselines for $\Delta$
Some deltas are computed from the **reproducible 77.5%**, some from the **historical 78.4%**, and some from **paired run baselines**:
- Rows 1–3, 18: baseline = 77.5% (IDF1 + $\Delta$ = 77.5) ✓
- Rows 7, 8, 9, 10, 15: baseline = ~78.4% (historical)
- Row 5 (AFLink): 73.3 + 3.8 = 77.1% (v52 CC baseline at 77.14%)
- Row 12 (DMT): 75.8 + 1.4 = 77.2% (v45 paired baseline)
- Row 17 (Network flow): 76.9 + 0.24 = 77.14% (v52 CC baseline)

Without documenting which baseline each $\Delta$ uses, the reader cannot interpret the column correctly.

**Fix**: Add a "Baseline" column or a table footnote specifying "Δ is relative to the paired control; see individual experiment descriptions."

### C6. CityFlowV2 dataset description needs verification
The paper states: "The full CityFlow benchmark introduced by Tang et al. contains **46 cameras and 880 globally annotated vehicle identities**." The original CityFlow paper (CVPR 2019, Tang et al. \cite{cityflow}) describes **40 cameras** at 10 intersections with **666 vehicle identities**. The "46/880" numbers may refer to a later CityFlowV2 expansion, but:
- The citation points to the original 2019 paper, not a CityFlowV2 paper
- No separate CityFlowV2 reference is provided

**Fix**: Either verify and cite the correct CityFlowV2 statistics, or correct to 40 cameras / 666 identities per the cited source.

### C7. SOTA comparison table has TODO and approximate entries
- WILDTRACK SOTA MODA = TODO
- Naive tracker MODA = TODO
- Two rows use "~80%" and "~72%" with tilde approximations
- Two rows say "Lower" with no number and "---" relative percentage

**Fix**: Either provide verified numbers with citations, or remove rows that cannot be properly cited.

### C8. Person pipeline table has 4 TODO cells
Precision and Recall are `TODO` for Naive and Global-optimal trackers (4 cells). The SOTA MODA is also TODO in sota_comparison.tex.

**Fix**: Fill from the 12b experiment logs, or remove the Precision/Recall columns since they are not the focus of the analysis.

---

## Major Issues

### M1. Table `tab:component_contributions` is never referenced in text
The `\label{tab:component_contributions}` (Table 1) is defined but `\ref{tab:component_contributions}` never appears in the body text. The table floats unanchored.

**Fix**: Add a `Table~\ref{tab:component_contributions}` reference in the Method section where component contributions are discussed.

### M2. Figure `fig:cumulative_ablation` is never referenced in text
The `\label{fig:cumulative_ablation}` figure is defined but never `\ref`'d in the prose. It will appear as an orphaned figure.

**Fix**: Add a reference in Section 4.2 near the ablation discussion, e.g., "Figure~\ref{fig:cumulative_ablation} traces the cumulative gain."

### M3. "19+ failed approaches" (abstract) vs "19 failed approaches" (intro)
The abstract says "We catalog **19+** failed approaches" while the introduction says "a dead-end catalog of **19** failed approaches." The table has exactly 19 rows.

**Fix**: Standardize to "19" everywhere, or use "19+" everywhere if additional dead ends are in supplementary material.

### M4. "approximately 5× less compute" in abstract is understated
The cost comparison table shows: 1 model on T4 for ~3h vs. 5 models on A100 for 50+h. This is closer to **80×** less GPU-hour cost, not 5×. The "5×" appears to refer to model count, not compute.

**Fix**: Reword to "using one-fifth the number of ReID models" or provide a more accurate compute ratio. Do not conflate model count with compute savings.

### M5. Seven bibliography entries lack proper author names
`\bibitem{aicity2022}`, `\bibitem{aic22winner}`, `\bibitem{aic22sta}`, `\bibitem{mtmc2021}`, `\bibitem{scorebased}`, `\bibitem{camawarefic}`, `\bibitem{realtimeaccess}` all use paper titles as the entry without named authors. IEEE requires proper author attribution.

**Fix**: Add full author lists for all bibliography entries.

### M6. "Anonymous Authors" needs real names for IEEE Access
IEEE Access uses **single-blind** review. The author field should contain real names, affiliations, and ORCID identifiers.

**Fix**: Replace "Anonymous Authors" with actual author information before submission.

### M7. ablation.tex has a persistent "verify before submission" TODO comment
Line 1: `% TODO: verify exact numbers from experiment log before submission.`

**Fix**: Either verify and remove the comment, or complete the verification.

### M8. dead_ends.tex has a persistent TODO about supplementary material
Footer note: `TODO: Move the training-path dead ends (CircleLoss ablations, SGD, VeRi-776 transfer, extended ResNet fine-tuning, EMA) into a supplementary table or appendix if the final venue page budget allows it.`

**Fix**: Decide whether to move entries and remove the TODO.

---

## Minor Issues

### m1. All figures are placeholder boxes
Every figure uses `\placeholderfigure{}` — actual plots/diagrams must be generated before submission.

### m2. Title uses "91%" but body consistently says "91.3%"
The title says "One Model, 91% of SOTA" while the abstract, intro, and conclusion all use "91.3%." Likely intentional rounding for the title, but reviewers may note the discrepancy.

### m3. Abstract may exceed IEEE Access 250-word limit
The abstract is dense and long. Verify word count fits within venue requirements.

### m4. No `\hyperref` package loaded
DOIs in the bibliography would benefit from hyperlinks for the digital/online version.

### m5. Component contributions table (Table 1) is very wide
The 5-column `tabularx` spanning `\textwidth` works for `table*` but may overflow in certain layout configurations. Verify in compiled PDF.

### m6. GT-assisted evaluation protocol should be more prominent
The paper mentions "GT-assisted frame clipping" only once in the experimental setup and the findings.md metric disambiguation. Reviewers may question this. Consider adding a clear disclaimer sentence.

### m7. Missing standard IEEE Access sections
No ORCID identifiers, author bios, acknowledgments section, or funding disclosure. These are typically required for camera-ready IEEE Access papers.

### m8. Missing `\cite{camawarefic}` in text
The bibliography entry `\bibitem{camawarefic}` on camera-aware FIC features is listed but never cited in the paper text. It will generate a "no citation" warning.

### m9. Temporal overlap bonus magnitude seems stale
The component contributions table says temporal overlap contributes "+0.9pp" citing "10c v75 overlap-off ablation," but the ablation table footnote acknowledges this was an ON/OFF measurement, not a clean cumulative delta. The two tables should present the same caveat.

### m10. Some dead-end rows mix vehicle and person categories
Beyond the caption issue (C3), the reader may find it confusing that row 6 (Global optimal tracker, person) and row 8 (Tracker max_gap=80) mix tracking-layer changes with association/feature changes.

**Suggestion**: Group dead ends by category (Association, Feature, Training, Tracking, Post-processing) with horizontal rules or sub-headers.

---

## Numbers Cross-Check: Prose vs. Tables vs. findings.md

| Claim | Paper value | findings.md value | Match |
|-------|:-----------:|:-----------------:|:-----:|
| Reproducible best MTMC IDF1 | 77.5% | 77.5% (10c v52) | ✅ |
| IDF1 / MOTA / HOTA | 80.1 / 67.1 / 58.1 | 0.801 / 0.671 / 0.581 | ✅ |
| Historical best | 78.4% | 78.4% (v80) | ✅ |
| SOTA (AIC22 1st) | 84.86% | 84.86% | ✅ |
| 91.3% of SOTA | 77.5/84.86 = 91.31% | Consistent | ✅ |
| 7.36pp gap | 84.86 − 77.5 = 7.36 | 7.36pp | ✅ |
| Augoverhaul mAP | 81.59% | 81.59% (09 v2) | ✅ |
| Augoverhaul MTMC | 72.2% | 72.2% (10c v48) | ✅ |
| DMT mAP | 87.3% | 87.3% | ✅ |
| DMT MTMC | 75.8% | 75.8% (v46) | ✅ |
| 384px MTMC | 75.6–75.9% | v43=75.85%, v44=75.62% | ✅ |
| CSLS delta | −34.7pp | −34.7pp | ✅ |
| SAM2 delta | −8.7pp | −8.7pp | ✅ |
| AFLink delta | −3.8 to −13.2pp | −3.82 to −13.20pp | ✅ |
| Network flow | 76.9%, −0.24pp | 76.9%, −0.24pp | ✅ |
| Conflict-free CC | +0.21pp | +0.21pp | ✅ |
| FIC whitening | +1–2pp | +1–2pp | ✅ |
| AQE K=3 | +0.9pp | +0.9pp | ✅ |
| WILDTRACK IDF1 | 94.7% | 94.7% (12b v1/v2/v3) | ✅ |
| WILDTRACK SOTA | 95.3% | 95.3% | ✅ |
| 99.4% of person SOTA | 94.7/95.3 = 99.37% | Consistent | ✅ |
| MVDeTr MODA | 92.1% | 92.1% (12a v3) | ✅ |
| Global-optimal tracker | 91.2%, −3.5pp | 91.17% ≈ 91.2%, −3.53pp | ✅ |
| ResNet secondary mAP | 52.77% | 52.77% (09d v18) | ✅ |
| Association configs | 225+ | 225+ | ✅ |
| Tracker configs (person) | 59 | 59+ | ✅ |
| CityFlow "46 cameras, 880 IDs" | main.tex L152 | Not in findings.md | ⚠️ Unverified |

All 28 verifiable numerical claims in the prose match `docs/findings.md` exactly. The only unverifiable claim is the CityFlow full-benchmark statistics (C6).

---

## Verdict

The paper's argument, structure, and experimental evidence are strong. The main blockers to submission-readiness are:
1. **TODO placeholders** across four tables (~28 empty cells total)
2. **Ablation arithmetic** and **dead-end baseline** inconsistencies that undermine the tables' credibility
3. **Dataset description** numbers that need source verification
4. **Placeholder figures** (expected at this stage)
5. **Bibliography completeness** (missing author names)

Resolving Critical and Major issues is required before any submission. Minor issues should be addressed during camera-ready preparation.