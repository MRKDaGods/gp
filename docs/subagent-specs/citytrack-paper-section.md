# Spec: CityTrack (AIC22 Winner) Comparison Section — Paper Edit + Doc Updates

> Read-only research is complete. This spec tells the implementer exactly what LaTeX to insert, where, which bib entry to add, and which research docs to update. No code/flag changes — the 3 new components (SSA, BT, occlusion) stay default-off; baseline MTMC IDF1 0.77936 is protected.

## 1. Target paper file & insertion point

- **File to edit:** `gp__Copy_/chapters/ch5_testing.tex` (the MTMC thesis, `book` document class via `gp__Copy_/main.tex`).
  - NOT the root `main.tex` — that is a separate IEEE VeRi-776 ReID paper with no MTMC content.
- **Insertion point:** at the END of the file, immediately AFTER the existing `\subsection{Association Hyperparameter Ablation}` block (after the `\end{figure}` that closes `\label{fig:ablation}`, currently the last line ~line 142).
- **New subsection title:** `\subsection{Comparison to the AIC22 Winner: Method Completeness and Component Ablation}`
- **Table convention to match (verified in this chapter):** plain `tabular` + `\hline` (the thesis preamble loads only `geometry, setspace, graphicx, hyperref, tocloft` — it does NOT load `booktabs` or `pifont`, so do NOT use `\toprule`/`\midrule`/`\cmark`). Labels follow `\label{tab:...}` / `\label{fig:...}` naming (e.g. `tab:mtmc_results`, `fig:ablation`).

## 2. LaTeX to insert (append after the ablation subsection)

```latex
\subsection{Comparison to the AIC22 Winner: Method Completeness and Component Ablation}

The AI~City Challenge 2022 Track~1 winner (CityTrack, Team~28~\cite{yang2022citytrack})
reports 84.86\% IDF1 using a five-model ensemble and seven engineering components.
Our single-model system reaches 77.94\% IDF1, a gap of 6.93~percentage points.
To understand whether this gap is closable through association engineering or is
fundamentally feature-quality-limited, we audited all seven CityTrack components
against our pipeline and re-implemented the three that were genuinely absent.
Table~\ref{tab:citytrack_audit} summarises the audit.

\begin{table}[htbp]
\centering
\caption{Audit of our pipeline against the seven components of the AIC22 Track~1
winner (CityTrack). ``Enabled'' = active in the production configuration;
``Implemented (off)'' = present but disabled because it was found neutral or harmful;
``Added (off)'' = newly implemented for this study behind a default-off flag.}
\label{tab:citytrack_audit}
\begin{tabular}{p{0.32\textwidth}p{0.22\textwidth}p{0.34\textwidth}}
\hline
\textbf{CityTrack component} & \textbf{Status in our system} & \textbf{Notes} \\ \hline
1. Zone-based search-space reduction & Implemented (off) & Soft $\pm0.03$ similarity bonus, not hard pruning; $-0.4$~pp when enabled. \\
2. Spatio-temporal time window & Enabled & Stronger variant: hard gate outside window $+$ Gaussian score inside, vs.\ their $\times 2$ penalty. \\
3. Stationary Sensitive Association (SSA) & Added (off) & Freezes stationary boxes to recent high-confidence detections. \\
4. Trajectory Re-Link (TRL) & Partial (intra on) & Intra-camera ReID-cosine relink enabled; cross-camera motion relink disabled ($-3.8$ to $-13.2$~pp). \\
5. Bidirectional tracking (BT) & Added (off) & Forward $+$ backward passes merged by IoU$+$ReID gating. \\
6. Occlusion-aware distance matrix & Added (off) & Per-tracklet occlusion flag applies a similarity penalty to occluded pairs. \\
7. Box-grained matching $+$ $k$-reciprocal & Implemented (off) & Mean-pooled embeddings $+$ conflict-free connected components by default; $k$-reciprocal and box-grained multi-query both hurt our 280D-PCA features. \\ \hline
\end{tabular}
\end{table}

Four of the seven components were already present (two enabled, two implemented but
disabled after being found harmful with our current features). The three genuinely
missing components --- SSA, bidirectional tracking, and the occlusion-aware distance
matrix --- were re-implemented behind default-off configuration flags and validated
with 84 unit tests, then ablated on a three-camera CityFlowV2 subset (scene S02:
cameras c006/c007/c008). Table~\ref{tab:citytrack_ablation} reports this ablation.
We stress that the S02-subset IDF1 is \emph{not} comparable to the full-dataset
77.94\% headline; it is an internal diagnostic on a hard scene used only to rank the
new components against one another.

\begin{table}[htbp]
\centering
\caption{Component ablation on the CityFlowV2 S02 three-camera subset
(c006/c007/c008). Metric is subset MTMC IDF1; deltas are relative to the
fresh-feature baseline. ID switches (IDsw) are reported as a reliability check.
The standalone occlusion row is shown struck through because it was measured on
\emph{cached} Stage-2 features and is not comparable; the fair occlusion estimate
is (all~$-$~BT) $= -1.03$~pp.}
\label{tab:citytrack_ablation}
\begin{tabular}{lccc}
\hline
\textbf{Configuration} & \textbf{S02 IDF1 (\%)} & \textbf{$\Delta$ (pp)} & \textbf{IDsw} \\ \hline
Baseline (no new components) & 61.79 & ---    & 43  \\
SSA only                     & 61.97 & $+0.18$ & 43  \\
Bidirectional only           & 66.89 & $+5.10$ & 149 \\
SSA $+$ BT $+$ Occlusion     & 65.87 & $+4.08$ & 142 \\
Occlusion only (cached feats) & \textit{68.29} & \textit{($+6.50$)} & \textit{100} \\ \hline
\end{tabular}
\end{table}

The ablation yields three honest verdicts. \textbf{SSA is neutral}: $+0.18$~pp is
within run-to-run noise and the ID-switch count is unchanged (43). \textbf{Bidirectional
tracking is the only component with a sizeable subset gain} ($+5.10$~pp), but it
triples local ID switches (43~$\rightarrow$~149), which means the apparent IDF1 gain
is partly recovered detections offset by new identity fragmentation; it therefore
requires full-dataset confirmation before it can be trusted, and remains default-off.
\textbf{The occlusion-aware matrix is neutral-to-negative under a fair comparison}:
its eye-catching standalone $+6.50$~pp was a measurement artifact --- that run reused
cached Stage-2 features (its ID switches jumped 43~$\rightarrow$~100 even though
occlusion is a pure Stage-4 operation that cannot change upstream tracking), whereas
the baseline recomputed features fresh. On an equal-footing comparison the occlusion
contribution is (all~$-$~BT)~$=65.87-66.89=-1.03$~pp. We deliberately do not claim the
$+6.50$~pp as a result.

Taken together, the audit supports our central finding: the 6.93~pp gap to the AIC22
winner is \emph{not} attributable to missing association machinery. Six of seven
components are present or were added, and none of the three additions produces a
confirmed, feature-fair full-pipeline improvement. The dominant remaining lever is
feature quality --- specifically the gap between our single TransReID ViT-B/16 model
and CityTrack's five-model ensemble --- consistent with the five-axis association
plateau established in the preceding ablation.
```

## 3. Bibliography

- **Bib file:** `gp__Copy_/references.bib`.
- **Situation:** there is NO CityTrack/Team28 entry. The existing `li2022aicity` ("Multi-Camera Vehicle Tracking with Powerful Visual Features and Spatial-Temporal Cues") is a different AICity paper and its author list looks unverified; do NOT reuse it for CityTrack. `tang2019cityflow` (CityFlow benchmark) already exists and is fine for dataset citations.
- **Action:** add a new entry with key `yang2022citytrack`. Title, venue, and year are verified from the CVF open-access page; the FULL AUTHOR LIST must be confirmed against the official CVF PDF before camera-ready (the snippet below uses a placeholder author field flagged for verification).

```bibtex
@inproceedings{yang2022citytrack,
  title     = {Box-Grained Reranking Matching for Multi-Camera Multi-Target Tracking},
  author    = {Yang, Fan and others},  % TODO: verify full author list from CVF PDF before camera-ready
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), AI City Challenge},
  pages     = {3096--3106},
  year      = {2022}
}
```

Source: https://openaccess.thecvf.com/content/CVPR2022W/AICity/html/Yang_Box-Grained_Reranking_Matching_for_Multi-Camera_Multi-Target_Tracking_CVPRW_2022_paper.html

## 4. Verified code locations (for the audit table accuracy)

All `file:line` confirmed on disk on branch `paper-tests`:

| # | Component | Status | Code location | Config flag |
|---|-----------|--------|---------------|-------------|
| 1 | Zone reduction | Impl (off) | `src/stage4_association/zone_scoring.py:1`; wired `src/stage4_association/pipeline.py:619` | `stage4.association.zone_model.enabled: false` |
| 2 | Spatio-temporal window | Enabled | `src/stage4_association/spatial_temporal.py:62`; applied `src/stage4_association/similarity.py:189` | `stage4.association.spatiotemporal.*` |
| 3 | SSA | Added (off) | `src/stage1_tracking/ssa.py:56` (`apply_ssa`); wired `src/stage1_tracking/pipeline.py:210` | `stage1.ssa.enabled: false` (`configs/datasets/cityflowv2.yaml:93`) |
| 4 | TRL | Partial | intra-cam `src/stage4_association/pipeline.py:808` (on); cross-cam `src/stage4_association/aflink.py:48` (off) | `intra_camera_merge` (on); `aflink.enabled: false` |
| 5 | Bidirectional tracking | Added (off) | `src/stage1_tracking/bidirectional.py:21` (`run_backward_pass`) + `src/stage1_tracking/bidirectional_merge.py:187` (`merge_bidirectional`); wired `src/stage1_tracking/pipeline.py:191` and `src/stage2_features/pipeline.py:760` | `stage1.bidirectional.enabled: false` (`configs/datasets/cityflowv2.yaml:99`) |
| 6 | Occlusion-aware matrix | Added (off) | `src/stage4_association/occlusion.py:33` (`compute_tracklet_occlusion`); wired `src/stage4_association/pipeline.py:160`; penalty applied `src/stage4_association/similarity.py:257` (`score *= occ_penalty`) | `stage4.association.occlusion_aware.enabled: false`, `occ_box_thresh: 0.6` (`configs/datasets/cityflowv2.yaml:217`) |
| 7 | Box-grained + k-reciprocal | Impl (off) | `src/stage4_association/reranking.py:30` (k-recip); `src/stage4_association/pipeline.py:84` (multi-query box-level) | `reranking.enabled: false`; `multi_query.enabled: false` |

Note: our occlusion penalty is applied in SIMILARITY space (`score *= occ_penalty`, penalty $<1$ for occluded pairs) — the analog of CityTrack's distance-space $D_{\text{final}} = D \times (1 + 0.1\,\mathbb{I}(\text{occ}\geq0.6))$.

Full prior evidence: `docs/subagent-specs/citytrack-audit-evidence.md`.

## 5. Research docs to update

### `docs/findings.md`
Add a dated bullet under the strategic narrative:
> **Update 2026-05-31 — CityTrack 7-component audit + S02 ablation.** Audited our pipeline against the AIC22 Track-1 winner (CityTrack, Team28, 0.8486 IDF1). 4/7 components already present (zone soft-bonus off, temporal window on/stronger, intra-cam TRL on, k-recip/box-grained off-harmful). Implemented the 3 missing ones behind default-off flags (commit b5aef3e, branch paper-tests, 84 tests pass): SSA (`src/stage1_tracking/ssa.py`), BT (`src/stage1_tracking/bidirectional.py` + `bidirectional_merge.py`), occlusion (`src/stage4_association/occlusion.py`). S02 3-cam subset ablation: SSA NEUTRAL (+0.18pp, IDsw unchanged), BT +5.10pp but IDsw 43→149 (needs full-dataset confirm), occlusion fair delta = (all−BT) = −1.03pp (standalone +6.50pp was a cached-feature artifact, discarded). Conclusion reinforced: 6.93pp gap is feature-quality (single vs 5-model ensemble), not association machinery. All 3 remain default-off; 0.77936 protected.

### `docs/what-worked.md`
No new confirmed win. Add a short note under "Tracking / Stage 1" or a new subsection so it is not mistaken for a win:
> - **Bidirectional tracking (BT)**: +5.10pp on S02 3-cam SUBSET only, but tripled local ID switches (43→149) — NOT a confirmed win; default-off pending full-dataset confirmation.

### `docs/dead-ends.md`
Add under "Association / Stage 4" (and a Stage-1 line):
> - **Occlusion-aware distance matrix (CityTrack #6)**: fair full-pipeline delta = (all−BT) = −1.03pp on S02 subset; standalone +6.50pp was a cached-feature measurement artifact (IDsw 43→100 from a pure Stage-4 op = upstream feature mismatch). Neutral-to-negative; default-off.
> - **SSA / Stationary Sensitive Association (CityTrack #3)**: +0.18pp on S02 subset = within noise, IDsw unchanged (43). Neutral; default-off. (Our Stage-5 stationary FILTER is the opposite operation and stays enabled.)

### `docs/performance-state.md`
Add under the Vehicle "Plateau confirmation" area:
> 6. **CityTrack 7-component audit (2026-05-31)**: 4/7 components already covered; 3 missing ones (SSA, BT, occlusion) added default-off behind flags. S02-subset ablation: SSA neutral, BT +5.10pp (IDsw tripled, unconfirmed), occlusion −1.03pp fair. No feature-fair full-pipeline gain; gap remains ensemble-vs-single-model feature-quality-limited. Baseline 0.77936 unchanged.

## 6. Risks & rollback

- Paper edits are fully reversible (git revert of the ch5_testing.tex + references.bib hunks).
- The new `yang2022citytrack` bib entry must have its full author list verified from the CVF PDF before camera-ready; the placeholder `{Yang, Fan and others}` is flagged inline.
- No code or config flags change in this task — SSA/BT/occlusion stay default-off; the protected baseline (MTMC IDF1 0.77936) is untouched.
- The S02-subset numbers must always be presented as a diagnostic subset, never conflated with the full-dataset 0.77936 headline (the paper prose and table caption both state this explicitly).
- Do NOT claim occlusion's standalone +6.50pp as a result anywhere — it is a cached-feature artifact.