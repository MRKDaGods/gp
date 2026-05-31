# VeRi-776 Wave 3 Paper Sync — Implementation Spec

Updates `main.tex` to reflect final Wave 3 ablation + seed-variance measurements and fixes the tie-band logic in `scripts/paper/aggregate_paper_results.py`. Optionally pushes a slim Kaggle feature-dump kernel to unblock retrieval-panel generation.

**Inputs (already on disk, do not regenerate):**
- `experiments/veri776-paper/results.csv` / `results.json` / `REPORT.md`
- `experiments/veri776-paper/{A1,A2,A3,A4,A5alpha,A5beta,S1_seed42,S2_seed123,S3_seed456}/eval_results.json`
- A5α / A5β / S1–S3 checkpoints at `experiments/veri776-paper/<exp>/checkpoint.pth` (5 valid 330 MB files)

**Headline fusion result UNCHANGED:** 93.30 % mAP / 98.45 % R1 (two-stream).

**Tie-band verdict:** A5β single = 88.94 mAP; A5α seeds mean (n=3) = 88.89 ± 0.54 mAP. |Δ| = 0.05 pp → deep inside 0.10 pp tie band. **A5α (deployed) remains headline.** No Wave 4.

---

## ⚠️ Numbering note (read first)

The user's request refers to "Table V". The on-disk paper has the **ablation table labelled `tab:ablation` at line 631, which is Table VI** (counting in order: I `tab:augmentation`, II `tab:hyperparams`, III `tab:implementation`, IV `tab:sota`, V `tab:vit_compare`, **VI `tab:ablation`**, VII `tab:fusion_sweep`, VIII `tab:pp_contrib`, IX `tab:qual`). The `--/--` placeholders the user described live in `tab:ablation`, so **all "Table V" instructions in this spec target `tab:ablation` (rendered as Table VI)**. The coder must not insert a fresh "Table V" — only edit `tab:ablation`.

---

## Task A — `main.tex` edits

File: `e:\dev\src\gp\main.tex` (root, NOT `paper/main.tex`).

### A.1 — Resolve the §3.2.2 TODO and add the tie-band footnote

Located in subsubsection "Loss Function" (`\subsubsection{Loss Function}`, line 312), inside the paragraph that ends with `CircleLoss was evaluated but produced…`.

**Before (lines 339–342):**
```latex
delayed) and layer-wise learning-rate decay 0.75. A5$\alpha$ and A5$\beta$ are
therefore evaluated as separate ablation rows; the headline Stream~1 recipe is
the one with higher measured mAP under the held evaluation. \textcolor{red}{TODO:
insert A5 winner identity after Wave~3.} CircleLoss was evaluated but produced
```

**After:**
```latex
delayed) and layer-wise learning-rate decay 0.75. A5$\alpha$ and A5$\beta$ are
therefore evaluated as separate ablation rows. The Wave~3 ablation campaign
measured A5$\alpha$ at 88.74\% mAP and A5$\beta$ at 88.94\% mAP under
stream-1-only AQE+rerank evaluation (\,$|\Delta|\!=\!0.20$\,pp single-run,
collapsing to $|\Delta|\!=\!0.05$\,pp once A5$\alpha$ is averaged over three
additional seeds; see Table~\ref{tab:ablation} and
Section~\ref{sec:seed_variance}). Because the difference falls well inside
the $\pm 0.54$\,pp three-seed standard deviation of A5$\alpha$, the two
recipes are statistically equivalent and A5$\alpha$---the actually deployed
checkpoint---is retained as the headline Stream~1 recipe.
CircleLoss was evaluated but produced
```

### A.2 — Rewrite `tab:ablation` (Table VI, line 628–650)

The current table mixes (a) the cumulative 82.3 → 93.3 trajectory and (b) a single isolated A5β placeholder row. The Wave 3 numbers require a new isolated-component block **above** the cumulative trajectory. Keep the cumulative trajectory intact (needed for the 82.3 → 93.3 narrative), but:
1. **Remove** the orphan placeholder row `Full (paper recipe, ... A5$\beta$) --- measured & -- & -- & -- \\`.
2. **Insert** a new "Isolated Component Ablation (Stream 1 only, AQE+rerank)" block before the TransReID baseline row.
3. Add a `\midrule` between the new block and the cumulative trajectory.

**Before (lines 628–650):**
```latex
\begin{table}[!t]
  \centering
  \caption{Stepwise Ablation: 82.3\% $\to$ 93.3\% mAP on VeRi-776}
  \label{tab:ablation}
  \begin{tabular}{L{4.2cm}C{1.1cm}C{1.0cm}C{1.2cm}}
    \toprule
    Configuration & mAP & R1 & $\Delta$ \\
    \midrule
    TransReID baseline (DeiT-B*)~\cite{he2021transreid} & 82.30 & 97.10 & --- \\
    Base cosine, ViT-B (no PP)                           & 82.22 & 97.50 & $-0.08$ \\
    \quad+AQE $k{=}3$                                    & 84.08 & 97.20 & $+1.86$ \\
    \quad+Re-rank $k_1{=}80,k_2{=}15,\lambda{=}0.2$     & 89.28 & 97.79 & $+5.20$ \\
    \quad+Concat-patch + flip TTA (1536D; A5$\alpha$, actual deployed recipe) & 89.97 & 97.80 & $+0.69$ \\
    Full (paper recipe, CE-LS + SupCon, LLRD=0.65; A5$\beta$) --- measured & -- & -- & -- \\
    \midrule
    CLIP-SENet v6, base cosine~\cite{lu2025clipsenet}    & 82.34 & 96.54 & --- \\
    \quad+AQE $k{=}10$                                   & 89.21 & 96.90 & $+6.87$ \\
    \quad+Re-rank $k_1{=}50,k_2{=}10,\lambda{=}0.1$     & 91.54 & 97.32 & $+2.33$ \\
    \midrule
    \textbf{Fusion ($w_1{=}0.3$, $w_2{=}0.7$)}          & \textbf{93.30} & \textbf{98.45} & $+1.76$ \\
    \bottomrule
  \end{tabular}
\end{table}
```

**After:**
```latex
\begin{table}[!t]
  \centering
  \caption{Stepwise Ablation: 82.3\% $\to$ 93.3\% mAP on VeRi-776.
    Rows A1--A5$\beta$ report isolated-component contributions measured
    under a common operating point (Stream~1 only, AQE $k{=}3$ +
    $k$-reciprocal re-ranking, single-flip 768D CLS); deltas are relative
    to baseline A1. The cumulative trajectory below combines components
    progressively and adds concat-patch flip TTA, CLIP-SENet, and fusion.}
  \label{tab:ablation}
  \begin{tabular}{L{4.2cm}C{1.1cm}C{1.0cm}C{1.2cm}}
    \toprule
    Configuration & mAP & R1 & $\Delta$ vs A1 \\
    \midrule
    \multicolumn{4}{l}{\emph{Isolated Component Ablation (Stream~1, AQE+rerank)}} \\
    A1 --- Baseline: DeiT + SGD + CE + Triplet            & 57.25 & 79.86 & --- \\
    A2 --- +CLIP init only                                & 71.55 & 87.43 & $+14.30$ \\
    A3 --- +SupCon only (DeiT + SGD + CE-LS + SupCon)    & 59.25 & 79.26 & $+2.00$ \\
    A4 --- +AdamW/LLRD only (DeiT + CE + Triplet)        & 81.35 & 94.70 & $+24.10$ \\
    A5$\alpha$ --- Full (CLIP + AdamW/LLRD=0.75 + CE-LS + Triplet + CenterLoss; \emph{deployed}) & 88.74 & 98.15 & $+31.49$ \\
    A5$\beta$ --- Full (CLIP + AdamW/LLRD=0.65 + CE-LS + SupCon; paper-described) & 88.94 & 97.14 & $+31.70$ \\
    \midrule
    \multicolumn{4}{l}{\emph{Cumulative Trajectory $\to$ Headline}} \\
    TransReID baseline (DeiT-B*)~\cite{he2021transreid} & 82.30 & 97.10 & --- \\
    Base cosine, ViT-B (no PP)                           & 82.22 & 97.50 & $-0.08$ \\
    \quad+AQE $k{=}3$                                    & 84.08 & 97.20 & $+1.86$ \\
    \quad+Re-rank $k_1{=}80,k_2{=}15,\lambda{=}0.2$     & 89.28 & 97.79 & $+5.20$ \\
    \quad+Concat-patch + flip TTA (1536D; A5$\alpha$, deployed) & 89.97 & 97.80 & $+0.69$ \\
    \midrule
    CLIP-SENet v6, base cosine~\cite{lu2025clipsenet}    & 82.34 & 96.54 & --- \\
    \quad+AQE $k{=}10$                                   & 89.21 & 96.90 & $+6.87$ \\
    \quad+Re-rank $k_1{=}50,k_2{=}10,\lambda{=}0.1$     & 91.54 & 97.32 & $+2.33$ \\
    \midrule
    \textbf{Fusion ($w_1{=}0.3$, $w_2{=}0.7$)}          & \textbf{93.30} & \textbf{98.45} & $+1.76$ \\
    \bottomrule
  \end{tabular}
\end{table}
```

Note: the $\Delta$ column header changes from `$\Delta$` to `$\Delta$ vs A1` since the new top block uses A1 as anchor. The cumulative trajectory deltas remain numerically meaningful as step-to-step deltas; the caption clarifies.

### A.3 — Update ablation discussion paragraph (immediately after the table, ~lines 652–657)

**Before:**
```latex
The most impactful single step is $k$-reciprocal re-ranking~\cite{zhong2017rerank}
on Stream~1 ($+5.20$\,pp). AQE~\cite{arandjelovic2012aqe} adds $+1.86$\,pp;
TTA contributes $+0.69$\,pp. Fusion provides $+1.76$\,pp above the stronger
post-processed CLIP-SENet stream, confirming genuine complementarity between
the two feature spaces.
```

**After:**
```latex
The isolated-component block clarifies the recipe drivers. AdamW with
layer-wise LR decay (A4) is the single largest training-recipe contributor,
adding $+24.10$\,pp mAP over the SGD baseline A1; CLIP initialisation (A2)
is the second largest at $+14.30$\,pp. SupCon in isolation (A3) barely
moves the needle ($+2.00$\,pp) but combines well once CLIP and AdamW are in
place: the full A5$\beta$ recipe (CLIP + AdamW + CE-LS + SupCon) reaches
88.94\% mAP, statistically indistinguishable from the deployed A5$\alpha$
recipe (CLIP + AdamW + CE-LS + Triplet + CenterLoss, 88.74\% mAP)---a
0.20\,pp single-run difference that collapses to 0.05\,pp once A5$\alpha$
is averaged over three additional seeds (Section~\ref{sec:seed_variance}).
Within the cumulative trajectory, the most impactful post-processing step
is $k$-reciprocal re-ranking~\cite{zhong2017rerank} on Stream~1
($+5.20$\,pp); AQE~\cite{arandjelovic2012aqe} adds $+1.86$\,pp; TTA
contributes $+0.69$\,pp. Fusion provides $+1.76$\,pp above the stronger
post-processed CLIP-SENet stream, confirming genuine complementarity
between the two feature spaces.
```

### A.4 — Add new §5.4 "Training Seed Variance" subsection

Insert immediately after the ablation discussion paragraph (after the `\end{table}` for tab:ablation and the discussion paragraph from A.3, i.e. before `\subsection{Fusion Weight Sensitivity}` at line 658).

**Insertion:**
```latex
\subsection{Training Seed Variance}
\label{sec:seed_variance}

To quantify training-time stochasticity, the deployed A5$\alpha$ recipe was
retrained from scratch with three additional seeds
$\{42, 123, 456\}$ under identical hyperparameters. Each run was evaluated
under the same Stream~1-only AQE+rerank operating point used for the
isolated-component ablation (Table~\ref{tab:ablation}, top block).

\begin{table}[!t]
  \centering
  \caption{A5$\alpha$ Training Seed Variance on VeRi-776
    (Stream~1 only, AQE $k{=}3$ + $k$-reciprocal re-ranking)}
  \label{tab:seed_variance}
  \begin{tabular}{lcc}
    \toprule
    Seed & mAP~(\%) & R1~(\%) \\
    \midrule
    42  & 89.52 & 97.79 \\
    123 & 88.53 & 97.38 \\
    456 & 88.64 & 97.20 \\
    \midrule
    \textbf{Mean $\pm$ std (n=3)} & \textbf{88.89 $\pm$ 0.54} & \textbf{97.46 $\pm$ 0.31} \\
    \bottomrule
  \end{tabular}
\end{table}

The three-seed standard deviation of 0.54\,pp mAP exceeds the 0.20\,pp
A5$\alpha$--A5$\beta$ single-run gap by more than $2\times$, confirming that
the two recipes are statistically equivalent under realistic training noise.
This variance is training-time variance only; once a checkpoint is fixed,
the evaluation pipeline is fully deterministic. The fusion headline of
93.30\% mAP / 98.45\% R1 uses the deployed A5$\alpha$ seed-0 checkpoint and
is therefore not subject to retraining variance.
```

This renumbers downstream subsections (Fusion Weight Sensitivity becomes §5.5, Post-processing Contribution Analysis §5.6, Qualitative Analysis §5.7). No `\label` changes are required because cross-references use `\ref{tab:...}` not `\ref{sec:5.4}`.

### A.5 — Update §6.4 Limitations item 4 (lines 816–824)

**Before:**
```latex
  \item An earlier draft of the Stream~1 training recipe described the metric
    loss and layer-wise learning-rate decay imprecisely. The present version
    corrects that disclosure by reporting isolated component ablations for
    CLIP initialisation (A2), SupCon versus triplet loss (A3), and AdamW with
    LLRD (A4), together with dual Full references: the actually deployed
    CE-LS + Triplet + CenterLoss recipe (A5$\alpha$) and the paper-described
    CE-LS + SupCon recipe (A5$\beta$). The headline result is selected by the
    higher measured mAP under the held evaluation, while the two-stream fusion
    cost and the VeRi-776--CityFlowV2 domain-gap constraint remain as noted
    above.
```

**After:**
```latex
  \item An earlier draft of the Stream~1 training recipe described the metric
    loss and layer-wise learning-rate decay imprecisely. The present version
    corrects that disclosure by reporting isolated component ablations for
    CLIP initialisation (A2), SupCon versus triplet loss (A3), and AdamW with
    LLRD (A4), together with dual Full references: the actually deployed
    CE-LS + Triplet + CenterLoss recipe (A5$\alpha$, 88.74\% mAP) and the
    paper-described CE-LS + SupCon recipe (A5$\beta$, 88.94\% mAP). The
    Wave~3 measurement places the two recipes within the 0.10\,pp tie band
    once A5$\alpha$ is averaged over three additional seeds
    (88.89 $\pm$ 0.54\,pp mAP), so A5$\alpha$---the deployed
    checkpoint---is retained as the headline Stream~1 recipe. The two-stream
    fusion cost and the VeRi-776--CityFlowV2 domain-gap constraint remain
    as noted above.
```

### A.6 — Headline-claim verification (no edits expected)

Confirm the following lines remain **unchanged** (fusion = 93.30 % / 98.45 % is correct as-is):
- Abstract, line 64: `98.45\% Rank-1}. This establishes a new state of the art in mAP on VeRi-776`
- §1.3 contribution C2, lines 141–145
- §5.1 SOTA table caption + paragraph (lines 583–590)
- §7 Conclusion, lines 836–842

If any of these have drifted (e.g. someone replaced 93.30 % with the stream-1 88.94 %), revert. No edit otherwise.

---

## Task B — `scripts/paper/aggregate_paper_results.py` tie-band fix

File: `e:\dev\src\gp\scripts\paper\aggregate_paper_results.py`.

Two functions change: `recommended_values` (currently uses single-run A5α vs A5β) and `recipe_outcome_lines` (currently prints the static `-0.21 pp` text).

### B.1 — Rewrite `recommended_values` (lines 472–518)

Replace the body so that the tie-band decision uses the A5α seed mean (n=3, ddof=1) instead of the A5α single seed-0 run. Add explicit thresholds:
- $|\Delta| \le 0.10$ pp → tied (A5α stays headline as the deployed recipe)
- $0.10 < |\Delta| \le 0.30$ pp → winner = whichever recipe is higher (A5α stays only if it wins)
- $|\Delta| > 0.30$ pp and A5β wins → **Wave 4 trigger** (record this in the returned dict as `"wave4_trigger": True`)

**Replacement function (full body, drop in place):**
```python
def recommended_values(records: list[ExperimentRecord]) -> dict[str, Any]:
    by_id = {record.exp_id: record for record in records}
    alpha_map = completed_metric(by_id, "A5alpha", "mAP")
    beta_map = completed_metric(by_id, "A5beta", "mAP")
    alpha_rank = completed_metric(by_id, "A5alpha", "rank1")
    beta_rank = completed_metric(by_id, "A5beta", "rank1")

    variance = seed_variance(records)
    seeds_mean_map = variance.get("mAP_mean")
    seeds_std_map = variance.get("mAP_std")
    seeds_mean_rank = variance.get("rank1_mean")
    n_seeds = variance.get("n_completed", 0) or 0

    # Use seeds-mean as the A5alpha reference when available; fall back to single seed-0.
    alpha_reference_map = seeds_mean_map if seeds_mean_map is not None else alpha_map
    alpha_reference_rank = seeds_mean_rank if seeds_mean_rank is not None else alpha_rank
    alpha_reference_source = "seeds_mean_n{}".format(n_seeds) if seeds_mean_map is not None else "single_seed0"

    tie_band = False
    wave4_trigger = False
    delta_beta_minus_alpha = None
    decision_reason = "insufficient_data"
    if alpha_reference_map is not None and beta_map is not None:
        delta_beta_minus_alpha = beta_map - alpha_reference_map
        abs_delta = abs(delta_beta_minus_alpha)
        if abs_delta <= 0.10:
            tie_band = True
            headline = "A5alpha"  # deployed recipe wins ties
            decision_reason = "tied_within_0.10pp"
        elif abs_delta <= 0.30:
            headline = "A5beta" if delta_beta_minus_alpha > 0 else "A5alpha"
            decision_reason = "winner_within_0.30pp"
        elif delta_beta_minus_alpha > 0:
            headline = "A5beta"
            wave4_trigger = True
            decision_reason = "wave4_trigger_beta_leads_over_0.30pp"
        else:
            headline = "A5alpha"
            decision_reason = "alpha_leads_over_0.30pp"
    elif alpha_reference_map is not None:
        headline = "A5alpha"
        decision_reason = "only_alpha_completed"
    elif beta_map is not None:
        headline = "A5beta"
        decision_reason = "only_beta_completed"
    else:
        headline = "none"

    fusion_candidates = [record for record in records if record.status == "completed" and record.data.get("fusion") is True]
    best_fusion = max(fusion_candidates, key=lambda record: float(record.data["mAP"]), default=None)
    fusion_map = float(best_fusion.data["mAP"]) if best_fusion is not None else None
    fusion_rank = float(best_fusion.data["rank1"]) if best_fusion is not None else None

    if headline == "A5alpha":
        stream_map = alpha_reference_map
        stream_rank = alpha_reference_rank
    elif headline == "A5beta":
        stream_map = beta_map
        stream_rank = beta_rank
    else:
        stream_map = None
        stream_rank = None

    return {
        "headline_recipe": headline,
        "stream1_full_mAP": stream_map,
        "stream1_full_rank1": stream_rank,
        "fusion_mAP": fusion_map,
        "fusion_rank1": fusion_rank,
        "tie_band_applies": tie_band,
        "wave4_trigger": wave4_trigger,
        "delta_beta_minus_alpha_mAP": delta_beta_minus_alpha,
        "alpha_reference_source": alpha_reference_source,
        "alpha_reference_mAP": alpha_reference_map,
        "alpha_reference_rank1": alpha_reference_rank,
        "alpha_seeds_std_mAP": seeds_std_map,
        "decision_reason": decision_reason,
    }
```

### B.2 — Rewrite `recipe_outcome_lines` (lines 544–562)

**Replacement function (full body):**
```python
def recipe_outcome_lines(by_id: dict[str, ExperimentRecord], recommended: dict[str, Any]) -> list[str]:
    alpha = by_id["A5alpha"]
    beta = by_id["A5beta"]
    lines = []

    delta = recommended.get("delta_beta_minus_alpha_mAP")
    source = recommended.get("alpha_reference_source", "single_seed0")
    alpha_ref = recommended.get("alpha_reference_mAP")
    alpha_std = recommended.get("alpha_seeds_std_mAP")
    headline = recommended.get("headline_recipe", "unknown")
    tie = bool(recommended.get("tie_band_applies"))
    wave4 = bool(recommended.get("wave4_trigger"))
    reason = recommended.get("decision_reason", "unknown")

    if alpha.status == "completed" and beta.status == "completed" and delta is not None and alpha_ref is not None:
        if source.startswith("seeds_mean"):
            ref_desc = f"A5alpha seeds mean ({source.replace('seeds_mean_', '')}) = {alpha_ref:.2f} +/- {alpha_std:.2f} pp" if alpha_std is not None else f"A5alpha seeds mean ({source.replace('seeds_mean_', '')}) = {alpha_ref:.2f} pp"
        else:
            ref_desc = f"A5alpha single seed-0 = {alpha_ref:.2f} pp"
        lines.append(f"Delta = {delta:+.2f} pp mAP (A5beta - {source}); A5beta single = {float(beta.data['mAP']):.2f} pp; {ref_desc}.")
        if tie:
            lines.append(f"Inside +/-0.10 pp tie band; A5alpha remains headline (deployed recipe). Reason: {reason}.")
        elif wave4:
            lines.append(f"OUTSIDE 0.30 pp tie band AND A5beta leads -> WAVE 4 TRIGGER. Reason: {reason}.")
        else:
            lines.append(f"Headline recipe: {headline}. Reason: {reason}.")
    elif alpha.status == "completed" or beta.status == "completed":
        completed = "A5alpha" if alpha.status == "completed" else "A5beta"
        missing = "A5beta" if completed == "A5alpha" else "A5alpha"
        lines.append(f"Only {completed} is completed; {missing} is not available for the alpha/beta decision.")
        lines.append("Tie band applies: false.")
    else:
        lines.append("Neither A5alpha nor A5beta is completed, so no recipe decision can be made yet.")
        lines.append("Tie band applies: false.")
    return lines
```

### B.3 — Expected REPORT.md output after fix

The "Recipe Disclosure Outcome" section should print (current data):
```
Delta = +0.05 pp mAP (A5beta - seeds_mean_n3); A5beta single = 88.94 pp; A5alpha seeds mean (n3) = 88.89 +/- 0.54 pp.
Inside +/-0.10 pp tie band; A5alpha remains headline (deployed recipe). Reason: tied_within_0.10pp.
```

The `recommended_paper_table_values` block in `results.json` should now include keys `wave4_trigger`, `delta_beta_minus_alpha_mAP`, `alpha_reference_source`, `alpha_reference_mAP`, `alpha_reference_rank1`, `alpha_seeds_std_mAP`, `decision_reason`.

### B.4 — Optional unit-test additions (not required for spec acceptance)

If the coder wants to harden this, add to `tests/test_paper/test_aggregate.py` (create if absent):
- `test_tie_band_within_0_10_keeps_alpha`: synthetic A5α seeds mean = 88.89, A5β = 88.94 → headline = "A5alpha", tie_band_applies = True, wave4_trigger = False.
- `test_wave4_trigger_when_beta_leads_0_5pp`: A5α seeds mean = 88.50, A5β = 89.10 → wave4_trigger = True, headline = "A5beta".
- `test_falls_back_to_single_seed_when_no_seeds`: no S1/S2/S3 records → uses A5α single seed-0; `alpha_reference_source` = "single_seed0".

---

## Task C — Retrieval panels (Wave 3b)

**Recommendation: Option 1 — push a slim Kaggle feature-dump kernel.** Justification:
- We already have 5 valid 330 MB checkpoints (A5α, A5β, S1, S2, S3); A5α is the deployed/headline.
- `scripts/paper/generate_retrieval_panels.py` already exists and only needs `features/{stream1,stream2}/{query,gallery}.npy` + `features/index_map.json` under `experiments/veri776-paper/A5alpha/features/`.
- Local extraction is blocked: GTX 1050 Ti is forbidden per `copilot-instructions.md` GPU rule, and the script expects the .npy files regardless.
- The qualitative table `tab:qual` (line 724) currently claims "Full retrieval panel figures are available in the project repository (see supplementary materials)" — this is **false** today and a reviewer-visible defect.
- Cost: ~30 min on a single T4, no training, minimal Kaggle quota.

### C.1 — Kernel sketch

Create `notebooks/kaggle/veri776_feature_dump.ipynb` (built from a new `_build_veri776_feature_dump.py` following the existing `_build_verifier_kernel.py` pattern). Kernel must:

1. Mount the A5α checkpoint as a Kaggle dataset input (re-use the existing `veri776-paper-checkpoints` dataset or whichever holds `A5alpha/checkpoint.pth`).
2. Mount the VeRi-776 dataset (existing `veri776` Kaggle dataset).
3. Mount the CLIP-SENet stream-2 checkpoint (existing dataset, same as 14t / 14u kernels).
4. Build Stream 1 model exactly as 14p (TransReID ViT-B/16 CLIP, SIE=20 cams, JPM groups=4, BNNeck, 768D CLS) and load A5α weights.
5. Build Stream 2 model exactly as 14t (CLIP-SENet ResNet101-IBN-a + TinyCLIP, 2048D).
6. For each of (stream1, stream2), iterate query (1678) and gallery (11579) splits with batch_size=64, flip TTA for stream1 (concat-patch 1536D), single-pass for stream2. Save:
   - `features/stream1/query.npy` (1678 × 1536 float16)
   - `features/stream1/gallery.npy` (11579 × 1536 float16)
   - `features/stream2/query.npy` (1678 × 2048 float16)
   - `features/stream2/gallery.npy` (11579 × 2048 float16)
   - `features/index_map.json` — list of dicts with `{row, image_path, vehicle_id, camera_id, split}` aligned to the .npy row order. Must satisfy the `Item` dataclass in `scripts/paper/generate_retrieval_panels.py` lines 51–57.
7. Total output is ~120 MB (float16). Write to `/kaggle/working/features/` and Kaggle output will tarball it.

### C.2 — Post-download steps

After kernel completes and `kaggle kernels output` pulls the .npy + index_map.json:
1. Move to `experiments/veri776-paper/A5alpha/features/`.
2. Run `python scripts/paper/generate_retrieval_panels.py --exp-id A5alpha`. Expected output: `figures/paper/retrieval/panel_{1..6}.pdf` + `panels_log.json`.
3. Update `tab:qual` in `main.tex` to remove the "(see supplementary materials)" hedge and add a `\begin{figure}` referencing `figures/paper/retrieval/panel_1.pdf` etc. (out of scope for this spec — separate Wave 3c follow-up).

### C.3 — If Option 1 declined

Fall back to Option 2: update `tab:qual` caption footnote (line 750–754) from:
> Full retrieval panel figures are available in the project repository (see supplementary materials).

to:
> Quantitative failure-mode analysis only; full retrieval panels are deferred to future work pending stream-1 / stream-2 feature dumps from the A5$\alpha$ checkpoint.

Do **not** pick Option 3 (local extraction) — violates the GPU pipeline rule.

---

## Verification steps for the coder

After Task A + Task B edits (Task C is optional and runs separately):

1. **Aggregation script regenerates REPORT.md with new tie-band text:**
   ```pwsh
   .\.venv\Scripts\python.exe scripts/paper/aggregate_paper_results.py
   Get-Content experiments/veri776-paper/REPORT.md | Select-String -Pattern "Delta = |tie band|WAVE 4|headline"
   ```
   Expected: prints the four "Recipe Disclosure Outcome" lines from B.3.

2. **results.json contains the new keys:**
   ```pwsh
   .\.venv\Scripts\python.exe -c "import json; d=json.load(open('experiments/veri776-paper/results.json'))['recommended_paper_table_values']; print({k: d[k] for k in ['headline_recipe','tie_band_applies','wave4_trigger','delta_beta_minus_alpha_mAP','alpha_reference_source','decision_reason']})"
   ```
   Expected: `{'headline_recipe': 'A5alpha', 'tie_band_applies': True, 'wave4_trigger': False, 'delta_beta_minus_alpha_mAP': ~0.05, 'alpha_reference_source': 'seeds_mean_n3', 'decision_reason': 'tied_within_0.10pp'}`.

3. **main.tex sanity checks** (no compile required):
   ```pwsh
   Select-String -Path main.tex -Pattern "TODO: insert A5 winner|-- & -- & --"
   ```
   Expected: zero matches (TODO removed, orphan placeholder removed).

   ```pwsh
   Select-String -Path main.tex -Pattern "tab:seed_variance|sec:seed_variance|A1 --- Baseline|A2 --- \+CLIP|A5\$\\alpha\$ --- Full"
   ```
   Expected: ≥ 5 matches (new label, new section anchor, isolated-component rows present).

4. **No stray UTF-8 / BOM issues introduced** (per orchestrator memory — Windows `charmap` traps):
   ```pwsh
   .\.venv\Scripts\python.exe -c "open('main.tex','r',encoding='utf-8').read()" 
   ```
   Should exit 0 with no decode error.

5. **(Optional) pdflatex smoke** — coder may run if a TeX toolchain is configured, but spec does not require it. If run, `pdflatex -interaction=nonstopmode main.tex` must complete with `Output written on main.pdf` and no `Undefined control sequence` errors.