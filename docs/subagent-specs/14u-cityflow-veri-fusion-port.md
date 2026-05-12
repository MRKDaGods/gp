# 14u: Port 14t Fusion Features (CLIP-SENet v6 × TransReID 09v v17) to CityFlowV2 MTMC

**Date**: 2026-05-12
**Author**: MTMC Planner
**Status**: PROPOSED — pending user decision (see § 4 Recommendation; default action = run **Option C only**)

## 0. TL;DR (read this first)

14t WIN on VeRi-776 (`w_clip=0.7, w_trans=0.3` → mAP 93.30 / R1 98.45) discovered a genuinely complementary fusion *geometry* between CLIP-SENet v6 and TransReID 09v v17 — both VeRi-776-trained. **The question for 14u is whether that geometry transfers to CityFlowV2 cross-camera matching, or whether the domain gap dominates as it did in 13d/13f/13h.**

Algebraic reduction (see § 1.3) shows that the user-proposed Options **A** (feature-level concat replacing primary) and **B** (score-level fused-as-new-primary) **both replace the CityFlow-domain-adapted production primary with a fusion of two VeRi-only experts**. The strong negative prior from 13d (`w_cs > 0` monotonically degrades) and 13h (CityFlow-fine-tuned CLIP-SENet still couldn't clear 0.7703 in fusion) means Options A and B are very likely to FAIL.

**This spec introduces a third de-risked variant — Option C — that the user did not consider**, in which the 14t-fused CLIP-SENet × TransReID-09v stream is **added as a new score-fusion stream alongside the existing production primary + DINOv2 tertiary**, not as a replacement. Option C is the only variant where the residual question ("does 14t fusion geometry add cross-camera signal?") is asked cleanly without confounding it with "is the production CityFlow primary worth keeping?" (it provably is, per the entire 13d/13f/13h evidence chain).

**Recommendation**: run **Option C only**. Skip Options A and B unless Option C produces a WIN, in which case A/B become natural follow-ups.

**Do NOT abort 14u entirely**: 14u Option C is genuinely new and not algebraically reducible to 13d/13f/13h because (i) the fusion-then-rerank pattern on the *combined* feature is novel (14t recipe applies AQE k=3 + rerank to the fused similarity, which 13d never tested), and (ii) Option C adds rather than replaces, preserving the CityFlow expert.

---

## 1. Hypothesis

### 1.1 Honest framing of prior evidence

| Prior | Setup | Result | Implication for 14u |
|:--|:--|:--|:--|
| **13d v2** | CLIP-SENet v6 (VeRi-only) added at `w_cs ∈ [0.2, 1.0]` to production (CityFlow-trained CLIP TransReID + DINOv2) | **Monotonic degradation**: −0.13pp at `w_cs=0.2`, −1.77pp at 0.6, −3.68pp at 0.8, −8.24pp standalone | A strong VeRi-776 expert (91.54% mAP post-rerank) does NOT transfer to CityFlow MTMC at any positive weight |
| **13f→13h** | CLIP-SENet v6 *fine-tuned 12 epochs on 666 CityFlow IDs*, then fused | Standalone IDF1 0.6855 → 0.7099 (+2.44pp); fusion peak at `w_cs_ft=0.30 → 0.7691` (−0.12pp below production 0.7703) | Domain adaptation works but the fine-tuned feature stream is too correlated with existing CLIP+DINOv2 pair to clear production |
| **14t (VeRi-776)** | CLIP-SENet v6 score-fused with TransReID 09v v17 at `w_clip=0.7, w_trans=0.3`, AQE k=3 + rerank | **WIN: mAP 93.30 / R1 98.45**, +3.33pp mAP over 09v v17 alone | Two single-cam-strong VeRi experts produce complementary embeddings *on VeRi-776* |

### 1.2 What 14u tests vs. what 13d/13f/13h already tested

| Question | Already answered? |
|:--|:--|
| Does VeRi-only CLIP-SENet help CityFlow fusion at any positive weight? | **YES — NO** (13d, monotonic degradation) |
| Does CityFlow-fine-tuned CLIP-SENet help CityFlow fusion? | **YES — NO** (13h, peak −0.12pp below production) |
| Does VeRi-only TransReID 09v v17 help CityFlow fusion? | **NO** (never tested as a CityFlow stream) |
| Does the *14t score-fused* `(0.7·CLIP-SENet + 0.3·TransReID-09v)` similarity, with rerank applied to the fused matrix, help CityFlow fusion? | **NO** (the fusion-then-rerank pattern is genuinely novel; rerank on fused similarity reshapes neighborhood structure differently from independent score streams) |

The third row matters: **rerank on the fused similarity matrix is structurally different from independent score-stream fusion**. 13d/13f/13h applied AQE k=3 once per stream and then linearly combined scores. 14t's WIN required AQE k=3 + rerank on the *combined* similarity, which 13d never tried. This is the only mechanism by which 14u could deliver where 13d/13f/13h failed.

### 1.3 Algebraic reduction of user-proposed Options A and B

User's Option A (feature-level concat → new primary):
$$S_{\text{primary}}^{14u\text{-A}} = \hat{f}_{\text{concat}} \cdot \hat{g}_{\text{concat}}^\top, \quad \hat{f}_{\text{concat}} = \text{L2}([\alpha \cdot f_{\text{CLIPSENet}}, (1-\alpha) \cdot f_{\text{TransReID-09v}}])$$

For equal-magnitude L2-normalized blocks at $\alpha=0.5$, this collapses to $\tfrac{1}{2}(S_{\text{CLIPSENet}} + S_{\text{TransReID-09v}})$, i.e. 50:50 score fusion of two VeRi-only streams. At other $\alpha$ it is a different linear combination but still a fusion of two VeRi-only streams.

User's Option B (score-level fused-as-new-primary):
$$S_{\text{final}}^{14u\text{-B}} = w_p \cdot \underbrace{[w_{\text{clip}} S_{\text{CLIPSENet}} + w_{\text{trans}} S_{\text{TransReID-09v}}]}_{\text{14t-fused primary}} + w_t \cdot S_{\text{DINOv2}}$$
$$= (w_p w_{\text{clip}}) S_{\text{CLIPSENet}} + (w_p w_{\text{trans}}) S_{\text{TransReID-09v}} + w_t \cdot S_{\text{DINOv2}}$$

This is **structurally identical to a three-way score fusion** with weights $(w_p w_{\text{clip}}, w_p w_{\text{trans}}, w_t)$. It differs from 13d only in (i) using TransReID-09v-VeRi-only instead of CityFlow-trained TransReID as the second stream, and (ii) freezing the CLIP-SENet:TransReID-09v ratio at 7:3. **Both differences make it strictly weaker than 13d, not stronger**, because the CityFlow-domain-adapted TransReID is dropped from the recipe entirely. Strong negative prior.

Option B is "rescuable" only by adding **AQE + rerank on the *fused* primary similarity** before linear combination with DINOv2. That is the genuinely novel piece. But it is also exactly what Option C tests without dropping the CityFlow primary.

### 1.4 Hypothesis statements

- **Option C primary hypothesis (the one this spec recommends running)**:
  - $H_0^C$: Adding the 14t-fused-and-reranked similarity stream as a fourth score-fusion stream at any positive weight `w_14t ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30}` produces no MTMC IDF1 lift above the noise band of 14e B1 plateau 0.77936 ± 0.0024.
  - $H_1^C$: The 14t fusion geometry encodes cross-camera-invariant vehicle signal that is complementary to the production CityFlow CLIP TransReID + DINOv2 pair, lifting MTMC IDF1 to ≥ 0.781 at some `w_14t ∈ [0.10, 0.25]`.
- **Option A/B hypothesis (not recommended)**: Replacing the production primary with the 14t-fused VeRi-only-pair primary will drop MTMC IDF1 by 1–5pp, reproducing the 13d trend. (We are NOT running these.)

---

## 2. Recipe

### 2.1 Three candidate variants

| Option | What replaces / adds | Algebraic novelty vs 13d/13f/13h | Prior | Recommendation |
|:--|:--|:--|:--|:--|
| **A** (user-proposed) | Concat $[α·f_{\text{CLIPSENet}}, (1-α)·f_{\text{TransReID-09v}}]$ → new primary; drops production CityFlow CLIP TransReID | Low — reduces to 50:50 score fusion of two VeRi-only streams at $α=0.5$ | Strong-negative (13d/13h) | **SKIP** unless Option C wins |
| **B** (user-proposed) | $S_{\text{14t-fused}} = 0.7·S_{\text{CLIPSENet}} + 0.3·S_{\text{TransReID-09v}}$ as new primary; drops production CityFlow CLIP TransReID | Medium — adds AQE+rerank on fused similarity but still replaces production primary | Strong-negative (13d/13h) | **SKIP** unless Option C wins |
| **C** (this spec's addition) | Add $S_{\text{14t-fused-AQE3-rerank}}$ as a 4th score-fusion stream alongside production primary + DINOv2 | **High** — preserves production primary; tests only whether 14t fusion geometry adds signal; rerank-on-fused-similarity is genuinely untested | Weak-negative-by-extrapolation | **RUN** (see § 4 Recommendation) |

### 2.2 Option C recipe details

1. **Extract CLIP-SENet v6 features on CityFlow crops**: REUSE the existing 13c output (kernel `yahiaakhalafallah/13c-clip-senet-cityflow-features` → produces per-tracklet pooled CLIP-SENet 2048-d features for all 929 tracklets across 6 cameras, already aligned with Stage-2 tracklet index). No re-extraction needed.
2. **Extract TransReID 09v v17 features on CityFlow crops**: NEW extraction required. The existing CityFlow primary uses a different TransReID checkpoint (CLIP TransReID ViT-B/16 @ 256² CityFlow-trained, mAP=80.14% local). 14t used `vehicle_transreid_vit_base_veri776.pth` (VeRi-only, 224²). This file is in `mrkdagods/mtmc-weights`. We need a new Kaggle GPU kernel that loads this checkpoint and emits per-tracklet pooled 768-d features over the same 929 tracklets at the same TTA settings 14c v2 used (4-view TTA: original, hflip, scale_0.95, scale_1.05; L2-mean-pool then L2-renorm).
3. **Build the 14t-fused similarity**: on CPU,
   - L2-normalize both feature sets per-tracklet
   - Compute $S_{\text{CLIPSENet}}$ and $S_{\text{TransReID-09v}}$ as $f f^\top$ tracklet-tracklet similarity (929×929 each)
   - Score-fuse: $S_{\text{14t}} = 0.7 \cdot S_{\text{CLIPSENet}} + 0.3 \cdot S_{\text{TransReID-09v}}$
   - Apply AQE k=3 (one iteration) on the L2-normed *combined* feature space: build the combined feature vector by score-weighted concat $[\sqrt{0.7} f_{\text{CLIPSENet}}, \sqrt{0.3} f_{\text{TransReID-09v}}]$ then L2-renorm — this makes $f f^\top$ algebraically equal to the score-fused $S_{\text{14t}}$. Then AQE on that combined feature.
   - Apply k-reciprocal reranking (k1=80, k2=15, λ=0.2) on the AQE'd combined similarity. Output is a single 929×929 reranked-AQE-fused-VeRi similarity matrix, call it $S^*_{\text{14t}}$.
4. **Add to Stage 4 association as a fourth stream**: alongside production primary $S_{\text{CityFlow-CLIP}}$ and DINOv2 tertiary $S_{\text{DINOv2}}$:
   $$S_{\text{final}}^{14u\text{-C}} = w_p \cdot S_{\text{CityFlow-CLIP}} + w_t \cdot S_{\text{DINOv2}} + w_{14t} \cdot S^*_{\text{14t}}, \quad w_p + w_t + w_{14t} = 1$$
   with $w_p / w_t$ rescaled by $(1 - w_{14t})$ to preserve the 14e B1 anchor ratio when $w_{14t}=0$.
5. **Stage 4 anchor**: keep `aqe_k=2`, `similarity_threshold=0.48`, `fic_regularisation=0.5`, conflict-free CC, gallery expansion (0.48/0.38), temporal overlap bonus 0.05, intra-merge (0.80/30), `mtmc_only_submission=false`. These are the locked 14e B1 settings.
6. **Drift gate**: `w_14t=0.00` must reproduce 0.77936 ± 0.0005 with `id_switches=154` (the 14e B1 / 14f A20 / 14h M0 / 14i F0 / 14j W0 / 14k K0 plateau).

### 2.3 Why score-stream-with-internal-rerank, not concat-into-existing-primary

If the 14t-fused stream is simply concatenated into the production CityFlow primary at feature level, the AQE+rerank applied at Stage 4 operates on the *combined* primary similarity — which mixes a CityFlow-domain-adapted feature space with a VeRi-only feature space at the embedding level. This is the failure mode 13d's "feature concatenation" dead end demonstrated (−1.6pp). Score-level addition with the 14t stream *pre-reranked internally* keeps the two feature spaces separate at the similarity level, which is the only theoretically sound way to combine them given the domain mismatch.

---

## 3. Anchor sweep

### 3.1 Drift gate (required first run)

| Label | $w_{14t}$ | $w_p$ | $w_t$ | thr | aqe_k | fic_reg | Acceptance |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--|
| **U0** | **0.00** | 0.475 | 0.525 | 0.48 | 2 | 0.5 | MTMC IDF1 = 0.77936 ± 0.0005, `id_switches=154` EXACT |

If U0 fails the drift gate, halt the sweep and diagnose. Do NOT promote any U-row.

### 3.2 Sweep matrix — Option C only (16 configs, CPU-only after feature extraction)

For $w_{14t} \in \{0.05, 0.10, 0.15, 0.20, 0.25, 0.30\}$ × `similarity_threshold ∈ {0.46, 0.48, 0.50}`:

```json
[
  {"label":"U0","w_14t":0.00,"w_primary":0.475,"w_tertiary":0.525,"similarity_threshold":0.48,"notes":"DRIFT GATE — must reproduce 0.77936 / id_sw=154 EXACT"},
  {"label":"U1","w_14t":0.05,"w_primary":0.45125,"w_tertiary":0.49875,"similarity_threshold":0.46},
  {"label":"U2","w_14t":0.05,"w_primary":0.45125,"w_tertiary":0.49875,"similarity_threshold":0.48},
  {"label":"U3","w_14t":0.05,"w_primary":0.45125,"w_tertiary":0.49875,"similarity_threshold":0.50},
  {"label":"U4","w_14t":0.10,"w_primary":0.4275,"w_tertiary":0.4725,"similarity_threshold":0.46},
  {"label":"U5","w_14t":0.10,"w_primary":0.4275,"w_tertiary":0.4725,"similarity_threshold":0.48},
  {"label":"U6","w_14t":0.10,"w_primary":0.4275,"w_tertiary":0.4725,"similarity_threshold":0.50},
  {"label":"U7","w_14t":0.15,"w_primary":0.40375,"w_tertiary":0.44625,"similarity_threshold":0.48},
  {"label":"U8","w_14t":0.20,"w_primary":0.38,"w_tertiary":0.42,"similarity_threshold":0.46},
  {"label":"U9","w_14t":0.20,"w_primary":0.38,"w_tertiary":0.42,"similarity_threshold":0.48},
  {"label":"U10","w_14t":0.20,"w_primary":0.38,"w_tertiary":0.42,"similarity_threshold":0.50},
  {"label":"U11","w_14t":0.25,"w_primary":0.35625,"w_tertiary":0.39375,"similarity_threshold":0.48},
  {"label":"U12","w_14t":0.30,"w_primary":0.3325,"w_tertiary":0.3675,"similarity_threshold":0.46},
  {"label":"U13","w_14t":0.30,"w_primary":0.3325,"w_tertiary":0.3675,"similarity_threshold":0.48},
  {"label":"U14","w_14t":0.30,"w_primary":0.3325,"w_tertiary":0.3675,"similarity_threshold":0.50},
  {"label":"U15","w_14t":0.30,"w_primary":0.3325,"w_tertiary":0.3675,"similarity_threshold":0.48,"notes":"K13-style balance sanity probe: rerun U13 with primary suppression check"}
]
```

All configs fix `aqe_k=2`, `fic_regularisation=0.5`, `pca_components=384`, `algorithm=conflict_free_cc`, gallery expansion enabled (0.48 / 0.38), temporal overlap bonus 0.05, intra-merge (0.80 / 30), `mtmc_only_submission=false`, `w_secondary=0.0`.

### 3.3 Boundary check

If best config sits at $w_{14t}=0.30$ (the upper boundary), follow up with an extension sweep at $w_{14t} \in \{0.35, 0.40, 0.45\}$ analogous to 14j→14k. Do not pre-launch the extension; treat it as a contingent follow-up like 14k.

---

## 4. Verdict bands

| Verdict | Best MTMC IDF1 | Action |
|:--:|:--:|:--|
| **WIN** | ≥ **0.7810** (+0.16pp vs 14e B1 0.77936) | Promote to new headline; replicate on a second seed; update `docs/findings.md`, `docs/experiment-log.md`, `.github/copilot-instructions.md`. Re-evaluate whether Options A/B are now worth running (probably still no — Option C's lift would show the 14t geometry adds signal *without* needing primary replacement). |
| **MARGINAL** | 0.7800–0.7810 | Document but do not promote; 14e B1 stays headline. Treat 14t geometry as a real but insufficient cross-camera signal. |
| **FAIL** | < 0.7800 | Confirms domain-gap thesis at maximum strength: even *14t's discovered fusion geometry*, with rerank on the fused similarity, doesn't help CityFlow. Document as the **final CityFlow VeRi-fusion dead end**. Close the entire VeRi→CityFlow-port branch. Next viable feature-side lever is genuinely-new-architecture (Option A from `post-14k-next.md` Candidate 2 EVA-02-L/14) or learned association (GNN edge classifier). |
| **DRIFT** | U0 not within 0.0005 of 0.77936 OR `id_switches ≠ 154` | Halt sweep; diagnose Stage-2 alignment, FAISS index, or feature-loading order. |

### 4.1 Recommendation

**Run Option C only.** Skip Options A and B for now — both are dominated by Option C in expected information value:

- Option C provides the cleanest test of the *fusion-geometry-transfer* hypothesis, the only piece not already falsified by 13d/13f/13h.
- Options A and B both drop the CityFlow-domain-adapted production primary, which is provably the most CityFlow-cross-camera-invariant single stream we have. Dropping it almost guarantees a regression regardless of what replaces it.
- If Option C WINs, then re-examining A/B becomes interesting (does the production primary even matter once the 14t-fused stream is in?). If Option C FAILs, A and B fail strictly worse (algebraic dominance argument from § 1.3).

**Cost-benefit**:
- Option C: 1 GPU kernel (~1.5–2h P100 for TransReID 09v feature extraction on CityFlow crops) + 1 CPU kernel (~10–15 min, 16 configs). Total ~2.5h, one GPU slot for ~2h on `yahiaakhalafallah`.
- Options A and B: each would require its own Stage 2 + Stage 4 rerun (~3–4h GPU), with strong negative prior. Information value < Option C.

---

## 5. Implementation plan

### 5.1 Inputs and dataset slugs

| Asset | Location | Status |
|:--|:--|:--|
| CityFlowV2 data | `thanhnguyenle/data-aicity-2023-track-2` | Confirmed via 13c, 14j, 14h kernel metadata |
| TransReID 09v v17 weights (`vehicle_transreid_vit_base_veri776.pth`) | `mrkdagods/mtmc-weights` | Confirmed via 14t kernel metadata; **also exists under `yahiaakhalafallah/mtmc-weights` per 14h kernel metadata — need to verify which is current**. |
| CLIP-SENet v6 checkpoint | Kernel source `yahiaakhalafallah/13-clip-senet-train` (output `best_mAP.pth` or `vehicle_clip_senet_veri776.pth`) | Confirmed via 14t kernel metadata |
| TransReID model code | In-repo: `src/stage2_features/transreid_model.py` (used by 14t) | Confirmed |
| CLIP-SENet model code | Bundled inline in 14t notebook | Confirmed; will need to copy or import |
| Stage-1 tracklets + 14c v2 TTA crops/multi-query (929 tracklets) | Kernel source `yahiaakhalafallah/14h-robust-tracklet-pooling` (which itself sources `mtmc-10a-stages-0-2` + `09s-dinov2-large-cityflowv2` + `mtmc-10b-stage-3-faiss-indexing`) | Confirmed canonical source for 14e B1 anchor reproduction |
| CLIP-SENet CityFlow per-tracklet features | Kernel `yahiaakhalafallah/13c-clip-senet-cityflow-features` (existing output, 6 cameras × per-cam NPZ, 929 total tracklets, 2048-d) | **Need to verify** that the feature set is aligned with the same 929 tracklets as 14h v3 output (13d v2 reported 6/6 cameras 100% key-set match — implies yes) |
| Stage-3 FAISS + Stage-4/5 association code | Existing repo + `mtmc-10b-stage-3-faiss-indexing` | Confirmed |

### 5.2 Kaggle kernel structure

**Kernel 1 — `yahiaakhalafallah/14u-transreid-09v-cityflow-features`** (GPU T4, ~1.5–2h):

- Inputs:
  - `dataset_sources`: `mrkdagods/mtmc-weights` (for the 09v VeRi-only TransReID checkpoint), `thanhnguyenle/data-aicity-2023-track-2` (CityFlow crops if needed)
  - `kernel_sources`: `yahiaakhalafallah/mtmc-10a-stages-0-2` (tracklets + crops + Stage 2 alignment metadata), `yahiaakhalafallah/14h-robust-tracklet-pooling` (canonical 14e B1 feature stack for cross-checks)
  - `enable_gpu`: true, `machine_shape`: `NvidiaTeslaT4`
- Logic:
  1. Locate `vehicle_transreid_vit_base_veri776.pth` under `/kaggle/input`
  2. Load via `src.stage2_features.transreid_model.build_transreid` with `vit_model="vit_base_patch16_clip_224.openai"`, `img_size=(224,224)` (matching 14t)
  3. Iterate over all 929 tracklets × all frames × 4 TTA views (original, hflip, scale_0.95, scale_1.05) on CityFlow crops at 224²
  4. Per-tracklet pool with the existing softmax-quality-weighted mean (mirror the production Stage 2 pooler exactly so the index aligns with 14e B1's 929-tracklet ordering)
  5. Emit `transreid_09v_cityflow_features.npy` shape `(929, 768)`, L2-normed, plus `embedding_index.json` matching the canonical 14h v3 index order
  6. Validation gate: assert `embedding_index.json` is identical row-for-row with the 14h v3 reference index; assert feature norms in `[0.999, 1.001]`; emit summary JSON

**Kernel 2 — `yahiaakhalafallah/14u-fusion-sweep`** (CPU, ~15–25 min):

- Inputs:
  - `kernel_sources`: `yahiaakhalafallah/14u-transreid-09v-cityflow-features`, `yahiaakhalafallah/13c-clip-senet-cityflow-features`, `yahiaakhalafallah/14h-robust-tracklet-pooling`, `yahiaakhalafallah/mtmc-10b-stage-3-faiss-indexing`
  - `dataset_sources`: `thanhnguyenle/data-aicity-2023-track-2` (ground-truth for trackeval)
  - `enable_gpu`: false
- Logic:
  1. Load 929×768 TransReID-09v features, 929×2048 CLIP-SENet features, 929×D primary CityFlow CLIP TransReID features, 929×D DINOv2 features — assert all align to the same `embedding_index.json`
  2. Build the 14t-fused-and-reranked similarity $S^*_{\text{14t}}$ following § 2.2 step 3 (combined feature $[\sqrt{0.7} f_{\text{CS}}, \sqrt{0.3} f_{\text{TR09v}}]$ L2-renorm → AQE k=3 → rerank k1=80,k2=15,λ=0.2 → 929×929 reranked similarity)
  3. For each of 16 configs (U0..U15), build $S_{\text{final}} = w_p S_{\text{CityFlow-CLIP}} + w_t S_{\text{DINOv2}} + w_{14t} S^*_{\text{14t}}$, run Stage 4 association (conflict-free CC, gallery expansion, temporal overlap, intra-merge) + Stage 5 trackeval
  4. Drift gate: U0 must reproduce 0.77936 / id_sw=154 EXACT
  5. Output `14u_summary.json` with per-config MTMC IDF1, trackeval IDF1, id_switches, MOTA, HOTA, num_clusters

### 5.3 Account choice

Per user's prompt: yahiaakhalafallah (has v6 + TransReID + CityFlow data). Confirmed: 13c, 13d, 14h, 14j, 14t all ran under this account. No multi-account auth needed.

### 5.4 Walltime budget

| Stage | Walltime | GPU slot |
|:--|:--|:--|
| Kernel 1 (TransReID 09v feature extraction on CityFlow) | ~1.5–2 h | 1 × T4 |
| Kernel 2 (fusion + sweep, CPU) | ~15–25 min | none |
| **Total** | **~2 – 2.5 h wall, 1 GPU slot for 2 h** | |

### 5.5 Disk budget

- Kernel 1 output: `transreid_09v_cityflow_features.npy` ≈ 929 × 768 × 4 B ≈ 2.85 MB + index ≈ 1 MB → negligible.
- Kernel 2 output: 16 × Stage-4 prediction MOT files (~2–3 MB each) + 16 × Stage-5 eval JSON → ~50 MB. Cleanup not required.

---

## 6. Blockers to resolve before push

1. **Verify the 09v VeRi-only TransReID weights dataset slug.** The findings doc references `mrkdagods/mtmc-weights`, the 14t notebook uses `mrkdagods/mtmc-weights` for `vehicle_transreid_vit_base_veri776.pth`, and the 14h notebook mounts `yahiaakhalafallah/mtmc-weights`. Confirm which copy contains the 09v v17 checkpoint and that the file naming matches the path discovery logic in 14t (`find_file_under_input("vehicle_transreid_vit_base_veri776.pth", preferred_slug="mtmc-weights")`).
2. **Confirm the 13c CLIP-SENet feature index aligns with the 14h v3 / 14e B1 929-tracklet ordering.** 13d v2 reported "6/6 cameras 100% key-set match" against the prior tracklet manifest — but the 14h v3 multi-query stack may have a different row ordering. Before Kernel 2, write a one-cell preflight that loads both indices and asserts row-for-row equality. If they differ, the fix is a permutation map, not a re-extraction.
3. **Decide on the AQE+rerank exact algorithm**: copy the 14t kernel's `compute_reranking_torch` + `average_query_expansion` verbatim (k=3, k1=80, k2=15, λ=0.2 — the 14t WIN settings), or use the existing Stage 4 reranker (different parameters). For consistency with the 14t recipe, **use the 14t kernel's exact rerank/AQE code** (copy-paste, do not refactor). This is the lever 13d/13f/13h never tested.
4. **TTA matching**: 14c v2 used 4-view TTA on the primary CLIP TransReID and 2-view TTA on DINOv2. For the new 14u TransReID-09v stream, use **4-view TTA matching the primary** to keep TTA noise floor consistent. Do NOT mix 1-view 09v with 4-view primary at fusion.
5. **`image_size` mismatch warning**: 14t used TransReID at 224² (matches the 09v VeRi-only training resolution); the production CityFlow CLIP TransReID primary uses 256². The 14u-TransReID-09v stream MUST run at **224²** to match the 14t WIN setup and the checkpoint's training resolution. Resizing CityFlow crops to 224² (vs 256²) is acceptable — the goal is to apply the 14t feature pipeline as-is.
6. **Stage 4 plumbing for four streams**: confirm the existing Stage 4 fusion code accepts an additional precomputed 929×929 similarity matrix as input alongside the existing primary+secondary+tertiary triple. If not, Kernel 2 must inline a stripped-down Stage 4 that does the linear similarity combination + AQE k=2 + conflict-free CC association. (14j already did exactly this for the R50-IBN quaternary stream, so the pattern is established — copy the 14j Kernel 2 association code.)

---

## 7. Estimated budget

- **Walltime**: ~2–2.5 h total (1.5–2 h GPU + 15–25 min CPU).
- **Disk**: < 100 MB output across both kernels.
- **Kaggle quota**: ~2 h × 1 GPU on yahiaakhalafallah. No multi-account work required.
- **Spec/coder time**: spec is this document; coder time to build both kernels ~2–3 hours.

---

## 8. Rollback plan

- If Kernel 1 fails (e.g. 09v checkpoint missing, OOM at 224², or feature shape mismatch): mark 14u BLOCKED in `docs/findings.md`, do NOT push Kernel 2, log the blocker for next planning turn.
- If Kernel 2 U0 drift gate fails (`U0 ≠ 0.77936 ± 0.0005` or `id_sw ≠ 154`): mark sweep DRIFT-AFFECTED; do not promote any config; diagnose the difference (likely index permutation or AQE/rerank parameter drift from the 14t recipe).
- If 14u verdict = FAIL (best < 0.7800): mark the entire VeRi→CityFlow port branch as **closed** in `docs/findings.md` § Dead Ends with explicit cross-reference to 13d/13f/13h/14u; do NOT retry CLIP-SENet or TransReID-09v fusion ports in any form.
- The 14e B1 0.77936 production-deployed config remains the production baseline regardless of 14u outcome.

---

## 9. Coder handoff checklist

1. Build `notebooks/kaggle/14u_transreid_09v_features/` (Kernel 1). Copy the 14t feature-extraction cells (`extract_transreid_features` with `mode="single_flip"`) but adapt to iterate over Stage-1 tracklet crops instead of VeRi-776 query/gallery splits. Wire `kernel_sources` to `mtmc-10a-stages-0-2` + `14h-robust-tracklet-pooling`. Set `enable_gpu: true`, `machine_shape: NvidiaTeslaT4`.
2. Inside Kernel 1, validate that the output `embedding_index.json` is row-for-row identical to the 14h v3 reference. Fail fast if not.
3. Build `notebooks/kaggle/14u_fusion_sweep/` (Kernel 2). Copy the 14j Stage-4 association code as the starting point (already supports 3-way score fusion with conflict-free CC at AQE k=2). Add a fourth precomputed similarity slot for $S^*_{\text{14t}}$. Set `enable_gpu: false`.
4. Inside Kernel 2, build $S^*_{\text{14t}}$ exactly per § 2.2 step 3 using the 14t kernel's verbatim `compute_reranking_torch` + `average_query_expansion` code (copy from the existing 14t notebook).
5. Run U0 first; halt if drift gate fails.
6. Push **once per kernel**; watch for `not valid dataset sources` warnings; cancel + refix per the kaggle push safety rules in `.github/copilot-instructions.md`.
7. On WIN (≥0.7810): replicate on a second seed before promoting to headline. Update `docs/findings.md`, `docs/experiment-log.md`, `.github/copilot-instructions.md`.
8. On MARGINAL/NEUTRAL/FAIL: keep 14e B1 0.77936 as headline. If FAIL, write the post-14u next spec proposing EVA-02-L/14 (per `post-14k-next.md` Candidate 2) or GNN edge classifier.

---

## 10. Negative-result value

If 14u FAILs, the result is still publishable: it closes the **5th and final** CityFlow VeRi-fusion experiment (after 13d, 13f, 13h, and the implicit OSNet branch via 14m) with maximum statistical strength — even rerank-on-fused-similarity, the most aggressive AQE/rerank pattern available, doesn't bridge the VeRi-776 → CityFlowV2 domain gap. That negative result strengthens the paper's "feature quality (specifically cross-camera invariance training methodology), not association tuning, is the MTMC bottleneck" thesis.