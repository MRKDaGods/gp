# Baseline Drift Fix Spec — CLIP-only MTMC IDF1 (v52 0.7714 → v15 0.7663, -0.51pp)

**Status**: Diagnosed. Root cause identified with HIGH confidence.
**Branch**: `fix/baseline-drift`
**Owner**: Coder (implementation), Planner (review)

## TL;DR

The drift is **not an algorithmic regression**. It is a **Stage 2 feature-distribution shift** caused by `stage2.camera_bn.enabled` being `true` in the dataset default config but `false` in the canonical v80-restored recipe that produced v52. The current 10c v15 baseline consumes 10a features extracted with camera-BN ON; the v52 0.7714 baseline consumed features extracted with camera-BN OFF. Same association code, different input feature distribution → different score.

This is **already partially documented** in `docs/findings.md`:
> "10c v9 baseline 76.625% is ~0.74pp below expected 77.36% due to feature distribution shift (camera_bn=true)."

The v15 result (0.7663) is statistically indistinguishable from the v9 result (0.76625). Same drift, same cause.

## Root Cause (HIGH confidence)

**File**: `configs/datasets/cityflowv2.yaml` line 149
```yaml
camera_bn:
  enabled: true       # current default
```

**Canonical v80-restored recipe** (`scripts/vehicle_v80_restored.py` line 27):
```python
"stage2.camera_bn.enabled=false",
```

**Mechanism**:
1. Stage 2 with `camera_bn=true` re-centers each camera's embeddings to per-camera mean before PCA.
2. PCA fitted on these re-centered embeddings has different principal axes than PCA fitted on raw embeddings.
3. FAISS index built on whitened, re-centered embeddings retrieves a different nearest-neighbor topology.
4. Stage 4 association similarity scores shift uniformly → similarity_threshold=0.50 cuts the graph differently → different connected components → different MTMC IDF1.

The v80-restored sweep (sim_thresh=0.50, app_w=0.70, fic_reg=0.50, aqe_k=3, gallery=0.48, orphan=0.38) is **calibrated for camera_bn=false features**. Applying it to camera_bn=true features produces ~-0.74pp.

## Top 3 Hypotheses (Ranked)

| # | Hypothesis | File:Line Evidence | Confidence |
|---|------------|--------------------|------------|
| 1 | **camera_bn=true in current 10a, =false in v52** | `configs/datasets/cityflowv2.yaml:149`, `scripts/vehicle_v80_restored.py:27`, `docs/findings.md` ("76.625% due to feature distribution shift (camera_bn=true)") | **HIGH (95%)** |
| 2 | Tertiary (DINOv2) extraction enabled in dataset config indirectly affects 10a outputs | `configs/datasets/cityflowv2.yaml:68-75` (vehicle3.enabled=true) | LOW (15%) — tertiary is saved to a separate `.npy`; primary PCA/FAISS pipeline does not see tertiary embeddings. The Explore-agent claim of "PCA contamination" was not substantiated by a code path. |
| 3 | Stage 4 algorithmic refactor between v52 and HEAD | None found by diff | LOW (5%) — no behavioral changes detected in `src/stage4_association/` |

## Fix Plan

### Step 1: Confirm the input features
Inspect the 10a kernel run that produced the embeddings used by 10c v15.
- Check the 10a notebook overrides cell for `stage2.camera_bn.enabled=...`.
- If absent, the runtime value is `true` (cityflowv2.yaml default).

### Step 2: Re-run 10a with camera-BN disabled (canonical recipe)
Push a 10a kernel with this single override added:
```
--override stage2.camera_bn.enabled=false
```
Keep all other settings identical to the run that produced v15's input.

### Step 3: Re-run 10c CLIP-only baseline against the new features
Apply the canonical v80-restored Stage 4 sweep (already in `scripts/vehicle_v80_restored.py`):
```
stage4.association.graph.similarity_threshold=0.50
stage4.association.weights.vehicle.appearance=0.70
stage4.association.fic.regularisation=0.50
stage4.association.query_expansion.k=3
stage4.association.gallery_expansion.threshold=0.48
stage4.association.gallery_expansion.orphan_match_threshold=0.38
stage4.association.intra_camera_merge.enabled=true
stage4.association.intra_camera_merge.threshold=0.80
stage4.association.intra_camera_merge.max_time_gap=30.0
stage4.association.query_expansion.dba=false
```

### Step 4 (preferred long-term): Realign defaults
Either:
- (a) Flip `configs/datasets/cityflowv2.yaml:149` to `enabled: false` to match the canonical recipe, OR
- (b) Re-tune the v80-restored Stage 4 sweep on `camera_bn=true` features to recover 0.7714+ (this is option to AVOID per project rules — association is exhausted).

**Recommendation: option (a)**. The camera_bn=true experiment was never validated to outperform camera_bn=false on this codebase; it was set as default speculatively. Reverting is a minimal, behavior-restoring change.

## Files to Touch

| File | Change | Type |
|------|--------|------|
| `configs/datasets/cityflowv2.yaml` line 149 | `enabled: true` → `enabled: false` | Revert |
| (none in `src/stage4_association/`) | — | No code change |

## Kaggle Test Plan

1. **10a re-extraction** (GPU, account: gumfreddy):
   - Push `notebooks/kaggle/10a_stages012/` once with `stage2.camera_bn.enabled=false`.
   - Estimate: ~50 min (matches 10a v5 timing).
2. **10b** (CPU): trigger when 10a completes; ~5 min.
3. **10c CLIP-only baseline** (CPU): push with v80-restored overrides; ~10 min.
4. **Total session budget**: ~65 min. Well within 3–4 hour budget.

**Expected MTMC IDF1**: 0.770–0.775. If <0.768 the fix has failed; rollback per below.

## Rollback Plan

If the re-run does not recover ≥0.770:
1. Revert `configs/datasets/cityflowv2.yaml` to `enabled: true` (current state).
2. Confirm tertiary (DINOv2) is genuinely independent: temporarily set `vehicle3.enabled=false` in cityflowv2.yaml AND `camera_bn.enabled=false`, re-run 10a + 10c. If still <0.770, the drift has another cause and isolation experiments are required (see below).

## Isolation Experiments (only if Step 3 fails)

Run in this order, each one 10a re-extract + 10c v80-restored:
- A. `camera_bn=false` + `vehicle3.enabled=false` (full v52-era state)
- B. `camera_bn=false` + `vehicle3.enabled=true` (current state minus camera_bn)
- C. `camera_bn=true` + `vehicle3.enabled=false` (isolate camera_bn alone)

Decision matrix:
- A=0.77+, B=0.77+ → camera_bn was the cause; deploy fix.
- A=0.77+, B<0.77 → tertiary extraction also contributes; need to investigate Stage 2 shared state.
- A<0.77 → drift is not in Stage 2; investigate Stage 1 (detection/tracking) or Stage 3 (FAISS index seeding/quantization).

## Confidence

- **Diagnosis**: HIGH (95%). Two independent lines of evidence (canonical recipe explicitly disables camera_bn; findings.md already attributes the same -0.74pp drift to this exact cause).
- **Fix recovers ≥0.770**: HIGH (85%). The remaining 15% accounts for run-to-run noise in detection/tracking and possible incidental changes since v52 not caught by the diff.
- **Fix recovers exactly 0.7714+**: MEDIUM (60%). Detection/tracking are non-deterministic; some residual drift is expected even with identical config.

## Constraints (per Planner directives)

- ✅ No new architectural changes proposed.
- ✅ No touch to the working 0.7703 ensemble result.
- ✅ Only revert/restore behavior to v52 levels.
- ✅ If fix fails, isolation experiments are specified instead of speculative additional fixes.

## GPU Budget

- 1× 10a re-extraction: ~50 min GPU
- 1× 10c re-run: ~10 min CPU
- Buffer for 1 isolation experiment if needed: +60 min GPU
- **Total worst-case: ~2 hours**, within 3–4 hour budget.