# 14t Production Wiring — Standalone Eval Script + Optional Backend Surface

**Status**: PROPOSED (planner spec)  
**Author**: MTMC Planner, 2026-05-16  
**Type**: Refactor + new CLI + optional API endpoint; no training  
**Estimated implementation effort**: 1 medium PR for Phase 1, plus 1 small follow-up PR for Phase 2 if Phase 1 lands and is requested

---

## 1. Scope & Non-Goals

### Phase 1 — Required

In scope:

- Create `scripts/eval/eval_14t_fusion_veri776.py`.
- The new CLI must produce the exact JSON schema required by [14t-verifier.md](14t-verifier.md#5-required-json-schema-for-eval_14t_fusion_veri776py).
- Refactor the two existing eval scripts to expose reusable feature-extraction helpers.
- Do not break either existing CLI surface.
- Create `configs/models/veri776_14t_fusion.yaml`.
- Confirm that `configs/model_registry.yaml` already contains a consistent `veri776_14t_fusion` entry around lines 267-325.
- Add a unit test for deterministic fusion math that needs no GPU and no real checkpoints.
- Add a note in `models/reid/README.md` documenting that this is a single-camera VeRi use case, not a CityFlow MTMC drop-in.

### Phase 2 — Optional, Separate PR

Out of scope for Phase 1:

- Backend endpoint `POST /api/v1/reid/veri_fusion` for ad-hoc single-camera ReID queries against a hosted gallery.
- `src/stage2_features/fusion_pipeline.py`.

Only build Phase 2 if Phase 1 verification passes and the user explicitly wants a dashboard/API surface for VeRi-style query-time ReID.

### Explicitly Not Done

Do not wire the fusion into `src/stage2_features/reid_model.py` or any CityFlow MTMC pipeline stage.

Reason: [docs/findings.md](../findings.md#L367-L377) records 14u as a CityFlow VeRi-fusion port DEAD END. The 14t fusion is excellent on VeRi-776 single-camera ReID, but does not transfer to CityFlowV2 MTMC. Adding it to the offline MTMC pipeline would silently hurt or fail to improve MTMC IDF1.

---

## 2. Refactor Existing Eval Scripts

Both scripts currently expose command-line evaluation behavior. Phase 1 must preserve that behavior while extracting helpers that the new 14t fusion CLI can import.

Run each script's `--help` before and after the refactor. If a small local smoke fixture exists, run the same smoke before and after. Any CLI behavior change is a regression unless deliberately approved.

### `scripts/eval/eval_clip_senet_veri776.py`

Expose these helpers at module level:

```python
def build_clipsenet_model(checkpoint: Path, device: str) -> nn.Module:
    ...

def extract_clipsenet_features(
    model: nn.Module,
    items,
    img_size: tuple[int, int],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    ...

def parse_veri_split(split_dir: Path) -> tuple[list[dict], int]:
    ...
```

Helper contracts:

- `build_clipsenet_model()` loads CLIP-SENet v6 from `best.pth`, `best_mAP.pth`, or `clipsenet_v6_veri776_best.pth` checkpoint payloads exactly as the current CLI does.
- `extract_clipsenet_features()` returns L2-normalized features with shape `[N, 2048]` plus `pids`, `camids`, and path strings aligned row-for-row with `items`.
- `parse_veri_split()` may wrap or rename the existing parser, but must remain importable at module level.
- Existing `parse_args()` and `main()` remain intact.
- Existing CLI flags remain intact: `--checkpoint`, `--veri-root`, `--device`, `--batch-size`, `--img-size H W`, `--output-json`, `--rerank/--no-rerank`, and `--aqe-k`.

The current CLI signature is visible near the parser block in `scripts/eval/eval_clip_senet_veri776.py` around lines 931-941.

### `scripts/eval/eval_09v_transreid_veri776.py`

Expose these helpers at module level:

```python
def build_09v_model(checkpoint: Path, device: str) -> nn.Module:
    ...

def extract_09v_features(
    model: nn.Module,
    items,
    device: str,
    batch_size: int,
    *,
    stream: Literal["global", "concat_patch_flip"],
) -> np.ndarray:
    ...
```

Helper contracts:

- `build_09v_model()` uses `src.stage2_features.transreid_model.build_transreid`, matching the current eval script.
- `extract_09v_features(..., stream="global")` returns the 768-d `single_flip` CLS BNNeck output. This corresponds to the 14t notebook arrays `q_tr_768` and `g_tr_768`.
- `extract_09v_features(..., stream="concat_patch_flip")` returns the 1536-d joint-optimum concat stream. This corresponds to the 14t notebook arrays `q_tr_1536` and `g_tr_1536`.
- All returned feature rows must be L2-normalized `float32`.
- Existing `parse_args()` and `main()` remain intact.
- Existing CLI flags remain intact: `--checkpoint`, `--veri-root`, `--device`, `--batch-size`, `--output-json`, `--rerank/--no-rerank`, and `--aqe-k`.

The current CLI signature is visible near the parser block in `scripts/eval/eval_09v_transreid_veri776.py` around lines 912-921.

### Stage 2 Surface Is Not a Fusion Surface

`src/stage2_features/reid_model.py` is a single-model wrapper. It exposes `ReIDModel`, model builders including `_build_transreid()`, batch feature extraction, tracklet embedding, and multi-query embedding helpers. It should not be modified for 14t.

`src/stage2_features/transreid_model.py` exposes `TransReID` and `build_transreid()`, which the 09v eval script already uses. The new fusion CLI should reuse the 09v eval helper rather than reaching around it into Stage 2 internals.

---

## 3. New Script: `scripts/eval/eval_14t_fusion_veri776.py`

The new script is a standalone VeRi-776 single-camera evaluation CLI. It is not a pipeline stage.

Lift the fusion math from `notebooks/kaggle/14t_veri_fusion/14t_veri_fusion.ipynb` around lines 1096-1275:

- `build_rerank_state_from_similarity`
- `compute_reranking_torch`
- `RERANK_CANONICAL = {"k1": 80, "k2": 15, "lambda_value": 0.2}`
- `AQE_K = 3`
- `WEIGHTS = [round(i / 10, 1) for i in range(11)]`
- `score_similarity`
- `score_all_similarity`
- AQE applied independently to both streams before recomputing fused similarity
- rerank built from the AQE'd all-similarity matrix

### Required Math Helpers

```python
def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    ...

def score_similarity(q_tr, g_tr, q_cs, g_cs, w: float) -> np.ndarray:
    return w * (q_cs @ g_cs.T) + (1.0 - w) * (q_tr @ g_tr.T)

def score_all_similarity(all_tr, all_cs, w: float) -> np.ndarray:
    return w * (all_cs @ all_cs.T) + (1.0 - w) * (all_tr @ all_tr.T)

def compute_distance_from_similarity(similarity: np.ndarray) -> np.ndarray:
    return 1.0 - similarity

def average_query_expansion(features: np.ndarray, k: int, iterations: int = 1) -> np.ndarray:
    ...

def build_rerank_state_from_similarity(similarity: np.ndarray, max_k1: int):
    ...

def compute_reranking_torch(original_dist, initial_rank, query_num, k1=80, k2=15, lambda_value=0.2):
    ...
```

Use `eval_market1501` from `src.training.evaluate_reid`. The 09v script already imports it.

### CLI Signature

```bash
python scripts/eval/eval_14t_fusion_veri776.py \
  --transreid-checkpoint PATH \
  --clipsenet-checkpoint PATH \
  --veri-root PATH \
  --device {cpu,cuda} \
  --w-clipsenet FLOAT \
  --transreid-stream {global,concat_patch_flip} \
  --aqe-k INT \
  --rerank-k1 INT \
  --rerank-k2 INT \
  --rerank-lambda FLOAT \
  --transreid-batch-size INT \
  --clipsenet-batch-size INT \
  --clipsenet-img-size H W \
  --output-json PATH \
  --skip-drift-parents \
  --weights-sweep \
  --concat-sweep
```

Defaults:

| Flag | Default |
|---|---:|
| `--device` | `cuda` |
| `--w-clipsenet` | `0.7` |
| `--transreid-stream` | `global` |
| `--aqe-k` | `3` |
| `--rerank-k1` | `80` |
| `--rerank-k2` | `15` |
| `--rerank-lambda` | `0.2` |
| `--transreid-batch-size` | `64` |
| `--clipsenet-batch-size` | `64` |
| `--clipsenet-img-size` | `320 320` |
| `--output-json` | required |
| `--skip-drift-parents` | false |
| `--weights-sweep` | false |
| `--concat-sweep` | false |

### Execution Flow

1. Seed `numpy` and `torch` with `1234`.
2. Resolve `query_dir = veri_root / "image_query"` and `gallery_dir = veri_root / "image_test"`.
3. Parse query and gallery splits using the refactored helper. If both eval scripts have near-identical parsers, choose one canonical parser and assert PID/camid/path equality after parsing through both formats.
4. Build the CLIP-SENet model on the requested device.
5. Build the 09v TransReID model on the requested device.
6. Extract `q_cs` and `g_cs` through `extract_clipsenet_features()` as `[N, 2048]` L2-normalized arrays.
7. Extract `q_tr` and `g_tr` through `extract_09v_features()` for the requested stream:
   - `global` -> 768-d
   - `concat_patch_flip` -> 1536-d
8. Validate alignment:
   - query PID arrays match between streams
   - query camid arrays match between streams
   - gallery PID arrays match between streams
   - gallery camid arrays match between streams
   - feature norms are finite and close to 1
9. Run the headline fusion path.
10. Run parent drift checks unless `--skip-drift-parents` is set.
11. Optionally run weights and concat sweeps.
12. Write JSON.

### Headline Fusion Path

Always run this path:

1. Compute raw fused query-gallery similarity:

```python
S_fused = score_similarity(q_tr, g_tr, q_cs, g_cs, w_clipsenet)
```

2. Build combined all-features per stream:

```python
all_tr = np.concatenate([q_tr, g_tr], axis=0)
all_cs = np.concatenate([q_cs, g_cs], axis=0)
```

3. Apply AQE independently to both streams:

```python
aqe_tr = average_query_expansion(all_tr, k=aqe_k, iterations=1)
aqe_cs = average_query_expansion(all_cs, k=aqe_k, iterations=1)
```

4. Split AQE outputs back into query and gallery rows.
5. Recompute AQE'd fused query-gallery similarity and all-similarity:

```python
qg_aqe = score_similarity(q_tr_aqe, g_tr_aqe, q_cs_aqe, g_cs_aqe, w_clipsenet)
all_similarity_aqe = score_all_similarity(aqe_tr, aqe_cs, w_clipsenet)
```

6. Build rerank state from `all_similarity_aqe`.
7. Run k-reciprocal reranking with the CLI's `k1`, `k2`, and `lambda`.
8. Evaluate the resulting distance matrix via `eval_market1501`.
9. Store this row at `score_fusion.best` when no sweep is requested.

Important: do not implement score fusion by concatenating the two feature arrays for the headline row. The 14t WIN used similarity-level fusion with AQE on both streams and rerank on fused all-similarity.

### Drift Parents

Unless `--skip-drift-parents` is set, run two pinned parent rows regardless of the headline CLI args:

1. `transreid_09v_concat_patch_aqe3_rerank`
   - Use 1536-d `concat_patch_flip` stream.
   - AQE k=3.
   - Rerank `k1=80`, `k2=15`, `lambda=0.2`.
   - Target mAP in verifier: `0.8997`.
2. `clipsenet_v6_aqe10_rerank_k1_50_k2_10_lambda_0_1`
   - Use CLIP-SENet stream alone.
   - AQE k=10.
   - Rerank `k1=50`, `k2=10`, `lambda=0.1`.
   - Target mAP in verifier: `0.9154`.

These are informational in the verifier. They are useful for diagnosing checkpoint drift and parser/model regressions.

### Optional Sweeps

If `--weights-sweep` is set:

- Run `w_clipsenet` in `[0.0, 0.1, ..., 1.0]`.
- Use the same AQE + rerank path as the headline.
- Store all rows under `score_fusion.all_rows`.
- Set `score_fusion.best` to the highest mAP row, with R1 as the tie-breaker.

If `--concat-sweep` is set:

- Run concat-fusion strategy for alpha in `[0.3, 0.5, 0.7]`.
- Treat concat as diagnostic only.
- Store rows under `concat_fusion.all_rows` if present.

### Determinism

Set fixed seeds:

```python
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)
```

Do not force globally deterministic CUDA algorithms if they create avoidable runtime regressions. The fusion math is deterministic once features are extracted; only feature extraction has tiny GPU-architecture-dependent noise.

### Required JSON Output

The output JSON must match the schema consumed by [14t-verifier.md](14t-verifier.md#5-required-json-schema-for-eval_14t_fusion_veri776py):

```json
{
  "experiment": "14t_fusion_verify",
  "wall_time_sec": 2700.0,
  "params": {
    "w_clipsenet": 0.7,
    "w_transreid": 0.3,
    "transreid_stream": "global",
    "aqe_k": 3,
    "rerank": {"k1": 80, "k2": 15, "lambda_value": 0.2}
  },
  "score_fusion": {
    "best": {"mAP": 0.933, "R1": 0.984, "R5": 0.99, "R10": 0.99},
    "all_rows": []
  },
  "drift_parents": {
    "transreid_09v_concat_patch_aqe3_rerank": {"mAP": 0.8997, "R1": 0.9815},
    "clipsenet_v6_aqe10_rerank_k1_50_k2_10_lambda_0_1": {"mAP": 0.9154, "R1": 0.9732}
  },
  "checkpoints": {
    "transreid": {"path": "...", "sha256": "..."},
    "clipsenet": {"path": "...", "sha256": "..."}
  }
}
```

Include `sha256` hashes for both checkpoint files. This is important because the CLIP-SENet v6 source is a kernel output that may drift if republished.

---

## 4. Config File: `configs/models/veri776_14t_fusion.yaml`

Create this file as metadata only. Phase 1 script does not read it.

```yaml
# 14t VeRi-776 single-camera fusion config
# This is NOT a CityFlow MTMC pipeline override. The fusion is single-cam-only.
# See docs/findings.md §14t (WIN) and §14u (CityFlow DEAD END).

veri776_14t_fusion:
  task_type: single_cam_reid
  primary_reid:
    name: transreid_09v_v17
    checkpoint: models/reid/vehicle_transreid_vit_base_veri776.pth
    stream: global  # 768-d; concat_patch_flip is 1536-d, weaker for this fusion
    img_size: [224, 224]
  secondary_reid:
    name: clipsenet_v6
    checkpoint: models/reid/clipsenet_v6_veri776_best.pth
    img_size: [320, 320]
  fusion:
    type: score_level
    formula: "w_clipsenet * S_clipsenet + (1 - w_clipsenet) * S_transreid"
    w_clipsenet: 0.7
  postprocessing:
    aqe:
      k: 3
      iterations: 1
    rerank:
      k1: 80
      k2: 15
      lambda: 0.2
  expected_metrics:
    map: 0.9330
    r1: 0.9845
    tolerance: 0.005
  provenance:
    source_kernel: yahiaakhalafallah/14t-veri-fusion-clip-senet-x-transreid
    docs: docs/findings.md  # §14t
    spec: docs/subagent-specs/14t-veri-clipsenet-transreid-fusion.md
```

Phase 2 may consume this YAML for backend configuration. Phase 1 should keep the CLI explicit and self-contained.

---

## 5. Model Registry Update

The `veri776_14t_fusion` entry already exists in `configs/model_registry.yaml` around lines 267-325. The current registry records:

- `id: veri776_14t_fusion`
- `task_type: single_cam_reid`
- `dataset: veri776`
- mAP `0.9330`
- R1 `0.9845`
- primary checkpoint `models/reid/vehicle_transreid_vit_base_veri776.pth`
- secondary checkpoint `models/reid/clipsenet_v6_veri776_best.pth`
- status `research`

Patch only if needed:

- Set `pipeline_config: configs/models/veri776_14t_fusion.yaml` after Phase 1 lands.
- Keep `status: research`.
- Keep `runnable_locally: false` unless a local helper-driven CPU tiny-subset eval is actually verified. Do not flip this based on the Kaggle verifier alone.
- Add a notes field warning that the entry is single-camera-only and must not be wired into CityFlow MTMC. Cite [docs/findings.md](../findings.md#L367-L377).

Do not promote the registry entry as a production MTMC model.

---

## 6. Test Coverage

### Unit Test: `tests/test_stage2/test_14t_fusion_math.py`

This test must be CPU-only and deterministic. It should import only math helpers that do not require model construction.

Required cases:

1. Create mock query features:
   - 4 rows x 8 dims for TransReID stream.
   - 4 rows x 8 dims for CLIP-SENet stream.
   - deterministic seed.
2. Create mock gallery features:
   - 12 rows x 8 dims for TransReID stream.
   - 12 rows x 8 dims for CLIP-SENet stream.
3. L2-normalize each matrix.
4. Assert the scalar formula:

```python
S_fused[0, 1] == w * (q_cs[0] @ g_cs[1]) + (1 - w) * (q_tr[0] @ g_tr[1])
```

5. Assert AQE k=1 on stacked features reduces to the identity-with-self-as-top-1 property.
6. Assert rerank distance behavior on a self-match matrix has diagonal self-distances at or below same-row off-diagonal distances.
7. Assert `eval_market1501` on a hand-crafted perfect-match case returns `mAP=1.0` and `R1=1.0`.

Keep the test independent of real checkpoints and VeRi images.

### Optional Smoke Test

Gate with:

```text
MTMC_RUN_14T_SMOKE=1
```

Smoke behavior:

- If `tests/fixtures/veri_mini/` exists, load a 32-image query/gallery mini split.
- Run the CLI with `--device cpu --weights-sweep`.
- Skip rerank if the CLI provides a future no-rerank switch, or keep the mini matrix tiny enough for runtime under 60s.
- Assert the fused mAP is strictly higher than either parent on the synthetic/mini subset only if the fixture was deliberately constructed for orthogonality.
- If no fixture exists, `pytest.skip`.

Do not add a slow checkpoint-dependent test to default CI.

---

## 7. Backend Wiring — Phase 2 Optional

Defer unless Phase 1 verification passes and the user asks for query-time VeRi-776-style ReID in the dashboard.

Sketch only:

- Add `backend/api/v1/reid.py` endpoint `POST /api/v1/reid/veri_fusion`.
- Input: query image and optional hosted-gallery slug.
- Output: top-k gallery IDs, similarity scores, and optional metadata.
- Add `src/stage2_features/fusion_pipeline.py` with `FusionReIDInferencer`.
- Cache gallery embedding stores under `data/outputs/<run_id>/veri_fusion_gallery/`.

Strict MTMC guardrail:

- `scripts/run_pipeline.py` must not pick this fusion automatically.
- If a future config adds `cfg.stage2.fusion_enabled`, default it to `false`.
- If `fusion_enabled=true` and `cfg.dataset.name != "veri776_single_cam"`, hard-error.
- Do not let CityFlow MTMC use this path by accident.

---

## 8. Migration & Rollback

Phase 1 is mostly additive:

- New CLI.
- New metadata config.
- New unit test.
- Helper extraction from two eval scripts.
- README and registry metadata notes.

Rollback is a normal git revert of the PR.

Pre-merge checks:

- Existing CLIP-SENet eval script `--help` still works.
- Existing 09v TransReID eval script `--help` still works.
- Existing eval JSON schema from those scripts is unchanged.
- New math unit tests pass.
- The 14aa Kaggle verifier is the full end-to-end acceptance test.

README note requirement:

- Document that `models/reid/clipsenet_v6_veri776_best.pth` is consumed by the fusion script.
- Document the expected source kernel/checkpoint path.
- Document that this is a single-camera VeRi-776 result and not a CityFlow MTMC feature stream.

---

## 9. What Not To Do

Hard rules:

- Do not add CLIP-SENet, fused or standalone, as a CityFlow MTMC tertiary/quaternary feature stream.
- Do not modify `src/stage2_features/reid_model.py`.
- Do not promote `veri776_14t_fusion` as production in the model registry.
- Do not use this fusion to replace the production CityFlow primary.
- Do not add the fusion script to the default `scripts/run_pipeline.py` chain.
- Do not use VeRi-only experts as evidence for CityFlow MTMC production readiness.
- Do not repeat the 14u CityFlow port unless a new hypothesis materially differs from score-stream fusion.

Why:

- 13d, 13f, 13h, and 14u jointly rule out CLIP-SENet/VeRi-fusion transfer to CityFlow MTMC.
- 14t remains valuable as a standalone VeRi-776 single-camera result and paper contrast.

---

## 10. Files Coder Must Create or Modify

Create:

- `scripts/eval/eval_14t_fusion_veri776.py`
- `configs/models/veri776_14t_fusion.yaml`
- `tests/test_stage2/test_14t_fusion_math.py`

Modify:

- `scripts/eval/eval_clip_senet_veri776.py` — extract helpers, no CLI change.
- `scripts/eval/eval_09v_transreid_veri776.py` — extract helpers, no CLI change.
- `configs/model_registry.yaml` — point `veri776_14t_fusion.pipeline_config` at the new YAML and add a single-camera-only warning if missing.
- `models/reid/README.md` — append a 14t fusion use-case section and scope warning.

Acceptance for Phase 1:

- Unit tests for fusion math pass locally.
- Existing eval scripts keep their CLI surfaces.
- The 14aa verifier in [14t-verifier.md](14t-verifier.md) passes on Kaggle.