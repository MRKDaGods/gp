"""Patch 10c notebook:
1. Add --dataset-config to run command (Tier 0a free gain)
2. Replace stale v34 scan grid with Phase 3 grid (temporal_split + CamTTA)
"""
import json, re
from pathlib import Path

NB_PATH = Path("notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb")

with open(NB_PATH) as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Patch cell 11: Add --dataset-config to run command ───────────────────────
run_cell = cells[11]
assert run_cell["cell_type"] == "code", f"Expected code cell at 11, got {run_cell['cell_type']}"
src = "".join(run_cell["source"])
assert "--config" in src and "run_pipeline.py" in src, "Cell 11 doesn't look like run command"

# Insert --dataset-config right after --config line
if "--dataset-config" not in src:
    src = src.replace(
        '    "--config", "configs/default.yaml",',
        '    "--config", "configs/default.yaml",\n    "--dataset-config", "configs/datasets/cityflowv2.yaml",'
    )
    lines = src.split("\n")
    run_cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    print("✓ Added --dataset-config to cell 11")
else:
    print("  cell 11 already has --dataset-config")

# ── Patch cell 15: Replace scan grid with Phase 3 grid ───────────────────────
scan_cell = cells[15]
assert scan_cell["cell_type"] == "code", f"Expected code cell at 15, got {scan_cell['cell_type']}"

new_scan_src = '''\
# ============================================================
# Phase 3 Scan: Test temporal_split + CamTTA with 384px features
# SET SCAN_ENABLED = True to run the grid search.
# ============================================================
SCAN_ENABLED = False

if SCAN_ENABLED:
    import itertools

    # Phase 3 fixed optimal baseline config (v48 best result)
    P3_SIM_THRESH  = 0.53   # optimal from v46 sweep
    P3_FIC_REG     = 3.0    # optimal FIC regularisation
    P3_AQE_K       = 2      # optimal after QE self-exclusion fix

    # Phase 3 scan grid: temporal_split + CamTTA combos
    # ~24 combinations, ~35-45 min on Kaggle T4
    scan_grid = {
        "temporal_split":          [False, True],
        "split_threshold":         [0.40, 0.45, 0.50],
        "camera_tta":              [False, True],
        "sim_thresh":              [0.51, 0.53, 0.55],
        "use_secondary":           [True],  # keep OSNet ensemble
    }

    keys   = list(scan_grid.keys())
    combos = list(itertools.product(*[scan_grid[k] for k in keys]))
    # Filter: split_threshold only matters when temporal_split=True
    combos = [c for c in combos if dict(zip(keys, c))["temporal_split"] or dict(zip(keys, c))["split_threshold"] == 0.45]
    print(f"Phase 3 scan: {len(combos)} combinations ...")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        ts     = params["temporal_split"]
        st     = params["split_threshold"]
        ctta   = params["camera_tta"]
        st_w   = P3_SIM_THRESH if params["sim_thresh"] is None else params["sim_thresh"]
        sec    = params["use_secondary"]

        scan_run = (f"scan_ts{int(ts)}_st{int(st*100)}"
                    f"_ctta{int(ctta)}_sim{int(st_w*100)}")
        (DATA_OUT / scan_run).mkdir(parents=True, exist_ok=True)

        cmd_scan = [
            sys.executable, "scripts/run_pipeline.py",
            "--config", "configs/default.yaml",
            "--dataset-config", "configs/datasets/cityflowv2.yaml",
            "--stages", "4,5",
            "--override", f"project.run_name={scan_run}",
            "--override", f"project.output_dir={DATA_OUT}",
            "--override", "stage0.cameras=[S01_c001,S01_c002,S01_c003,S02_c006,S02_c007,S02_c008]",
            "--override", f"stage4.association.graph.similarity_threshold={st_w}",
            "--override", f"stage4.association.fic.regularisation={P3_FIC_REG}",
            "--override", "stage4.association.fac.enabled=true",
            "--override", "stage4.association.fac.knn=20",
            "--override", "stage4.association.fac.learning_rate=0.5",
            "--override", "stage4.association.fac.beta=0.08",
            "--override", f"stage4.association.query_expansion.k={P3_AQE_K}",
            "--override", "stage4.association.intra_camera_merge.enabled=true",
            "--override", "stage4.association.intra_camera_merge.threshold=0.75",
            "--override", "stage4.association.intra_camera_merge.max_time_gap=60",
            "--override", "stage4.association.fic.enabled=true",
            "--override", "stage4.association.reranking.enabled=false",
            "--override", f"stage4.association.temporal_split.enabled={str(ts).lower()}",
            "--override", f"stage4.association.temporal_split.split_threshold={st}",
            "--override", f"stage2.camera_tta.enabled={str(ctta).lower()}",
            "--override", f"stage5.mtmc_only_submission={str(MTMC_ONLY).lower()}",
            "--override", "stage5.gt_frame_clip=true",
            "--override", "stage5.gt_frame_clip_min_iou=0.5",
            "--override", "stage5.gt_zone_filter=true",
            "--override", "stage5.gt_zone_margin_frac=0.2",
        ]
        if sec and has_secondary:
            sec_emb = str(RUN_DIR / "stage2" / "embeddings_secondary.npy")
            cmd_scan += [
                "--override", f"stage4.association.secondary_embeddings.path={sec_emb}",
                "--override", f"stage4.association.secondary_embeddings.weight=0.25",
            ]
        if GT_DIR:
            cmd_scan += ["--override", f"stage5.ground_truth_dir={GT_DIR}"]

        t0 = time.time()
        r  = subprocess.run(cmd_scan, cwd=str(PROJECT), capture_output=True)
        elapsed = time.time() - t0

        report = DATA_OUT / scan_run / "stage5" / "evaluation_report.json"
        idf1 = mota = hota = 0.0
        if report.exists():
            rp = json.loads(report.read_text())
            m  = rp.get("metrics", rp)
            idf1 = m.get("mtmc_idf1", m.get("IDF1", m.get("idf1", 0.0)))
            mota = m.get("MOTA", m.get("mota", 0.0))
            hota = m.get("HOTA", m.get("hota", 0.0))

        results.append({**params, "IDF1": idf1, "MOTA": mota, "HOTA": hota, "time": elapsed})
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  [{status}] ts={int(ts)} st={st} ctta={int(ctta)} sim={st_w} -> IDF1={idf1:.4f} ({elapsed:.0f}s)")

    has_hota = any(r2["HOTA"] > 0 for r2 in results)
    sort_key = "HOTA" if has_hota else "IDF1"
    results.sort(key=lambda x: x[sort_key], reverse=True)
    print()
    print("=" * 80)
    print(f"PHASE 3 SCAN RESULTS (sorted by {sort_key})")
    print("=" * 80)
    for idx, r2 in enumerate(results[:20]):
        print(f"  #{idx+1}: ts={int(r2['temporal_split'])} st={r2['split_threshold']} ctta={int(r2['camera_tta'])} sim={r2['sim_thresh']} -> IDF1={r2['IDF1']:.4f} MOTA={r2['MOTA']:.4f} HOTA={r2['HOTA']:.4f}")
    best = results[0]
    print(f"\\nBEST: ts={int(best['temporal_split'])} st={best['split_threshold']} ctta={int(best['camera_tta'])} sim={best['sim_thresh']} -> IDF1={best['IDF1']:.4f}")

    import json as _json
    results_path = DATA_OUT / "scan_phase3_results.json"
    with open(results_path, "w") as _f:
        _json.dump({"sort_key": sort_key, "results": results}, _f, indent=2)
    import shutil as _shutil
    _shutil.copy2(str(results_path), "/kaggle/working/scan_phase3_results.json")
    print("Scan results saved to /kaggle/working/scan_phase3_results.json")
else:
    print("Phase 3 scan disabled. Set SCAN_ENABLED = True to run.")
'''

lines = new_scan_src.split("\n")
scan_cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
print("✓ Replaced cell 15 with Phase 3 scan grid")

# ── Save ──────────────────────────────────────────────────────────────────────
with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"\n✓ Saved: {NB_PATH}")
print(f"  Total cells: {len(cells)}")
