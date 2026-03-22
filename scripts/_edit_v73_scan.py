"""
v73: Update 10c scan cell for appearance_weight sweep.
This will run on top of the reverted PCA 384 features + intra-merge v72 wins.
Also sweeps DBA iterations and mutual_nn top_k.
"""
import json

nb_path = "notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find scan cell
scan_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "SCAN_ENABLED" in src:
        scan_idx = i
        break

print(f"Scan cell: {scan_idx}")

new_source = r'''# ============================================================
# v73: Appearance weight + DBA iterations + top_k sweep
# Baseline v72: mtmc_idf1=78.28% (PCA 384, intra 0.80/30)
# PCA 512 HURT (-0.6pp) -> stay on 384.
# Sweep last untouched 10c-only knobs.
# ============================================================
SCAN_ENABLED = True

if SCAN_ENABLED:
    import itertools

    configs = []

    # Phase 1: Appearance weight sweep (st_weight auto-computed)
    for app_w in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        configs.append({
            "tag": f"app_{app_w:.2f}",
            "appearance_w": app_w,
            "hsv_w": 0.0,
            "st_w": round(1.0 - app_w - 0.0, 4),
            "dba_iterations": 1,
            "top_k": 20,
        })

    # Phase 2: DBA iterations (with app_w=0.75 baseline)
    for dba in [0, 2, 3]:
        configs.append({
            "tag": f"dba_{dba}",
            "appearance_w": 0.75,
            "hsv_w": 0.0,
            "st_w": 0.25,
            "dba_iterations": dba,
            "top_k": 20,
        })

    # Phase 3: mutual_nn top_k (affects sparse graph density)
    for tk in [10, 15, 25, 30, 40]:
        configs.append({
            "tag": f"topk_{tk}",
            "appearance_w": 0.75,
            "hsv_w": 0.0,
            "st_w": 0.25,
            "dba_iterations": 1,
            "top_k": tk,
        })

    print(f"v73 scan: {len(configs)} configs (7 app_w + 3 DBA + 5 top_k)")

    results = []
    for cfg in configs:
        scan_run = f"scan_{cfg['tag']}"

        scan_dir = DATA_OUT / scan_run
        scan_dir.mkdir(parents=True, exist_ok=True)
        for stage_sub in ("stage1", "stage2", "stage3"):
            src = DATA_OUT / RUN_NAME / stage_sub
            dst = scan_dir / stage_sub
            if src.exists() and not dst.exists():
                dst.symlink_to(src)

        cmd_scan = [
            sys.executable, "scripts/run_pipeline.py",
            "--config", "configs/default.yaml",
            "--dataset-config", "configs/datasets/cityflowv2.yaml",
            "--stages", "4,5",
            "--override", f"project.run_name={scan_run}",
            "--override", f"project.output_dir={DATA_OUT}",
            "--override", f"stage4.association.query_expansion.k={AQE_K}",
            "--override", "stage4.association.query_expansion.alpha=5.0",
            "--override", f"stage4.association.query_expansion.dba_iterations={cfg['dba_iterations']}",
            "--override", f"stage4.association.graph.similarity_threshold={SIM_THRESH}",
            "--override", f"stage4.association.graph.algorithm={ALGORITHM}",
            "--override", f"stage4.association.graph.louvain_resolution={LOUVAIN_RES}",
            "--override", f"stage4.association.graph.bridge_prune_margin={BRIDGE_PRUNE}",
            "--override", f"stage4.association.graph.max_component_size={MAX_COMP_SIZE}",
            "--override", f"stage4.association.weights.vehicle.appearance={cfg['appearance_w']}",
            "--override", f"stage4.association.weights.vehicle.hsv={cfg['hsv_w']}",
            "--override", f"stage4.association.weights.vehicle.spatiotemporal={cfg['st_w']}",
            "--override", f"stage4.association.mutual_nn.top_k_per_query={cfg['top_k']}",
            "--override", "stage4.association.fic.enabled=true",
            "--override", "stage4.association.fic.regularisation=0.1",
            "--override", "stage4.association.fac.enabled=false",
            "--override", "stage4.association.fac.knn=20",
            "--override", "stage4.association.fac.learning_rate=0.5",
            "--override", "stage4.association.fac.beta=0.08",
            "--override", f"stage4.association.secondary_embeddings.path={RUN_DIR}/stage2/embeddings_secondary.npy",
            "--override", f"stage4.association.secondary_embeddings.weight={FUSION_WEIGHT}",
            "--override", f"stage4.association.camera_bias.enabled={str(CAMERA_BIAS).lower()}",
            "--override", f"stage4.association.camera_bias.iterations={CAMERA_BIAS_ITERS}",
            "--override", f"stage4.association.zone_model.enabled={str(ZONE_MODEL).lower()}",
            "--override", "stage4.association.zone_model.zone_data_path=configs/datasets/cityflowv2_zones.json",
            "--override", f"stage4.association.zone_model.bonus={ZONE_BONUS}",
            "--override", f"stage4.association.zone_model.penalty={ZONE_PENALTY}",
            "--override", f"stage4.association.hierarchical.enabled={str(HIERARCHICAL).lower()}",
            "--override", f"stage4.association.hierarchical.centroid_threshold={HIER_CENTROID_TH}",
            "--override", f"stage4.association.hierarchical.merge_threshold={HIER_MERGE_TH}",
            "--override", f"stage4.association.hierarchical.orphan_threshold={HIER_ORPHAN_TH}",
            "--override", "stage4.association.hierarchical.max_merge_size=12",
            "--override", f"stage4.association.intra_camera_merge.enabled={str(INTRA_MERGE).lower()}",
            "--override", f"stage4.association.intra_camera_merge.threshold={INTRA_MERGE_THRESH}",
            "--override", f"stage4.association.intra_camera_merge.max_time_gap={INTRA_MERGE_GAP}",
            "--override", f"stage5.mtmc_only_submission={str(MTMC_ONLY).lower()}",
            "--override", "stage5.stationary_filter.enabled=true",
            "--override", "stage5.stationary_filter.min_displacement_px=150",
            "--override", "stage5.stationary_filter.max_mean_velocity_px=2.0",
            "--override", "stage5.min_submission_confidence=0.15",
            "--override", "stage5.cross_id_nms_iou=0.40",
            "--override", "stage5.min_trajectory_confidence=0.30",
            "--override", "stage5.min_trajectory_frames=40",
            "--override", "stage5.track_edge_trim.enabled=false",
            "--override", "stage5.track_smoothing.enabled=false",
            "--override", "stage5.gt_frame_clip=true",
            "--override", "stage5.gt_zone_filter=true",
        ]
        if GT_DIR:
            cmd_scan += ["--override", f"stage5.ground_truth_dir={GT_DIR}"]
        t0 = time.time()
        r = subprocess.run(cmd_scan, cwd=str(PROJECT), capture_output=True)
        elapsed = time.time() - t0

        report = DATA_OUT / scan_run / "stage5" / "evaluation_report.json"
        mtmc_idf1 = idf1 = mota = hota = 0.0
        ids = 0
        if report.exists():
            rp = json.loads(report.read_text())
            m = rp.get("metrics", rp)
            mtmc_idf1 = m.get("mtmc_idf1", 0.0)
            idf1 = m.get("IDF1", m.get("idf1", 0.0))
            mota = m.get("MOTA", m.get("mota", 0.0))
            hota = m.get("HOTA", m.get("hota", 0.0))
            ids = m.get("IDSW", m.get("id_switches", 0))

        results.append({
            **cfg,
            "mtmc_idf1": mtmc_idf1, "IDF1": idf1, "MOTA": mota,
            "HOTA": hota, "IDSW": ids, "time": elapsed,
        })
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  [{status}] {cfg['tag']:<15} -> mtmc_idf1={mtmc_idf1:.4f} IDF1={idf1:.4f} HOTA={hota:.4f} ids={ids} ({elapsed:.0f}s)")

    results.sort(key=lambda x: x["mtmc_idf1"], reverse=True)
    print("\n" + "=" * 90)
    print("v73 SCAN RESULTS (sorted by mtmc_idf1)")
    print("=" * 90)
    print(f"{'tag':<15} {'app_w':<7} {'st_w':<7} {'dba':<5} {'topk':<6} {'mtmc_idf1':>10} {'IDF1':>7} {'HOTA':>7} {'IDSW':>5}")
    for r2 in results:
        print(f"{r2['tag']:<15} {r2['appearance_w']:<7} {r2['st_w']:<7} {r2['dba_iterations']:<5} {r2['top_k']:<6} "
              f"{r2['mtmc_idf1']:>10.4f} {r2['IDF1']:>7.4f} {r2['HOTA']:>7.4f} {r2['IDSW']:>5}")
    best = results[0]
    print(f"\nBEST: {best['tag']} -> mtmc_idf1={best['mtmc_idf1']:.4f}")

    print("\n--- APPEARANCE WEIGHT sensitivity ---")
    for r2 in results:
        if r2["tag"].startswith("app_"):
            print(f"  app_w={r2['appearance_w']:<5} st_w={r2['st_w']:<5} -> mtmc_idf1={r2['mtmc_idf1']:.4f}  IDF1={r2['IDF1']:.4f}  ids={r2['IDSW']}")

    print("\n--- DBA ITERATIONS sensitivity ---")
    for r2 in results:
        if r2["tag"].startswith("dba_"):
            print(f"  dba={r2['dba_iterations']} -> mtmc_idf1={r2['mtmc_idf1']:.4f}  IDF1={r2['IDF1']:.4f}  ids={r2['IDSW']}")

    print("\n--- TOP_K sensitivity ---")
    for r2 in results:
        if r2["tag"].startswith("topk_"):
            print(f"  top_k={r2['top_k']:<3} -> mtmc_idf1={r2['mtmc_idf1']:.4f}  IDF1={r2['IDF1']:.4f}  ids={r2['IDSW']}")

    import json as _json
    results_path = DATA_OUT / "scan_results.json"
    with open(results_path, "w") as _f:
        _json.dump({"scan": "v73_appw_dba_topk", "results": results}, _f, indent=2)
    print(f"\nResults saved to {results_path}")
    import shutil as _shutil
    _out = Path("/kaggle/working")
    _shutil.copy2(str(results_path), str(_out / "scan_results.json"))
    print(f"Copied to {_out}")
else:
    print("Scan disabled. Set SCAN_ENABLED = True to run v73 sweep.")
'''

lines = new_source.strip().split("\n")
source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]]

nb["cells"][scan_idx]["source"] = source_lines
nb["cells"][scan_idx]["outputs"] = []
nb["cells"][scan_idx]["execution_count"] = None

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"Done! Updated scan cell with v73 appearance_weight + DBA + top_k sweep")
print(f"  7 appearance_w + 3 DBA + 5 top_k = 15 configs")
