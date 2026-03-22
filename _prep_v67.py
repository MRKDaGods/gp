"""
v67: Replace scan cell (Cell 15) with focused algorithm sweep.
Tests conflict_free_cc vs Louvain (community_detection) at 5 resolutions
+ connected_components as reference.

All other params fixed at v66 best values.
"""
import json

NB_PATH = r'e:\dev\src\gp\notebooks\kaggle\10c_stages45\mtmc-10c-stages-4-5-association-eval.ipynb'

nb = json.load(open(NB_PATH, encoding='utf-8'))

# --- New scan cell source ---
scan_source = r'''# ============================================================
# v67: ALGORITHM SWEEP — Louvain vs conflict_free_cc
# Tests community_detection at 5 resolutions + baselines
# All other params fixed at v66 best.  ~7 runs × 25s = ~3 min
# ============================================================
SCAN_ENABLED = True

if SCAN_ENABLED:
    experiments = [
        {"name": "cfcc_baseline",  "algorithm": "conflict_free_cc",      "louvain_res": 0.7},
        {"name": "cc_baseline",    "algorithm": "connected_components",   "louvain_res": 0.7},
        {"name": "louvain_050",    "algorithm": "community_detection",    "louvain_res": 0.5},
        {"name": "louvain_070",    "algorithm": "community_detection",    "louvain_res": 0.7},
        {"name": "louvain_100",    "algorithm": "community_detection",    "louvain_res": 1.0},
        {"name": "louvain_120",    "algorithm": "community_detection",    "louvain_res": 1.2},
        {"name": "louvain_150",    "algorithm": "community_detection",    "louvain_res": 1.5},
    ]
    print(f"Running {len(experiments)} algorithm experiments ...")

    results = []
    for exp in experiments:
        scan_run = f"scan_{exp['name']}"

        # Symlink upstream stages
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
            # v66 fixed params
            "--override", f"stage4.association.query_expansion.k={AQE_K}",
            "--override", f"stage4.association.graph.similarity_threshold={SIM_THRESH}",
            "--override", f"stage4.association.graph.bridge_prune_margin={BRIDGE_PRUNE}",
            "--override", f"stage4.association.graph.max_component_size={MAX_COMP_SIZE}",
            "--override", f"stage4.association.weights.vehicle.appearance={APPEARANCE_WEIGHT}",
            "--override", f"stage4.association.weights.vehicle.hsv={HSV_WEIGHT}",
            "--override", f"stage4.association.weights.vehicle.spatiotemporal={ST_WEIGHT}",
            "--override", "stage4.association.mutual_nn.top_k_per_query=20",
            "--override", "stage4.association.fic.enabled=true",
            "--override", "stage4.association.fic.regularisation=3.0",
            "--override", "stage4.association.fac.enabled=false",
            "--override", f"stage4.association.intra_camera_merge.enabled={str(INTRA_MERGE).lower()}",
            "--override", f"stage4.association.intra_camera_merge.threshold={INTRA_MERGE_THRESH}",
            "--override", f"stage4.association.intra_camera_merge.max_time_gap={INTRA_MERGE_GAP}",
            "--override", f"stage4.association.secondary_embeddings.path={RUN_DIR}/stage2/embeddings_secondary.npy",
            "--override", f"stage4.association.secondary_embeddings.weight={FUSION_WEIGHT}",
            "--override", f"stage4.association.camera_bias.enabled={str(CAMERA_BIAS).lower()}",
            "--override", f"stage4.association.zone_model.enabled={str(ZONE_MODEL).lower()}",
            "--override", f"stage4.association.hierarchical.enabled={str(HIERARCHICAL).lower()}",
            # Variable params for this experiment
            "--override", f"stage4.association.graph.algorithm={exp['algorithm']}",
            "--override", f"stage4.association.graph.louvain_resolution={exp['louvain_res']}",
            # Stage 5: fixed at v66 values
            "--override", f"stage5.mtmc_only_submission={str(MTMC_ONLY).lower()}",
            "--override", "stage5.stationary_filter.enabled=true",
            "--override", "stage5.stationary_filter.min_displacement_px=150",
            "--override", "stage5.stationary_filter.max_mean_velocity_px=2.0",
            "--override", "stage5.min_submission_confidence=0.15",
            "--override", "stage5.cross_id_nms_iou=0.35",
            "--override", "stage5.min_trajectory_confidence=0.30",
            "--override", "stage5.min_trajectory_frames=10",
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

        # Read metrics
        report = DATA_OUT / scan_run / "stage5" / "evaluation_report.json"
        idf1 = mota = hota = mtmc_idf1 = 0.0
        id_switches = 0
        if report.exists():
            rp = json.loads(report.read_text())
            m = rp.get("metrics", rp)
            idf1 = m.get("IDF1", m.get("idf1", 0.0))
            mota = m.get("MOTA", m.get("mota", 0.0))
            hota = m.get("HOTA", m.get("hota", 0.0))
            mtmc_idf1 = m.get("mtmc_idf1", 0.0)
            id_switches = m.get("id_switches", 0)

        results.append({
            "name": exp["name"], "algorithm": exp["algorithm"],
            "louvain_res": exp["louvain_res"],
            "IDF1": idf1, "MOTA": mota, "HOTA": hota,
            "mtmc_idf1": mtmc_idf1, "id_switches": id_switches,
            "time": elapsed
        })
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  [{status}] {exp['name']:20s} alg={exp['algorithm']:25s} res={exp['louvain_res']:.1f} "
              f"-> IDF1={idf1:.4f} mtmc_idf1={mtmc_idf1:.4f} HOTA={hota:.4f} ids={id_switches} ({elapsed:.0f}s)")

    # Sort by mtmc_idf1
    results.sort(key=lambda x: x["mtmc_idf1"], reverse=True)
    print("\n" + "=" * 90)
    print("ALGORITHM SWEEP RESULTS (sorted by mtmc_idf1)")
    print("=" * 90)
    print(f"{'Name':<20s} {'Algorithm':<25s} {'Res':>5s} {'IDF1':>7s} {'mtmc_idf1':>10s} {'HOTA':>7s} {'IDs':>5s}")
    for r2 in results:
        print(f"{r2['name']:<20s} {r2['algorithm']:<25s} {r2['louvain_res']:>5.1f} "
              f"{r2['IDF1']:>7.4f} {r2['mtmc_idf1']:>10.4f} {r2['HOTA']:>7.4f} {r2['id_switches']:>5d}")
    best = results[0]
    print(f"\nBEST: {best['name']} ({best['algorithm']} res={best['louvain_res']}) "
          f"-> mtmc_idf1={best['mtmc_idf1']:.4f} IDF1={best['IDF1']:.4f}")

    # Save results
    import json as _json
    results_path = DATA_OUT / "scan_results.json"
    with open(results_path, "w") as _f:
        _json.dump({"scan_type": "algorithm_sweep_v67", "results": results}, _f, indent=2)
    print(f"\nSaved to {results_path}")
    import shutil as _shutil
    _out = Path("/kaggle/working")
    _shutil.copy2(str(results_path), str(_out / "scan_results.json"))
else:
    print("Scan disabled. Set SCAN_ENABLED = True to run algorithm sweep.")
'''

# Split into lines with proper \n endings
lines = scan_source.strip().split('\n')
source_list = [line + '\n' for line in lines[:-1]] + [lines[-1]]

# Replace Cell 15
nb['cells'][15]['source'] = source_list

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"Updated Cell 15 with v67 algorithm sweep ({len(source_list)} lines)")
print("SCAN_ENABLED = True")
