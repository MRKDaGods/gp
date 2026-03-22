"""Edit the 10c notebook scan cell for v72 intra-merge + QE alpha sweep."""
import json

nb_path = "notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the scan cell (contains SCAN_ENABLED)
scan_idx = None
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "SCAN_ENABLED" in src:
        scan_idx = i
        break

print(f"Scan cell index: {scan_idx}")
print(f"Total cells: {len(nb['cells'])}")

if scan_idx is None:
    raise RuntimeError("Could not find SCAN_ENABLED cell")

# Show first 5 lines
for line in nb["cells"][scan_idx]["source"][:5]:
    print(repr(line))

# New scan cell source for v72: intra-merge sweep + QE alpha sweep
new_source = r'''# ============================================================
# v72: Intra-camera merge threshold/gap sweep + QE alpha sweep
# Current best v71: INTRA_MERGE_THRESH=0.75, INTRA_MERGE_GAP=60, QE_ALPHA=5.0
# mtmc_idf1=78.14%. These are the last untouched 10c-only knobs.
# ============================================================
SCAN_ENABLED = True

if SCAN_ENABLED:
    import itertools

    # Phase 1: Intra-merge threshold x gap (with QE alpha=5.0 fixed)
    intra_grid = {
        "intra_thresh": [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
        "intra_gap":    [30, 60, 90, 120, 180],
    }
    # Phase 2: QE alpha sweep (with intra-merge at current best 0.75/60)
    qe_alphas = [1.0, 3.0, 5.0, 8.0, 12.0]

    # Build all configs
    configs = []
    # Phase 1: intra-merge grid
    for thresh, gap in itertools.product(intra_grid["intra_thresh"], intra_grid["intra_gap"]):
        configs.append({
            "tag": f"im_t{thresh:.2f}_g{gap}",
            "intra_enabled": True,
            "intra_thresh": thresh,
            "intra_gap": gap,
            "qe_alpha": 5.0,
        })
    # Phase 2: QE alpha (with current best intra-merge)
    for alpha in qe_alphas:
        configs.append({
            "tag": f"qe_a{alpha:.1f}",
            "intra_enabled": True,
            "intra_thresh": 0.75,
            "intra_gap": 60,
            "qe_alpha": alpha,
        })
    # Control: intra-merge disabled
    configs.append({
        "tag": "im_OFF",
        "intra_enabled": False,
        "intra_thresh": 0.75,
        "intra_gap": 60,
        "qe_alpha": 5.0,
    })

    print(f"v72 scan: {len(configs)} configs ({len(intra_grid['intra_thresh'])}x{len(intra_grid['intra_gap'])} intra-merge + {len(qe_alphas)} QE alpha + 1 control)")

    results = []
    for cfg in configs:
        scan_run = f"scan_{cfg['tag']}"

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
            "--override", f"stage4.association.query_expansion.k={AQE_K}",
            "--override", f"stage4.association.query_expansion.alpha={cfg['qe_alpha']}",
            "--override", f"stage4.association.graph.similarity_threshold={SIM_THRESH}",
            "--override", f"stage4.association.graph.algorithm={ALGORITHM}",
            "--override", f"stage4.association.graph.louvain_resolution={LOUVAIN_RES}",
            "--override", f"stage4.association.graph.bridge_prune_margin={BRIDGE_PRUNE}",
            "--override", f"stage4.association.graph.max_component_size={MAX_COMP_SIZE}",
            "--override", f"stage4.association.weights.vehicle.appearance={APPEARANCE_WEIGHT}",
            "--override", f"stage4.association.weights.vehicle.hsv={HSV_WEIGHT}",
            "--override", f"stage4.association.weights.vehicle.spatiotemporal={ST_WEIGHT}",
            "--override", "stage4.association.mutual_nn.top_k_per_query=20",
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
            "--override", f"stage4.association.intra_camera_merge.enabled={str(cfg['intra_enabled']).lower()}",
            "--override", f"stage4.association.intra_camera_merge.threshold={cfg['intra_thresh']}",
            "--override", f"stage4.association.intra_camera_merge.max_time_gap={cfg['intra_gap']}",
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

        # Read metrics
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
        print(f"  [{status}] {cfg['tag']:<20} -> mtmc_idf1={mtmc_idf1:.4f} IDF1={idf1:.4f} HOTA={hota:.4f} ids={ids} ({elapsed:.0f}s)")

    # Sort by mtmc_idf1
    results.sort(key=lambda x: x["mtmc_idf1"], reverse=True)
    print("\n" + "=" * 90)
    print("v72 SCAN RESULTS (sorted by mtmc_idf1)")
    print("=" * 90)
    print(f"{'tag':<22} {'intra':<6} {'thresh':<7} {'gap':<5} {'qe_a':<6} {'mtmc_idf1':>10} {'IDF1':>7} {'HOTA':>7} {'IDSW':>5}")
    for r2 in results:
        en = "ON" if r2["intra_enabled"] else "OFF"
        print(f"{r2['tag']:<22} {en:<6} {r2['intra_thresh']:<7} {r2['intra_gap']:<5} {r2['qe_alpha']:<6} "
              f"{r2['mtmc_idf1']:>10.4f} {r2['IDF1']:>7.4f} {r2['HOTA']:>7.4f} {r2['IDSW']:>5}")
    best = results[0]
    print(f"\nBEST: {best['tag']} -> mtmc_idf1={best['mtmc_idf1']:.4f}")

    # Phase-specific analysis
    print("\n--- INTRA-MERGE: threshold sensitivity (avg across gaps) ---")
    for thresh in sorted(set(r2["intra_thresh"] for r2 in results if r2["tag"].startswith("im_t"))):
        subset = [r2 for r2 in results if r2["tag"].startswith("im_t") and r2["intra_thresh"] == thresh]
        if subset:
            avg = sum(r2["mtmc_idf1"] for r2 in subset) / len(subset)
            best_v = max(r2["mtmc_idf1"] for r2 in subset)
            print(f"  thresh={thresh:.2f}  avg={avg:.4f}  best={best_v:.4f}  (n={len(subset)})")

    print("\n--- INTRA-MERGE: gap sensitivity (avg across thresholds) ---")
    for gap in sorted(set(r2["intra_gap"] for r2 in results if r2["tag"].startswith("im_t"))):
        subset = [r2 for r2 in results if r2["tag"].startswith("im_t") and r2["intra_gap"] == gap]
        if subset:
            avg = sum(r2["mtmc_idf1"] for r2 in subset) / len(subset)
            best_v = max(r2["mtmc_idf1"] for r2 in subset)
            print(f"  gap={gap:<5}  avg={avg:.4f}  best={best_v:.4f}  (n={len(subset)})")

    print("\n--- QE ALPHA sensitivity ---")
    for r2 in results:
        if r2["tag"].startswith("qe_"):
            print(f"  alpha={r2['qe_alpha']:<5} -> mtmc_idf1={r2['mtmc_idf1']:.4f}  IDF1={r2['IDF1']:.4f}  ids={r2['IDSW']}")

    print("\n--- CONTROL (intra-merge OFF) ---")
    ctrl = [r2 for r2 in results if r2["tag"] == "im_OFF"]
    if ctrl:
        print(f"  mtmc_idf1={ctrl[0]['mtmc_idf1']:.4f}  (vs best ON: {best['mtmc_idf1']:.4f}, delta={best['mtmc_idf1']-ctrl[0]['mtmc_idf1']:+.4f})")

    # Save results
    import json as _json
    results_path = DATA_OUT / "scan_results.json"
    with open(results_path, "w") as _f:
        _json.dump({"scan": "v72_intramerge_qe", "results": results}, _f, indent=2)
    print(f"\nResults saved to {results_path}")
    import shutil as _shutil
    _out = Path("/kaggle/working")
    _shutil.copy2(str(results_path), str(_out / "scan_results.json"))
    print(f"Copied to {_out}")
else:
    print("Scan disabled. Set SCAN_ENABLED = True to run v72 sweep.")
'''

# Split into lines with proper \n endings for notebook format
lines = new_source.strip().split("\n")
source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]]

# Replace the scan cell source
nb["cells"][scan_idx]["source"] = source_lines
nb["cells"][scan_idx]["outputs"] = []
nb["cells"][scan_idx]["execution_count"] = None

# Write back
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print(f"\nDone! Updated cell {scan_idx} with v72 intra-merge + QE alpha sweep")
print(f"  Intra-merge: 6 thresholds x 5 gaps = 30 configs")
print(f"  QE alpha: 5 configs")
print(f"  Control: 1 config")
print(f"  Total: 36 configs")
