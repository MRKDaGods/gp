"""Edit 10c notebook scan cell for v74: unexplored features scan.

Tests CSLS hubness reduction, cluster_verify, temporal_split,
gallery_expansion thresholds, and length_weight_power — all untouched
since project start. All 10c-only (no GPU needed).
"""
import json, sys

NB = "notebooks/kaggle/10c_stages45/mtmc-10c-stages-4-5-association-eval.ipynb"

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find scan cell (c16)
scan_idx = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("id") == "c16":
        scan_idx = i
        break

if scan_idx is None:
    print("ERROR: scan cell c16 not found")
    sys.exit(1)

new_source = r'''# ============================================================
# v74: Unexplored features scan -- CSLS, cluster_verify,
#       temporal_split, gallery_expansion, length_weight_power
# Baseline v73: mtmc_idf1=78.0% (app_w=0.70, intra 0.80/30)
# All configs are 10c-only (no GPU needed).
# ============================================================
SCAN_ENABLED = True

if SCAN_ENABLED:
    configs = []

    # Each config: {"tag": str, "overrides": [str, ...]}
    # Overrides are ADDITIONAL to the base command (on top of v73 params)

    # 1. Control (exact same as v37 baseline)
    configs.append({"tag": "control", "overrides": []})

    # --- Phase A: Binary feature tests ---
    # 2. CSLS hubness reduction (penalizes hub embeddings)
    configs.append({"tag": "csls_on", "overrides": [
        "stage4.association.csls.enabled=true",
        "stage4.association.csls.k=10",
    ]})

    # 3. Cluster post-verification (eject weak members)
    configs.append({"tag": "clverify_030", "overrides": [
        "stage4.association.cluster_verify.enabled=true",
        "stage4.association.cluster_verify.min_connectivity=0.30",
    ]})
    configs.append({"tag": "clverify_025", "overrides": [
        "stage4.association.cluster_verify.enabled=true",
        "stage4.association.cluster_verify.min_connectivity=0.25",
    ]})
    configs.append({"tag": "clverify_035", "overrides": [
        "stage4.association.cluster_verify.enabled=true",
        "stage4.association.cluster_verify.min_connectivity=0.35",
    ]})

    # 4. Temporal split (split conflated same-looking vehicles)
    configs.append({"tag": "tsplit_60_050", "overrides": [
        "stage4.association.temporal_split.enabled=true",
        "stage4.association.temporal_split.min_gap=60",
        "stage4.association.temporal_split.split_threshold=0.50",
    ]})
    configs.append({"tag": "tsplit_30_050", "overrides": [
        "stage4.association.temporal_split.enabled=true",
        "stage4.association.temporal_split.min_gap=30",
        "stage4.association.temporal_split.split_threshold=0.50",
    ]})
    configs.append({"tag": "tsplit_60_045", "overrides": [
        "stage4.association.temporal_split.enabled=true",
        "stage4.association.temporal_split.min_gap=60",
        "stage4.association.temporal_split.split_threshold=0.45",
    ]})

    # --- Phase B: length_weight_power sweep ---
    # Default is 0.3; controls how much longer tracklets influence similarity
    for lwp in [0.0, 0.1, 0.5, 0.7, 1.0]:
        configs.append({"tag": f"lwp_{lwp:.1f}", "overrides": [
            f"stage4.association.weights.length_weight_power={lwp}",
        ]})

    # --- Phase C: gallery_expansion threshold sweep ---
    # Default: threshold=0.50, orphan=0.40
    for ge_th in [0.40, 0.45, 0.55, 0.60]:
        configs.append({"tag": f"galexp_{ge_th:.2f}", "overrides": [
            f"stage4.association.gallery_expansion.threshold={ge_th}",
        ]})

    # --- Phase D: exhaustive_min_similarity ---
    for ems in [0.05, 0.15, 0.20]:
        configs.append({"tag": f"exhmin_{ems:.2f}", "overrides": [
            f"stage4.association.exhaustive_min_similarity={ems}",
        ]})

    print(f"v74 scan: {len(configs)} configs")
    print("  Phase A: 1 control + 2 CSLS + 3 cluster_verify + 3 temporal_split")
    print("  Phase B: 5 length_weight_power")
    print("  Phase C: 4 gallery_expansion")
    print("  Phase D: 3 exhaustive_min_similarity")

    results = []
    for ci, cfg in enumerate(configs):
        tag = cfg["tag"]
        extra = cfg["overrides"]
        scan_run = f"scan_{tag}"
        scan_dir = DATA_OUT / scan_run
        scan_dir.mkdir(parents=True, exist_ok=True)
        for stage_sub in ("stage1", "stage2", "stage3"):
            src = DATA_OUT / RUN_NAME / stage_sub
            dst = scan_dir / stage_sub
            if src.exists() and not dst.exists():
                dst.symlink_to(src)

        # Base command (v73 params)
        cmd_scan = [
            sys.executable, "scripts/run_pipeline.py",
            "--config", "configs/default.yaml",
            "--dataset-config", "configs/datasets/cityflowv2.yaml",
            "--stages", "4,5",
            "--override", f"project.run_name={scan_run}",
            "--override", f"project.output_dir={DATA_OUT}",
            "--override", f"stage4.association.query_expansion.k={AQE_K}",
            "--override", "stage4.association.query_expansion.alpha=5.0",
            "--override", f"stage4.association.graph.similarity_threshold={SIM_THRESH}",
            "--override", f"stage4.association.graph.algorithm={ALGORITHM}",
            "--override", f"stage4.association.graph.louvain_resolution={LOUVAIN_RES}",
            "--override", f"stage4.association.graph.bridge_prune_margin={BRIDGE_PRUNE}",
            "--override", f"stage4.association.graph.max_component_size={MAX_COMP_SIZE}",
            "--override", f"stage4.association.weights.vehicle.appearance={APPEARANCE_WEIGHT}",
            "--override", "stage4.association.weights.vehicle.hsv=0.0",
            "--override", f"stage4.association.weights.vehicle.spatiotemporal={ST_WEIGHT}",
            "--override", "stage4.association.mutual_nn.top_k_per_query=20",
            "--override", "stage4.association.fic.enabled=true",
            "--override", "stage4.association.fic.regularisation=0.1",
            "--override", "stage4.association.fac.enabled=false",
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
        # Add config-specific overrides
        for ov in extra:
            cmd_scan += ["--override", ov]

        print(f"\n[{ci+1}/{len(configs)}] {tag} (+{len(extra)} overrides)")
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
            "tag": tag, "overrides": extra,
            "mtmc_idf1": mtmc_idf1, "IDF1": idf1, "MOTA": mota,
            "HOTA": hota, "IDSW": ids, "time": elapsed,
        })
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  [{status}] {tag:<18} -> mtmc_idf1={mtmc_idf1:.4f} IDF1={idf1:.4f} HOTA={hota:.4f} ids={ids} ({elapsed:.0f}s)")

    # Sort and print summary
    results.sort(key=lambda x: x["mtmc_idf1"], reverse=True)
    print("\n" + "=" * 90)
    print("v74 SCAN RESULTS (sorted by mtmc_idf1)")
    print("=" * 90)
    print(f"{'tag':<18} {'mtmc_idf1':>10} {'IDF1':>7} {'HOTA':>7} {'IDSW':>5} {'time':>6}  overrides")
    for r2 in results:
        ov_str = ", ".join(r2["overrides"]) if r2["overrides"] else "(baseline)"
        print(f"{r2['tag']:<18} {r2['mtmc_idf1']:>10.4f} {r2['IDF1']:>7.4f} {r2['HOTA']:>7.4f} {r2['IDSW']:>5} {r2['time']:>5.0f}s  {ov_str}")
    best = results[0]
    print(f"\nBEST: {best['tag']} -> mtmc_idf1={best['mtmc_idf1']:.4f}")
    ctrl = [r2 for r2 in results if r2["tag"] == "control"]
    if ctrl:
        delta = best["mtmc_idf1"] - ctrl[0]["mtmc_idf1"]
        print(f"  vs control: {delta:+.4f}")

    # Phase-by-phase analysis
    for phase, prefix in [("CSLS", "csls"), ("CLUSTER_VERIFY", "clverify"),
                          ("TEMPORAL_SPLIT", "tsplit"), ("LENGTH_WEIGHT_POWER", "lwp"),
                          ("GALLERY_EXPANSION", "galexp"), ("EXHAUSTIVE_MIN_SIM", "exhmin")]:
        phase_results = [r2 for r2 in results if r2["tag"].startswith(prefix)]
        if phase_results:
            print(f"\n--- {phase} ---")
            for r2 in sorted(phase_results, key=lambda x: x["mtmc_idf1"], reverse=True):
                ov = ", ".join(r2["overrides"])
                print(f"  {r2['tag']:<18} mtmc_idf1={r2['mtmc_idf1']:.4f}  IDF1={r2['IDF1']:.4f}  ids={r2['IDSW']}  ({ov})")

    import json as _json
    results_path = DATA_OUT / "scan_results.json"
    with open(results_path, "w") as _f:
        _json.dump({"scan": "v74_new_features", "results": results}, _f, indent=2)
    print(f"\nResults saved to {results_path}")
    import shutil as _shutil
    _out = Path("/kaggle/working")
    _shutil.copy2(str(results_path), str(_out / "scan_results.json"))
    print(f"Copied to {_out}")
else:
    print("Scan disabled. Set SCAN_ENABLED = True to run v74 feature exploration.")
'''

# Split into lines and format for notebook JSON
lines = new_source.split('\n')
source_lines = [line + "\n" for line in lines[:-1]] + [lines[-1]]

nb["cells"][scan_idx]["source"] = source_lines
print(f"Updated cell {scan_idx} (c16) with v74 scan ({len(lines)} lines)")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

print("Saved notebook")
