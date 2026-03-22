"""
v69: Combine v67+v68 winners + fine-scan min_traj_frames around 15 and cross_id_nms_iou around 0.40
- Cell 9: ALGORITHM = conflict_free_cc (already from v68)
- Cell 11: update min_traj_frames=15, cross_id_nms_iou=0.40
- Cell 15: fine-scan min_traj_frames [12,13,14,15,16,18,20] × cross_id_nms [0.35,0.40,0.45,0.50]
"""
import json

NB_PATH = r'e:\dev\src\gp\notebooks\kaggle\10c_stages45\mtmc-10c-stages-4-5-association-eval.ipynb'
nb = json.load(open(NB_PATH, encoding='utf-8'))

# --- 1. Update Cell 11 (run cell): min_traj_frames and cross_id_nms ---
cell11_src = nb['cells'][11]['source']
new_cell11 = []
for line in cell11_src:
    if 'stage5.cross_id_nms_iou=' in line:
        new_cell11.append(line.replace('cross_id_nms_iou=0.35', 'cross_id_nms_iou=0.40'))
    elif 'stage5.min_trajectory_frames=' in line:
        new_cell11.append(line.replace('min_trajectory_frames=10', 'min_trajectory_frames=15'))
    else:
        new_cell11.append(line)
nb['cells'][11]['source'] = new_cell11

# --- 2. Replace Cell 15 with fine-grained scan ---
scan_source = r'''# ============================================================
# v69: FINE SCAN — min_traj_frames around 15, NMS around 0.40
# All else fixed at v67/v68 best: conflict_free_cc, AQE_K=3, etc.
# ============================================================
SCAN_ENABLED = True

if SCAN_ENABLED:
    import itertools

    scan_grid = {
        "min_traj_frames": [12, 13, 14, 15, 16, 18, 20, 25],  # 8 values
        "cross_id_nms_iou": [0.35, 0.40, 0.45, 0.50],          # 4 values
    }
    # 8 × 4 = 32 combos at ~25s = ~13 min

    keys = list(scan_grid.keys())
    combos = list(itertools.product(*[scan_grid[k] for k in keys]))
    print(f"Running {len(combos)} fine-scan experiments ...")

    results = []
    for combo in combos:
        params = dict(zip(keys, combo))
        scan_run = f"scan_f{params['min_traj_frames']}_nms{params['cross_id_nms_iou']:.2f}".replace(".", "p")

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
            # v66+v67 Stage 4 fixed params
            "--override", f"stage4.association.query_expansion.k={AQE_K}",
            "--override", f"stage4.association.graph.similarity_threshold={SIM_THRESH}",
            "--override", f"stage4.association.graph.bridge_prune_margin={BRIDGE_PRUNE}",
            "--override", f"stage4.association.graph.algorithm={ALGORITHM}",
            "--override", f"stage4.association.graph.louvain_resolution={LOUVAIN_RES}",
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
            # Stage 5: fixed params + variable sweep params
            "--override", f"stage5.mtmc_only_submission={str(MTMC_ONLY).lower()}",
            "--override", "stage5.stationary_filter.enabled=true",
            "--override", "stage5.stationary_filter.min_displacement_px=150",
            "--override", "stage5.stationary_filter.max_mean_velocity_px=2.0",
            "--override", "stage5.min_submission_confidence=0.15",
            "--override", f"stage5.cross_id_nms_iou={params['cross_id_nms_iou']}",
            "--override", "stage5.min_trajectory_confidence=0.30",
            "--override", f"stage5.min_trajectory_frames={params['min_traj_frames']}",
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
            "frames": params["min_traj_frames"],
            "nms": params["cross_id_nms_iou"],
            "IDF1": idf1, "MOTA": mota, "HOTA": hota,
            "mtmc_idf1": mtmc_idf1, "id_switches": id_switches,
            "time": elapsed
        })
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  [{status}] frames={params['min_traj_frames']:2d} nms={params['cross_id_nms_iou']:.2f} "
              f"-> IDF1={idf1:.4f} mtmc={mtmc_idf1:.4f} HOTA={hota:.4f} ids={id_switches} ({elapsed:.0f}s)")

    # Sort by mtmc_idf1
    results.sort(key=lambda x: x["mtmc_idf1"], reverse=True)
    print("\n" + "=" * 90)
    print("FINE SCAN RESULTS (sorted by mtmc_idf1)")
    print("=" * 90)
    print(f"{'Frames':>6s} {'NMS':>5s} {'IDF1':>7s} {'mtmc_idf1':>10s} {'HOTA':>7s} {'IDs':>5s}")
    for r2 in results:
        marker = " <-- v68 baseline" if r2["frames"] == 10 and r2["nms"] == 0.35 else ""
        print(f"{r2['frames']:>6d} {r2['nms']:>5.2f} {r2['IDF1']:>7.4f} {r2['mtmc_idf1']:>10.4f} {r2['HOTA']:>7.4f} {r2['id_switches']:>5d}{marker}")
    best = results[0]
    print(f"\nBEST: frames={best['frames']} nms={best['nms']:.2f} -> mtmc_idf1={best['mtmc_idf1']:.4f} IDF1={best['IDF1']:.4f}")

    # Per-parameter sensitivity
    print("\n--- min_traj_frames sensitivity (avg over NMS values) ---")
    for fv in sorted(set(r2["frames"] for r2 in results)):
        subset = [r2 for r2 in results if r2["frames"] == fv]
        avg_mtmc = sum(r2["mtmc_idf1"] for r2 in subset) / len(subset)
        best_mtmc = max(r2["mtmc_idf1"] for r2 in subset)
        avg_ids = sum(r2["id_switches"] for r2 in subset) / len(subset)
        print(f"  frames={fv:2d}: avg_mtmc={avg_mtmc:.4f} best_mtmc={best_mtmc:.4f} avg_ids={avg_ids:.0f}")

    print("\n--- cross_id_nms_iou sensitivity (avg over frames) ---")
    for nv in sorted(set(r2["nms"] for r2 in results)):
        subset = [r2 for r2 in results if r2["nms"] == nv]
        avg_mtmc = sum(r2["mtmc_idf1"] for r2 in subset) / len(subset)
        best_mtmc = max(r2["mtmc_idf1"] for r2 in subset)
        print(f"  nms={nv:.2f}: avg_mtmc={avg_mtmc:.4f} best_mtmc={best_mtmc:.4f}")

    # Save
    import json as _json
    results_path = DATA_OUT / "scan_results.json"
    with open(results_path, "w") as _f:
        _json.dump({"scan_type": "fine_scan_v69", "results": results}, _f, indent=2)
    print(f"\nSaved to {results_path}")
    import shutil as _shutil
    _out = Path("/kaggle/working")
    _shutil.copy2(str(results_path), str(_out / "scan_results.json"))
else:
    print("Scan disabled. Set SCAN_ENABLED = True to run fine scan.")
'''

lines = scan_source.strip().split('\n')
source_list = [line + '\n' for line in lines[:-1]] + [lines[-1]]
nb['cells'][15]['source'] = source_list

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

# Verify
nb2 = json.load(open(NB_PATH, encoding='utf-8'))
cell11_src = ''.join(nb2['cells'][11]['source'])
assert 'cross_id_nms_iou=0.40' in cell11_src, "Cell 11 NMS not updated!"
assert 'min_trajectory_frames=15' in cell11_src, "Cell 11 frames not updated!"
cell9_src = ''.join(nb2['cells'][9]['source'])
assert 'conflict_free_cc' in cell9_src, "Cell 9 algorithm wrong!"

print("v69 notebook prepared:")
print("  Cell 9: ALGORITHM = conflict_free_cc")
print("  Cell 11: cross_id_nms_iou=0.40, min_trajectory_frames=15")
print(f"  Cell 15: Fine scan (32 experiments)")
