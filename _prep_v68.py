"""
v68: 
1. Update Cell 9 ALGORITHM from connected_components -> conflict_free_cc (+0.21pp from v67 scan)
2. Replace Cell 15 scan with Stage 5 filter sweep
   Tests cross_id_nms_iou, min_trajectory_confidence, min_trajectory_frames, 
   stationary_filter thresholds
"""
import json

NB_PATH = r'e:\dev\src\gp\notebooks\kaggle\10c_stages45\mtmc-10c-stages-4-5-association-eval.ipynb'

nb = json.load(open(NB_PATH, encoding='utf-8'))

# --- 1. Update Cell 9: ALGORITHM -> conflict_free_cc ---
cell9_src = nb['cells'][9]['source']
new_cell9 = []
for line in cell9_src:
    if line.startswith('ALGORITHM') and 'connected_components' in line:
        new_cell9.append('ALGORITHM         = "conflict_free_cc"   # v67: +0.21pp over connected_components\n')
    else:
        new_cell9.append(line)
nb['cells'][9]['source'] = new_cell9

# --- 2. Replace Cell 15 with Stage 5 filter sweep ---
scan_source = r'''# ============================================================
# v68: STAGE 5 FILTER SWEEP
# Tests cross_id_nms_iou, min_trajectory_confidence,
# min_trajectory_frames, stationary displacement
# Algorithm fixed at conflict_free_cc (v67 winner)
# All Stage 4 params fixed at v66 best.
# ============================================================
SCAN_ENABLED = True

if SCAN_ENABLED:
    import itertools

    # Stage 5 filter grid
    scan_grid = {
        "cross_id_nms_iou":   [0.20, 0.25, 0.30, 0.35, 0.40, 0.50],    # 6: current=0.35
        "min_traj_conf":      [0.15, 0.20, 0.25, 0.30, 0.40],           # 5: current=0.30
        "min_traj_frames":    [5, 8, 10, 15],                             # 4: current=10
        "stat_min_disp":      [100, 150, 200],                            # 3: current=150
    }
    # Total: 6 x 5 x 4 x 3 = 360 ... too many!
    # Instead, sweep each dimension independently (hold others at v66 defaults)
    # This gives: 6 + 5 + 4 + 3 - 4(baselines) + 1(combined baseline) = 15 runs

    experiments = []
    defaults = {
        "cross_id_nms_iou": 0.35,
        "min_traj_conf": 0.30,
        "min_traj_frames": 10,
        "stat_min_disp": 150,
    }

    # Baseline
    experiments.append({"name": "baseline", **defaults})

    # Sweep each dimension
    for param_name, values in scan_grid.items():
        for val in values:
            if val == defaults[param_name]:
                continue  # skip duplicate of baseline
            exp = dict(defaults)
            exp[param_name] = val
            tag = f"{param_name}_{val}".replace(".", "p")
            experiments.append({"name": tag, **exp})

    # Also test a few promising combinations
    experiments.append({"name": "combo_aggressive",
        "cross_id_nms_iou": 0.25, "min_traj_conf": 0.20,
        "min_traj_frames": 8, "stat_min_disp": 100})
    experiments.append({"name": "combo_conservative",
        "cross_id_nms_iou": 0.40, "min_traj_conf": 0.40,
        "min_traj_frames": 15, "stat_min_disp": 200})
    experiments.append({"name": "combo_nms_low_conf_low",
        "cross_id_nms_iou": 0.25, "min_traj_conf": 0.20,
        "min_traj_frames": 10, "stat_min_disp": 150})
    experiments.append({"name": "combo_nms_low_frames_low",
        "cross_id_nms_iou": 0.25, "min_traj_conf": 0.30,
        "min_traj_frames": 5, "stat_min_disp": 150})

    print(f"Running {len(experiments)} Stage 5 filter experiments ...")

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
            # v66 Stage 4 fixed params + v67 algorithm fix
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
            # Stage 5: variable params for this experiment
            "--override", f"stage5.mtmc_only_submission={str(MTMC_ONLY).lower()}",
            "--override", "stage5.stationary_filter.enabled=true",
            "--override", f"stage5.stationary_filter.min_displacement_px={exp['stat_min_disp']}",
            "--override", "stage5.stationary_filter.max_mean_velocity_px=2.0",
            "--override", "stage5.min_submission_confidence=0.15",
            "--override", f"stage5.cross_id_nms_iou={exp['cross_id_nms_iou']}",
            "--override", f"stage5.min_trajectory_confidence={exp['min_traj_conf']}",
            "--override", f"stage5.min_trajectory_frames={exp['min_traj_frames']}",
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
            "name": exp["name"],
            "cross_id_nms_iou": exp["cross_id_nms_iou"],
            "min_traj_conf": exp["min_traj_conf"],
            "min_traj_frames": exp["min_traj_frames"],
            "stat_min_disp": exp["stat_min_disp"],
            "IDF1": idf1, "MOTA": mota, "HOTA": hota,
            "mtmc_idf1": mtmc_idf1, "id_switches": id_switches,
            "time": elapsed
        })
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  [{status}] {exp['name']:30s} nms={exp['cross_id_nms_iou']:.2f} conf={exp['min_traj_conf']:.2f} "
              f"frames={exp['min_traj_frames']:2d} disp={exp['stat_min_disp']:3d} "
              f"-> IDF1={idf1:.4f} mtmc={mtmc_idf1:.4f} HOTA={hota:.4f} ids={id_switches} ({elapsed:.0f}s)")

    # Sort by mtmc_idf1
    results.sort(key=lambda x: x["mtmc_idf1"], reverse=True)
    print("\n" + "=" * 110)
    print("STAGE 5 FILTER SWEEP RESULTS (sorted by mtmc_idf1)")
    print("=" * 110)
    print(f"{'Name':<30s} {'NMS':>5s} {'Conf':>5s} {'Frm':>4s} {'Disp':>5s} {'IDF1':>7s} {'mtmc_idf1':>10s} {'HOTA':>7s} {'IDs':>5s}")
    for r2 in results:
        print(f"{r2['name']:<30s} {r2['cross_id_nms_iou']:>5.2f} {r2['min_traj_conf']:>5.2f} "
              f"{r2['min_traj_frames']:>4d} {r2['stat_min_disp']:>5d} "
              f"{r2['IDF1']:>7.4f} {r2['mtmc_idf1']:>10.4f} {r2['HOTA']:>7.4f} {r2['id_switches']:>5d}")
    best = results[0]
    print(f"\nBEST: {best['name']} (nms={best['cross_id_nms_iou']} conf={best['min_traj_conf']} "
          f"frames={best['min_traj_frames']} disp={best['stat_min_disp']}) "
          f"-> mtmc_idf1={best['mtmc_idf1']:.4f} IDF1={best['IDF1']:.4f}")

    # Per-parameter sensitivity
    print("\n" + "=" * 110)
    print("PARAMETER SENSITIVITY (one-at-a-time sweeps only)")
    print("=" * 110)
    defaults_set = {"cross_id_nms_iou": 0.35, "min_traj_conf": 0.30, "min_traj_frames": 10, "stat_min_disp": 150}
    for param_name, default_val in defaults_set.items():
        print(f"\n--- {param_name} (default={default_val}) ---")
        for r2 in results:
            # Show only single-param experiments for this param
            is_single = all(
                r2[p] == defaults_set[p]
                for p in defaults_set if p != param_name
            )
            if is_single or r2["name"] == "baseline":
                marker = " <-- default" if r2[param_name] == default_val and r2["name"] != "baseline" else ""
                if r2["name"] == "baseline":
                    marker = " <-- baseline"
                print(f"  {param_name}={r2[param_name]:<8} mtmc_idf1={r2['mtmc_idf1']:.4f} IDF1={r2['IDF1']:.4f} "
                      f"HOTA={r2['HOTA']:.4f} ids={r2['id_switches']}{marker}")

    # Save results
    import json as _json
    results_path = DATA_OUT / "scan_results.json"
    with open(results_path, "w") as _f:
        _json.dump({"scan_type": "stage5_filter_sweep_v68", "results": results}, _f, indent=2)
    print(f"\nSaved to {results_path}")
    import shutil as _shutil
    _out = Path("/kaggle/working")
    _shutil.copy2(str(results_path), str(_out / "scan_results.json"))
else:
    print("Scan disabled. Set SCAN_ENABLED = True to run Stage 5 filter sweep.")
'''

# Split into lines with proper \n endings
lines = scan_source.strip().split('\n')
source_list = [line + '\n' for line in lines[:-1]] + [lines[-1]]

# Replace Cell 15
nb['cells'][15]['source'] = source_list

# Save
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

# Verify
nb2 = json.load(open(NB_PATH, encoding='utf-8'))
cell9_src = ''.join(nb2['cells'][9]['source'])
cell15_src = ''.join(nb2['cells'][15]['source'])
assert 'conflict_free_cc' in cell9_src, "Cell 9 ALGORITHM not updated!"
assert 'STAGE 5 FILTER SWEEP' in cell15_src, "Cell 15 scan not updated!"
assert 'SCAN_ENABLED = True' in cell15_src, "Scan not enabled!"

print("v68 notebook prepared:")
print("  Cell 9: ALGORITHM = conflict_free_cc")
print(f"  Cell 15: Stage 5 filter sweep ({len(source_list)} lines)")
