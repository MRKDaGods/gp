"""
v70: Sweep FIC regularization, sim_thresh, and extended min_traj_frames
- FIC reg: [0.5, 1.0, 2.0, 3.0, 5.0, 10.0] (current=3.0)
- sim_thresh: [0.48, 0.50, 0.53, 0.55, 0.58] (current=0.53)
- min_traj_frames: [25, 30, 35] (current=25 from v69)
- All independent sweeps (one-at-a-time) to keep runs manageable
"""
import json

NB_PATH = r'e:\dev\src\gp\notebooks\kaggle\10c_stages45\mtmc-10c-stages-4-5-association-eval.ipynb'
nb = json.load(open(NB_PATH, encoding='utf-8'))

# --- Update Cell 11: set baseline to v69 best (frames=25, nms=0.40) ---
# (Already updated in v69 to frames=15 nms=0.40, need to change to 25)
cell11_src = nb['cells'][11]['source']
new_cell11 = []
for line in cell11_src:
    if 'stage5.min_trajectory_frames=' in line:
        new_cell11.append(line.replace('min_trajectory_frames=15', 'min_trajectory_frames=25'))
    else:
        new_cell11.append(line)
nb['cells'][11]['source'] = new_cell11

# --- Replace Cell 15 with multi-param sweep ---
scan_source = r'''# ============================================================
# v70: FIC REGULARIZATION + SIM_THRESH + EXTENDED FRAMES SWEEP
# One-at-a-time sweeps to map parameter landscape
# Baseline: conflict_free_cc, frames=25, nms=0.40, fic_reg=3.0, sim=0.53
# ============================================================
SCAN_ENABLED = True

if SCAN_ENABLED:
    experiments = []

    # Defaults (v69 best)
    defaults = {
        "fic_reg": 3.0,
        "sim_thresh_v": SIM_THRESH,  # 0.53
        "min_traj_frames_v": 25,
    }

    # Baseline experiment
    experiments.append({"name": "baseline", **defaults})

    # FIC regularization sweep (hold sim=0.53, frames=25)
    for fic in [0.1, 0.5, 1.0, 2.0, 5.0, 8.0, 15.0]:
        if fic == defaults["fic_reg"]:
            continue
        experiments.append({"name": f"fic_{fic}".replace(".", "p"),
            "fic_reg": fic, "sim_thresh_v": defaults["sim_thresh_v"],
            "min_traj_frames_v": defaults["min_traj_frames_v"]})

    # sim_thresh sweep (hold fic=3.0, frames=25)
    for sim in [0.45, 0.48, 0.50, 0.52, 0.54, 0.55, 0.58, 0.60]:
        experiments.append({"name": f"sim_{sim}".replace(".", "p"),
            "fic_reg": defaults["fic_reg"], "sim_thresh_v": sim,
            "min_traj_frames_v": defaults["min_traj_frames_v"]})

    # Extended frames (hold fic=3.0, sim=0.53)
    for f in [30, 35, 40]:
        experiments.append({"name": f"frames_{f}",
            "fic_reg": defaults["fic_reg"], "sim_thresh_v": defaults["sim_thresh_v"],
            "min_traj_frames_v": f})

    print(f"Running {len(experiments)} experiments ...")

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
            # Stage 4 params (v69 best + sweep variables)
            "--override", f"stage4.association.query_expansion.k={AQE_K}",
            "--override", f"stage4.association.graph.similarity_threshold={exp['sim_thresh_v']}",
            "--override", f"stage4.association.graph.bridge_prune_margin={BRIDGE_PRUNE}",
            "--override", f"stage4.association.graph.algorithm={ALGORITHM}",
            "--override", f"stage4.association.graph.louvain_resolution={LOUVAIN_RES}",
            "--override", f"stage4.association.graph.max_component_size={MAX_COMP_SIZE}",
            "--override", f"stage4.association.weights.vehicle.appearance={APPEARANCE_WEIGHT}",
            "--override", f"stage4.association.weights.vehicle.hsv={HSV_WEIGHT}",
            "--override", f"stage4.association.weights.vehicle.spatiotemporal={ST_WEIGHT}",
            "--override", "stage4.association.mutual_nn.top_k_per_query=20",
            "--override", "stage4.association.fic.enabled=true",
            "--override", f"stage4.association.fic.regularisation={exp['fic_reg']}",
            "--override", "stage4.association.fac.enabled=false",
            "--override", f"stage4.association.intra_camera_merge.enabled={str(INTRA_MERGE).lower()}",
            "--override", f"stage4.association.intra_camera_merge.threshold={INTRA_MERGE_THRESH}",
            "--override", f"stage4.association.intra_camera_merge.max_time_gap={INTRA_MERGE_GAP}",
            "--override", f"stage4.association.secondary_embeddings.path={RUN_DIR}/stage2/embeddings_secondary.npy",
            "--override", f"stage4.association.secondary_embeddings.weight={FUSION_WEIGHT}",
            "--override", f"stage4.association.camera_bias.enabled={str(CAMERA_BIAS).lower()}",
            "--override", f"stage4.association.zone_model.enabled={str(ZONE_MODEL).lower()}",
            "--override", f"stage4.association.hierarchical.enabled={str(HIERARCHICAL).lower()}",
            # Stage 5: v69 best + sweep variable
            "--override", f"stage5.mtmc_only_submission={str(MTMC_ONLY).lower()}",
            "--override", "stage5.stationary_filter.enabled=true",
            "--override", "stage5.stationary_filter.min_displacement_px=150",
            "--override", "stage5.stationary_filter.max_mean_velocity_px=2.0",
            "--override", "stage5.min_submission_confidence=0.15",
            "--override", "stage5.cross_id_nms_iou=0.40",
            "--override", "stage5.min_trajectory_confidence=0.30",
            "--override", f"stage5.min_trajectory_frames={exp['min_traj_frames_v']}",
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
            "fic_reg": exp["fic_reg"], "sim_thresh": exp["sim_thresh_v"],
            "frames": exp["min_traj_frames_v"],
            "IDF1": idf1, "MOTA": mota, "HOTA": hota,
            "mtmc_idf1": mtmc_idf1, "id_switches": id_switches,
            "time": elapsed
        })
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  [{status}] {exp['name']:20s} fic={exp['fic_reg']:5.1f} sim={exp['sim_thresh_v']:.2f} "
              f"frames={exp['min_traj_frames_v']:2d} -> IDF1={idf1:.4f} mtmc={mtmc_idf1:.4f} "
              f"HOTA={hota:.4f} ids={id_switches} ({elapsed:.0f}s)")

    # Sort by mtmc_idf1
    results.sort(key=lambda x: x["mtmc_idf1"], reverse=True)
    print("\n" + "=" * 100)
    print("v70 SWEEP RESULTS (sorted by mtmc_idf1)")
    print("=" * 100)
    print(f"{'Name':<20s} {'FIC':>5s} {'Sim':>5s} {'Frm':>4s} {'IDF1':>7s} {'mtmc_idf1':>10s} {'HOTA':>7s} {'IDs':>5s}")
    for r2 in results:
        marker = " <--" if r2["name"] == "baseline" else ""
        print(f"{r2['name']:<20s} {r2['fic_reg']:>5.1f} {r2['sim_thresh']:>5.2f} {r2['frames']:>4d} "
              f"{r2['IDF1']:>7.4f} {r2['mtmc_idf1']:>10.4f} {r2['HOTA']:>7.4f} {r2['id_switches']:>5d}{marker}")
    best = results[0]
    print(f"\nBEST: {best['name']} (fic={best['fic_reg']} sim={best['sim_thresh']} frames={best['frames']}) "
          f"-> mtmc_idf1={best['mtmc_idf1']:.4f} IDF1={best['IDF1']:.4f}")

    # Per-param sensitivity
    for param_name, default_val in [("fic_reg", 3.0), ("sim_thresh", 0.53), ("frames", 25)]:
        print(f"\n--- {param_name} sensitivity (default={default_val}) ---")
        for r2 in sorted(
            [r2 for r2 in results
             if all(r2[p] == dv for p, dv in [("fic_reg", 3.0), ("sim_thresh", 0.53), ("frames", 25)] if p != param_name)],
            key=lambda x: x["mtmc_idf1"], reverse=True
        ):
            marker = " <-- default" if r2[param_name] == default_val else ""
            print(f"  {param_name}={r2[param_name]:<8} mtmc_idf1={r2['mtmc_idf1']:.4f} IDF1={r2['IDF1']:.4f} "
                  f"HOTA={r2['HOTA']:.4f} ids={r2['id_switches']}{marker}")

    # Save
    import json as _json
    results_path = DATA_OUT / "scan_results.json"
    with open(results_path, "w") as _f:
        _json.dump({"scan_type": "fic_sim_frames_v70", "results": results}, _f, indent=2)
    print(f"\nSaved to {results_path}")
    import shutil as _shutil
    _out = Path("/kaggle/working")
    _shutil.copy2(str(results_path), str(_out / "scan_results.json"))
else:
    print("Scan disabled. Set SCAN_ENABLED = True to run v70 sweep.")
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
assert 'min_trajectory_frames=25' in cell11_src, "Cell 11 frames not updated!"
cell15_src = ''.join(nb2['cells'][15]['source'])
assert 'FIC REGULARIZATION' in cell15_src, "Cell 15 not updated!"

# Count experiments
exps = 1 + 7 + 8 + 3  # baseline + FIC + sim + frames
print(f"v70 prepared: {exps} experiments (~{exps * 28 // 60} min)")
print("  Cell 11: frames=25, nms=0.40")
print("  Cell 15: FIC [0.1-15.0] + sim [0.45-0.60] + frames [30,35,40]")
