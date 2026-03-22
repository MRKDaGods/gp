"""
v71: Combine all wins + extend frames search
- FIC reg=0.1 (+0.08pp from v70)
- Extend frames: [40, 45, 50, 60, 80] to find plateau
- Also test fic=[0.05, 0.1, 0.2] × frames=[35, 40, 50] to check interaction
"""
import json

NB_PATH = r'e:\dev\src\gp\notebooks\kaggle\10c_stages45\mtmc-10c-stages-4-5-association-eval.ipynb'
nb = json.load(open(NB_PATH, encoding='utf-8'))

# --- Update Cell 11: set frames=40, update FIC to 0.1 ---
cell11_src = nb['cells'][11]['source']
new_cell11 = []
for line in cell11_src:
    if 'stage5.min_trajectory_frames=' in line:
        new_cell11.append(line.replace('min_trajectory_frames=25', 'min_trajectory_frames=40'))
    elif 'stage4.association.fic.regularisation=' in line:
        new_cell11.append(line.replace('fic.regularisation=3.0', 'fic.regularisation=0.1'))
    else:
        new_cell11.append(line)
nb['cells'][11]['source'] = new_cell11

# --- Replace Cell 15 with combined sweep ---
scan_source = r'''# ============================================================
# v71: COMBINED WINS + EXTENDED FRAMES + FIC INTERACTION
# Baseline: fic=0.1, frames=40, sim=0.53, cfcc, nms=0.40
# ============================================================
SCAN_ENABLED = True

if SCAN_ENABLED:
    import itertools

    experiments = []

    # Baseline (v70 combined best)
    experiments.append({"name": "baseline_fic01_f40", "fic_reg": 0.1, "frames": 40})

    # Extended frames with fic=0.1
    for f in [35, 45, 50, 60, 80]:
        experiments.append({"name": f"fic01_f{f}", "fic_reg": 0.1, "frames": f})

    # FIC × frames interaction grid
    for fic in [0.05, 0.2, 0.5, 3.0]:
        for f in [35, 40, 50]:
            experiments.append({"name": f"fic{fic}_f{f}".replace(".", "p"), "fic_reg": fic, "frames": f})

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
            # Stage 4 fixed params (v69 best)
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
            # Stage 5 fixed + frames variable
            "--override", f"stage5.mtmc_only_submission={str(MTMC_ONLY).lower()}",
            "--override", "stage5.stationary_filter.enabled=true",
            "--override", "stage5.stationary_filter.min_displacement_px=150",
            "--override", "stage5.stationary_filter.max_mean_velocity_px=2.0",
            "--override", "stage5.min_submission_confidence=0.15",
            "--override", "stage5.cross_id_nms_iou=0.40",
            "--override", "stage5.min_trajectory_confidence=0.30",
            "--override", f"stage5.min_trajectory_frames={exp['frames']}",
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
            "name": exp["name"], "fic_reg": exp["fic_reg"], "frames": exp["frames"],
            "IDF1": idf1, "MOTA": mota, "HOTA": hota,
            "mtmc_idf1": mtmc_idf1, "id_switches": id_switches,
            "time": elapsed
        })
        status = "OK" if r.returncode == 0 else "FAIL"
        print(f"  [{status}] {exp['name']:25s} fic={exp['fic_reg']:5.2f} frames={exp['frames']:2d} "
              f"-> IDF1={idf1:.4f} mtmc={mtmc_idf1:.4f} HOTA={hota:.4f} ids={id_switches} ({elapsed:.0f}s)")

    # Sort by mtmc_idf1
    results.sort(key=lambda x: x["mtmc_idf1"], reverse=True)
    print("\n" + "=" * 100)
    print("v71 RESULTS (sorted by mtmc_idf1)")
    print("=" * 100)
    print(f"{'Name':<25s} {'FIC':>5s} {'Frm':>4s} {'IDF1':>7s} {'mtmc_idf1':>10s} {'HOTA':>7s} {'IDs':>5s}")
    for r2 in results:
        print(f"{r2['name']:<25s} {r2['fic_reg']:>5.2f} {r2['frames']:>4d} "
              f"{r2['IDF1']:>7.4f} {r2['mtmc_idf1']:>10.4f} {r2['HOTA']:>7.4f} {r2['id_switches']:>5d}")
    best = results[0]
    print(f"\nBEST: {best['name']} (fic={best['fic_reg']} frames={best['frames']}) "
          f"-> mtmc_idf1={best['mtmc_idf1']:.4f} IDF1={best['IDF1']:.4f}")

    # Save
    import json as _json
    results_path = DATA_OUT / "scan_results.json"
    with open(results_path, "w") as _f:
        _json.dump({"scan_type": "combined_v71", "results": results}, _f, indent=2)
    print(f"\nSaved to {results_path}")
    import shutil as _shutil
    _out = Path("/kaggle/working")
    _shutil.copy2(str(results_path), str(_out / "scan_results.json"))
else:
    print("Scan disabled.")
'''

lines = scan_source.strip().split('\n')
source_list = [line + '\n' for line in lines[:-1]] + [lines[-1]]
nb['cells'][15]['source'] = source_list

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

# Verify
nb2 = json.load(open(NB_PATH, encoding='utf-8'))
cell11_src = ''.join(nb2['cells'][11]['source'])
assert 'min_trajectory_frames=40' in cell11_src
assert 'fic.regularisation=0.1' in cell11_src
print("v71 prepared: 18 experiments")
print("  Cell 11: fic=0.1, frames=40")
print("  Cell 15: fic×frames interaction + extended frames [35-80]")
