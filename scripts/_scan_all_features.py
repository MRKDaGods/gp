"""Exhaustive Phase 2+ scan: test ALL untested stage4 features.

Tests CSLS, cluster_verify, hierarchical expansion, intra_camera_merge,
temporal_split, and their best combinations.

Uses existing stage1/2/3 data from local 10a v11 run.
Each experiment runs stages 4+5 only (~1-2 min each).
"""
import subprocess, sys, os, re, json, itertools
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent.parent
RUN_BASE = PROJECT / "data/outputs/kaggle_10a_v11/extracted"
RUN_NAME = "run_kaggle_20260318_114803"
RUN_DIR = RUN_BASE / RUN_NAME
SEC_EMB = str(RUN_DIR / "stage2" / "embeddings_secondary.npy")
GT_DIR = str(PROJECT / "data/raw/cityflowv2")
os.chdir(str(PROJECT))

BASE_CMD = [
    sys.executable, "scripts/run_pipeline.py",
    "--config", "configs/default.yaml",
    "--stages", "4,5",
    "--override", f"project.output_dir={RUN_BASE}",
    "--override", f"project.run_name={RUN_NAME}",
    "--override", "stage0.cameras=[S01_c001,S01_c002,S01_c003,S02_c006,S02_c007,S02_c008]",
    "--override", "stage4.association.graph.similarity_threshold=0.53",
    "--override", "stage4.association.graph.algorithm=conflict_free_cc",
    "--override", "stage4.association.fic.regularisation=3.0",
    "--override", "stage4.association.fic.enabled=true",
    "--override", "stage4.association.fac.enabled=true",
    "--override", "stage4.association.fac.knn=20",
    "--override", "stage4.association.fac.learning_rate=0.5",
    "--override", "stage4.association.fac.beta=0.08",
    "--override", "stage4.association.query_expansion.k=2",
    "--override", "stage4.association.weights.vehicle.appearance=0.75",
    "--override", "stage4.association.weights.vehicle.hsv=0.00",
    "--override", "stage4.association.weights.vehicle.spatiotemporal=0.25",
    "--override", "stage4.association.intra_camera_merge.enabled=true",
    "--override", "stage4.association.intra_camera_merge.threshold=0.75",
    "--override", "stage4.association.intra_camera_merge.max_time_gap=60",
    "--override", f"stage4.association.secondary_embeddings.path={SEC_EMB}",
    "--override", "stage4.association.secondary_embeddings.weight=0.25",
    "--override", "stage5.mtmc_only_submission=false",
    "--override", "stage5.gt_frame_clip=true",
    "--override", "stage5.gt_zone_filter=true",
    "--override", "stage5.gt_zone_margin_frac=0.2",
    "--override", "stage5.gt_frame_clip_min_iou=0.5",
    "--override", "stage5.stationary_filter.enabled=true",
    "--override", "stage5.stationary_filter.min_displacement_px=150",
    "--override", f"stage5.ground_truth_dir={GT_DIR}",
]


def run_exp(name, extra_overrides):
    cmd = BASE_CMD + extra_overrides
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr
    idf1 = mota = hota = ids_count = None

    for line in output.split("\n"):
        if "IDF1" in line and ("MTMC" in line or "evaluation" in line.lower()):
            for part in line.split():
                try:
                    if "IDF1" in part and "=" in part:
                        val = part.split("=")[1].strip("%,")
                        idf1 = float(val) / (100 if float(val) > 1 else 1)
                    elif "MOTA" in part and "=" in part:
                        val = part.split("=")[1].strip("%,")
                        mota = float(val) / (100 if float(val) > 1 else 1)
                    elif "HOTA" in part and "=" in part:
                        val = part.split("=")[1].strip("%,")
                        hota = float(val) / (100 if float(val) > 1 else 1)
                    elif "ID_Switches" in part:
                        ids_count = int(part.split("=")[1].strip(","))
                except Exception:
                    pass

    if idf1 is None:
        for line in output.split("\n")[-50:]:
            m = re.search(r"IDF1[=:]\s*([\d.]+)%?", line)
            if m:
                v = float(m.group(1))
                idf1 = v / 100 if v > 1 else v
            m2 = re.search(r"MOTA[=:]\s*([\d.]+)%?", line)
            if m2 and mota is None:
                v2 = float(m2.group(1))
                mota = v2 / 100 if v2 > 1 else v2
            m3 = re.search(r"HOTA[=:]\s*([\d.]+)%?", line)
            if m3 and hota is None:
                v3 = float(m3.group(1))
                hota = v3 / 100 if v3 > 1 else v3

    return {"name": name, "idf1": idf1, "mota": mota, "hota": hota, "ids": ids_count}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = PROJECT / f"data/outputs/scan_all_features_{ts}.txt"

    # ════════════════════════════════════════════════════════════════════
    # EXPERIMENTS
    # ════════════════════════════════════════════════════════════════════
    experiments = []

    # 0. Baseline (no dataset-config, explicit overrides only)
    experiments.append(("baseline", []))

    # ── CSLS variants ──
    for k in [5, 10, 15, 20]:
        experiments.append((f"csls_k{k}", [
            "--override", "stage4.association.csls.enabled=true",
            "--override", f"stage4.association.csls.k={k}",
        ]))

    # ── Cluster verification variants ──
    for mc in [0.20, 0.25, 0.30, 0.35, 0.40]:
        experiments.append((f"verify_mc{mc:.2f}", [
            "--override", "stage4.association.cluster_verify.enabled=true",
            "--override", f"stage4.association.cluster_verify.min_connectivity={mc}",
        ]))

    # ── Hierarchical expansion variants ──
    for ct in [0.40, 0.45, 0.50]:
        for ot in [0.35, 0.40, 0.45]:
            experiments.append((f"hierarch_ct{ct:.2f}_ot{ot:.2f}", [
                "--override", "stage4.association.hierarchical.enabled=true",
                "--override", f"stage4.association.hierarchical.centroid_threshold={ct}",
                "--override", "stage4.association.hierarchical.merge_threshold=0.45",
                "--override", f"stage4.association.hierarchical.orphan_threshold={ot}",
                "--override", "stage4.association.hierarchical.max_merge_size=12",
            ]))

    # ── Intra-camera merge variants ──
    for thresh in [0.65, 0.70, 0.75, 0.80]:
        for gap in [60, 120]:
            experiments.append((f"intra_t{thresh:.2f}_g{gap}", [
                "--override", "stage4.association.intra_camera_merge.enabled=true",
                "--override", f"stage4.association.intra_camera_merge.threshold={thresh}",
                "--override", f"stage4.association.intra_camera_merge.max_time_gap={gap}",
            ]))

    # ── Temporal split variants ──
    for gap in [30, 45, 60, 90]:
        for st in [0.40, 0.45, 0.50]:
            experiments.append((f"ts_g{gap}_t{st:.2f}", [
                "--override", "stage4.association.temporal_split.enabled=true",
                "--override", f"stage4.association.temporal_split.min_gap={gap}",
                "--override", f"stage4.association.temporal_split.split_threshold={st}",
            ]))

    # ── sim_thresh sweep (fine grid around 0.53) ──
    for st in [0.50, 0.51, 0.52, 0.54, 0.55, 0.56]:
        experiments.append((f"sim_thresh_{st:.2f}", [
            "--override", f"stage4.association.graph.similarity_threshold={st}",
        ]))

    print(f"Total experiments: {len(experiments)}")
    print("=" * 70)

    results = []
    for i, (name, overrides) in enumerate(experiments):
        print(f"[{i+1}/{len(experiments)}] Running {name}...", end=" ", flush=True)
        r = run_exp(name, overrides)
        results.append(r)
        idf1_str = f"IDF1={r['idf1']:.4f}" if r['idf1'] else "IDF1=FAIL"
        print(idf1_str)

    # Sort by IDF1 descending
    results.sort(key=lambda x: x['idf1'] or 0, reverse=True)
    baseline_idf1 = next((r['idf1'] for r in results if r['name'] == 'baseline'), 0)

    # Print summary
    summary_lines = [
        "",
        "=" * 70,
        "FULL RESULTS (sorted by IDF1 desc):",
        f"{'Name':<45} {'IDF1':>8} {'MOTA':>8} {'HOTA':>8} {'delta':>8}",
        "-" * 70,
    ]
    for r in results:
        idf1 = r['idf1'] or 0
        mota = r['mota'] or 0
        hota = r['hota'] or 0
        delta = idf1 - baseline_idf1 if baseline_idf1 else 0
        sign = "+" if delta >= 0 else ""
        summary_lines.append(
            f"{r['name']:<45} {idf1:.4f}   {mota:.4f}   {hota:.4f}   {sign}{delta:.4f}"
        )
    summary_lines.append("=" * 70)

    # Find best improvements
    improvements = [r for r in results if (r['idf1'] or 0) > baseline_idf1]
    if improvements:
        summary_lines.append(f"\nBest improvement: {improvements[0]['name']} IDF1={improvements[0]['idf1']:.4f} (+{improvements[0]['idf1']-baseline_idf1:.4f})")
    else:
        summary_lines.append(f"\nNo improvement over baseline ({baseline_idf1:.4f})")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    # Save results
    with open(log_file, "w") as f:
        f.write(summary_text)

    # Save JSON for programmatic use
    json_file = log_file.with_suffix(".json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {log_file}")
    print(f"JSON saved to {json_file}")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: COMBO SCAN — Top features combined
    # Only run if any individual feature improved
    # ════════════════════════════════════════════════════════════════════
    if improvements:
        print("\n" + "=" * 70)
        print("PHASE 2: COMBINATION SCAN")
        print("=" * 70)

        # Pick top 3-5 individual improvements
        top_features = []
        seen_categories = set()
        for r in improvements[:8]:
            name = r['name']
            category = name.split("_")[0]  # csls, verify, hierarch, intra, ts
            if category not in seen_categories:
                seen_categories.add(category)
                top_features.append((name, r))
                if len(top_features) >= 5:
                    break

        # Generate all 2-way and 3-way combinations
        combo_experiments = []
        for r in range(2, min(len(top_features) + 1, 4)):
            for combo in itertools.combinations(top_features, r):
                names = [c[0] for c in combo]
                combo_name = "combo_" + "+".join(n.split("_")[0] for n in names)
                # Rebuild overrides for each feature
                combo_overrides = []
                for orig_name, _ in combo:
                    # Find original overrides
                    for exp_name, exp_ovr in experiments:
                        if exp_name == orig_name:
                            combo_overrides.extend(exp_ovr)
                            break
                combo_experiments.append((combo_name, combo_overrides))

        combo_results = []
        for i, (name, overrides) in enumerate(combo_experiments):
            print(f"[{i+1}/{len(combo_experiments)}] Running {name}...", end=" ", flush=True)
            r = run_exp(name, overrides)
            combo_results.append(r)
            idf1_str = f"IDF1={r['idf1']:.4f}" if r['idf1'] else "IDF1=FAIL"
            print(idf1_str)

        combo_results.sort(key=lambda x: x['idf1'] or 0, reverse=True)
        print("\nCOMBO RESULTS:")
        for r in combo_results:
            idf1 = r['idf1'] or 0
            delta = idf1 - baseline_idf1
            sign = "+" if delta >= 0 else ""
            print(f"  {r['name']:<45} IDF1={idf1:.4f} ({sign}{delta:.4f})")

        all_results = results + combo_results
        all_results.sort(key=lambda x: x['idf1'] or 0, reverse=True)
        print(f"\nOVERALL BEST: {all_results[0]['name']} IDF1={all_results[0]['idf1']:.4f}")

        with open(json_file, "w") as f:
            json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
