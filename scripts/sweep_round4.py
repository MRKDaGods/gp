"""Round 4: Test untapped features — hierarchical expansion, MNN top_k, QE alpha, etc.
Best from round 3: ic_gap060 → IDF1=0.813
"""
import subprocess
import sys
import json
import time
from pathlib import Path

PYTHON = sys.executable
BASE_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = "data/outputs/kaggle_10a_v11/extracted"
RUN_NAME = "run_kaggle_20260318_114803"
SECONDARY_EMB = str(
    BASE_DIR / RUN_DIR / RUN_NAME / "stage2" / "embeddings_secondary.npy"
)

FIXED_OVERRIDES = [
    f"project.output_dir={RUN_DIR}",
    f"project.run_name={RUN_NAME}",
    "stage0.cameras=[S01_c001,S01_c002,S01_c003,S02_c006,S02_c007,S02_c008]",
    f"stage4.association.secondary_embeddings.path={SECONDARY_EMB}",
    "stage4.association.secondary_embeddings.weight=0.3",
    "stage5.mtmc_only_submission=false",
    "stage5.ground_truth_dir=data/raw/cityflowv2",
    "stage5.gt_frame_clip=true",
    "stage5.gt_zone_filter=true",
    "stage5.gt_zone_margin_frac=0.2",
    "stage5.gt_frame_clip_min_iou=0.5",
]

# Best config from round 3 (ic_gap060)
BEST = {
    "stage4.association.graph.similarity_threshold": "0.53",
    "stage4.association.fic.regularisation": "0.15",
    "stage4.association.fac.enabled": "true",
    "stage4.association.fac.knn": "20",
    "stage4.association.fac.learning_rate": "0.5",
    "stage4.association.fac.beta": "0.08",
    "stage4.association.query_expansion.k": "3",
    "stage4.association.intra_camera_merge.enabled": "true",
    "stage4.association.intra_camera_merge.threshold": "0.75",
    "stage4.association.intra_camera_merge.max_time_gap": "60",
}

BASELINE_IDF1 = 0.813


def run_experiment(name, overrides):
    cmd = [
        PYTHON, "scripts/run_pipeline.py",
        "-c", "configs/default.yaml", "-s", "4,5",
    ]
    for o in FIXED_OVERRIDES:
        cmd += ["--override", o]
    for k, v in overrides.items():
        cmd += ["--override", f"{k}={v}"]

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
    elapsed = time.time() - t0

    all_output = r.stdout + "\n" + r.stderr
    mtmc_idf1 = mtmc_mota = frag = conflated = None
    for line in all_output.split("\n"):
        if "MTMC evaluation:" in line:
            for part in line.split(","):
                part = part.strip()
                if "IDF1=" in part:
                    try: mtmc_idf1 = float(part.split("IDF1=")[1].split(",")[0].strip())
                    except: pass
                if "MOTA=" in part:
                    try: mtmc_mota = float(part.split("MOTA=")[1].split(",")[0].strip())
                    except: pass
        if "fragmented GT IDs" in line:
            try:
                frag = int(line.split("analysis:")[1].split("fragmented")[0].strip())
                conflated = int(line.split(",")[1].strip().split(" ")[0])
            except: pass

    if mtmc_idf1 is not None:
        delta = mtmc_idf1 - BASELINE_IDF1
        status = "BETTER" if delta > 0.0005 else ("WORSE" if delta < -0.0005 else "SAME")
        print(
            f"  [{status}] {name}: IDF1={mtmc_idf1:.3f} ({delta:+.3f}) "
            f"frag={frag} conf={conflated} ({elapsed:.0f}s)", flush=True,
        )
        return {"name": name, "idf1": mtmc_idf1, "mota": mtmc_mota or 0.,
                "fragmented": frag or 0, "conflated": conflated or 0,
                "time_s": round(elapsed, 1), "overrides": overrides}
    else:
        print(f"  [FAIL] {name}: no IDF1 ({elapsed:.0f}s)", flush=True)
        return None


def main():
    print("=" * 70, flush=True)
    print("ROUND 4 SWEEP — UNTAPPED FEATURES", flush=True)
    print(f"Base: IDF1={BASELINE_IDF1} (ic_gap060 best)", flush=True)
    print("=" * 70, flush=True)

    results = []

    experiments = [
        # Confirm baseline
        ("r4_base", {**BEST}),

        # --- 1. HIERARCHICAL CENTROID EXPANSION (highest priority) ---
        ("hier_default", {**BEST,
            "stage4.association.hierarchical.enabled": "true",
            "stage4.association.hierarchical.centroid_threshold": "0.35",
            "stage4.association.hierarchical.merge_threshold": "0.35",
            "stage4.association.hierarchical.orphan_threshold": "0.30",
            "stage4.association.hierarchical.max_merge_size": "12"}),
        ("hier_tight", {**BEST,
            "stage4.association.hierarchical.enabled": "true",
            "stage4.association.hierarchical.centroid_threshold": "0.40",
            "stage4.association.hierarchical.merge_threshold": "0.40",
            "stage4.association.hierarchical.orphan_threshold": "0.35",
            "stage4.association.hierarchical.max_merge_size": "12"}),
        ("hier_loose", {**BEST,
            "stage4.association.hierarchical.enabled": "true",
            "stage4.association.hierarchical.centroid_threshold": "0.30",
            "stage4.association.hierarchical.merge_threshold": "0.30",
            "stage4.association.hierarchical.orphan_threshold": "0.25",
            "stage4.association.hierarchical.max_merge_size": "12"}),
        ("hier_veryloose", {**BEST,
            "stage4.association.hierarchical.enabled": "true",
            "stage4.association.hierarchical.centroid_threshold": "0.25",
            "stage4.association.hierarchical.merge_threshold": "0.25",
            "stage4.association.hierarchical.orphan_threshold": "0.20",
            "stage4.association.hierarchical.max_merge_size": "16"}),

        # --- 2. MUTUAL NN TOP_K (expand nearest neighbor pool) ---
        ("mnn_k30", {**BEST,
            "stage4.association.mutual_nn.top_k_per_query": "30"}),
        ("mnn_k40", {**BEST,
            "stage4.association.mutual_nn.top_k_per_query": "40"}),
        ("mnn_k50", {**BEST,
            "stage4.association.mutual_nn.top_k_per_query": "50"}),

        # --- 3. QE ALPHA (stronger smoothing) ---
        ("qe_alpha3", {**BEST,
            "stage4.association.query_expansion.alpha": "3.0"}),
        ("qe_alpha2", {**BEST,
            "stage4.association.query_expansion.alpha": "2.0"}),

        # --- 4. LENGTH WEIGHT POWER ---
        ("lwp_0", {**BEST,
            "stage4.association.weights.length_weight_power": "0.0"}),
        ("lwp_05", {**BEST,
            "stage4.association.weights.length_weight_power": "0.5"}),

        # --- 5. TEMPORAL OVERLAP TUNING ---
        ("temp_bonus08", {**BEST,
            "stage4.association.temporal_overlap.bonus": "0.08",
            "stage4.association.temporal_overlap.max_mean_time": "10.0"}),
        ("temp_bonus10", {**BEST,
            "stage4.association.temporal_overlap.bonus": "0.10",
            "stage4.association.temporal_overlap.max_mean_time": "10.0"}),

        # --- 6. ORPHAN MATCH THRESHOLD ---
        ("orphan_t035", {**BEST,
            "stage4.association.gallery_expansion.orphan_match_threshold": "0.35"}),
        ("orphan_t030", {**BEST,
            "stage4.association.gallery_expansion.orphan_match_threshold": "0.30"}),

        # --- 7. COMBINATIONS of best single features ---
        # Hier + MNN k30
        ("hier_mnn30", {**BEST,
            "stage4.association.hierarchical.enabled": "true",
            "stage4.association.hierarchical.centroid_threshold": "0.35",
            "stage4.association.hierarchical.merge_threshold": "0.35",
            "stage4.association.hierarchical.orphan_threshold": "0.30",
            "stage4.association.hierarchical.max_merge_size": "12",
            "stage4.association.mutual_nn.top_k_per_query": "30"}),
        # Hier + temporal bonus
        ("hier_temp", {**BEST,
            "stage4.association.hierarchical.enabled": "true",
            "stage4.association.hierarchical.centroid_threshold": "0.35",
            "stage4.association.hierarchical.merge_threshold": "0.35",
            "stage4.association.hierarchical.orphan_threshold": "0.30",
            "stage4.association.hierarchical.max_merge_size": "12",
            "stage4.association.temporal_overlap.bonus": "0.08",
            "stage4.association.temporal_overlap.max_mean_time": "10.0"}),
        # MNN k30 + temporal bonus
        ("mnn30_temp", {**BEST,
            "stage4.association.mutual_nn.top_k_per_query": "30",
            "stage4.association.temporal_overlap.bonus": "0.08",
            "stage4.association.temporal_overlap.max_mean_time": "10.0"}),
    ]

    for name, overrides in experiments:
        result = run_experiment(name, overrides)
        if result:
            results.append(result)

    print("\n" + "=" * 70, flush=True)
    print("ROUND 4 SUMMARY (sorted by IDF1)", flush=True)
    print("=" * 70, flush=True)
    results.sort(key=lambda x: -x["idf1"])
    for i, r in enumerate(results, 1):
        delta = r["idf1"] - BASELINE_IDF1
        print(
            f"  {i:>3}. {r['name']:<28s} IDF1={r['idf1']:.4f} ({delta:+.4f}) "
            f"frag={r['fragmented']:>3} conf={r['conflated']:>3}",
            flush=True,
        )

    # Save results
    out = BASE_DIR / "data" / "outputs" / "sweep_round4.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}", flush=True)

    if results:
        best = results[0]
        print(f"\n*** BEST: {best['name']} IDF1={best['idf1']:.4f} ***", flush=True)
        print(f"    Overrides: {best['overrides']}", flush=True)


if __name__ == "__main__":
    main()
