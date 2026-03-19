"""Round 5: Strategic threshold + expansion experiments.
Hypothesis: higher graph threshold → fewer false merges, then gallery expansion recovers orphans.
Also test: per-pair thresholds, combined score tuning, and weight balancing.

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

# Best config (ic_gap060)
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
    mtmc_idf1 = mtmc_mota = frag = conflated = unmatched_pred = None
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
        if "unmatched pred" in line:
            try:
                parts = line.split(",")
                for p in parts:
                    if "unmatched pred" in p:
                        unmatched_pred = int(p.strip().split(" ")[0])
            except: pass

    if mtmc_idf1 is not None:
        delta = mtmc_idf1 - BASELINE_IDF1
        status = "BETTER" if delta > 0.0005 else ("WORSE" if delta < -0.0005 else "SAME")
        print(
            f"  [{status}] {name}: IDF1={mtmc_idf1:.3f} ({delta:+.3f}) "
            f"frag={frag} conf={conflated} unm_p={unmatched_pred} ({elapsed:.0f}s)", flush=True,
        )
        return {"name": name, "idf1": mtmc_idf1, "mota": mtmc_mota or 0.,
                "fragmented": frag or 0, "conflated": conflated or 0,
                "unmatched_pred": unmatched_pred or 0,
                "time_s": round(elapsed, 1), "overrides": overrides}
    else:
        print(f"  [FAIL] {name}: no IDF1 ({elapsed:.0f}s)", flush=True)
        # Print last 20 lines of output for debugging
        lines = all_output.strip().split("\n")
        for l in lines[-20:]:
            print(f"    {l}", flush=True)
        return None


def main():
    print("=" * 70, flush=True)
    print("ROUND 5 SWEEP — STRATEGIC THRESHOLD + EXPANSION", flush=True)
    print(f"Base: IDF1={BASELINE_IDF1} (ic_gap060 best)", flush=True)
    print("=" * 70, flush=True)

    results = []

    experiments = [
        # Confirm baseline
        ("r5_base", {**BEST}),

        # --- Strategy 1: Higher graph threshold + wider gallery expansion ---
        # With thresh=0.53, we get 26 conflated, 81 fragmented.
        # Higher threshold = fewer false edges = fewer conflated, more fragmented.
        # But gallery expansion can recover orphans using centroid/member sims.
        ("t055_gal045", {**BEST,
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.gallery_expansion.threshold": "0.45",
            "stage4.association.gallery_expansion.orphan_match_threshold": "0.35",
            "stage4.association.gallery_expansion.max_rounds": "3"}),
        ("t057_gal045", {**BEST,
            "stage4.association.graph.similarity_threshold": "0.57",
            "stage4.association.gallery_expansion.threshold": "0.45",
            "stage4.association.gallery_expansion.orphan_match_threshold": "0.35",
            "stage4.association.gallery_expansion.max_rounds": "3"}),
        ("t060_gal045", {**BEST,
            "stage4.association.graph.similarity_threshold": "0.60",
            "stage4.association.gallery_expansion.threshold": "0.45",
            "stage4.association.gallery_expansion.orphan_match_threshold": "0.35",
            "stage4.association.gallery_expansion.max_rounds": "3"}),

        # --- Strategy 2: Weight tuning for combined similarity ---
        # Current: appearance=0.75 (vehicle), hsv=0.0, st=0.25
        # Maybe less appearance weight with better calibrated embeddings?
        ("w_app065_st035", {**BEST,
            "stage4.association.weights.vehicle.appearance": "0.65",
            "stage4.association.weights.vehicle.spatiotemporal": "0.35"}),
        ("w_app080_st020", {**BEST,
            "stage4.association.weights.vehicle.appearance": "0.80",
            "stage4.association.weights.vehicle.spatiotemporal": "0.20"}),
        ("w_app070_hsv005_st025", {**BEST,
            "stage4.association.weights.vehicle.appearance": "0.70",
            "stage4.association.weights.vehicle.hsv": "0.05",
            "stage4.association.weights.vehicle.spatiotemporal": "0.25"}),

        # --- Strategy 3: Secondary embedding weight tuning ---
        # Current: 0.3 (30% OSNet). Maybe different blend helps.
        ("sec_w015", {**BEST,
            "stage4.association.secondary_embeddings.weight": "0.15"}),
        ("sec_w025", {**BEST,
            "stage4.association.secondary_embeddings.weight": "0.25"}),
        ("sec_w035", {**BEST,
            "stage4.association.secondary_embeddings.weight": "0.35"}),

        # --- Strategy 4: FAC tuning (finer grid around optimal) ---
        ("fac_knn15_lr04", {**BEST,
            "stage4.association.fac.knn": "15",
            "stage4.association.fac.learning_rate": "0.4"}),
        ("fac_knn25_lr06", {**BEST,
            "stage4.association.fac.knn": "25",
            "stage4.association.fac.learning_rate": "0.6"}),
        ("fac_beta005", {**BEST,
            "stage4.association.fac.beta": "0.05"}),
        ("fac_beta012", {**BEST,
            "stage4.association.fac.beta": "0.12"}),

        # --- Strategy 5: FIC regularisation fine-tuning ---
        ("fic012", {**BEST,
            "stage4.association.fic.regularisation": "0.12"}),
        ("fic018", {**BEST,
            "stage4.association.fic.regularisation": "0.18"}),
        ("fic020", {**BEST,
            "stage4.association.fic.regularisation": "0.20"}),

        # --- Strategy 6: Intra-camera merge with higher threshold ---
        ("ic085_gap60", {**BEST,
            "stage4.association.intra_camera_merge.threshold": "0.85",
            "stage4.association.intra_camera_merge.max_time_gap": "60"}),
        ("ic082_gap45", {**BEST,
            "stage4.association.intra_camera_merge.threshold": "0.82",
            "stage4.association.intra_camera_merge.max_time_gap": "45"}),

        # --- Strategy 7: Combined best candidates ---
        # Higher sec weight if it helps + adjusted FAC
        ("combo_sec025_fac15lr04", {**BEST,
            "stage4.association.secondary_embeddings.weight": "0.25",
            "stage4.association.fac.knn": "15",
            "stage4.association.fac.learning_rate": "0.4"}),
    ]

    for name, overrides in experiments:
        result = run_experiment(name, overrides)
        if result:
            results.append(result)

    print("\n" + "=" * 70, flush=True)
    print("ROUND 5 SUMMARY (sorted by IDF1)", flush=True)
    print("=" * 70, flush=True)
    results.sort(key=lambda x: -x["idf1"])
    for i, r in enumerate(results, 1):
        delta = r["idf1"] - BASELINE_IDF1
        print(
            f"  {i:>3}. {r['name']:<28s} IDF1={r['idf1']:.4f} ({delta:+.4f}) "
            f"frag={r['fragmented']:>3} conf={r['conflated']:>3} unm={r['unmatched_pred']:>3}",
            flush=True,
        )

    # Save results
    out = BASE_DIR / "data" / "outputs" / "sweep_round5.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}", flush=True)

    if results:
        best = results[0]
        print(f"\n*** BEST: {best['name']} IDF1={best['idf1']:.4f} ***", flush=True)
        print(f"    Overrides: {best['overrides']}", flush=True)


if __name__ == "__main__":
    main()
