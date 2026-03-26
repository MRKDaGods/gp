"""Round 3: Optimize intracam merge + threshold combinations.
Best from round 2: intracam_075 + thresh053 → IDF1=0.812
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

# Best combined config so far
BEST_COMBINED = {
    "stage4.association.graph.similarity_threshold": "0.53",
    "stage4.association.fic.regularisation": "0.15",
    "stage4.association.fac.enabled": "true",
    "stage4.association.fac.knn": "20",
    "stage4.association.fac.learning_rate": "0.5",
    "stage4.association.fac.beta": "0.08",
    "stage4.association.query_expansion.k": "3",
    "stage4.association.intra_camera_merge.enabled": "true",
    "stage4.association.intra_camera_merge.threshold": "0.75",
    "stage4.association.intra_camera_merge.max_time_gap": "120",
}

BASELINE_IDF1 = 0.812


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
    print("ROUND 3 SWEEP — OPTIMIZE INTRACAM + THRESHOLD", flush=True)
    print(f"Base: IDF1={BASELINE_IDF1} (intracam_075 + thresh053)", flush=True)
    print("=" * 70, flush=True)

    results = []

    experiments = [
        # Confirm base
        ("r3_base", {**BEST_COMBINED}),

        # --- Threshold variations with intracam ---
        ("thresh050_ic075", {**BEST_COMBINED,
            "stage4.association.graph.similarity_threshold": "0.50"}),
        ("thresh051_ic075", {**BEST_COMBINED,
            "stage4.association.graph.similarity_threshold": "0.51"}),
        ("thresh052_ic075", {**BEST_COMBINED,
            "stage4.association.graph.similarity_threshold": "0.52"}),
        ("thresh054_ic075", {**BEST_COMBINED,
            "stage4.association.graph.similarity_threshold": "0.54"}),

        # --- Intracam threshold variations ---
        ("thresh053_ic070", {**BEST_COMBINED,
            "stage4.association.intra_camera_merge.threshold": "0.70"}),
        ("thresh053_ic078", {**BEST_COMBINED,
            "stage4.association.intra_camera_merge.threshold": "0.78"}),
        ("thresh053_ic080", {**BEST_COMBINED,
            "stage4.association.intra_camera_merge.threshold": "0.80"}),
        ("thresh053_ic060", {**BEST_COMBINED,
            "stage4.association.intra_camera_merge.threshold": "0.60"}),

        # --- Intracam time gap variations ---
        ("ic_gap060", {**BEST_COMBINED,
            "stage4.association.intra_camera_merge.max_time_gap": "60"}),
        ("ic_gap180", {**BEST_COMBINED,
            "stage4.association.intra_camera_merge.max_time_gap": "180"}),
        ("ic_gap240", {**BEST_COMBINED,
            "stage4.association.intra_camera_merge.max_time_gap": "240"}),

        # --- FAC aggressive + intracam ---
        ("fac_agg_ic", {**BEST_COMBINED,
            "stage4.association.fac.knn": "30",
            "stage4.association.fac.learning_rate": "0.7",
            "stage4.association.fac.beta": "0.10"}),

        # --- FIC reg 0.10 with intracam+thresh053 ---
        ("fic010_ic", {**BEST_COMBINED,
            "stage4.association.fic.regularisation": "0.10"}),

        # --- AQE k=5 (default) with intracam+thresh053 ---
        ("aqek5_ic", {**BEST_COMBINED,
            "stage4.association.query_expansion.k": "5"}),

        # --- Gallery expansion tuning with intracam ---
        ("gallery_loose_ic", {**BEST_COMBINED,
            "stage4.association.gallery_expansion.threshold": "0.45",
            "stage4.association.gallery_expansion.max_rounds": "3",
            "stage4.association.gallery_expansion.orphan_match_threshold": "0.35"}),

        # --- Best threshold + intracam 0.78 (conservative) ---
        ("thresh052_ic078", {**BEST_COMBINED,
            "stage4.association.graph.similarity_threshold": "0.52",
            "stage4.association.intra_camera_merge.threshold": "0.78"}),

        # --- Secondary weight tuning ---
        ("sec_w04", {**BEST_COMBINED,
            "stage4.association.secondary_embeddings.weight": "0.4"}),
        ("sec_w02", {**BEST_COMBINED,
            "stage4.association.secondary_embeddings.weight": "0.2"}),

        # --- Max component size increase ---
        ("maxcomp16", {**BEST_COMBINED,
            "stage4.association.graph.max_component_size": "16"}),
    ]

    for name, overrides in experiments:
        result = run_experiment(name, overrides)
        if result:
            results.append(result)
        print(flush=True)

    print("\n" + "=" * 70, flush=True)
    print("ROUND 3 SUMMARY (sorted by IDF1)", flush=True)
    print("=" * 70, flush=True)
    results.sort(key=lambda x: x["idf1"], reverse=True)
    for i, r in enumerate(results):
        delta = r["idf1"] - BASELINE_IDF1
        marker = " ***" if delta > 0.001 else ""
        print(
            f"  {i+1:2d}. {r['name']:25s} IDF1={r['idf1']:.4f} "
            f"({delta:+.4f}) frag={r['fragmented']:3d} conf={r['conflated']:3d}{marker}",
            flush=True,
        )

    out_path = BASE_DIR / "data" / "outputs" / "sweep_round3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    best = results[0] if results else None
    if best:
        print(f"\n*** BEST: {best['name']} IDF1={best['idf1']:.4f} ***", flush=True)
        print(f"    Overrides: {best['overrides']}", flush=True)


if __name__ == "__main__":
    main()
