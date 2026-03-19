"""Automated stage 4-5 parameter sweep for MTMC IDF1 optimization.

Usage:
    python scripts/sweep_stage45.py
"""
import subprocess
import sys
import json
import time
from pathlib import Path
import itertools

PYTHON = sys.executable
BASE_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = "data/outputs/kaggle_10a_v11/extracted"
RUN_NAME = "run_kaggle_20260318_114803"
SECONDARY_EMB = str(
    BASE_DIR / RUN_DIR / RUN_NAME / "stage2" / "embeddings_secondary.npy"
)

# Fixed overrides that match Kaggle evaluation protocol
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


def run_experiment(name: str, overrides: dict) -> dict | None:
    """Run a single stage 4-5 experiment and return metrics."""
    cmd = [
        PYTHON, "scripts/run_pipeline.py",
        "-c", "configs/default.yaml",
        "-s", "4,5",
    ]
    for o in FIXED_OVERRIDES:
        cmd += ["--override", o]
    for k, v in overrides.items():
        cmd += ["--override", f"{k}={v}"]

    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BASE_DIR))
    elapsed = time.time() - t0

    # Parse MTMC IDF1 from output (loguru writes to stderr)
    all_output = r.stdout + "\n" + r.stderr
    mtmc_idf1 = None
    mtmc_mota = None
    id_sw = None
    frag = None
    conflated = None
    for line in all_output.split("\n"):
        if "MTMC evaluation:" in line:
            # Extract IDF1
            for part in line.split(","):
                part = part.strip()
                if "IDF1=" in part:
                    try:
                        mtmc_idf1 = float(part.split("IDF1=")[1].split(",")[0].strip())
                    except (ValueError, IndexError):
                        pass
                if "MOTA=" in part:
                    try:
                        mtmc_mota = float(part.split("MOTA=")[1].split(",")[0].strip())
                    except (ValueError, IndexError):
                        pass
                if "ID Switches=" in part:
                    try:
                        id_sw = int(part.split("ID Switches=")[1].strip())
                    except (ValueError, IndexError):
                        pass
        if "fragmented GT IDs" in line:
            try:
                frag = int(line.split("analysis:")[1].split("fragmented")[0].strip())
                conflated = int(line.split(",")[1].strip().split(" ")[0])
            except (ValueError, IndexError):
                pass

    if mtmc_idf1 is not None:
        result = {
            "name": name,
            "idf1": mtmc_idf1,
            "mota": mtmc_mota,
            "id_sw": id_sw,
            "fragmented": frag,
            "conflated": conflated,
            "time_s": round(elapsed, 1),
            "overrides": overrides,
        }
        status = "BETTER" if mtmc_idf1 > 0.789 else "same/worse"
        print(f"  [{status}] {name}: IDF1={mtmc_idf1:.3f} MOTA={mtmc_mota:.3f} IDsw={id_sw} frag={frag} conf={conflated} ({elapsed:.0f}s)", flush=True)
        return result
    else:
        print(f"  [FAIL] {name}: no MTMC IDF1 parsed ({elapsed:.0f}s)", flush=True)
        if r.returncode != 0:
            # Print last 20 lines of stderr
            err_lines = r.stderr.strip().split("\n")[-20:]
            for el in err_lines:
                print(f"    ERR: {el}", flush=True)
        return None


def main():
    print("=" * 70, flush=True)
    print("STAGE 4-5 PARAMETER SWEEP", flush=True)
    print(f"Baseline: MTMC IDF1=0.789, MOTA=0.835", flush=True)
    print("=" * 70, flush=True)

    results = []

    # v42 baseline (already tested, but confirm)
    experiments = [
        # ---- v42 Baseline (confirmed 0.789) ----
        ("v42_baseline", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
        }),

        # ---- FAC variants (fac_default gave 0.796!) ----
        ("fac_default", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "20",
            "stage4.association.fac.learning_rate": "0.5",
            "stage4.association.fac.beta": "0.08",
        }),
        ("fac_conservative", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "10",
            "stage4.association.fac.learning_rate": "0.3",
            "stage4.association.fac.beta": "0.05",
        }),
        ("fac_aggressive", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "30",
            "stage4.association.fac.learning_rate": "0.7",
            "stage4.association.fac.beta": "0.10",
        }),
        ("fac_knn15_lr04", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "15",
            "stage4.association.fac.learning_rate": "0.4",
            "stage4.association.fac.beta": "0.06",
        }),
        ("fac_knn25_lr06", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "25",
            "stage4.association.fac.learning_rate": "0.6",
            "stage4.association.fac.beta": "0.08",
        }),

        # ---- FAC + lower threshold (more merging) ----
        ("fac_thresh050", {
            "stage4.association.graph.similarity_threshold": "0.50",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "20",
            "stage4.association.fac.learning_rate": "0.5",
            "stage4.association.fac.beta": "0.08",
        }),
        ("fac_thresh052", {
            "stage4.association.graph.similarity_threshold": "0.52",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "20",
            "stage4.association.fac.learning_rate": "0.5",
            "stage4.association.fac.beta": "0.08",
        }),
        ("fac_thresh058", {
            "stage4.association.graph.similarity_threshold": "0.58",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "20",
            "stage4.association.fac.learning_rate": "0.5",
            "stage4.association.fac.beta": "0.08",
        }),

        # ---- FAC + gallery expansion tuning ----
        ("fac_gallery_loose", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "20",
            "stage4.association.fac.learning_rate": "0.5",
            "stage4.association.fac.beta": "0.08",
            "stage4.association.gallery_expansion.threshold": "0.45",
            "stage4.association.gallery_expansion.max_rounds": "3",
            "stage4.association.gallery_expansion.orphan_match_threshold": "0.35",
        }),

        # ---- FAC + camera bias combo ----
        ("fac_camera_bias", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "20",
            "stage4.association.fac.learning_rate": "0.5",
            "stage4.association.fac.beta": "0.08",
            "stage4.association.camera_bias.enabled": "true",
        }),

        # ---- FAC + AQE k tuning ----
        ("fac_aqe_k7", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "20",
            "stage4.association.fac.learning_rate": "0.5",
            "stage4.association.fac.beta": "0.08",
            "stage4.association.query_expansion.k": "7",
        }),
        ("fac_aqe_k3", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "20",
            "stage4.association.fac.learning_rate": "0.5",
            "stage4.association.fac.beta": "0.08",
            "stage4.association.query_expansion.k": "3",
        }),

        # ---- FAC + appearance weight tuning ----
        ("fac_app080", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "20",
            "stage4.association.fac.learning_rate": "0.5",
            "stage4.association.fac.beta": "0.08",
            "stage4.association.weights.vehicle.appearance": "0.80",
            "stage4.association.weights.vehicle.hsv": "0.00",
            "stage4.association.weights.vehicle.spatiotemporal": "0.20",
        }),

        # ---- FAC + FIC tuning ----
        ("fac_fic005", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.05",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "20",
            "stage4.association.fac.learning_rate": "0.5",
            "stage4.association.fac.beta": "0.08",
        }),
        ("fac_fic015", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.15",
            "stage4.association.fac.enabled": "true",
            "stage4.association.fac.knn": "20",
            "stage4.association.fac.learning_rate": "0.5",
            "stage4.association.fac.beta": "0.08",
        }),

        # ---- No FAC baselines for comparison ----
        ("nofac_thresh050", {
            "stage4.association.graph.similarity_threshold": "0.50",
            "stage4.association.fic.regularisation": "0.1",
        }),
        ("nofac_thresh052", {
            "stage4.association.graph.similarity_threshold": "0.52",
            "stage4.association.fic.regularisation": "0.1",
        }),
        ("nofac_aqe_k7", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.query_expansion.k": "7",
        }),
        ("nofac_gallery_loose", {
            "stage4.association.graph.similarity_threshold": "0.55",
            "stage4.association.fic.regularisation": "0.1",
            "stage4.association.gallery_expansion.threshold": "0.45",
            "stage4.association.gallery_expansion.max_rounds": "3",
            "stage4.association.gallery_expansion.orphan_match_threshold": "0.35",
        }),
    ]

    for name, overrides in experiments:
        result = run_experiment(name, overrides)
        if result:
            results.append(result)
        print(flush=True)  # blank line between experiments

    # Summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY (sorted by IDF1)")
    print("=" * 70)
    results.sort(key=lambda x: x["idf1"], reverse=True)
    for i, r in enumerate(results):
        marker = " *" if r["idf1"] > 0.789 else ""
        print(f"  {i+1:2d}. {r['name']:25s} IDF1={r['idf1']:.4f} MOTA={r['mota']:.3f} IDsw={r['id_sw']:4d} frag={r['fragmented']:3d} conf={r['conflated']:3d}{marker}")

    # Save results
    out_path = BASE_DIR / "data" / "outputs" / "sweep_beat_sota.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    best = results[0] if results else None
    if best and best["idf1"] > 0.789:
        print(f"\n*** IMPROVEMENT FOUND: {best['name']} IDF1={best['idf1']:.4f} (+{best['idf1']-0.789:.4f}pp) ***")
        print(f"    Overrides: {best['overrides']}")
    else:
        print("\nNo improvement over baseline 0.789 found in this sweep.")


if __name__ == "__main__":
    main()
