"""Round 2 sweep: test CSLS, intra-camera merge, cluster verification,
and k-reciprocal reranking on top of the best config from round 1.

Best base: FAC default + AQE k=3 + FIC 0.15 → IDF1=0.799, 26 conflated
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

# Best base config from round 1 (FAC default + AQE k=3 + FIC 0.15)
BEST_BASE = {
    "stage4.association.graph.similarity_threshold": "0.55",
    "stage4.association.fic.regularisation": "0.15",
    "stage4.association.fac.enabled": "true",
    "stage4.association.fac.knn": "20",
    "stage4.association.fac.learning_rate": "0.5",
    "stage4.association.fac.beta": "0.08",
    "stage4.association.query_expansion.k": "3",
}

BASELINE_IDF1 = 0.799


def run_experiment(name: str, overrides: dict) -> dict | None:
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

    all_output = r.stdout + "\n" + r.stderr
    mtmc_idf1 = mtmc_mota = id_sw = frag = conflated = None
    for line in all_output.split("\n"):
        if "MTMC evaluation:" in line:
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
            "mota": mtmc_mota or 0.,
            "id_sw": id_sw or 0,
            "fragmented": frag or 0,
            "conflated": conflated or 0,
            "time_s": round(elapsed, 1),
            "overrides": overrides,
        }
        delta = mtmc_idf1 - BASELINE_IDF1
        status = "BETTER" if delta > 0.0005 else ("WORSE" if delta < -0.0005 else "SAME")
        print(
            f"  [{status}] {name}: IDF1={mtmc_idf1:.3f} ({delta:+.3f}) "
            f"MOTA={mtmc_mota:.3f} frag={frag} conf={conflated} ({elapsed:.0f}s)",
            flush=True,
        )
        return result
    else:
        print(f"  [FAIL] {name}: no MTMC IDF1 parsed ({elapsed:.0f}s)", flush=True)
        if r.returncode != 0:
            for el in r.stderr.strip().split("\n")[-10:]:
                print(f"    ERR: {el}", flush=True)
        return None


def main():
    print("=" * 70, flush=True)
    print("ROUND 2 SWEEP — NEW FEATURES", flush=True)
    print(f"Base: IDF1={BASELINE_IDF1} (FAC+AQE_k3+FIC015)", flush=True)
    print("=" * 70, flush=True)

    results = []

    experiments = [
        # ---- Confirm baseline ----
        ("r2_base", {**BEST_BASE}),

        # ---- CSLS hubness reduction ----
        ("csls_k5", {**BEST_BASE,
            "stage4.association.csls.enabled": "true",
            "stage4.association.csls.k": "5",
        }),
        ("csls_k10", {**BEST_BASE,
            "stage4.association.csls.enabled": "true",
            "stage4.association.csls.k": "10",
        }),
        ("csls_k15", {**BEST_BASE,
            "stage4.association.csls.enabled": "true",
            "stage4.association.csls.k": "15",
        }),

        # ---- Intra-camera merge ----
        ("intracam_065", {**BEST_BASE,
            "stage4.association.intra_camera_merge.enabled": "true",
            "stage4.association.intra_camera_merge.threshold": "0.65",
            "stage4.association.intra_camera_merge.max_time_gap": "120",
        }),
        ("intracam_070", {**BEST_BASE,
            "stage4.association.intra_camera_merge.enabled": "true",
            "stage4.association.intra_camera_merge.threshold": "0.70",
            "stage4.association.intra_camera_merge.max_time_gap": "120",
        }),
        ("intracam_075", {**BEST_BASE,
            "stage4.association.intra_camera_merge.enabled": "true",
            "stage4.association.intra_camera_merge.threshold": "0.75",
            "stage4.association.intra_camera_merge.max_time_gap": "120",
        }),

        # ---- Post-cluster verification ----
        ("verify_025", {**BEST_BASE,
            "stage4.association.cluster_verify.enabled": "true",
            "stage4.association.cluster_verify.min_connectivity": "0.25",
        }),
        ("verify_030", {**BEST_BASE,
            "stage4.association.cluster_verify.enabled": "true",
            "stage4.association.cluster_verify.min_connectivity": "0.30",
        }),
        ("verify_035", {**BEST_BASE,
            "stage4.association.cluster_verify.enabled": "true",
            "stage4.association.cluster_verify.min_connectivity": "0.35",
        }),

        # ---- k-Reciprocal re-ranking (high lambda = more original) ----
        ("rerank_l07", {**BEST_BASE,
            "stage4.association.reranking.enabled": "true",
            "stage4.association.reranking.k1": "15",
            "stage4.association.reranking.k2": "6",
            "stage4.association.reranking.lambda_value": "0.7",
        }),
        ("rerank_l08", {**BEST_BASE,
            "stage4.association.reranking.enabled": "true",
            "stage4.association.reranking.k1": "15",
            "stage4.association.reranking.k2": "6",
            "stage4.association.reranking.lambda_value": "0.8",
        }),
        ("rerank_l09", {**BEST_BASE,
            "stage4.association.reranking.enabled": "true",
            "stage4.association.reranking.k1": "10",
            "stage4.association.reranking.k2": "4",
            "stage4.association.reranking.lambda_value": "0.9",
        }),

        # ---- Agglomerative complete-linkage graph solver ----
        ("agglom", {**BEST_BASE,
            "stage4.association.graph.algorithm": "agglomerative",
        }),
        ("agglom_t050", {**BEST_BASE,
            "stage4.association.graph.algorithm": "agglomerative",
            "stage4.association.graph.similarity_threshold": "0.50",
        }),

        # ---- AQE k=2 (even tighter NN) ----
        ("aqe_k2", {**BEST_BASE,
            "stage4.association.query_expansion.k": "2",
        }),

        # ---- More aggressive FAC + AQE k=3 but less aggressive lr ----
        ("fac_knn30_lr05_aqek3", {**BEST_BASE,
            "stage4.association.fac.knn": "30",
            "stage4.association.fac.learning_rate": "0.5",
        }),

        # ---- Threshold 0.53 with base (might help merge more) ----
        ("thresh053", {**BEST_BASE,
            "stage4.association.graph.similarity_threshold": "0.53",
        }),

        # ---- FIC reg 0.20 (more whitening) ----
        ("fic020", {**BEST_BASE,
            "stage4.association.fic.regularisation": "0.20",
        }),
    ]

    for name, overrides in experiments:
        result = run_experiment(name, overrides)
        if result:
            results.append(result)
        print(flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("ROUND 2 SUMMARY (sorted by IDF1)", flush=True)
    print("=" * 70, flush=True)
    results.sort(key=lambda x: x["idf1"], reverse=True)
    for i, r in enumerate(results):
        delta = r["idf1"] - BASELINE_IDF1
        marker = " ***" if delta > 0.001 else (" *" if delta > 0.0 else "")
        print(
            f"  {i+1:2d}. {r['name']:25s} IDF1={r['idf1']:.4f} "
            f"({delta:+.4f}) frag={r['fragmented']:3d} conf={r['conflated']:3d}{marker}",
            flush=True,
        )

    out_path = BASE_DIR / "data" / "outputs" / "sweep_round2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    best = results[0] if results else None
    if best and best["idf1"] > BASELINE_IDF1:
        print(
            f"\n*** BEST: {best['name']} IDF1={best['idf1']:.4f} "
            f"(+{best['idf1']-BASELINE_IDF1:.4f}pp) ***",
            flush=True,
        )
        print(f"    Overrides: {best['overrides']}", flush=True)


if __name__ == "__main__":
    main()
