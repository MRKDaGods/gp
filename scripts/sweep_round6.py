"""Round 6: Micro-tuning around sec_w025 best (IDF1=0.814).
Also try some creative ideas: disable gallery expansion, aggressive combinations.
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
    "stage5.mtmc_only_submission=false",
    "stage5.ground_truth_dir=data/raw/cityflowv2",
    "stage5.gt_frame_clip=true",
    "stage5.gt_zone_filter=true",
    "stage5.gt_zone_margin_frac=0.2",
    "stage5.gt_frame_clip_min_iou=0.5",
]

# Best config from round 5 (sec_w025)
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
    "stage4.association.secondary_embeddings.weight": "0.25",
}

BASELINE_IDF1 = 0.814


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
                for p in line.split(","):
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
        lines = all_output.strip().split("\n")
        for l in lines[-10:]:
            print(f"    {l}", flush=True)
        return None


def main():
    print("=" * 70, flush=True)
    print("ROUND 6 SWEEP — MICRO-TUNING + CREATIVE IDEAS", flush=True)
    print(f"Base: IDF1={BASELINE_IDF1} (sec_w025 best)", flush=True)
    print("=" * 70, flush=True)

    results = []

    experiments = [
        # Confirm baseline
        ("r6_base", {**BEST}),

        # --- sec_weight micro-tuning ---
        ("sec_w020", {**BEST,
            "stage4.association.secondary_embeddings.weight": "0.20"}),
        ("sec_w022", {**BEST,
            "stage4.association.secondary_embeddings.weight": "0.22"}),
        ("sec_w028", {**BEST,
            "stage4.association.secondary_embeddings.weight": "0.28"}),

        # --- Disable gallery expansion (already have intra-camera merge) ---
        ("no_gallery", {**BEST,
            "stage4.association.gallery_expansion.enabled": "false"}),

        # --- Lower graph threshold with sec_w025 ---
        ("t052_sw025", {**BEST,
            "stage4.association.graph.similarity_threshold": "0.52"}),
        ("t054_sw025", {**BEST,
            "stage4.association.graph.similarity_threshold": "0.54"}),

        # --- FIC + sec_weight combo ---
        ("fic012_sw025", {**BEST,
            "stage4.association.fic.regularisation": "0.12"}),
        ("fic013_sw025", {**BEST,
            "stage4.association.fic.regularisation": "0.13"}),

        # --- Disable mutual NN entirely (allow all edges through) ---
        ("no_mnn", {**BEST,
            "stage4.association.mutual_nn.enabled": "false"}),

        # --- QE k=2 with sec_w025 ---
        ("qe_k2_sw025", {**BEST,
            "stage4.association.query_expansion.k": "2"}),
        ("qe_k4_sw025", {**BEST,
            "stage4.association.query_expansion.k": "4"}),

        # --- Intra-camera merge + higher similarity threshold combo ---
        ("ic077_t054_sw025", {**BEST,
            "stage4.association.intra_camera_merge.threshold": "0.77",
            "stage4.association.graph.similarity_threshold": "0.54"}),

        # --- max_component_size tuning ---
        ("maxcomp10", {**BEST,
            "stage4.association.graph.max_component_size": "10"}),
        ("maxcomp14", {**BEST,
            "stage4.association.graph.max_component_size": "14"}),

        # --- Disable secondary entirely (primary only) ---
        ("no_secondary", {**BEST,
            "stage4.association.secondary_embeddings.weight": "0.0"}),
    ]

    for name, overrides in experiments:
        result = run_experiment(name, overrides)
        if result:
            results.append(result)

    print("\n" + "=" * 70, flush=True)
    print("ROUND 6 SUMMARY (sorted by IDF1)", flush=True)
    print("=" * 70, flush=True)
    results.sort(key=lambda x: -x["idf1"])
    for i, r in enumerate(results, 1):
        delta = r["idf1"] - BASELINE_IDF1
        print(
            f"  {i:>3}. {r['name']:<28s} IDF1={r['idf1']:.4f} ({delta:+.4f}) "
            f"frag={r['fragmented']:>3} conf={r['conflated']:>3} unm={r['unmatched_pred']:>3}",
            flush=True,
        )

    out = BASE_DIR / "data" / "outputs" / "sweep_round6.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}", flush=True)

    if results:
        best = results[0]
        print(f"\n*** BEST: {best['name']} IDF1={best['idf1']:.4f} ***", flush=True)
        print(f"    Overrides: {best['overrides']}", flush=True)


if __name__ == "__main__":
    main()
