"""Phase 2 scan: test sub-cluster temporal splitting thresholds.

Baseline: IDF1=0.830 (v48 config + secondary embeddings).
Reuses the existing stage1/2/3 data from the local 10a v11 run.

Experiments (stages 4+5 only, ~2min each):
  - Temporal split: min_gap ∈ {30, 45, 60, 90, 120}s × split_threshold ∈ {0.45, 0.50, 0.55}
"""
import subprocess, sys, os, re, json
from pathlib import Path

PROJECT = Path(__file__).parent.parent
RUN_BASE = PROJECT / "data/outputs/kaggle_10a_v11/extracted"
RUN_NAME = "run_kaggle_20260318_114803"
RUN_DIR = RUN_BASE / RUN_NAME
SEC_EMB = str(RUN_DIR / "stage2" / "embeddings_secondary.npy")
GT_DIR = str(PROJECT / "data/raw/cityflowv2")
os.chdir(str(PROJECT))

BASE_OVERRIDES = [
    "--stages", "4,5",
    "--override", f"project.output_dir={RUN_BASE}",
    "--override", f"project.run_name={RUN_NAME}",
    "--override", "stage0.cameras=[S01_c001,S01_c002,S01_c003,S02_c006,S02_c007,S02_c008]",
    "--override", "stage4.association.graph.similarity_threshold=0.53",
    "--override", "stage4.association.fic.regularisation=3.0",
    "--override", "stage4.association.fic.enabled=true",
    "--override", "stage4.association.fac.enabled=true",
    "--override", "stage4.association.fac.knn=20",
    "--override", "stage4.association.fac.learning_rate=0.5",
    "--override", "stage4.association.fac.beta=0.08",
    "--override", "stage4.association.query_expansion.k=2",
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


def run_scan(name, extra_overrides):
    cmd = [
        sys.executable, "scripts/run_pipeline.py",
        "--config", "configs/default.yaml",
    ] + BASE_OVERRIDES + extra_overrides

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    idf1 = mota = hota = ids = None

    # Try to find IDF1 from MTMC evaluation output line
    for line in output.split("\n"):
        if "MTMC evaluation:" in line or "[MTMC]" in line:
            for part in line.split():
                try:
                    if part.startswith("IDF1="):
                        idf1 = float(part.split("=")[1].strip("%,")) / 100
                    elif part.startswith("MOTA="):
                        mota = float(part.split("=")[1].strip("%,")) / 100
                    elif part.startswith("HOTA="):
                        hota = float(part.split("=")[1].strip("%,")) / 100
                    elif part.startswith("ID_Switches="):
                        ids = int(part.split("=")[1].strip(","))
                except Exception:
                    pass

    if idf1 is None:
        # Fallback: search last 40 lines for IDF1 pattern
        for line in output.split("\n")[-40:]:
            m = re.search(r"IDF1[=:]\s*([\d.]+)%?", line)
            if m:
                idf1 = float(m.group(1)) / 100
                break

    # Also try reading evaluation_report.json directly
    if idf1 is None:
        report_path = RUN_DIR / "stage5" / "evaluation_report.json"
        if report_path.exists():
            try:
                with open(report_path) as f:
                    report = json.load(f)
                idf1 = report.get("mtmc_idf1", report.get("idf1"))
                mota = report.get("mtmc_mota", report.get("mota"))
                hota = report.get("mtmc_hota", report.get("hota"))
            except Exception:
                pass

    return {
        "name": name,
        "idf1": idf1,
        "mota": mota,
        "hota": hota,
        "ids": ids,
        "rc": result.returncode,
    }


# Baseline (no temporal split)
experiments = [
    (
        "baseline",
        [],
    ),
]

# Temporal split: min_gap × split_threshold grid
for min_gap in [30, 45, 60, 90]:
    for split_thresh in [0.45, 0.50, 0.55]:
        tag = f"ts_gap{int(min_gap)}_thresh{int(split_thresh*100)}"
        experiments.append((
            tag,
            [
                "--override", "stage4.association.temporal_split.enabled=true",
                "--override", f"stage4.association.temporal_split.min_gap={min_gap}.0",
                "--override", f"stage4.association.temporal_split.split_threshold={split_thresh}",
            ],
        ))

print(f"Phase 2 scan: {len(experiments)} experiments")
print("=" * 60)

results = []
for name, extra in experiments:
    print(f"Running {name}...", end=" ", flush=True)
    r = run_scan(name, extra)
    results.append(r)
    idf1_str = f"{r['idf1']:.4f}" if r["idf1"] is not None else "FAILED"
    ids_str = f"  IDS={r['ids']}" if r["ids"] is not None else ""
    print(f"IDF1={idf1_str}{ids_str}")

print("\n" + "=" * 60)
print("SUMMARY (sorted by IDF1 desc):")
print(f"{'Name':<35} {'IDF1':>8} {'MOTA':>8} {'HOTA':>8} {'IDS':>6}")
print("-" * 65)

baseline_idf1 = next((r["idf1"] for r in results if r["name"] == "baseline"), None)

for r in sorted(results, key=lambda x: (x["idf1"] or 0), reverse=True):
    idf1 = f"{r['idf1']:.4f}" if r["idf1"] is not None else "FAILED"
    mota = f"{r['mota']:.4f}" if r["mota"] is not None else "  N/A"
    hota = f"{r['hota']:.4f}" if r["hota"] is not None else "  N/A"
    ids = f"{r['ids']}" if r["ids"] is not None else "  N/A"

    delta_str = ""
    if r["idf1"] is not None and baseline_idf1 is not None and r["name"] != "baseline":
        delta = r["idf1"] - baseline_idf1
        delta_str = f"  ({delta:+.4f})"

    print(f"{r['name']:<35} {idf1:>8} {mota:>8} {hota:>8} {ids:>6}{delta_str}")

print("=" * 60)
best = max((r for r in results if r["idf1"] is not None), key=lambda x: x["idf1"], default=None)
if best:
    print(f"Best: {best['name']}  IDF1={best.get('idf1', 'N/A'):.4f}")
