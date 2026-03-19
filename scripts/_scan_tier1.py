"""Quick scan to test Tier 1 features: CSLS and cluster_verify.
Baseline: IDF1=0.830 (v46 config + secondary embeddings).
Reuses the existing stage1/2/3 data in the original run directory.
"""
import subprocess, sys, os, re
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
    cmd = [sys.executable, "scripts/run_pipeline.py", "--config", "configs/default.yaml"] + \
          BASE_OVERRIDES + extra_overrides

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    idf1 = mota = hota = ids = None
    for line in output.split('\n'):
        if 'MTMC evaluation:' in line or '[MTMC]' in line:
            for part in line.split():
                try:
                    if part.startswith('IDF1='): idf1 = float(part.split('=')[1].strip('%,')) / 100
                    elif part.startswith('MOTA='): mota = float(part.split('=')[1].strip('%,')) / 100
                    elif part.startswith('HOTA='): hota = float(part.split('=')[1].strip('%,')) / 100
                    elif part.startswith('ID_Switches='): ids = int(part.split('=')[1].strip(','))
                except: pass
    if idf1 is None:
        # Try to find in last few lines of stderr
        for line in output.split('\n')[-30:]:
            m = re.search(r'IDF1[=:]\s*([\d.]+)%?', line)
            if m:
                idf1 = float(m.group(1)) / 100
    return {"name": name, "idf1": idf1, "mota": mota, "hota": hota, "ids": ids}

experiments = [
    # CSLS tests
    ("csls_k10",  ["--override", "stage4.association.csls.enabled=true",  "--override", "stage4.association.csls.k=10"]),
    ("csls_k5",   ["--override", "stage4.association.csls.enabled=true",  "--override", "stage4.association.csls.k=5"]),
    ("csls_k20",  ["--override", "stage4.association.csls.enabled=true",  "--override", "stage4.association.csls.k=20"]),
    # Cluster verify tests
    ("clv030",    ["--override", "stage4.association.cluster_verify.enabled=true", "--override", "stage4.association.cluster_verify.min_connectivity=0.30"]),
    ("clv025",    ["--override", "stage4.association.cluster_verify.enabled=true", "--override", "stage4.association.cluster_verify.min_connectivity=0.25"]),
    ("clv020",    ["--override", "stage4.association.cluster_verify.enabled=true", "--override", "stage4.association.cluster_verify.min_connectivity=0.20"]),
    # CSLS + cluster verify combined
    ("csls10_clv030", ["--override", "stage4.association.csls.enabled=true", "--override", "stage4.association.csls.k=10",
                       "--override", "stage4.association.cluster_verify.enabled=true", "--override", "stage4.association.cluster_verify.min_connectivity=0.30"]),
]

BASELINE = 0.830
results = []
for name, extra in experiments:
    print(f"\nRunning: {name} ... ", end="", flush=True)
    r = run_scan(name, extra)
    results.append(r)
    if r['idf1']:
        delta = r['idf1'] - BASELINE
        print(f"IDF1={r['idf1']:.4f} ({delta:+.4f})")
    else:
        print("FAILED")

print("\n\n=== RESULTS (Baseline IDF1=0.8300) ===")
print(f"{'Name':<22} {'IDF1':>8} {'Delta':>8} {'MOTA':>8} {'HOTA':>8} {'IDS':>6}")
print("-" * 62)
for r in sorted(results, key=lambda x: x['idf1'] or 0, reverse=True):
    if r['idf1']:
        delta = r['idf1'] - BASELINE
        ids_str = str(r['ids']) if r['ids'] is not None else "?"
        mota_str = f"{r['mota']:.4f}" if r['mota'] else "?"
        hota_str = f"{r['hota']:.4f}" if r['hota'] else "?"
        print(f"{r['name']:<22} {r['idf1']:>8.4f} {delta:>+8.4f} {mota_str:>8} {hota_str:>8} {ids_str:>6}")
