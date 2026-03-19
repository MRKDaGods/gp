"""Supplementary scan: FIC regularisation + weight configs + other untested dims.
Runs AFTER the main scan finishes (uses same BASE_CMD approach)."""
import subprocess, sys, os, re, json
from pathlib import Path
from datetime import datetime

PROJECT = Path(__file__).parent.parent
RUN_BASE = PROJECT / "data/outputs/kaggle_10a_v11/extracted"
RUN_NAME = "run_kaggle_20260318_114803"
RUN_DIR = RUN_BASE / RUN_NAME
SEC_EMB = str(RUN_DIR / "stage2" / "embeddings_secondary.npy")
GT_DIR = str(PROJECT / "data/raw/cityflowv2")
os.chdir(str(PROJECT))

# Same baseline as main scan (0.8300 confirmed)
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
    idf1 = None
    for line in output.split("\n"):
        m = re.search(r"IDF1[=:]\s*([\d.]+)%?", line)
        if m:
            v = float(m.group(1))
            idf1 = v / 100 if v > 1 else v
    return {"name": name, "idf1": idf1}


experiments = []

# ── FIC regularisation sweep ──
for fic_reg in [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
    experiments.append((f"fic_reg_{fic_reg}", [
        "--override", f"stage4.association.fic.regularisation={fic_reg}",
    ]))

# ── Weight configurations (appearance / hsv / spatiotemporal) ──
weight_configs = [
    (0.80, 0.00, 0.20, "w80_0_20"),
    (0.70, 0.00, 0.30, "w70_0_30"),
    (0.70, 0.05, 0.25, "w70_5_25"),
    (0.75, 0.05, 0.20, "w75_5_20"),
    (0.65, 0.00, 0.35, "w65_0_35"),
    (0.60, 0.00, 0.40, "w60_0_40"),
    (0.85, 0.00, 0.15, "w85_0_15"),
]
for app, hsv, st, label in weight_configs:
    experiments.append((f"weights_{label}", [
        "--override", f"stage4.association.weights.vehicle.appearance={app}",
        "--override", f"stage4.association.weights.vehicle.hsv={hsv}",
        "--override", f"stage4.association.weights.vehicle.spatiotemporal={st}",
    ]))

# ── Secondary embedding weight ──
for sec_w in [0.0, 0.10, 0.15, 0.20, 0.30, 0.35, 0.40]:
    experiments.append((f"sec_weight_{sec_w}", [
        "--override", f"stage4.association.secondary_embeddings.weight={sec_w}",
    ]))

# ── FAC variations ──
for fac_knn in [10, 15, 30]:
    experiments.append((f"fac_knn_{fac_knn}", [
        "--override", f"stage4.association.fac.knn={fac_knn}",
    ]))
for fac_lr in [0.3, 0.7, 1.0]:
    experiments.append((f"fac_lr_{fac_lr}", [
        "--override", f"stage4.association.fac.learning_rate={fac_lr}",
    ]))

# ── QE k values ──
for qe_k in [0, 1, 3, 4, 5]:
    experiments.append((f"qe_k_{qe_k}", [
        "--override", f"stage4.association.query_expansion.k={qe_k}",
    ]))

# ── Max component size ──
for mcs in [8, 10, 15, 20]:
    experiments.append((f"max_comp_{mcs}", [
        "--override", f"stage4.association.graph.max_component_size={mcs}",
    ]))

# ── bridge_prune_margin (test small positive values) ──
for bpm in [0.01, 0.02, 0.03]:
    experiments.append((f"bridge_prune_{bpm}", [
        "--override", f"stage4.association.graph.bridge_prune_margin={bpm}",
    ]))

print(f"Total experiments: {len(experiments)}")
print("=" * 70)

results = []
baseline_idf1 = 0.8300  # confirmed from main scan

for i, (name, overrides) in enumerate(experiments):
    print(f"[{i+1}/{len(experiments)}] Running {name}...", end=" ", flush=True)
    r = run_exp(name, overrides)
    results.append(r)
    idf1 = r['idf1'] or 0
    delta = idf1 - baseline_idf1
    sign = "+" if delta >= 0 else ""
    print(f"IDF1={idf1:.4f} ({sign}{delta:.4f})")

results.sort(key=lambda x: x['idf1'] or 0, reverse=True)
print("\n" + "=" * 70)
print("RESULTS (sorted by IDF1):")
print(f"{'Name':<35} {'IDF1':>8} {'Delta':>8}")
print("-" * 55)
for r in results:
    idf1 = r['idf1'] or 0
    delta = idf1 - baseline_idf1
    sign = "+" if delta >= 0 else ""
    print(f"{r['name']:<35} {idf1:.4f}   {sign}{delta:.4f}")

# Save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"data/outputs/scan_supplementary_{ts}.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to data/outputs/scan_supplementary_{ts}.json")
