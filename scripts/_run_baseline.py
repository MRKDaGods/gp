"""Run stage4+5 with baseline config (with secondary embeddings).
Reproduces the known-good 0.8297 result so we have a confirmed baseline
to measure v48 improvements against.
"""
import subprocess, sys, os
from pathlib import Path

PROJECT = Path(__file__).parent.parent
RUN_BASE = PROJECT / "data/outputs/kaggle_10a_v11/extracted"
RUN_NAME = "run_kaggle_20260318_114803"
RUN_DIR = RUN_BASE / RUN_NAME
SEC_EMB = str(RUN_DIR / "stage2" / "embeddings_secondary.npy")
GT_DIR = str(PROJECT / "data/raw/cityflowv2")

has_sec = Path(SEC_EMB).exists()
print(f"Secondary embeddings: {'FOUND' if has_sec else 'MISSING'}")

cmd = [
    sys.executable, "scripts/run_pipeline.py",
    "--config", "configs/default.yaml",
    "--stages", "4,5",
    "--override", f"project.run_name={RUN_NAME}",
    "--override", f"project.output_dir={RUN_BASE}",
    "--override", "stage0.cameras=[S01_c001,S01_c002,S01_c003,S02_c006,S02_c007,S02_c008]",
    # Stage 4 — v46 optimized config
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
    # Stage 5
    "--override", "stage5.mtmc_only_submission=false",
    "--override", "stage5.gt_frame_clip=true",
    "--override", "stage5.gt_zone_filter=true",
    "--override", "stage5.gt_zone_margin_frac=0.2",
    "--override", "stage5.gt_frame_clip_min_iou=0.5",
    "--override", "stage5.stationary_filter.enabled=true",
    "--override", "stage5.stationary_filter.min_displacement_px=150",
    "--override", f"stage5.ground_truth_dir={GT_DIR}",
]

if has_sec:
    cmd += [
        "--override", f"stage4.association.secondary_embeddings.path={SEC_EMB}",
        "--override", "stage4.association.secondary_embeddings.weight=0.25",
    ]

os.chdir(str(PROJECT))
r = subprocess.run(cmd)
sys.exit(r.returncode)
