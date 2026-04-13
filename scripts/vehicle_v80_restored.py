from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_config, save_config


BASE_OVERRIDES: list[str] = [
    "stage1.tracker.min_hits=2",
    "stage1.interpolation.max_gap=50",
    "stage1.intra_merge.max_time_gap=40.0",
    "stage2.reid.vehicle.weights_path=models/reid/transreid_cityflowv2_best.pth",
    "stage2.reid.vehicle.weights_fallback=models/reid/transreid_cityflowv2_best.pth",
    "stage2.reid.vehicle.input_size=[256,256]",
    "stage2.reid.color_augment=false",
    "stage2.crop.samples_per_tracklet=48",
    "stage2.pca.n_components=384",
    "stage2.camera_bn.enabled=false",
    "stage2.camera_tta.enabled=false",
    "stage2.power_norm.alpha=0.0",
    "stage2.reid.vehicle2.enabled=false",
    "stage4.association.query_expansion.k=3",
    "stage4.association.query_expansion.alpha=5.0",
    "stage4.association.query_expansion.dba=false",
    "stage4.association.graph.similarity_threshold=0.53",
    "stage4.association.graph.algorithm=conflict_free_cc",
    "stage4.association.graph.bridge_prune_margin=0.0",
    "stage4.association.graph.max_component_size=12",
    "stage4.association.fic.enabled=true",
    "stage4.association.fic.regularisation=0.1",
    "stage4.association.fac.enabled=false",
    "stage4.association.reranking.enabled=false",
    "stage4.association.camera_bias.enabled=false",
    "stage4.association.zone_model.enabled=false",
    "stage4.association.camera_pair_norm.enabled=false",
    "stage4.association.secondary_embeddings.weight=0.0",
    "stage4.association.weights.vehicle.appearance=0.70",
    "stage4.association.weights.vehicle.hsv=0.00",
    "stage4.association.gallery_expansion.threshold=0.50",
    "stage4.association.gallery_expansion.orphan_match_threshold=0.40",
    "stage5.mtmc_only_submission=false",
    "stage5.min_trajectory_confidence=0.30",
    "stage5.min_trajectory_frames=40",
    "stage5.min_submission_confidence=0.15",
    "stage5.cross_id_nms_iou=0.40",
    "stage5.stationary_filter.enabled=true",
    "stage5.stationary_filter.min_displacement_px=150",
    "stage5.stationary_filter.max_mean_velocity_px=2.0",
    "stage5.track_edge_trim.enabled=false",
    "stage5.track_smoothing.enabled=false",
    "stage5.gt_zone_filter=true",
    "stage5.gt_frame_clip=true",
]

LEGACY_SECONDARY_OVERRIDES: list[str] = [
    "stage2.reid.vehicle2.enabled=true",
    "stage4.association.secondary_embeddings.weight=0.10",
]


def build_overrides(*, include_legacy_secondary: bool, extra_overrides: Sequence[str]) -> list[str]:
    overrides = list(BASE_OVERRIDES)
    if include_legacy_secondary:
        overrides.extend(LEGACY_SECONDARY_OVERRIDES)
    overrides.extend(extra_overrides)
    return overrides


def default_python() -> str:
    venv_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit or run a high-confidence v80-restored vehicle recipe without editing Kaggle notebooks. "
            "This intentionally leaves the historically ambiguous secondary-fusion path opt-in."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-config", default="configs/datasets/cityflowv2.yaml")
    parser.add_argument(
        "--write-config",
        default="configs/experiments/vehicle_v80_restored_candidate.yaml",
        help="Write a fully merged config to this path.",
    )
    parser.add_argument("--run", action="store_true", help="Execute scripts/run_pipeline.py with the restored overrides.")
    parser.add_argument("--python", default=default_python(), help="Python executable to use when --run is set.")
    parser.add_argument("--stages", help="Optional pipeline stage list, for example 3,4,5.")
    parser.add_argument("--run-name", help="Optional project.run_name override.")
    parser.add_argument("--output-dir", help="Optional project.output_dir override.")
    parser.add_argument(
        "--legacy-secondary-fusion",
        action="store_true",
        help=(
            "Opt into the historical 10%% score-level secondary fusion. Disabled by default because the "
            "current repo secondary model differs from the original v80 setup."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional dotlist override. May be provided multiple times.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    extra_overrides = list(args.override)
    if args.run_name:
        extra_overrides.append(f"project.run_name={args.run_name}")
    if args.output_dir:
        extra_overrides.append(f"project.output_dir={args.output_dir}")

    overrides = build_overrides(
        include_legacy_secondary=args.legacy_secondary_fusion,
        extra_overrides=extra_overrides,
    )

    config_path = REPO_ROOT / args.config
    dataset_config_path = REPO_ROOT / args.dataset_config

    if args.write_config:
        merged_cfg = load_config(
            config_path=config_path,
            dataset_config=dataset_config_path,
            overrides=overrides,
        )
        output_path = REPO_ROOT / args.write_config
        save_config(merged_cfg, output_path)
        print(f"Wrote merged config: {output_path}")

    print("\nHigh-confidence v80-restored overrides:")
    for override in overrides:
        print(f"  - {override}")

    cmd = [
        args.python,
        "scripts/run_pipeline.py",
        "--config",
        args.config,
        "--dataset-config",
        args.dataset_config,
    ]
    if args.stages:
        cmd.extend(["--stages", args.stages])
    for override in overrides:
        cmd.extend(["--override", override])

    print("\nSuggested command:")
    print(" ".join(cmd))

    if not args.run:
        return 0

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())