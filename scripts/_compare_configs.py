"""Quick config comparison: with vs without dataset-config."""
import sys
sys.path.insert(0, ".")
from omegaconf import OmegaConf
from src.core.config import load_config

cfg_no_ds = load_config("configs/default.yaml", overrides=[
    "stage4.association.fac.enabled=true",
    "stage4.association.graph.similarity_threshold=0.53",
])

cfg_ds = load_config("configs/default.yaml", dataset_config="configs/datasets/cityflowv2.yaml", overrides=[
    "stage4.association.fac.enabled=true",
    "stage4.association.graph.similarity_threshold=0.53",
])

keys_to_check = [
    "stage4.association.graph.similarity_threshold",
    "stage4.association.graph.algorithm",
    "stage4.association.graph.bridge_prune_margin",
    "stage4.association.graph.max_component_size",
    "stage4.association.fac.enabled",
    "stage4.association.fic.regularisation",
    "stage4.association.weights.vehicle.appearance",
    "stage4.association.weights.vehicle.hsv",
    "stage4.association.weights.vehicle.spatiotemporal",
    "stage4.association.gallery_expansion.threshold",
    "stage4.association.gallery_expansion.orphan_match_threshold",
    "stage4.association.camera_bias.enabled",
    "stage5.mtmc_only_submission",
    "stage5.gt_zone_filter",
    "stage5.gt_frame_clip",
    "stage5.gt_zone_margin_frac",
    "stage5.stationary_filter.min_displacement_px",
    "stage5.track_smoothing.enabled",
    "stage2.pca.n_components",
    "stage2.pca.enabled",
]

print(f"{'Key':<60} {'No DS':>10} {'With DS':>10} {'Match?':>8}")
print("-" * 92)
for k in keys_to_check:
    v1 = OmegaConf.select(cfg_no_ds, k)
    v2 = OmegaConf.select(cfg_ds, k)
    match = "  OK" if str(v1) == str(v2) else "  DIFF!"
    print(f"{k:<60} {str(v1):>10} {str(v2):>10} {match:>8}")
