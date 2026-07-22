"""Built-in run profiles + profile loading.

``multiclass`` is the default operational profile: one YOLO26 pass over
COCO person+vehicle classes, per-class branches (person: TransReID
Market1501 checkpoint + HSV; vehicles: TransReID VeRi checkpoint + HSV).
Custom profiles load from YAML files with the same RunProfile schema.

Each builtin ships its PROFILE_DEFAULT config layer — the bottom layer of
the ResolvedConfig stack (deployment/case/run overrides stack above it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from athar.core.types import EntityClass
from athar.profiles.base import ClassBranch, ComponentSpec, RunProfile


def multiclass_profile() -> RunProfile:
    return RunProfile(
        name="multiclass",
        frame_source=ComponentSpec(name="video"),
        detector=ComponentSpec(
            name="yolo_v1",
            config={"model_path": "models/detection/yolo26m.pt", "img_size": 640},
        ),
        branches=[
            ClassBranch(
                entity_classes=[EntityClass.PERSON],
                tracker=ComponentSpec(name="boxmot_v1", config={"algorithm": "botsort"}),
                embedders=[
                    ComponentSpec(
                        name="transreid_v1",
                        config={
                            # v1 wildtrack.yaml stage2.reid.person recipe
                            "weights_path": "models/reid/person_transreid_vit_base_market1501.pth",
                            "stream_name": "transreid_person",
                            "input_size": [256, 128],  # list, not tuple — YAML-stable
                            "num_cameras": 6,
                            "vit_model": "vit_base_patch16_224",
                            "clip_normalization": False,
                        },
                    ),
                    ComponentSpec(name="hsv_v1"),
                ],
                score_terms=[ComponentSpec(name="appearance")],
                solver=ComponentSpec(name="graph_cc"),
            ),
            ClassBranch(
                entity_classes=[EntityClass.CAR, EntityClass.BUS, EntityClass.TRUCK],
                tracker=ComponentSpec(name="boxmot_v1", config={"algorithm": "botsort"}),
                embedders=[
                    ComponentSpec(
                        name="transreid_v1",
                        config={
                            "weights_path": "models/reid/vehicle_transreid_vit_base_veri776.pth",
                            "input_size": [224, 224],  # list, not tuple — YAML-stable
                            "num_cameras": 20,
                        },
                    ),
                    ComponentSpec(name="hsv_v1"),
                ],
                score_terms=[ComponentSpec(name="appearance")],
                solver=ComponentSpec(name="graph_cc"),
            ),
        ],
    )


MULTICLASS_DEFAULTS: dict[str, Any] = {
    "detect_track": {"batch_size": 16, "device": "cpu"},
    "embed": {
        "samples_per_tracklet": 16,
        "tracklet_chunk": 4,
        "min_area": 500,
        "device": "cpu",
    },
    "associate": {
        "top_k": 20,
        "mutual_top_k": 10,
        "similarity_threshold": 0.5,
        "min_time_gap": 0.0,
        "max_time_gap": 600.0,
        "weights": {
            "person": {"appearance": 0.65, "hsv": 0.10, "spatiotemporal": 0.25},
            "vehicle": {"appearance": 0.60, "hsv": 0.15, "spatiotemporal": 0.25},
        },
    },
}

def production_profile() -> RunProfile:
    """multiclass + the CLIP-SENet and DINOv2 fusion streams on the
    vehicle branch.

    Production ≠ parity (D18): gates keep running ``multiclass`` with the
    v1 components; upgrades land here. Fusion weighting references the v1
    recipes: w_clipsenet = 0.7 (14t) and w_dinov2 = 0.525 (14e tertiary)
    against the TransReID primary.
    """
    profile = multiclass_profile()
    profile.name = "production"
    for branch in profile.branches:
        if EntityClass.CAR in branch.entity_classes:
            branch.embedders[1:1] = [
                ComponentSpec(
                    name="clipsenet_v1",
                    config={
                        "weights_path": "models/reid/clipsenet_v6_veri776_best.pth",
                    },
                ),
                ComponentSpec(
                    name="dinov2_v1",
                    config={
                        "weights_path": "models/reid/vehicle_transreid_dinov2_large_cityflowv2_final.pth",
                    },
                ),
            ]
    return profile


PRODUCTION_DEFAULTS: dict[str, Any] = {
    **MULTICLASS_DEFAULTS,
    "associate": {
        **MULTICLASS_DEFAULTS["associate"],
        # weighted per-stream fusion (renormalized over streams carrying
        # both tracklets); absent streams keep weight 1.0
        "stream_weights": {
            "transreid_primary": 1.0,
            "clipsenet": 0.7,
            "dinov2": 0.525,
        },
    },
}


BUILTIN_PROFILES: dict[str, tuple[Callable[[], RunProfile], dict[str, Any]]] = {
    "multiclass": (multiclass_profile, MULTICLASS_DEFAULTS),
    "production": (production_profile, PRODUCTION_DEFAULTS),
}


class ProfileError(ValueError):
    pass


def load_profile(name_or_path: str) -> tuple[RunProfile, dict[str, Any]]:
    """A builtin name, or a YAML file with {profile: {...}, defaults: {...}}."""
    if name_or_path in BUILTIN_PROFILES:
        factory, defaults = BUILTIN_PROFILES[name_or_path]
        return factory(), defaults
    path = Path(name_or_path)
    if not path.is_file():
        raise ProfileError(
            f"unknown profile {name_or_path!r} — not a builtin "
            f"({', '.join(sorted(BUILTIN_PROFILES))}) and not a file"
        )
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "profile" not in data:
        raise ProfileError(f"{path}: expected a mapping with a 'profile' key")
    profile = RunProfile.model_validate(data["profile"])
    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ProfileError(f"{path}: 'defaults' must be a mapping")
    return profile, defaults
