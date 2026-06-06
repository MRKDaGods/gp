"""Bundled-fusion model resolution tests.

Covers the fix for registered models that BUNDLE a multi-stream Stage-4 ensemble
inside their `model_overrides` (one `model_id`, not a user FusionConfig):

* `vehicle_mtmc_14e_b1` (production) - DINOv2 tertiary stream. Its Stage-4
  tertiary path must be wired DYNAMICALLY to the run-scoped stage2 output, not the
  stale `data/outputs/run_latest/...` path baked into cityflowv2.yaml (which never
  exists -> the tertiary silently dropped -> ensemble degraded to primary-only).
* `vehicle_mtmc_14k_v1_k7` (research) - DINOv2 tertiary + FastReID R50-IBN
  quaternary. The quaternary stream must enable the Stage-2 vehicle2 slot and get
  a dynamic Stage-4 path; there is no vehicle4 slot, so quaternary reuses vehicle2
  -> embeddings_secondary.npy.
* A weighted stream with no wireable extractor must FAIL LOUD, never degrade.
"""

from __future__ import annotations

import pytest

from backend.models.registry import (
    CheckpointRef,
    ModelEntry,
    Provenance,
    Requirements,
)
from backend.services import pipeline_service
from backend.services.pipeline_service import (
    PipelineModelValidationError,
    resolve_pipeline_model,
)

DYN_TERTIARY = (
    "stage4.association.tertiary_embeddings.path="
    "${project.output_dir}/${project.run_name}/stage2/embeddings_tertiary.npy"
)
DYN_QUATERNARY = (
    "stage4.association.quaternary_embeddings.path="
    "${project.output_dir}/${project.run_name}/stage2/embeddings_secondary.npy"
)
STALE_RUN_LATEST = "run_latest"


def _override_kv(overrides, dotted_key):
    """Return the value for an exact dotted override key, or None."""
    for override in overrides:
        key, _, value = override.partition("=")
        if key == dotted_key:
            return value
    return None


# 14e B1 (production) - dynamic tertiary path, no stale run_latest

def test_14e_b1_wires_dynamic_tertiary_path() -> None:
    result = resolve_pipeline_model("vehicle_mtmc_14e_b1", dataset="cityflowv2")
    overrides = result.applied_overrides

    # The DINOv2 tertiary stream is wired to the run-scoped stage2 output.
    assert DYN_TERTIARY in overrides, (
        f"dynamic tertiary path missing. Got: {overrides}"
    )
    assert (
        _override_kv(overrides, "stage4.association.tertiary_embeddings.enabled")
        == "true"
    )

    # The stale, never-existing run_latest path must NOT be emitted anywhere.
    assert not any(STALE_RUN_LATEST in o for o in overrides), (
        f"stale run_latest path leaked into overrides: {overrides}"
    )

    # The original bare weight override is preserved (loader needs path + weight>0).
    assert (
        _override_kv(overrides, "stage4.association.tertiary_embeddings.weight")
        == "0.525"
    )


def test_14e_b1_fusion_resolved_populated() -> None:
    result = resolve_pipeline_model("vehicle_mtmc_14e_b1", dataset="cityflowv2")
    fr = result.fusion_resolved
    assert fr is not None, "14e B1 must resolve as a (bundled) fusion, not single"
    assert fr["mode"] == "bundled"
    assert fr["primary_model_id"] == "vehicle_mtmc_14e_b1"
    slots = {s["slot"] for s in fr["streams"]}
    assert slots == {"tertiary"}
    tert = next(s for s in fr["streams"] if s["slot"] == "tertiary")
    # DINOv2 is enabled by cityflowv2.yaml (vehicle3), so it is wired via config.
    assert tert["wired_via"] == "pipeline_config"
    assert tert["stage2_slot"] == "vehicle3"


def test_14e_b1_does_not_emit_stage2_overrides_for_config_wired_stream() -> None:
    # The tertiary slot is already enabled in cityflowv2.yaml, so no Stage-2
    # vehicle3 enable overrides should be emitted by the wiring.
    result = resolve_pipeline_model("vehicle_mtmc_14e_b1", dataset="cityflowv2")
    assert not any(
        o.startswith("stage2.reid.vehicle3.") for o in result.applied_overrides
    ), f"unexpected stage2 vehicle3 overrides: {result.applied_overrides}"


# K7 (research) - DINOv2 tertiary + R50-IBN quaternary

def test_k7_wires_quaternary_r50ibn_stream() -> None:
    result = resolve_pipeline_model("vehicle_mtmc_14k_v1_k7", dataset="cityflowv2")
    overrides = result.applied_overrides

    # Quaternary R50-IBN -> Stage-2 vehicle2 slot enabled with the builder-correct
    # model_name (arch resnet50_ibn maps to fastreid_sbs_r50_ibn).
    assert _override_kv(overrides, "stage2.reid.vehicle2.enabled") == "true"
    assert (
        _override_kv(overrides, "stage2.reid.vehicle2.model_name")
        == "fastreid_sbs_r50_ibn"
    )
    assert _override_kv(overrides, "stage2.reid.vehicle2.save_separate") == "true"
    assert _override_kv(overrides, "stage2.reid.vehicle2.embedding_dim") == "2048"
    assert (
        _override_kv(overrides, "stage2.reid.vehicle2.weights_path")
        == "models/reid/fastreid_r50_ibn_cityflowv2_final.pth"
    )

    # Quaternary reuses the vehicle2 -> embeddings_secondary.npy producer.
    assert DYN_QUATERNARY in overrides, (
        f"dynamic quaternary path missing. Got: {overrides}"
    )
    assert (
        _override_kv(overrides, "stage4.association.quaternary_embeddings.enabled")
        == "true"
    )

    # Tertiary (DINOv2) still wired dynamically via the pipeline_config slot.
    assert DYN_TERTIARY in overrides
    assert not any(STALE_RUN_LATEST in o for o in overrides)


def test_k7_fusion_resolved_has_both_streams() -> None:
    result = resolve_pipeline_model("vehicle_mtmc_14k_v1_k7", dataset="cityflowv2")
    fr = result.fusion_resolved
    assert fr is not None
    by_slot = {s["slot"]: s for s in fr["streams"]}
    assert set(by_slot) == {"tertiary", "quaternary"}
    assert by_slot["tertiary"]["wired_via"] == "pipeline_config"
    assert by_slot["quaternary"]["wired_via"] == "checkpoint_architecture"
    assert by_slot["quaternary"]["stage2_slot"] == "vehicle2"
    assert by_slot["quaternary"]["model_name"] == "fastreid_sbs_r50_ibn"


# Fail-loud guard - a weighted stream that cannot be wired must raise

def _synthetic_unwireable_model() -> ModelEntry:
    """A cityflowv2 model that sets quaternary weight but provides NO quaternary
    checkpoint architecture (and the pipeline_config has no vehicle-slot producer
    for it) -> must fail loud rather than silently dropping the stream."""
    return ModelEntry(
        id="synthetic_unwireable_quaternary",
        name="Synthetic unwireable quaternary",
        task_type="mtmc_vehicle",
        dataset="cityflowv2",
        description="Test-only model with an unwireable quaternary stream.",
        metrics=[],
        pipeline_config="configs/datasets/cityflowv2.yaml",
        model_overrides=[
            "stage4.association.quaternary_embeddings.weight=0.40",
        ],
        checkpoint_refs=[
            CheckpointRef(
                role="quaternary_reid",
                local_path="models/reid/does_not_matter.pth",
                # NO architecture block -> not wireable from the checkpoint.
            ),
        ],
        requirements=Requirements(gpu_required=False, min_vram_gb=0),
        status="research",
        runnable_locally=True,
        notebook_or_kernel_ref=None,
        provenance=Provenance(created_at="2026-06-05", verified_by="test"),
    )


def test_unwireable_stream_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    synthetic = _synthetic_unwireable_model()

    def _fake_get_model(model_id: str):
        if model_id == synthetic.id:
            return synthetic
        return None

    # _lookup_registry_model -> get_model (imported into pipeline_service).
    monkeypatch.setattr(pipeline_service, "get_model", _fake_get_model)

    with pytest.raises(PipelineModelValidationError) as exc:
        resolve_pipeline_model(synthetic.id, dataset="cityflowv2")
    msg = str(exc.value).lower()
    assert "quaternary" in msg
    assert "not wired" in msg


def test_unwireable_stream_does_not_silently_degrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must fire BEFORE returning - never return a primary-only
    resolution that drops the weighted stream."""
    synthetic = _synthetic_unwireable_model()
    monkeypatch.setattr(
        pipeline_service,
        "get_model",
        lambda mid: synthetic if mid == synthetic.id else None,
    )
    with pytest.raises(PipelineModelValidationError):
        resolve_pipeline_model(synthetic.id, dataset="cityflowv2")


# Non-fusion models are untouched

def test_person_model_has_no_bundled_fusion() -> None:
    # WILDTRACK person model declares no *_embeddings.weight streams.
    result = resolve_pipeline_model("person_mtmc_12b", dataset="wildtrack")
    assert result.fusion_resolved is None
    assert not any(
        "_embeddings.path=" in o for o in result.applied_overrides
    ), f"unexpected ensemble path override: {result.applied_overrides}"
