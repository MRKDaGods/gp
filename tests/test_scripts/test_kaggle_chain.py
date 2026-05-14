from __future__ import annotations

import importlib.util
from pathlib import Path


def load_kaggle_chain_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "kaggle_chain.py"
    spec = importlib.util.spec_from_file_location("kaggle_chain", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_kernel_ref_uses_requested_owner():
    kaggle_chain = load_kaggle_chain_module()

    assert (
        kaggle_chain.build_kernel_ref("gumfreddy", notebook_id="10a")
        == "gumfreddy/mtmc-10a-stages-0-2-tracking-reid-features"
    )
    assert (
        kaggle_chain.build_kernel_ref("gumfreddy", slug=kaggle_chain.SLUGS["10c"])
        == "gumfreddy/mtmc-10c-stages-4-5-association-eval"
    )


def test_apply_owner_to_metadata_rewrites_chain_kernel_refs_only():
    kaggle_chain = load_kaggle_chain_module()
    original_metadata = {
        "id": "mrkdagods/mtmc-10b-stage-3-faiss-indexing",
        "kernel_sources": [
            "mrkdagods/mtmc-10a-stages-0-2-tracking-reid-features",
            "otheruser/custom-upstream-kernel",
        ],
    }

    updated_metadata, changed = kaggle_chain.apply_owner_to_metadata(
        original_metadata,
        notebook_id="10b",
        owner="gumfreddy",
    )

    assert changed is True
    assert updated_metadata["id"] == "gumfreddy/mtmc-10b-stage-3-faiss-indexing"
    assert updated_metadata["kernel_sources"] == [
        "gumfreddy/mtmc-10a-stages-0-2-tracking-reid-features",
        "otheruser/custom-upstream-kernel",
    ]
    assert original_metadata["id"] == "mrkdagods/mtmc-10b-stage-3-faiss-indexing"