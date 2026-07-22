"""CLIP-SENet offline-construction equivalence check (Phase 4 blocker).

The serving builder constructs CLIP-SENet with pretrained backbone
downloads and then strict-loads our checkpoint. The air-gap requirement
needs construction with ``appearance_pretrained=False`` /
``semantic_pretrained=False`` and NO network — but the TinyCLIP branch
tries an hf-hub config first, so the offline fallback may pick a different
provider/config. This script answers, empirically:

1. does construction SUCCEED with no network and cold caches;
2. which provider each branch selects in each mode;
3. after the strict checkpoint load, are the forward outputs IDENTICAL.

Usage (run once per mode, then compare)::

    python scripts/eval/check_clipsenet_offline_build.py --mode online  --out a.npz
    python scripts/eval/check_clipsenet_offline_build.py --mode offline --out b.npz
    python scripts/eval/check_clipsenet_offline_build.py --compare a.npz b.npz

``--mode offline`` must be a FRESH process: it blocks sockets and points
every model cache at an empty temp dir BEFORE importing torch, simulating
a cold air-gapped box. The npz carries outputs + a JSON provenance blob
(providers, state-dict sha256, torch version).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = PROJECT_ROOT / "models" / "reid" / "clipsenet_v6_veri776_best.pth"

NUM_IMAGES = 8
IMAGE_SIZE = 224
SEED = 0


def _block_network_and_caches() -> None:
    """Simulate a cold air-gapped machine. MUST run before torch/timm/
    open_clip/huggingface imports (they read cache env vars at import)."""
    cache_root = Path(tempfile.mkdtemp(prefix="athar-coldcache-"))
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
        os.environ[var] = str(cache_root / var.lower())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import socket

    def refuse(*_args, **_kwargs):
        raise OSError("network blocked: offline-construction check")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]


def _state_dict_sha256(model) -> str:
    hasher = hashlib.sha256()
    state = model.state_dict()
    for key in sorted(state):
        hasher.update(key.encode("utf-8"))
        hasher.update(state[key].detach().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _provenance(model) -> dict:
    appearance = model.appearance_branch.loaded_backbone
    semantic = getattr(model.semantic_branch, "loaded_backbone", None)
    return {
        "appearance": {
            "family": appearance.family,
            "model_name": appearance.model_name,
            "pretrained_tag": appearance.pretrained_tag,
        },
        "semantic": {
            "provider": getattr(model.semantic_branch, "provider", None),
            "model_name": getattr(semantic, "model_name", None),
            "pretrained_tag": getattr(semantic, "pretrained_tag", None),
            "output_dim": getattr(model.semantic_branch, "output_dim", None),
        },
    }


def run_mode(mode: str, out: Path) -> int:
    if mode == "offline":
        _block_network_and_caches()

    import numpy as np
    import torch

    sys.path.insert(0, str(PROJECT_ROOT))
    from athar.components.embedders.clip_senet_v6 import (
        build_clip_senet,
        load_checkpoint,
    )

    state_dict, _kind, num_classes = load_checkpoint(CHECKPOINT, map_location="cpu")
    pretrained = mode == "online"
    model = build_clip_senet(
        num_classes=num_classes,
        appearance_pretrained=pretrained,
        semantic_pretrained=pretrained,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"FAIL[{mode}]: non-strict checkpoint load; "
              f"missing={missing[:10]} unexpected={unexpected[:10]}")
        return 1
    model.eval()

    torch.manual_seed(SEED)
    images = torch.rand(NUM_IMAGES, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.inference_mode():
        outputs = model(images)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[-1]
    features = outputs.float().cpu().numpy()

    meta = {
        "mode": mode,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "state_dict_sha256": _state_dict_sha256(model),
        "providers": _provenance(model),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, features=features, meta=json.dumps(meta))
    print(f"[{mode}] wrote {out}")
    print(json.dumps(meta, indent=2))
    return 0


def compare(path_a: Path, path_b: Path) -> int:
    import numpy as np

    a, b = np.load(path_a), np.load(path_b)
    meta_a, meta_b = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    fa, fb = a["features"], b["features"]
    print(f"A: mode={meta_a['mode']} torch={meta_a['torch']}")
    print(f"B: mode={meta_b['mode']} torch={meta_b['torch']}")
    print(f"providers A: {json.dumps(meta_a['providers'])}")
    print(f"providers B: {json.dumps(meta_b['providers'])}")
    same_state = meta_a["state_dict_sha256"] == meta_b["state_dict_sha256"]
    print(f"state_dict sha256 equal: {same_state}")
    max_abs = float(np.max(np.abs(fa - fb))) if fa.shape == fb.shape else float("inf")
    bitwise = fa.shape == fb.shape and bool(np.array_equal(fa, fb))
    print(f"feature shape: {fa.shape} vs {fb.shape}")
    print(f"features bitwise equal: {bitwise}; max_abs_diff: {max_abs:.3e}")
    if same_state and bitwise:
        print("VERDICT: EQUIVALENT - offline construction is safe")
        return 0
    if same_state and max_abs < 1e-5:
        print("VERDICT: state-identical, tiny float drift in forward (env kernels)")
        return 0
    print("VERDICT: NOT equivalent - offline path builds a different model")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["online", "offline"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    args = parser.parse_args()
    if args.compare:
        return compare(*args.compare)
    if not args.mode or not args.out:
        parser.error("--mode and --out are required unless --compare is used")
    if not CHECKPOINT.is_file():
        print(f"checkpoint missing: {CHECKPOINT}")
        return 2
    return run_mode(args.mode, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
