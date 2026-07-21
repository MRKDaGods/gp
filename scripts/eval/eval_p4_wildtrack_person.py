"""Gate P4 — generic person tracking baseline on WILDTRACK-as-plain-video.

Scores a v2 pipeline run (the ``athar run`` DAG, person branch) against the
WILDTRACK image-plane MOT ground truth:

- per-camera SCT metrics (``evaluate_mot``: MOTA/IDF1/ID switches, IoU 0.5)
- cross-camera MTMC IDF1 (``evaluate_mtmc``: track ids remapped to the
  associate stage's global identities; GT personIDs are global already)

Frame convention: predictions come from a run whose cameras are WILDTRACK
``Image_subsets/<cam>`` image directories (400 annotated frames at 2 fps),
so the positional frame_index + 1 equals the 1-based GT frame id produced
by the v1 ``prepare_dataset.py`` recipe. Running on re-encoded video files
with a different frame plan would break this alignment — don't.

Usage:
    python scripts/eval/eval_p4_wildtrack_person.py \
        --run-dir <runs_root>/<run_id> \
        --gt-dir data/raw/wildtrack/manifests/ground_truth \
        --out-dir <somewhere>

Needs motmetrics (v1 env locally, pip on Kaggle). Pure CPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def artifact_relpath(manifest: dict, name: str) -> str:
    art = manifest["artifacts"][name]
    return art["relpath"] if isinstance(art, dict) else art


def collect_person_predictions(run_dir: Path, manifest: dict) -> dict[str, list[tuple]]:
    """Per-camera MOT rows (frame, track_id, x, y, w, h, conf, cls) for the
    person branch only. Frames are 1-based (positional index + 1)."""
    rows_by_cam: dict[str, list[tuple]] = defaultdict(list)
    for video in manifest["inputs"]:
        cam = video["camera_id"]
        name = f"tracklets.{cam}"
        if name not in manifest["artifacts"]:
            continue
        payload = json.loads(
            (run_dir / artifact_relpath(manifest, name)).read_text(encoding="utf-8")
        )
        person_ids = {
            t["key"]["track_id"]
            for t in payload["tracklets"]
            if t["entity_class"] == "person"
        }
        for track_id_str, observations in payload["observations"].items():
            track_id = int(track_id_str)
            if track_id not in person_ids:
                continue
            for obs in observations:
                bbox = obs["bbox"]
                rows_by_cam[cam].append(
                    (
                        int(obs["frame_index"]) + 1,
                        track_id,
                        bbox["x1"],
                        bbox["y1"],
                        bbox["x2"] - bbox["x1"],
                        bbox["y2"] - bbox["y1"],
                        float(obs.get("confidence", 1.0)),
                        0,
                    )
                )
    return dict(rows_by_cam)


def global_id_map(run_dir: Path, manifest: dict) -> dict[tuple[str, int], int]:
    """(camera_id, track_id) -> associate global_id. Tracks the associate
    stage never saw fall back to fresh singleton ids in ``remap_global``."""
    name = "associate.trajectories"
    if name not in manifest["artifacts"]:
        return {}
    payload = json.loads(
        (run_dir / artifact_relpath(manifest, name)).read_text(encoding="utf-8")
    )
    mapping: dict[tuple[str, int], int] = {}
    for traj in payload["trajectories"]:
        for member in traj["members"]:
            mapping[(member["camera_id"], int(member["track_id"]))] = int(
                traj["global_id"]
            )
    return mapping


def remap_global(
    rows_by_cam: dict[str, list[tuple]], mapping: dict[tuple[str, int], int]
) -> dict[str, list[tuple]]:
    next_singleton = (max(mapping.values()) + 1) if mapping else 0
    singleton_ids: dict[tuple[str, int], int] = {}
    out: dict[str, list[tuple]] = {}
    for cam, rows in rows_by_cam.items():
        remapped = []
        for row in rows:
            key = (cam, row[1])
            gid = mapping.get(key)
            if gid is None:
                if key not in singleton_ids:
                    singleton_ids[key] = next_singleton
                    next_singleton += 1
                gid = singleton_ids[key]
            remapped.append((row[0], gid, *row[2:]))
        out[cam] = remapped
    return out


def write_mot(rows_by_cam: dict[str, list[tuple]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cam, rows in rows_by_cam.items():
        lines = [
            f"{f},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},{cls}"
            for f, tid, x, y, w, h, conf, cls in sorted(rows)
        ]
        (out_dir / f"{cam}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    from athar.evaluation.metrics import evaluate_mot, evaluate_mtmc

    manifest = load_manifest(args.run_dir)
    rows_by_cam = collect_person_predictions(args.run_dir, manifest)
    if not rows_by_cam:
        print("error: no person tracklets in run", file=sys.stderr)
        return 1

    sct_dir = args.out_dir / "preds_sct"
    mtmc_dir = args.out_dir / "preds_mtmc"
    write_mot(rows_by_cam, sct_dir)
    write_mot(remap_global(rows_by_cam, global_id_map(args.run_dir, manifest)), mtmc_dir)

    sct = evaluate_mot(str(args.gt_dir), str(sct_dir), iou_threshold=args.iou)
    mtmc = evaluate_mtmc(str(args.gt_dir), str(mtmc_dir), iou_threshold=args.iou)

    result = {
        "gate": "P4",
        "run_id": manifest["run_id"],
        "config_hash": manifest.get("config", {}).get("config_hash"),
        "sct": {
            "mota": sct.mota,
            "idf1": sct.idf1,
            "id_switches": sct.id_switches,
            "per_camera": (sct.details or {}).get("per_camera", {}),
        },
        "mtmc": {
            "idf1": mtmc.idf1,
            "mota": mtmc.mota,
            "id_switches": mtmc.id_switches,
        },
        "num_cameras": len(rows_by_cam),
        "num_pred_rows": sum(len(r) for r in rows_by_cam.values()),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "p4_metrics.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"P4 person baseline: SCT IDF1={sct.idf1:.4f} MOTA={sct.mota:.4f} "
        f"IDSW={sct.id_switches} | MTMC IDF1={mtmc.idf1:.4f}"
    )
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
