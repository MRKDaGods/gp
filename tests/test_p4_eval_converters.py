"""Unit tests for the Gate P4 eval converters (run artifacts -> MOT rows)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "eval_p4", REPO_ROOT / "scripts" / "eval" / "eval_p4_wildtrack_person.py"
)
eval_p4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_p4)


def _obs(frame: int, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9):
    return {
        "frame_index": frame,
        "ts_scene_s": frame / 2.0,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "confidence": conf,
        "interpolated": False,
    }


def _tracklet(cam: str, track_id: int, entity_class: str):
    return {
        "key": {"run_id": "r1", "camera_id": cam, "track_id": track_id},
        "entity_class": entity_class,
        "start_ts_scene_s": 0.0,
        "end_ts_scene_s": 1.0,
        "observation_count": 2,
        "mean_confidence": 0.9,
    }


def _fake_run(tmp_path: Path) -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    payload = {
        "schema_version": 1,
        "camera_id": "C1",
        "tracklets": [
            _tracklet("C1", 10000001, "person"),
            _tracklet("C1", 1, "car"),  # must be filtered out
        ],
        "observations": {
            "10000001": [_obs(0, 10, 20, 50, 120), _obs(1, 12, 22, 52, 122)],
            "1": [_obs(0, 0, 0, 40, 40)],
        },
    }
    (run_dir / "tracklets_C1.json").write_text(json.dumps(payload), encoding="utf-8")
    traj = {
        "trajectories": [
            {
                "global_id": 0,
                "entity_class": "person",
                "members": [
                    {"run_id": "r1", "camera_id": "C1", "track_id": 10000001},
                    {"run_id": "r1", "camera_id": "C2", "track_id": 10000005},
                ],
            }
        ]
    }
    (run_dir / "trajectories.json").write_text(json.dumps(traj), encoding="utf-8")
    manifest = {
        "run_id": "r1",
        "inputs": [{"camera_id": "C1"}],
        "artifacts": {
            "tracklets.C1": {"relpath": "tracklets_C1.json"},
            "associate.trajectories": {"relpath": "trajectories.json"},
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, manifest


class TestCollect:
    def test_person_rows_one_based_xywh(self, tmp_path):
        run_dir, manifest = _fake_run(tmp_path)
        rows = eval_p4.collect_person_predictions(run_dir, manifest)
        assert set(rows) == {"C1"}
        assert len(rows["C1"]) == 2  # the car track is excluded
        frame, tid, x, y, w, h, conf, cls = rows["C1"][0]
        assert (frame, tid) == (1, 10000001)  # 1-based frame
        assert (x, y, w, h) == (10, 20, 40, 100)
        assert cls == 0


class TestGlobalRemap:
    def test_trajectory_members_share_id_and_singletons_are_fresh(self, tmp_path):
        run_dir, manifest = _fake_run(tmp_path)
        mapping = eval_p4.global_id_map(run_dir, manifest)
        assert mapping[("C1", 10000001)] == 0
        assert mapping[("C2", 10000005)] == 0

        rows = {
            "C1": [(1, 10000001, 0, 0, 5, 5, 1.0, 0), (1, 777, 0, 0, 5, 5, 1.0, 0)],
            "C2": [(1, 10000005, 0, 0, 5, 5, 1.0, 0)],
        }
        remapped = eval_p4.remap_global(rows, mapping)
        assert remapped["C1"][0][1] == 0
        assert remapped["C2"][0][1] == 0
        assert remapped["C1"][1][1] == 1  # singleton got a fresh id past max


class TestWrite:
    def test_mot_files_sorted_by_frame(self, tmp_path):
        rows = {"C1": [(2, 5, 1, 2, 3, 4, 0.5, 0), (1, 5, 1, 2, 3, 4, 0.5, 0)]}
        eval_p4.write_mot(rows, tmp_path / "out")
        lines = (tmp_path / "out" / "C1.txt").read_text().strip().splitlines()
        assert lines[0].startswith("1,5,") and lines[1].startswith("2,5,")
