"""Reporting: chain-of-custody HTML rendering + model SHA attestation +
(browser-gated) PDF printing."""

from __future__ import annotations

import pytest

from athar.reporting import (
    ReportError,
    html_to_pdf,
    load_weight_shas,
    models_from_config,
    render_report_html,
)

REPORT = {
    "schema_version": 1,
    "run": {
        "run_id": "run-r1",
        "role": "gallery",
        "profile": "multiclass",
        "config_hash": "cfghash" * 8,
        "created_at": "2026-07-22T10:00:00+00:00",
    },
    "evidence": [
        {
            "camera_id": "cam-01",
            "original_path": "evidence/c1.mp4",
            "sha256": "ev" * 32,
            "duration_s": 60.0,
            "fps": 25.0,
        }
    ],
    "identities": [
        {
            "global_id": 3,
            "entity_class": "person",
            "cross_camera": True,
            "members": [
                {
                    "camera_id": "cam-01",
                    "track_id": 7,
                    "start_ts_scene_s": 1.0,
                    "end_ts_scene_s": 9.5,
                    "thumbnail": "thumbs/cam-01/7.jpg",
                    "clip": None,
                }
            ],
        }
    ],
}


class TestHtml:
    def test_chain_of_custody_content(self, tmp_path):
        thumb = tmp_path / "thumbs" / "cam-01" / "7.jpg"
        thumb.parent.mkdir(parents=True)
        thumb.write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
        html = render_report_html(
            REPORT,
            models=[{"path": "models/reid/x.pth", "sha256": "ab" * 32}],
            run_dir=tmp_path,
            locale="ar",
        )
        assert 'dir="rtl"' in html and 'lang="ar"' in html
        assert "ev" * 32 in html            # video sha
        assert "cfghash" * 8 in html        # config hash
        assert "ab" * 32 in html            # model sha
        assert "data:image/jpeg;base64," in html  # embedded thumbnail
        assert "cam-01" in html and "9.5" in html

    def test_english_ltr_and_missing_pieces(self):
        html = render_report_html(REPORT, models=[], run_dir=None, locale="en")
        assert 'dir="ltr"' in html
        assert "no model checkpoints recorded" in html
        assert "data:image" not in html  # no run_dir -> no embedding

    def test_unpinned_model_is_flagged_not_invented(self):
        html = render_report_html(
            REPORT,
            models=[{"path": "models/reid/unknown.pth", "sha256": None}],
            locale="en",
        )
        assert "not pinned in the weights manifest" in html

    def test_escapes_html(self):
        report = dict(REPORT, run=dict(REPORT["run"], profile="<script>x"))
        html = render_report_html(report, locale="en")
        assert "<script>" not in html


class TestModelRefs:
    def test_from_config_values(self):
        values = {
            "embed.streams.appearance.weights": "models/reid/a.pth",
            "detect.model_path": "models/detection/yolo26m.pt",
            "detect.conf": 0.3,
            "notes": "not-a-checkpoint.txt",
        }
        models = models_from_config(values, {"a.pth": "aa" * 32})
        assert models == [
            {"path": "models/detection/yolo26m.pt", "sha256": None},
            {"path": "models/reid/a.pth", "sha256": "aa" * 32},
        ]

    def test_load_weight_shas(self, tmp_path):
        manifest = tmp_path / "weights.yaml"
        manifest.write_text(
            "sets:\n"
            "  s1:\n"
            "    files:\n"
            "      - name: a.pth\n"
            "        local_path: models/a.pth\n"
            "        sha256: " + "aa" * 32 + "\n",
            encoding="utf-8",
        )
        assert load_weight_shas(manifest) == {"a.pth": "aa" * 32}
        assert load_weight_shas(tmp_path / "missing.yaml") == {}


class TestPdf:
    def test_prints_arabic_html(self):
        pytest.importorskip("playwright")
        try:
            pdf = html_to_pdf(render_report_html(REPORT, locale="ar"))
        except ReportError as exc:
            pytest.skip(f"chromium unavailable: {exc}")
        assert pdf.startswith(b"%PDF")
        assert len(pdf) > 1000
