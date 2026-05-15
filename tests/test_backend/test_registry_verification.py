from __future__ import annotations

from pathlib import Path

from scripts.verify_registry_numbers import run_verification


def test_registry_numbers_verify(tmp_path: Path) -> None:
    report_path = tmp_path / "registry_verification.json"

    exit_code, results = run_verification(report_path=report_path)

    assert exit_code == 0
    assert report_path.exists()
    assert results
    assert not [item for item in results if item["status"] == "FAIL"]