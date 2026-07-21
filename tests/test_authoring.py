"""Tests for YAML config authoring and the CLI resolve command."""

from __future__ import annotations

import pytest

from athar.cli.main import main as cli_main
from athar.contracts.authoring import (
    ConfigAuthoringError,
    load_layer_file,
    parse_dotted_overrides,
    resolve_from_files,
)
from athar.contracts.config import ConfigLayer


class TestParseOverrides:
    def test_yaml_scalar_typing(self):
        nested = parse_dotted_overrides(
            ["detector.conf=0.4", "detector.size=x", "embed.enabled=true", "n=3"]
        )
        assert nested == {
            "detector": {"conf": 0.4, "size": "x"},
            "embed": {"enabled": True},
            "n": 3,
        }

    def test_malformed_pair_rejected(self):
        with pytest.raises(ConfigAuthoringError, match="key.path=value"):
            parse_dotted_overrides(["no-equals-sign"])

    def test_scalar_path_conflict_rejected(self):
        with pytest.raises(ConfigAuthoringError, match="conflicts"):
            parse_dotted_overrides(["a=1", "a.b=2"])


class TestLayerFiles:
    def test_missing_file_fails_at_submit(self, tmp_path):
        with pytest.raises(ConfigAuthoringError, match="not found"):
            load_layer_file(tmp_path / "nope.yaml")

    def test_non_mapping_root_rejected(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a list\n")
        with pytest.raises(ConfigAuthoringError, match="mapping"):
            load_layer_file(bad)

    def test_empty_file_is_empty_layer(self, tmp_path):
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        assert load_layer_file(empty) == {}


class TestResolveFromFiles:
    def test_full_stack_with_provenance(self, tmp_path):
        (tmp_path / "profile.yaml").write_text(
            "detector:\n  size: m\n  conf: 0.25\nembed:\n  streams: 2\n"
        )
        (tmp_path / "site.yaml").write_text("detector:\n  conf: 0.3\n")
        cfg = resolve_from_files(
            profile_defaults=tmp_path / "profile.yaml",
            deployment=tmp_path / "site.yaml",
            overrides=["detector.size=x"],
        )
        assert cfg.values["detector.size"] == "x"
        assert cfg.values["detector.conf"] == 0.3
        assert cfg.values["embed.streams"] == 2
        assert cfg.provenance["detector.size"] is ConfigLayer.RUN_OVERRIDE
        assert cfg.provenance["detector.conf"] is ConfigLayer.DEPLOYMENT
        assert cfg.provenance["embed.streams"] is ConfigLayer.PROFILE_DEFAULT


class TestCli:
    def test_config_resolve_prints_provenance(self, tmp_path, capsys):
        profile = tmp_path / "p.yaml"
        profile.write_text("detector:\n  size: m\n")
        code = cli_main(
            ["config", "resolve", "--profile", str(profile), "--set", "detector.size=x"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "detector.size" in out
        assert "[run_override]" in out
        assert "config_hash:" in out

    def test_config_resolve_bad_file_exits_2(self, tmp_path, capsys):
        code = cli_main(["config", "resolve", "--profile", str(tmp_path / "missing.yaml")])
        assert code == 2
        assert "error:" in capsys.readouterr().err

    def test_unimplemented_commands_point_to_roadmap(self, capsys):
        assert cli_main(["models"]) == 3
        assert "ROADMAP" in capsys.readouterr().err

    def test_run_requires_videos(self, capsys):
        assert cli_main(["run"]) == 2
        assert "--video" in capsys.readouterr().err
