"""Builtin profile + profile-YAML loading tests."""

from __future__ import annotations

import pytest
import yaml

from athar.core.types import EntityClass
from athar.profiles.builtin import BUILTIN_PROFILES, ProfileError, load_profile


class TestBuiltin:
    def test_multiclass_loads_and_covers_both_worlds(self):
        profile, defaults = load_profile("multiclass")
        assert profile.name == "multiclass"
        covered = {c for b in profile.branches for c in b.entity_classes}
        assert EntityClass.PERSON in covered and EntityClass.CAR in covered
        assert "detect_track" in defaults and "associate" in defaults

    def test_every_builtin_validates(self):
        for name in BUILTIN_PROFILES:
            profile, defaults = load_profile(name)
            assert profile.branches and isinstance(defaults, dict)


class TestProduction:
    def test_adds_clipsenet_to_vehicle_branch_only(self):
        profile, defaults = load_profile("production")
        assert profile.name == "production"
        vehicle = next(
            b for b in profile.branches if EntityClass.CAR in b.entity_classes
        )
        person = next(
            b for b in profile.branches if EntityClass.PERSON in b.entity_classes
        )
        assert [e.name for e in vehicle.embedders] == [
            "transreid_v1", "clipsenet_v1", "dinov2_v1", "hsv_v1",
        ]
        person_embedders = [e.name for e in person.embedders]
        assert "clipsenet_v1" not in person_embedders
        assert "dinov2_v1" not in person_embedders
        # 14t/14e recipe reference weighting, renormalized fusion in associate
        assert defaults["associate"]["stream_weights"] == {
            "transreid_primary": 1.0, "clipsenet": 0.7, "dinov2": 0.525,
        }

    def test_parity_profile_untouched(self):
        # D18: production upgrades must never leak into the parity profile
        profile, defaults = load_profile("multiclass")
        for branch in profile.branches:
            assert "clipsenet_v1" not in [e.name for e in branch.embedders]
        assert "stream_weights" not in defaults["associate"]


class TestYamlProfiles:
    def test_yaml_roundtrip(self, tmp_path):
        profile, defaults = load_profile("multiclass")
        doc = {"profile": profile.model_dump(mode="json"), "defaults": defaults}
        path = tmp_path / "custom.yaml"
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        loaded, loaded_defaults = load_profile(str(path))
        assert loaded == profile
        assert loaded_defaults == defaults

    def test_unknown_name_rejected(self):
        with pytest.raises(ProfileError, match="not a builtin"):
            load_profile("no-such-profile")

    def test_missing_profile_key_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("just: junk", encoding="utf-8")
        with pytest.raises(ProfileError, match="'profile' key"):
            load_profile(str(path))


class TestCliParser:
    def test_run_and_search_wired(self):
        from athar.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["run", "--video", "c01=x.mp4", "--role", "probe", "--set", "a.b=1"]
        )
        assert args.video == ["c01=x.mp4"] and args.role == "probe"
        args = parser.parse_args(["search", "--gallery", "g", "--probe", "p"])
        assert args.gallery == "g" and args.top_k == 10
