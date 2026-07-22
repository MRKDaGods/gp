"""DeviceManager tests: budgets, reservations, env override parsing."""

from __future__ import annotations

import pytest

from athar.serving.devices import DeviceManager, _env_vram_budget_bytes


class TestBudgets:
    def test_cpu_unlimited_by_default(self):
        dm = DeviceManager()
        assert dm.budget("cpu") is None
        assert dm.can_fit("cpu", 10**15)

    def test_explicit_budget_enforced(self):
        dm = DeviceManager(budgets={"cpu": 100})
        assert dm.can_fit("cpu", 100)
        dm.reserve("cpu", 60)
        assert dm.can_fit("cpu", 40)
        assert not dm.can_fit("cpu", 41)

    def test_release_returns_capacity(self):
        dm = DeviceManager(budgets={"cpu": 100})
        dm.reserve("cpu", 80)
        dm.release("cpu", 80)
        assert dm.reserved("cpu") == 0
        assert dm.can_fit("cpu", 100)

    def test_over_release_clamps(self):
        dm = DeviceManager(budgets={"cpu": 100})
        dm.reserve("cpu", 10)
        dm.release("cpu", 50)  # logged + clamped, never negative
        assert dm.reserved("cpu") == 0

    def test_negative_reserve_rejected(self):
        dm = DeviceManager()
        with pytest.raises(ValueError):
            dm.reserve("cpu", -1)

    def test_headroom_validated(self):
        with pytest.raises(ValueError, match="headroom"):
            DeviceManager(headroom=0.0)

    def test_snapshot_shape(self):
        dm = DeviceManager(budgets={"cpu": 100})
        dm.reserve("cpu", 30)
        snap = dm.snapshot()
        assert snap == {"cpu": {"budget_bytes": 100, "reserved_bytes": 30}}


class TestEnvOverride:
    def test_env_budget_parsed_mb(self, monkeypatch):
        monkeypatch.setenv("ATHAR_VRAM_BUDGET_MB", "512")
        assert _env_vram_budget_bytes() == 512 * 1024 * 1024

    def test_env_budget_garbage_ignored(self, monkeypatch):
        monkeypatch.setenv("ATHAR_VRAM_BUDGET_MB", "lots")
        assert _env_vram_budget_bytes() is None

    def test_env_budget_absent(self, monkeypatch):
        monkeypatch.delenv("ATHAR_VRAM_BUDGET_MB", raising=False)
        assert _env_vram_budget_bytes() is None

    def test_env_applies_to_cuda_device(self, monkeypatch):
        monkeypatch.setenv("ATHAR_VRAM_BUDGET_MB", "2")
        dm = DeviceManager()
        assert dm.budget("cuda:0") == 2 * 1024 * 1024
