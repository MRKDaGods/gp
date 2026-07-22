"""ReIDRuntime tests: LRU semantics, env sizing, thread safety, leases,
and VRAM-budget enforcement.

Uses an injected fake builder — no torch models are loaded. The runtime's
only torch touchpoint (_release_cuda) is exercised through clear().
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from athar.serving.devices import DeviceBudgetError, DeviceManager
from athar.serving.runtime import ReIDRuntime, _env_max_models


@dataclass(frozen=True)
class FakeLoaded:
    model_id: str
    device: str
    loader: str = "fake"


class Builds:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def __call__(self, model_id: str, device: str) -> FakeLoaded:
        with self.lock:
            self.calls.append((model_id, device))
        return FakeLoaded(model_id=model_id, device=device)


@pytest.fixture()
def builds():
    return Builds()


class TestLru:
    def test_hit_does_not_rebuild(self, builds):
        rt = ReIDRuntime(max_models=2, builder=builds)
        a1 = rt.load("a", "cpu")
        a2 = rt.load("a", "cpu")
        assert a1 is a2
        assert builds.calls == [("a", "cpu")]

    def test_capacity_evicts_least_recent(self, builds):
        rt = ReIDRuntime(max_models=2, builder=builds)
        rt.load("a", "cpu")
        rt.load("b", "cpu")
        rt.load("a", "cpu")  # refresh a; b is now oldest
        rt.load("c", "cpu")  # evicts b
        stats = rt.stats()
        loaded = {entry["model_id"] for entry in stats["loaded"]}
        assert loaded == {"a", "c"}
        rt.load("b", "cpu")
        assert builds.calls.count(("b", "cpu")) == 2  # b was rebuilt

    def test_clear_empties(self, builds):
        rt = ReIDRuntime(max_models=2, builder=builds)
        rt.load("a", "cpu")
        rt.clear()
        assert rt.stats()["loaded"] == []

    def test_capacity_must_be_positive(self, builds):
        with pytest.raises(ValueError, match=">= 1"):
            ReIDRuntime(max_models=0, builder=builds)


class TestEnvSizing:
    def test_env_honored(self, monkeypatch, builds):
        monkeypatch.setenv("REID_MODEL_CACHE_SIZE", "3")
        rt = ReIDRuntime(builder=builds)
        assert rt.max_models == 3

    def test_env_garbage_falls_back(self, monkeypatch):
        monkeypatch.setenv("REID_MODEL_CACHE_SIZE", "many")
        assert _env_max_models() == 2


class TestThreads:
    def test_concurrent_same_key_single_survivor(self, builds):
        rt = ReIDRuntime(max_models=2, builder=builds)
        results = []
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            results.append(rt.load("a", "cpu"))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # all callers got the SAME cached object (first insert wins)
        survivors = {id(r) for r in results}
        assert len(survivors) == 1
        assert len(rt.stats()["loaded"]) == 1

    def test_concurrent_same_key_builds_once(self, builds):
        """Waiters block on the first builder instead of deserializing the
        same checkpoint N times."""
        rt = ReIDRuntime(max_models=2, builder=builds)
        barrier = threading.Barrier(8)

        def worker():
            barrier.wait()
            rt.load("a", "cpu")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert builds.calls == [("a", "cpu")]

    def test_failed_build_unwinds_cleanly(self, builds):
        attempts = []

        def flaky(model_id, device):
            attempts.append(model_id)
            if len(attempts) == 1:
                raise RuntimeError("checkpoint corrupt")
            return FakeLoaded(model_id=model_id, device=device)

        rt = ReIDRuntime(max_models=2, builder=flaky)
        with pytest.raises(RuntimeError, match="checkpoint corrupt"):
            rt.load("a", "cpu")
        # failure fully unwound: next call retries and succeeds
        assert rt.load("a", "cpu").model_id == "a"
        assert rt.devices.reserved("cpu") == 0  # failed reservation released


class TestLeases:
    def test_lease_pins_against_lru_eviction(self, builds):
        rt = ReIDRuntime(max_models=2, builder=builds)
        with rt.acquire("a", "cpu"):
            rt.load("b", "cpu")
            rt.load("c", "cpu")  # would evict "a" (LRU-oldest) without the lease
            loaded = {e["model_id"] for e in rt.stats()["loaded"]}
            assert "a" in loaded
        # after release, the next insert can evict it again
        rt.load("d", "cpu")
        loaded = {e["model_id"] for e in rt.stats()["loaded"]}
        assert "a" not in loaded
        assert len(loaded) == 2

    def test_release_is_idempotent(self, builds):
        rt = ReIDRuntime(max_models=2, builder=builds)
        lease = rt.acquire("a", "cpu")
        lease.release()
        lease.release()
        assert rt.stats()["loaded"][0]["refcount"] == 0

    def test_nested_leases_refcount(self, builds):
        rt = ReIDRuntime(max_models=2, builder=builds)
        l1 = rt.acquire("a", "cpu")
        l2 = rt.acquire("a", "cpu")
        assert rt.stats()["loaded"][0]["refcount"] == 2
        l1.release()
        assert rt.stats()["loaded"][0]["refcount"] == 1
        l2.release()
        assert rt.stats()["loaded"][0]["refcount"] == 0

    def test_clear_keeps_leased_models(self, builds):
        rt = ReIDRuntime(max_models=2, builder=builds)
        lease = rt.acquire("a", "cpu")
        rt.load("b", "cpu")
        rt.clear()
        loaded = {e["model_id"] for e in rt.stats()["loaded"]}
        assert loaded == {"a"}
        lease.release()
        rt.clear()
        assert rt.stats()["loaded"] == []


class TestBudget:
    @staticmethod
    def _runtime(builds, budget: int, size: int, max_models: int = 4) -> ReIDRuntime:
        return ReIDRuntime(
            max_models=max_models,
            builder=builds,
            devices=DeviceManager(budgets={"cpu": budget}),
            size_estimator=lambda model_id: size,
        )

    def test_budget_evicts_unleased_to_fit(self, builds):
        rt = self._runtime(builds, budget=100, size=60)
        rt.load("a", "cpu")
        rt.load("b", "cpu")  # 120 > 100: evicts "a"
        loaded = {e["model_id"] for e in rt.stats()["loaded"]}
        assert loaded == {"b"}
        assert rt.devices.reserved("cpu") == 60

    def test_budget_refuses_when_everything_leased(self, builds):
        rt = self._runtime(builds, budget=100, size=60)
        with rt.acquire("a", "cpu"):
            with pytest.raises(DeviceBudgetError, match="leased"):
                rt.acquire("b", "cpu")
        # lease released: now it fits by evicting "a"
        assert rt.load("b", "cpu").model_id == "b"

    def test_eviction_releases_reservation(self, builds):
        rt = self._runtime(builds, budget=200, size=60)
        rt.load("a", "cpu")
        rt.load("b", "cpu")
        rt.load("c", "cpu")  # 3 x 60 = 180 <= 200: all resident
        assert rt.devices.reserved("cpu") == 180
        rt.clear()
        assert rt.devices.reserved("cpu") == 0
