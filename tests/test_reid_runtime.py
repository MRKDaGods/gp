"""ReIDRuntime tests: LRU semantics, env sizing, thread safety.

Uses an injected fake builder — no torch models are loaded. The runtime's
only torch touchpoint (_release_cuda) is exercised through clear().
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

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
