"""Tests for FAISS index."""

import numpy as np
import pytest

from src.stage3_indexing.faiss_index import FAISSIndex


def test_faiss_build_and_search():
    index = FAISSIndex(index_type="flat_ip")
    embeddings = np.random.randn(100, 64).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    index.build(embeddings)
    assert index.size == 100

    # Search with first vector — should find itself
    distances, indices = index.search_single(embeddings[0], top_k=5)
    assert indices[0] == 0
    np.testing.assert_almost_equal(distances[0], 1.0, decimal=3)


def test_faiss_save_load(tmp_path):
    index = FAISSIndex()
    data = np.random.randn(50, 32).astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    data = data / norms
    index.build(data)

    path = tmp_path / "test_index.bin"
    index.save(path)

    index2 = FAISSIndex()
    index2.load(path)
    assert index2.size == 50

    d1, i1 = index.search_single(data[0], top_k=3)
    d2, i2 = index2.search_single(data[0], top_k=3)
    np.testing.assert_array_equal(i1, i2)


def test_faiss_not_built():
    index = FAISSIndex()
    with pytest.raises(RuntimeError):
        index.search(np.random.randn(1, 32).astype(np.float32), top_k=5)
