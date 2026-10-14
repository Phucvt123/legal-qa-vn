import numpy as np
import pytest

from legal_qa.vectorstore import NumpyStore, QdrantStore

ACTIVE = "con_hieu_luc"


def _data(n=60, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    chunks = [{
        "id": f"doc#{i}", "text": f"chunk {i}",
        "so_hieu": "45/2019/QH14" if i % 3 else "10/2012/QH13",
        "status": ACTIVE if i % 3 else "het_hieu_luc",
    } for i in range(n)]
    return chunks, v, rng


def _query(rng, dim):
    q = rng.normal(size=dim).astype(np.float32)
    return q / np.linalg.norm(q)


@pytest.mark.parametrize("only_active", [True, False])
def test_qdrant_matches_numpy(only_active):
    chunks, v, rng = _data()
    a_store = NumpyStore(chunks, v, ACTIVE)
    b_store = QdrantStore(chunks, v, ACTIVE, collection="t_match")
    for _ in range(20):
        q = _query(rng, v.shape[1])
        a, b = a_store.search(q, 5, only_active), b_store.search(q, 5, only_active)
        assert [c["id"] for c, _ in a] == [c["id"] for c, _ in b]
        assert np.allclose([s for _, s in a], [s for _, s in b], atol=1e-4)


@pytest.mark.parametrize("store_cls", [NumpyStore, QdrantStore])
def test_filters(store_cls):
    chunks, v, rng = _data()
    store = store_cls(chunks, v, ACTIVE)
    q = _query(rng, v.shape[1])
    assert all(c["status"] == ACTIVE for c, _ in store.search(q, 10, only_active=True))
    old = store.search(q, 10, only_active=False, so_hieu="10/2012/QH13")
    assert old and all(c["so_hieu"] == "10/2012/QH13" for c, _ in old)


def test_payload_has_no_internal_fields():
    chunks, v, rng = _data()
    hit, _ = QdrantStore(chunks, v, ACTIVE, collection="t_payload").search(_query(rng, v.shape[1]), 1)[0]
    assert "index_version" not in hit and hit["text"].startswith("chunk")
