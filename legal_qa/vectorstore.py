"""Vector store có thể thay đổi: NumPy (brute-force) hoặc Qdrant (local / server / cloud).

File artifacts/ (chunks.json + embeddings.npy) là nguồn dữ liệu gốc.
Qdrant được nạp từ artifacts lúc khởi động, và bỏ qua bước nạp nếu dữ liệu không đổi.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)

Hit = tuple[dict, float]  # (chunk, score)


class VectorStore(ABC):
    @abstractmethod
    def search(self, qvec: np.ndarray, k: int, only_active: bool = True,
               so_hieu: str | None = None) -> list[Hit]:
        """Trả về top-k (chunk, cosine score), sắp xếp giảm dần."""


def fingerprint(vectors: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vectors, dtype=np.float32).tobytes()).hexdigest()[:16]


def _point_id(chunk: dict, i: int) -> str:
    """Qdrant chỉ nhận id là số nguyên hoặc UUID. Dùng lại id dạng UUID nếu có, nếu không thì sinh uuid5."""
    raw = str(chunk.get("id", i))
    try:
        return str(uuid.UUID(raw))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


# ---------------------------------------------------------------- NumPy
class NumpyStore(VectorStore):
    def __init__(self, chunks: list[dict], vectors: np.ndarray, active_status: str):
        self.chunks = chunks
        self.mat = vectors.astype(np.float32)
        self.active_status = active_status

    def search(self, qvec, k, only_active=True, so_hieu=None):
        mask = np.array([
            (not only_active or c.get("status") == self.active_status)
            and (so_hieu is None or c.get("so_hieu") == so_hieu)
            for c in self.chunks
        ])
        k = min(k, int(mask.sum()))
        if k == 0:
            return []
        scores = np.where(mask, self.mat @ qvec.astype(np.float32), -np.inf)
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.chunks[i], float(scores[i])) for i in idx]


# ---------------------------------------------------------------- Qdrant
class QdrantStore(VectorStore):
    def __init__(self, chunks: list[dict], vectors: np.ndarray, active_status: str,
                 collection: str = "legal_vn", url: str = "", api_key: str = "",
                 path: str = ":memory:"):
        from qdrant_client import QdrantClient, models

        self.m = models
        self.active_status = active_status
        self.collection = collection
        self.remote = bool(url)
        if self.remote:
            self.client = QdrantClient(url=url, api_key=api_key or None, timeout=30)
        elif path == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(path=path)
        self.version = fingerprint(vectors)
        self._ensure_index(chunks, vectors.astype(np.float32))

    def _ensure_index(self, chunks, vectors):
        m, c, name, n = self.m, self.client, self.collection, len(chunks)

        if c.collection_exists(name):
            same = c.count(name, exact=True, count_filter=m.Filter(must=[
                m.FieldCondition(key="index_version", match=m.MatchValue(value=self.version))
            ])).count
            total = c.count(name, exact=True).count
            if same == n and total == n:
                log.info("Qdrant '%s' đã khớp index (%d points), bỏ qua bước nạp", name, n)
                return
            log.info("Qdrant '%s' lệch index (%d/%d), tạo lại", name, same, total)
            c.delete_collection(name)

        c.create_collection(name, vectors_config=m.VectorParams(
            size=vectors.shape[1], distance=m.Distance.COSINE))
        if self.remote:  # local mode không hỗ trợ payload index
            for field in ("status", "so_hieu", "index_version"):
                c.create_payload_index(name, field_name=field,
                                       field_schema=m.PayloadSchemaType.KEYWORD)

        points = [
            m.PointStruct(id=_point_id(ch, i), vector=vectors[i].tolist(),
                          payload={**ch, "index_version": self.version})
            for i, ch in enumerate(chunks)
        ]
        c.upload_points(name, points=points, batch_size=128, wait=True)
        log.info("Đã nạp %d points vào Qdrant '%s' (%s)", n, name,
                 "server" if self.remote else "local")

    def search(self, qvec, k, only_active=True, so_hieu=None):
        m = self.m
        must = []
        if only_active:
            must.append(m.FieldCondition(key="status", match=m.MatchValue(value=self.active_status)))
        if so_hieu:
            must.append(m.FieldCondition(key="so_hieu", match=m.MatchValue(value=so_hieu)))
        res = self.client.query_points(
            self.collection, query=qvec.astype(np.float32).tolist(),
            query_filter=m.Filter(must=must) if must else None,
            limit=k, with_payload=True,
        )
        hits = []
        for p in res.points:
            payload = dict(p.payload)
            payload.pop("index_version", None)
            hits.append((payload, float(p.score)))
        return hits


# ---------------------------------------------------------------- factory
def build_store(chunks: list[dict], vectors: np.ndarray, settings) -> VectorStore:
    kind = settings.vector_store.lower()
    if kind == "numpy":
        return NumpyStore(chunks, vectors, settings.active_status)
    if kind == "qdrant":
        return QdrantStore(chunks, vectors, settings.active_status,
                           collection=settings.qdrant_collection, url=settings.qdrant_url,
                           api_key=settings.qdrant_api_key, path=settings.qdrant_path)
    raise ValueError(f"VECTOR_STORE không hợp lệ: {settings.vector_store!r} (dùng 'qdrant' hoặc 'numpy')")
