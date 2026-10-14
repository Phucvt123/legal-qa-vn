"""Retriever: encode câu hỏi bằng bge-m3 rồi tìm trong vector store."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from legal_qa.config import settings
from legal_qa.vectorstore import Hit, build_store

log = logging.getLogger(__name__)


def _read_manifest(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    if not raw.strip():
        raise RuntimeError(f"{path} rỗng hoặc không tồn tại. Chạy: python scripts/build_index.py --strict")
    if raw.startswith("version https://git-lfs"):
        raise RuntimeError(f"{path} là con trỏ Git LFS, chưa phải file thật.")
    return json.loads(raw)


class Retriever:
    def __init__(self, artifacts_dir: str | Path = "artifacts", model: SentenceTransformer | None = None):
        d = Path(artifacts_dir)
        manifest = _read_manifest(d / "manifest.json")
        if manifest.get("embed_model") != settings.embed_model:
            raise RuntimeError(f"Index build bằng {manifest.get('embed_model')}, "
                               f"nhưng EMBED_MODEL={settings.embed_model}. Build lại index.")

        self.chunks: list[dict] = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        vectors = np.load(d / "embeddings.npy").astype(np.float32)
        if len(self.chunks) != len(vectors):
            raise RuntimeError(f"Lệch dữ liệu: {len(self.chunks)} chunks nhưng {len(vectors)} vectors")

        self.store = build_store(self.chunks, vectors, settings)
        self.model = model or SentenceTransformer(settings.embed_model, device="cpu")
        self.model.max_seq_length = 512
        log.info("Đã load %s trên cpu, %d chunks, store=%s",
                 settings.embed_model, len(self.chunks), type(self.store).__name__)

    def encode(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)[0]

    def search(self, query: str, k: int | None = None, only_active: bool = True,
               so_hieu: str | None = None) -> list[dict]:
        """Trả về list chunk dạng dict (kèm key 'score'), giống kiểu của retriever cũ."""
        hits = self.store.search(self.encode(query), k or getattr(settings, "top_k", 5),
                                 only_active, so_hieu)
        return [{**chunk, "score": score} for chunk, score in hits]
