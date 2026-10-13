"""Dense retrieval bằng numpy (POC E3: dense 0.991 Recall@5, tốt hơn hybrid/rerank)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import ACTIVE_STATUS, settings

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, artifacts_dir: Path | None = None, model=None):
        d = artifacts_dir or settings.artifacts_dir
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if manifest["embed_model"] != settings.embed_model:
            raise RuntimeError(
                f"Index build bằng {manifest['embed_model']} nhưng config là {settings.embed_model}. Hãy build lại index.")
        self.chunks: list[dict] = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        self.matrix = np.load(d / "embeddings.npy").astype(np.float32)
        if len(self.chunks) != self.matrix.shape[0]:
            raise RuntimeError("Số chunk và số embedding không khớp. Hãy build lại index.")
        self.active = np.array([c["status"] == ACTIVE_STATUS for c in self.chunks])

        if model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(settings.embed_model, device=device)
            model.max_seq_length = 512
            logger.info("Đã load %s trên %s, %d chunks", settings.embed_model, device, len(self.chunks))
        self.model = model

    def search(self, query: str, k: int | None = None, only_active: bool = True) -> list[dict]:
        k = k or settings.top_k
        q = np.asarray(self.model.encode(query, normalize_embeddings=True), dtype=np.float32)
        scores = self.matrix @ q
        if only_active:
            scores = np.where(self.active, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top if np.isfinite(scores[i])]
