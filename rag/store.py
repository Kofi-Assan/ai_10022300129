# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence
"""FAISS vector store — manual build/search."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from rag.chunking import Chunk


class FaissStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Expected dim {self.dim}, got {vectors.shape[1]}")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, min(k, len(self.chunks)))
        return scores[0], indices[0]

    def save(self, dir_path: str | Path) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(d / "index.faiss"))
        payload = [
            {"text": c.text, "source": c.source, "meta": c.meta} for c in self.chunks
        ]
        (d / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, dir_path: str | Path) -> FaissStore:
        d = Path(dir_path)
        index = faiss.read_index(str(d / "index.faiss"))
        raw = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        store = cls(dim=index.d)
        store.index = index
        store.chunks = [
            Chunk(text=x["text"], source=x["source"], meta=x.get("meta") or {})
            for x in raw
        ]
        return store
