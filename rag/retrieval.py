# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence
"""
Top-k retrieval with cosine similarity (FAISS IP on normalized vectors)
plus hybrid fusion with manual BM25 keyword scores.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from rag.chunking import Chunk
from rag.embeddings import embed_query
from rag.store import FaissStore


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


@dataclass
class RetrievalHit:
    chunk: Chunk
    vector_score: float
    bm25_score: float
    fused_score: float
    rank: int


def _bm25_scores(
    query_tokens: list[str],
    corpus_tokens: list[list[str]],
    k1: float = 1.2,
    b: float = 0.75,
) -> np.ndarray:
    N = len(corpus_tokens)
    if N == 0:
        return np.array([])
    dl = np.array([len(doc) for doc in corpus_tokens], dtype=np.float64)
    avgdl = float(dl.mean()) if dl.mean() > 0 else 1.0
    df = Counter()
    for doc in corpus_tokens:
        df.update(set(doc))
    scores = np.zeros(N, dtype=np.float64)
    for qi in query_tokens:
        n_qi = df.get(qi, 0)
        idf = math.log((N - n_qi + 0.5) / (n_qi + 0.5) + 1.0)
        for i, doc in enumerate(corpus_tokens):
            f = doc.count(qi)
            if f == 0:
                continue
            denom = f + k1 * (1 - b + b * dl[i] / avgdl)
            scores[i] += idf * (f * (k1 + 1)) / denom
    return scores


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.ones_like(x)
    return (x - lo) / (hi - lo)


def _bm25_only_retrieve(store: FaissStore, query: str, k: int = 8) -> list[RetrievalHit]:
    """Fallback retrieval when query embedding backend is unavailable."""
    if not store.chunks:
        return []
    corpus_tokens = [tokenize(c.text) for c in store.chunks]
    q_tok = tokenize(query)
    bm25 = _bm25_scores(q_tok, corpus_tokens)
    bm25_n = _minmax(bm25)
    order = np.argsort(-bm25_n)[: min(k, len(store.chunks))]
    hits: list[RetrievalHit] = []
    for rank, idx in enumerate(order):
        i = int(idx)
        hits.append(
            RetrievalHit(
                chunk=store.chunks[i],
                vector_score=0.0,
                bm25_score=float(bm25[i]),
                fused_score=float(bm25_n[i]),
                rank=rank + 1,
            )
        )
    return hits


def hybrid_retrieve(
    store: FaissStore,
    query: str,
    k: int = 8,
    oversample: int = 64,
    vector_weight: float = 0.55,
) -> list[RetrievalHit]:
    """
    Hybrid: retrieve oversample by vector, score all with BM25, fuse min-max scores.
    Fixes many pure-vector failures on exact names/years (keyword signal).
    """
    if not store.chunks:
        return []
    if os.environ.get("FORCE_BM25_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        return _bm25_only_retrieve(store, query, k=k)
    try:
        q_vec = embed_query(query)
    except Exception:
        return _bm25_only_retrieve(store, query, k=k)
    vec_scores, vec_idx = store.search(q_vec, min(oversample, len(store.chunks)))
    candidate_idx = [int(i) for i in vec_idx if i >= 0]
    if not candidate_idx:
        return []

    corpus_tokens = [tokenize(c.text) for c in store.chunks]
    q_tok = tokenize(query)
    full_bm25 = _bm25_scores(q_tok, corpus_tokens)

    vec_full = np.zeros(len(store.chunks), dtype=np.float64)
    for s, idx in zip(vec_scores, vec_idx):
        if idx >= 0:
            vec_full[int(idx)] = float(s)

    bm25_cand = full_bm25[candidate_idx]
    vec_cand = np.array([vec_full[i] for i in candidate_idx])
    bm25_n = _minmax(bm25_cand)
    vec_n = _minmax(vec_cand)
    fused = vector_weight * vec_n + (1.0 - vector_weight) * bm25_n

    order = np.argsort(-fused)[:k]
    hits: list[RetrievalHit] = []
    for rank, j in enumerate(order):
        idx = candidate_idx[int(j)]
        hits.append(
            RetrievalHit(
                chunk=store.chunks[idx],
                vector_score=float(vec_full[idx]),
                bm25_score=float(full_bm25[idx]),
                fused_score=float(fused[j]),
                rank=rank + 1,
            )
        )
    return hits


def query_expansion_simple(query: str) -> str:
    """Lightweight expansion: election/budget domain terms (manual, not LLM)."""
    extra = []
    low = query.lower()
    if "vote" in low or "election" in low or "parliament" in low:
        extra.append("constituency candidate party votes")
    if "budget" in low or "fiscal" in low or "revenue" in low or "ghana" in low:
        extra.append("economic policy expenditure revenue")
    if not extra:
        return query
    return query + " " + " ".join(extra)


def retrieve_with_optional_expansion(
    store: FaissStore,
    query: str,
    k: int = 8,
    use_expansion: bool = False,
) -> tuple[list[RetrievalHit], str]:
    q = query_expansion_simple(query) if use_expansion else query
    return hybrid_retrieve(store, q, k=k), q


def pure_vector_topk(store: FaissStore, query: str, k: int = 8) -> list[RetrievalHit]:
    """Baseline for Part E comparison (no BM25 fusion)."""
    if os.environ.get("FORCE_BM25_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        return _bm25_only_retrieve(store, query, k=k)
    try:
        q_vec = embed_query(query)
    except Exception:
        return _bm25_only_retrieve(store, query, k=k)
    scores, indices = store.search(q_vec, min(k, len(store.chunks)))
    hits: list[RetrievalHit] = []
    for rank, (s, idx) in enumerate(zip(scores, indices)):
        if idx < 0:
            continue
        hits.append(
            RetrievalHit(
                chunk=store.chunks[int(idx)],
                vector_score=float(s),
                bm25_score=0.0,
                fused_score=float(s),
                rank=rank + 1,
            )
        )
    return hits
