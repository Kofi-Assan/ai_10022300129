# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence
"""
Embedding pipeline (manual).

Default: try local `sentence-transformers` (best offline).
Fallback: OpenAI embeddings (works on Python versions where PyTorch isn't available).
"""
from __future__ import annotations

import os

import numpy as np

_MODEL = None
_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
_OPENAI_EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
_BACKEND = os.environ.get("EMBEDDINGS_BACKEND", "auto").lower().strip()


def _get_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Local embeddings require `sentence-transformers` (and a compatible PyTorch build). "
                "Either install them on Python 3.11, or set EMBEDDINGS_BACKEND=openai and provide "
                "OPENAI_API_KEY to use OpenAI embeddings."
            ) from e

        _MODEL = SentenceTransformer(_MODEL_NAME)
    return _MODEL


def _normalize(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / denom


def _embed_openai(texts: list[str]) -> np.ndarray:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required to embed with OpenAI. "
            "Set it in `.env` (local) or as an environment variable (deploy)."
        )
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    out: list[np.ndarray] = []
    # Keep batches modest to avoid request limits.
    batch_size = int(os.environ.get("OPENAI_EMBED_BATCH", "64"))
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=_OPENAI_EMBED_MODEL, input=batch)
        out.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
    mat = np.vstack(out).astype(np.float32)
    return _normalize(mat)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    # Deployment guard: on low-memory hosts we can force BM25 retrieval only.
    if os.environ.get("FORCE_BM25_ONLY", "").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("FORCE_BM25_ONLY is enabled; skipping dense embeddings.")

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    backend = _BACKEND
    if backend == "auto":
        try:
            # Fast path if sentence-transformers is available.
            model = _get_model()
            vectors = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 50,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vectors.astype(np.float32)
        except Exception:
            backend = "openai"

    if backend in {"st", "sentence-transformers", "local"}:
        model = _get_model()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 50,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    if backend == "openai":
        return _embed_openai(texts)

    raise ValueError(
        f"Unknown EMBEDDINGS_BACKEND={_BACKEND!r}. Use 'auto', 'openai', or 'local'."
    )


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
