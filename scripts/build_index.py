# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence
"""Build FAISS index from downloaded CSV + PDF."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunking import chunk_pdf_text, chunks_from_csv
from rag.embeddings import embed_texts
from rag.store import FaissStore

RAW = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def main() -> None:
    csv_path = RAW / "Ghana_Election_Result.csv"
    pdf_path = RAW / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    if not csv_path.is_file() or not pdf_path.is_file():
        print("Run scripts/download_data.py first.", file=sys.stderr)
        sys.exit(1)

    # Part A: make chunking configurable for comparisons.
    # CLI: python scripts/build_index.py [chunk_size] [overlap]
    default_chunk = _int_env("PDF_CHUNK_SIZE", 900)
    default_overlap = _int_env("PDF_CHUNK_OVERLAP", 120)
    try:
        chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else default_chunk
        overlap = int(sys.argv[2]) if len(sys.argv) > 2 else default_overlap
    except ValueError:
        chunk_size, overlap = default_chunk, default_overlap

    chunks = []
    chunks.extend(chunks_from_csv(str(csv_path)))
    pdf_text = extract_pdf_text(pdf_path)
    chunks.extend(
        chunk_pdf_text(
            pdf_text,
            source_label="budget_2025",
            chunk_size=chunk_size,
            overlap=overlap,
        )
    )

    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunks…")
    vectors = embed_texts(texts)
    dim = vectors.shape[1]
    store = FaissStore(dim=dim)
    store.add(vectors, chunks)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save(INDEX_DIR)
    (INDEX_DIR / "build_config.json").write_text(
        json.dumps(
            {
                "pdf_chunk_size": chunk_size,
                "pdf_overlap": overlap,
                "embedding_backend": os.environ.get("EMBEDDINGS_BACKEND", "auto"),
                "embedding_model_local": os.environ.get("EMBEDDING_MODEL", ""),
                "embedding_model_openai": os.environ.get("OPENAI_EMBED_MODEL", ""),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved index to {INDEX_DIR} ({len(chunks)} chunks, dim={dim}).")


if __name__ == "__main__":
    main()
