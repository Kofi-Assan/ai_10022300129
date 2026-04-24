# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence
"""
Manual chunking for CSV rows and PDF text.
Design: CSV — one logical chunk per row (tabular context preserved).
PDF — character windows with overlap to keep cross-boundary phrases.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass
class Chunk:
    text: str
    source: str
    meta: dict


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def chunks_from_csv(path: str, source_label: str = "ghana_elections") -> list[Chunk]:
    df = pd.read_csv(path)
    df = df.dropna(how="all")
    chunks: list[Chunk] = []
    for idx, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])]
        text = clean_text(" | ".join(parts))
        if not text:
            continue
        chunks.append(
            Chunk(
                text=text,
                source=source_label,
                meta={"row": int(idx), "type": "csv_row"},
            )
        )
    return chunks


def chunk_pdf_text(
    raw_text: str,
    source_label: str,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """
    chunk_size ~900 chars: fits several sentences; overlap 120 (~13%) reduces
    cut mid-argument (justify in README / experiment logs).
    """
    raw_text = clean_text(raw_text)
    if not raw_text:
        return []
    chunks: list[Chunk] = []
    start = 0
    i = 0
    n = len(raw_text)
    while start < n:
        end = min(start + chunk_size, n)
        piece = raw_text[start:end]
        if end < n:
            boundary = max(piece.rfind(". "), piece.rfind("\n"), piece.rfind(" "))
            if boundary > chunk_size // 3:
                piece = piece[: boundary + 1].strip()
                end = start + len(piece)
        piece = clean_text(piece)
        if piece:
            chunks.append(
                Chunk(
                    text=piece,
                    source=source_label,
                    meta={"chunk_index": i, "type": "pdf_window", "start": start},
                )
            )
            i += 1
        start = max(end - overlap, start + 1)
    return chunks


def iter_all_chunks(
    csv_path: str | None,
    pdf_text: str | None,
) -> Iterator[Chunk]:
    if csv_path:
        yield from chunks_from_csv(csv_path)
    if pdf_text:
        yield from chunk_pdf_text(pdf_text, source_label="budget_2025")
