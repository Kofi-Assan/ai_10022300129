# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence
"""Prompt construction and context window management (manual)."""
from __future__ import annotations

from dataclasses import replace

from rag.retrieval import RetrievalHit


def select_context(
    hits: list[RetrievalHit],
    max_chars: int = 6000,
    min_score_floor: float = 0.0,
) -> list[RetrievalHit]:
    """Rank by fused_score, truncate to max_chars; drop very low fused scores."""
    ordered = sorted(
        [h for h in hits if h.fused_score >= min_score_floor],
        key=lambda h: -h.fused_score,
    )
    out: list[RetrievalHit] = []
    used = 0
    for h in ordered:
        block = h.chunk.text
        if used + len(block) + 2 > max_chars:
            remain = max_chars - used - 2
            if remain > 80:
                trimmed = replace(h.chunk, text=block[:remain] + "…")
                out.append(
                    RetrievalHit(
                        chunk=trimmed,
                        vector_score=h.vector_score,
                        bm25_score=h.bm25_score,
                        fused_score=h.fused_score,
                        rank=h.rank,
                    )
                )
            break
        out.append(h)
        used += len(block) + 2
    return out


def build_context_block(hits: list[RetrievalHit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        src = h.chunk.source
        parts.append(f"[{i}] (source={src})\n{h.chunk.text}")
    return "\n\n".join(parts)


def build_rag_prompt(
    user_query: str,
    context_block: str,
    style: str = "strict",
) -> str:
    """
    style=strict: stronger hallucination control.
    style=concise: shorter answers allowed.
    """
    base_rules = """You are an assistant for Academic City. Answer ONLY using the CONTEXT below.
If the CONTEXT does not contain enough information, say you do not know and suggest what is missing.
Do not invent statistics, dates, or names not present in CONTEXT. Quote figures only when they appear in CONTEXT."""

    if style == "concise":
        rules = base_rules + "\nKeep the answer brief (3-6 sentences) unless the user asks for detail."
    else:
        rules = (
            base_rules
            + "\nBefore answering, mentally list which context snippets support each claim."
        )

    return f"""{rules}

CONTEXT:
{context_block}

USER QUESTION:
{user_query}

ANSWER:"""


def build_no_context_prompt(user_query: str) -> str:
    """Baseline for Part E: same question without retrieval."""
    return f"""Answer the question helpfully. You may use general knowledge.

USER QUESTION:
{user_query}

ANSWER:"""
