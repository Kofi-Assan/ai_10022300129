# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence
"""Full RAG pipeline with stage logging (no LangChain)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from rag.prompts import (
    build_context_block,
    build_no_context_prompt,
    build_rag_prompt,
    select_context,
)
from rag.retrieval import RetrievalHit, hybrid_retrieve, pure_vector_topk
from rag.store import FaissStore

logger = logging.getLogger("rag.pipeline")


@dataclass
class PipelineLog:
    stages: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, data: dict[str, Any]) -> None:
        entry = {"stage": name, **data}
        self.stages.append(entry)
        logger.info("%s: %s", name, {k: v for k, v in data.items() if k != "final_prompt"})


def call_llm(prompt: str, model: str | None = None) -> str:
    provider = os.environ.get("LLM_PROVIDER", "auto").lower().strip()
    openai_key = os.environ.get("OPENAI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    if provider == "auto":
        if groq_key:
            provider = "groq"
        elif openai_key:
            provider = "openai"
        else:
            provider = "none"

    if provider == "groq":
        if not groq_key:
            return (
                "[Demo] Set GROQ_API_KEY in .env (or switch LLM_PROVIDER). "
                f"Prompt length: {len(prompt)} chars."
            )
        client = OpenAI(
            api_key=groq_key,
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )
        m = model or os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    elif provider == "openai":
        if not openai_key:
            return (
                "[Demo] Set OPENAI_API_KEY in .env (or switch LLM_PROVIDER). "
                f"Prompt length: {len(prompt)} chars."
            )
        client = OpenAI(api_key=openai_key)
        m = model or os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    else:
        return (
            "[Demo] No LLM provider configured. Set LLM_PROVIDER=groq or openai with matching API key. "
            f"Prompt length: {len(prompt)} chars."
        )

    try:
        resp = client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return (
            "[Demo] LLM call failed (often quota/network). "
            f"{type(e).__name__}: {e}. Prompt length: {len(prompt)} chars."
        )


def run_rag(
    store: FaissStore,
    user_query: str,
    top_k: int = 8,
    prompt_style: str = "strict",
    max_context_chars: int = 6000,
    use_hybrid: bool = True,
    plog: PipelineLog | None = None,
) -> tuple[str, PipelineLog]:
    plog = plog or PipelineLog()
    plog.add("query", {"text": user_query})

    if use_hybrid:
        hits = hybrid_retrieve(store, user_query, k=top_k)
    else:
        hits = pure_vector_topk(store, user_query, k=top_k)

    plog.add(
        "retrieval",
        {
            "mode": "hybrid" if use_hybrid else "vector_only",
            "hits": [
                {
                    "source": h.chunk.source,
                    "vector_score": round(h.vector_score, 4),
                    "bm25_score": round(h.bm25_score, 4),
                    "fused_score": round(h.fused_score, 4),
                    "text_preview": h.chunk.text[:220] + ("…" if len(h.chunk.text) > 220 else ""),
                }
                for h in hits
            ],
        },
    )

    selected = select_context(hits, max_chars=max_context_chars)
    plog.add(
        "context_selection",
        {
            "num_chunks": len(selected),
            "max_chars": max_context_chars,
            "sources": list({h.chunk.source for h in selected}),
        },
    )

    context_block = build_context_block(selected)
    final_prompt = build_rag_prompt(user_query, context_block, style=prompt_style)
    plog.add(
        "prompt",
        {
            "style": prompt_style,
            "final_prompt": final_prompt,
            "prompt_chars": len(final_prompt),
        },
    )

    answer = call_llm(final_prompt)
    plog.add("generation", {"answer_preview": answer[:500]})
    return answer, plog


def run_llm_only(user_query: str, plog: PipelineLog | None = None) -> tuple[str, PipelineLog]:
    plog = plog or PipelineLog()
    plog.add("query", {"text": user_query, "mode": "no_retrieval"})
    p = build_no_context_prompt(user_query)
    plog.add("prompt", {"final_prompt": p, "prompt_chars": len(p)})
    ans = call_llm(p)
    plog.add("generation", {"answer_preview": ans[:500]})
    return ans, plog


def apply_feedback_boost(
    hits: list[RetrievalHit],
    boost_sources: set[str],
    boost: float = 0.15,
) -> list[RetrievalHit]:
    """Part G: session feedback — up-weight chunks from preferred sources."""
    out: list[RetrievalHit] = []
    for h in hits:
        bonus = boost if h.chunk.source in boost_sources else 0.0
        out.append(
            RetrievalHit(
                chunk=h.chunk,
                vector_score=h.vector_score,
                bm25_score=h.bm25_score,
                fused_score=h.fused_score + bonus,
                rank=h.rank,
            )
        )
    out.sort(key=lambda x: -x.fused_score)
    for i, h in enumerate(out):
        h.rank = i + 1
    return out
