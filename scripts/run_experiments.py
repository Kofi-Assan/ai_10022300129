# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence
"""
Run reproducible RAG experiments and write evidence files.

Purpose: generate raw evidence (retrieval hits, prompts, answers) to support Parts B–E.
You must still write MANUAL reflections in experiment_logs/ (per exam instruction).

Usage examples:
  python scripts/run_experiments.py --queries-file experiment_logs/queries.txt --hybrid --prompt strict --llm-only
  python scripts/run_experiments.py --query "What is the budget deficit target?" --vector-only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT))

from rag.pipeline import PipelineLog, call_llm, run_llm_only  # noqa: E402
from rag.prompts import build_context_block, build_rag_prompt, select_context  # noqa: E402
from rag.retrieval import (  # noqa: E402
    RetrievalHit,
    hybrid_retrieve,
    pure_vector_topk,
    retrieve_with_optional_expansion,
)
from rag.store import FaissStore  # noqa: E402


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _read_queries(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    out = []
    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def _hit_to_dict(h: RetrievalHit) -> dict:
    return {
        "rank": h.rank,
        "source": h.chunk.source,
        "meta": h.chunk.meta,
        "vector_score": float(h.vector_score),
        "bm25_score": float(h.bm25_score),
        "fused_score": float(h.fused_score),
        "text": h.chunk.text,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default="", help="Single query to run")
    ap.add_argument(
        "--queries-file",
        type=str,
        default="",
        help="Text file with one query per line (blank lines and # comments ignored)",
    )
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--prompt", type=str, default="strict", choices=["strict", "concise"])
    ap.add_argument("--max-context-chars", type=int, default=6000)
    ap.add_argument("--min-score-floor", type=float, default=0.0)

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--hybrid", action="store_true", help="Hybrid BM25+vector retrieval")
    mode.add_argument("--vector-only", action="store_true", help="Vector-only baseline retrieval")

    ap.add_argument(
        "--query-expansion",
        action="store_true",
        help="Enable manual query expansion (Part B extension option)",
    )
    ap.add_argument("--llm-only", action="store_true", help="Also run LLM-only baseline (Part E)")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "experiment_logs" / "auto_runs"),
        help="Where to write evidence JSON files",
    )
    args = ap.parse_args()

    queries: list[str] = []
    if args.query.strip():
        queries = [args.query.strip()]
    elif args.queries_file:
        queries = _read_queries(Path(args.queries_file))
    else:
        raise SystemExit("Provide --query or --queries-file")

    index_dir = ROOT / "data" / "index"
    store = FaissStore.load(index_dir)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = _now_id()

    results = {
        "student": {"name": "Kofi Assan", "index": "10022300129"},
        "run_id": run_id,
        "index_dir": str(index_dir),
        "settings": {
            "top_k": args.top_k,
            "prompt_style": args.prompt,
            "max_context_chars": args.max_context_chars,
            "min_score_floor": args.min_score_floor,
            "mode": "vector_only" if args.vector_only else "hybrid",
            "query_expansion": bool(args.query_expansion),
            "llm_only_baseline": bool(args.llm_only),
        },
        "queries": [],
    }

    for q in queries:
        item: dict = {"query": q}
        plog = PipelineLog()
        plog.add("query", {"text": q})

        if args.vector_only:
            hits = pure_vector_topk(store, q, k=args.top_k)
            expanded_q = q
        else:
            if args.query_expansion:
                hits, expanded_q = retrieve_with_optional_expansion(
                    store, q, k=args.top_k, use_expansion=True
                )
            else:
                hits = hybrid_retrieve(store, q, k=args.top_k)
                expanded_q = q

        plog.add(
            "retrieval",
            {
                "mode": "vector_only" if args.vector_only else "hybrid",
                "expanded_query": expanded_q,
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

        selected = select_context(hits, max_chars=args.max_context_chars, min_score_floor=args.min_score_floor)
        context_block = build_context_block(selected)
        final_prompt = build_rag_prompt(expanded_q, context_block, style=args.prompt)
        plog.add(
            "prompt",
            {
                "style": args.prompt,
                "final_prompt": final_prompt,
                "prompt_chars": len(final_prompt),
            },
        )
        ans = call_llm(final_prompt)
        plog.add("generation", {"answer_preview": ans[:500]})

        item["rag"] = {
            "expanded_query": expanded_q,
            "hits": [_hit_to_dict(h) for h in hits],
            "selected_sources": sorted({h.chunk.source for h in selected}),
            "final_prompt": final_prompt,
            "answer": ans,
            "pipeline_log": plog.stages,
        }

        if args.llm_only:
            base_ans, base_log = run_llm_only(q)
            item["llm_only"] = {
                "answer": base_ans,
                "pipeline_log": base_log.stages,
            }

        results["queries"].append(item)

    out_path = out_dir / f"evidence_{run_id}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote evidence file: {out_path}")


if __name__ == "__main__":
    main()

