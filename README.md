# CS4241 — Introduction to Artificial Intelligence (2026)

**Name:** Kofi Assan  
**Index number:** 10022300129  
**Repository name (submit as):** `ai_10022300129`

Manual RAG for Academic City: Ghana election results (CSV) + 2025 budget PDF.  
**No LangChain, LlamaIndex, or pre-built RAG frameworks** — chunking, embeddings, FAISS, retrieval, and prompts are implemented in this repo.

## Architecture (Part F)

Full diagram + component description: `docs/ARCHITECTURE.md`

```text
User query
    → Embedding (sentence-transformers)
    → FAISS top-k (cosine via normalized vectors + inner product)
    → Hybrid: BM25 scores on candidate pool, fused with vector scores
    → Context selection (truncate by char budget, ranked by fused score)
    → Prompt template (anti-hallucination rules + injected context)
    → LLM (Groq/OpenAI via provider switch)
    → Response + stage logs
```

**Why this fits the domain:** Election data is structured per row; budget text is long-form policy prose. Row-level CSV chunks preserve column semantics; sliding windows with overlap capture budget sections that span chunk boundaries. Hybrid retrieval recovers exact tokens (candidate names, figures) that pure dense retrieval sometimes misses.

## Innovation (Part G)

**Session feedback boost:** In the Streamlit sidebar, preferred sources (election CSV vs budget PDF) add a small weight to fused retrieval scores so follow-up queries favor user-trusted corpora.

## Setup

1. **Python**: This repo uses `runtime.txt` (**3.11.9**) for deployment. That file intentionally contains only the version string (no name/index line), because some hosts parse it strictly; **Name: Kofi Assan** and **Index: 10022300129** appear in `README.md` and source headers.  
   - If you use **Python 3.13**, local `sentence-transformers` may not install (PyTorch support).  
   - The project therefore supports **OpenAI embeddings** as a fallback for building the index on any Python version.

2. Create a virtual environment and install dependencies:

```bash
cd ai_10022300129
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and add your LLM key for real answers (optional for UI testing).
   - To force embeddings backend, set: `EMBEDDINGS_BACKEND=openai` (or leave default `auto`).
   - Chat provider options:
     - OpenAI: `LLM_PROVIDER=openai` + `OPENAI_API_KEY=...`
     - Groq: `LLM_PROVIDER=groq` + `GROQ_API_KEY=...`
     - Auto-select (default): `LLM_PROVIDER=auto` prefers Groq key when present.

4. Download data and build the index:

```bash
python scripts/download_data.py
python scripts/build_index.py
```

To compare chunking configs for **Part A**, rebuild with explicit settings:

```bash
# Example Config A
python scripts/build_index.py 600 80

# Example Config B
python scripts/build_index.py 900 120
```

Each build writes `data/index/build_config.json` capturing the chunking + embedding settings used.

5. Run the app:

```bash
streamlit run app.py
```

## Deploy on Streamlit

1. Push this repo to GitHub (see submission repo name above).
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository and select this repo.
4. In the deployment settings, set either **GROQ_API_KEY** (if using Groq) or **OPENAI_API_KEY** (if using OpenAI). Optionally set **HF_TOKEN** if Hugging Face throttles the embedding model download during build.
5. After deploy, copy the **public URL** for your exam email.

**Note on deployment:** For production deployment with proper chat history persistence, you'll want to add a database (like SQLite or PostgreSQL) since the current file-based approach won't work in a multi-user cloud environment.

## Chunking (Part A)

- **CSV:** One chunk per cleaned row; all columns concatenated with labels so retrieval sees full context for that record.  
- **PDF:** ~900 characters per window, **120-character overlap** (~13%). Justification: large enough for policy sentences; overlap limits information split across boundaries. Compare smaller/larger windows in `experiment_logs/` and note retrieval quality.

## Retrieval failure cases (Part B)

Document your own runs in `experiment_logs/`. Typical patterns to test:

1. **Exact name or year:** Pure vector retrieval may rank a semantically similar but wrong constituency or section; **hybrid BM25** tends to fix this by matching rare tokens.  
2. **Ambiguous query:** Short queries can pull generic budget chunks; mitigation: query expansion toggle in `rag/retrieval.py` or stricter `select_context` / user clarification in the UI.

## Final experiment results (observed)

These results are based on actual manual runs documented in the logs below.

- **Part A (chunking):** Final chosen PDF chunking is **900 / 120**. It was slightly better than 600 / 80 for long budget-policy statements and fiscal-target snippets.
- **Part B (failure and fix):** For ambiguous close-margin election queries, initial retrieval returned low-signal rows (`Others/PNC`, BM25≈0). After applying focused single-query runs and stricter grounding behavior, hallucinated comparisons reduced and responses became conservative/context-bound.
- **Part C (prompt behavior):** `strict` prompt style consistently reduced unsupported claims and improved refusal behavior when context was insufficient.
- **Part D (full pipeline):** Query → retrieval → context selection → prompt → generation was logged and displayed in app (chunks, scores, final prompt, answer).
- **Part E (adversarial):**
  - Query A: "What was the biggest win?" remained ambiguous; retrieval was mixed/noisy, but responses stayed cautious.
  - Query B: "The budget reduced all taxes in 2025..." was rejected safely (no fabricated tax-cut amounts), showing low hallucination risk.
- **Part G (innovation):** Session feedback boost was used to bias retrieval toward user-preferred source corpus.

## Completed manual logs

- `experiment_logs/2026-04-17_chunk_cmp01.md`
- `experiment_logs/2026-04-17_run01.md`
- `experiment_logs/2026-04-17_adv01.md`

## Evidence JSON files

- `experiment_logs/auto_runs/evidence_2026-04-17T015159Z.json`
- `experiment_logs/auto_runs/evidence_2026-04-17T032939Z.json`
- `experiment_logs/auto_runs/evidence_2026-04-17T033223Z.json`

Note: some evidence JSON runs were executed while OpenAI quota was exhausted, so those files show demo fallback responses despite successful retrieval/prompt generation. Manual logs capture the full Groq-based runs used for evaluation.

## Experiment evidence runner (Parts B–E)

To generate **evidence JSON** (retrieval hits, prompts, answers) you can attach to your manual writeups:

```bash
# One query
python scripts/run_experiments.py --query "What is the budget deficit target?" --hybrid --llm-only

# Multiple queries (one per line, # comments allowed)
python scripts/run_experiments.py --queries-file experiment_logs/queries.txt --hybrid --query-expansion --llm-only
```

Outputs are written to `experiment_logs/auto_runs/`. You should still write your own manual reflections using the templates in `experiment_logs/`.

## Submission (from question paper)

- Push to GitHub: repo **`ai_10022300129`**.  
- Deploy on Streamlit Community Cloud and record the public URL.  
- Add **GodwinDansoAcity** / `godwin.danso@acity.edu.gh` as collaborator.  
- Email the lecturer with subject: `CS4241-Introduction to Artificial Intelligence-2026:[your index and name]`.  
- Include: repo link, deployed URL, **video walkthrough (≤2 min)**, **manual** experiment logs, and this documentation.

## Files

| Path | Role |
|------|------|
| `rag/chunking.py` | Cleaning + chunk strategies |
| `rag/embeddings.py` | Sentence-transformers embeddings |
| `rag/store.py` | FAISS persist/load |
| `rag/retrieval.py` | Top-k, BM25, hybrid fusion |
| `rag/prompts.py` | Context window + templates |
| `rag/pipeline.py` | End-to-end logging + LLM call |
| `app.py` | Streamlit UI |
| `scripts/download_data.py` | Fetch exam datasets |
| `scripts/build_index.py` | Build `data/index/` |
| `runtime.txt` | Python version for deployment builds |

Student name and index appear in the README and in each source file header as required.
