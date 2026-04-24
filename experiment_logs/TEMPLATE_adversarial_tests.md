# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence

# Part E — Adversarial Testing + RAG vs LLM-only (Manual)

**Date:**  
**Test ID:** (e.g., `adv01`)  

## Adversarial queries (2 required)

### Query A (ambiguous / underspecified)
**Query text:**

**Why it’s adversarial (1–2 lines):**

### Query B (misleading / incomplete)
**Query text:**

**Why it’s adversarial (1–2 lines):**

## Evidence runs
For each query, run:
- **RAG (hybrid)** and record retrieved hits + answer quality
- **LLM-only** (no retrieval) and record hallucination/guessing behavior

### Query A — RAG
- Retrieved hits summary (sources + why relevant):
- Answer summary:
- Accuracy:
- Hallucination rate/risk:
- Consistency across 2 runs (same settings):

### Query A — LLM-only baseline
- Answer summary:
- Accuracy:
- Hallucination rate/risk:
- Consistency across 2 runs:

### Query B — RAG
- Retrieved hits summary:
- Answer summary:
- Accuracy:
- Hallucination rate/risk:
- Consistency across 2 runs:

### Query B — LLM-only baseline
- Answer summary:
- Accuracy:
- Hallucination rate/risk:
- Consistency across 2 runs:

## Conclusion (evidence-based)
- Where RAG clearly beats LLM-only:
- Where RAG still fails:
- What fix you applied (if any) and whether it helped:

