# Name: Kofi Assan | Index: 10022300129 | IT3241-Introduction to Artificial Intelligence

# Part F — Architecture & System Design

## 1. High-level Architecture Diagram

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e0f2fe', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#f3f4f6'}}}%%
flowchart LR
  %% Node Definitions and Shapes
  U([🖥️ User Streamlit UI])
  Q{🔍 Query Handler}
  
  subgraph IndexBuild [⚙️ Offline / Build-time]
    direction TB
    D1[(📄 CSV: Elections)]
    D2[(📄 PDF: 2025 Budget)]
    C1[✂️ Clean & Chunk Rows]
    C2[✂️ Sliding Window Chunking]
    E[🧠 Embedding Pipeline]
    V[(🗄️ FAISS Index + chunks.json)]
    
    D1 --> C1
    D2 --> C2
    C1 --> E
    C2 --> E
    E --> V
  end

  subgraph Runtime [⚡ Online / Query-time]
    direction TB
    R1[🎯 Dense Retrieval FAISS]
    R2[🔑 Keyword Scoring BM25]
    F[⚖️ Fuse Scores & Rank]
    S[📏 Context Selection]
    P[📝 Strict Prompt Template]
    L((🤖 LLM API))
    A([💬 Final Answer])
    Lg[(📜 Pipeline Logs)]

    R1 --> F
    R2 --> F
    F --> S
    S --> P
    P --> L
    L --> A
    
    %% Logging paths
    F -.-> Lg
    S -.-> Lg
    P -.-> Lg
    L -.-> Lg
  end

  %% Cross-subgraph interactions
  U -->|Query| Q
  Q --> R1
  Q --> R2
  V -.->|Vector Matches| R1
  A -->|Response| U
  Lg -.->|Audit Trail| U

  %% Styling classes
  classDef ui fill:#8b5cf6,stroke:#7c3aed,stroke-width:2px,color:#fff;
  classDef query fill:#ec4899,stroke:#db2777,stroke-width:2px,color:#fff;
  classDef data fill:#f59e0b,stroke:#d97706,stroke-width:2px,color:#fff;
  classDef process fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#fff;
  classDef store fill:#10b981,stroke:#059669,stroke-width:2px,color:#fff;
  classDef llm fill:#ef4444,stroke:#dc2626,stroke-width:2px,color:#fff;
  classDef log fill:#6b7280,stroke:#4b5563,stroke-width:2px,color:#fff,stroke-dasharray: 5 5;

  %% Apply styles
  class U ui;
  class Q query;
  class D1,D2 data;
  class C1,C2,E,R1,R2,F,S,P process;
  class V store;
  class L llm;
  class A ui;
  class Lg log;
```

## 2. Components Interaction and Data Flow

The RAG application is split into two distinct phases: **Offline Build-time** and **Online Query-time**.

### Data Flow (Offline Build-time)
1. **Data Ingestion & Cleaning:** The Ghana Elections CSV and 2025 Budget PDF are loaded. Null values and whitespace are cleaned.
2. **Chunking Strategy:** 
   - The CSV is processed row-by-row. Each row is converted into a self-contained string chunk ensuring column semantics (like "Candidate Name" and "Votes") are preserved together.
   - The PDF uses a sliding window chunking technique (900 characters with a 120-character overlap) to maintain the context of long, spanning policy sentences.
3. **Embedding & Storage:** Both CSV and PDF chunks are embedded into dense vectors using `sentence-transformers`. The vectors and their metadata are stored in a local FAISS index (`index.faiss`) for fast retrieval.

### Data Flow (Online Query-time)
1. **User Query:** The user inputs a question via the Streamlit UI.
2. **Hybrid Retrieval:** The system performs two parallel searches:
   - **Dense Retrieval (FAISS):** Finds chunks with high semantic similarity to the query.
   - **Keyword Scoring (BM25):** Finds chunks with exact token matches (vital for specific numbers, names, or years).
3. **Score Fusion:** The semantic and keyword scores are normalized and combined. The chunks are re-ranked based on this fused score.
4. **Context Selection:** The system selects the highest-ranking chunks up to a strict character limit to ensure it fits within the LLM's context window.
5. **Prompt Construction:** The selected context is injected into a strict prompt template that forces the LLM to only answer based on the provided text, mitigating hallucination.
6. **LLM Generation:** The LLM receives the prompt and generates the final response, which is surfaced back to the user alongside the retrieved documents and similarity scores.

## 3. Justification: Why this design is suitable for the domain

This architecture was specifically tailored for querying election data and budget policies:

- **Handling Structured vs Unstructured Data:** The domain contains highly structured tabular data (elections) and unstructured long-form text (budget). The custom dual-chunking strategy (row-level for CSV, sliding windows for PDF) ensures that quantitative election margins aren't split mid-sentence, while dense budget paragraphs remain cohesive.
- **Why Hybrid Retrieval is Critical:** Election questions often rely on exact names (e.g., "John Mahama") or precise constituency figures. Pure dense vector retrieval often struggles with exact keyword matching and might retrieve a semantically similar but incorrect constituency. By fusing FAISS with BM25, the system excels at both conceptual queries ("What is the economic policy?") and exact fact-finding ("How many votes did candidate X get?").
- **Cost and Efficiency:** Using a local FAISS index and local sentence-transformers means the vector retrieval is completely free and runs offline, reducing API costs and latency.
- **Traceability:** In sensitive domains like government budgets and elections, hallucination is dangerous. Passing similarity scores, the selected chunks, and the exact prompt to the UI allows users to audit the LLM's answer and trust the output.
