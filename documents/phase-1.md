# Phase 1 — MVP Vertical Slice

**Duration:** Week 1-2  
**Git Tag:** `v0.2.0-mvp`  
**Status:** ✅ Complete

## What Was Built

### Backend — Full RAG Pipeline

#### Ingestion Pipeline
- **PDF Parser** (`ingestion/parsers/pdf_parser.py`) — Layout-aware parsing using PyMuPDF. Classifies elements by font size into titles/headings/paragraphs. OCR-ready fallback to plain text extraction.
- **Base Parser Interface** (`ingestion/parsers/base.py`) — Abstract `BaseParser` with `ParsedDocument`/`ParsedElement` dataclasses preserving element types (title, heading, paragraph, table, list, image_caption).
- **Recursive Chunker** (`ingestion/chunkers/recursive_chunker.py`) — Configurable chunk_size (default 512), overlap (default 50), hierarchical separators (paragraphs → sentences → words). Preserves element metadata per chunk.
- **Embedding Registry** (`ingestion/embedders/embedder.py`) — Swappable interface with `OllamaEmbedder` (self-hosted, nomic-embed-text, 768 dims) and `GeminiEmbedder` (API-based). One env var switch: `DEFAULT_EMBEDDING_PROVIDER=ollama|gemini`.
- **Pipeline Orchestrator** (`ingestion/pipeline.py`) — End-to-end: parse → chunk → embed → store in pgvector. Each step wrapped in OpenInference spans.

#### Retrieval
- **Dense Retriever** (`retrieval/dense.py`) — pgvector cosine similarity search (`<=>` operator). Returns top-k chunks with similarity scores.
- **Retrieval Pipeline** (`retrieval/pipeline.py`) — Orchestrator (dense-only in Phase 1, extensible for hybrid/RRF in Phase 2).

#### Generation
- **Generator** (`generation/generator.py`) — Ollama LLM inference with grounded QA prompt template. Enforces citation from retrieved context. Structured citation extraction via regex. SSE streaming support.

#### Workers
- **Ingest Task** (`workers/ingest_task.py`) — Celery task for async document ingestion with retry logic.

#### API Endpoints
- `POST /api/v1/ingest` — Upload PDF → Celery job
- `GET /api/v1/ingest/{job_id}` — Job status + chunk count
- `POST /api/v1/query` — Full RAG: retrieve → generate → return answer + citations + trace_id
- `POST /api/v1/query/stream` — SSE streaming variant
- `POST /api/v1/retrieve` — Debug endpoint: chunks + scores, no generation

### Frontend — Stitch MCP Design

Both pages designed using the **Lumina Nexus** design system (Stitch MCP):
- **Primary**: Teal `#00D4AA` / Cyan `#00B4D8`
- **Background**: Dark navy `#0A0F1C`
- **Fonts**: Inter (body) + JetBrains Mono (labels/code)
- **Style**: Glassmorphism, backdrop-blur, glowing borders
- **Zero purple** anywhere

#### Landing Page (`frontend/public/landing.html`)
- WebGL shader hero with flowing teal/cyan gradient mesh
- Glassmorphism feature grid (6 capabilities)
- RAG pipeline visualization (Document → Parse → Chunk → Embed → Retrieve → Generate)
- Performance metrics cards (150ms latency, 99% recall, 50% cost savings)
- Floating chat widget with teal glow
- Sticky nav with glass effect

#### Chat Playground (`frontend/public/chat.html`)
- Sidebar navigation (7 pages)
- Ingested documents list
- Chat interface with user/AI message bubbles
- Inline citation badges [1], [2]
- Faithfulness score + trace_id per AI message
- Glass input bar with send button
- Top telemetry bar (model, latency, tokens)

## Key Design Decisions
1. **Embedding registry over single model** — benchmarkable dual-model setup
2. **OpenInference spans from the first query** — tracing is architecture, not decoration
3. **Grounded prompt template** — forces citations, reduces hallucination
4. **Stitch MCP for UI design** — production-quality glassmorphism design system

## Files Created (20 files, 2386 lines)

```
backend/app/
├── api/v1/
│   ├── __init__.py (updated — mounts ingest, query, retrieve routers)
│   ├── ingest.py
│   ├── query.py
│   └── retrieve.py
├── generation/generator.py
├── ingestion/
│   ├── parsers/{base.py, pdf_parser.py, __init__.py}
│   ├── chunkers/{recursive_chunker.py, __init__.py}
│   ├── embedders/{embedder.py, __init__.py}
│   └── pipeline.py
├── retrieval/{dense.py, pipeline.py}
└── workers/ingest_task.py

frontend/public/
├── landing.html (24KB — WebGL shader + glassmorphism)
├── chat.html (20KB — chat playground)
├── landing_preview.png
└── chat_preview.png
```

## Next Phase
**Phase 2 — Advanced Retrieval**: BM25 + hybrid RRF, cross-encoder reranking, hierarchical chunking, contextual retrieval, query transformation, semantic cache, retrieval inspector page.
