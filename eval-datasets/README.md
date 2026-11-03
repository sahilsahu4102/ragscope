# RAGScope — Versioned golden evaluation datasets

Golden QA sets the eval harness scores retrieval and generation against. Each
is loaded into the database through `POST /api/v1/datasets/upload` (or created
by `POST /api/v1/datasets/generate`) and referenced by id in eval runs and A/B
experiments.

## Datasets

| File | Questions | Built by | Use |
|---|---|---|---|
| `golden-v1.jsonl` | 30 | LLM-generated from seeded chunk samples (`/datasets/generate`) | replication set |
| `golden-v2.jsonl` | 28 | v1 after hand review: 5 revised, 2 dropped (`app/scripts/revise_golden.py`) | headline single-hop set |
| `multihop-v1.jsonl` | 18 | two-hop questions written from chosen chunk pairs, with automated checks (below) | multi-hop retrieval |

Corpus: the four PDFs in `test_docs/` (5,869 chunks).

## Format

One JSON object per line:

```json
{
  "question": "What is the range of per-GPU model state size saved during Llama 3 checkpointing?",
  "gold_answer": "Each GPU's model state ranges from 1 MB to 4 GB per GPU.",
  "gold_contexts": ["durations. Checkpointing saves each GPU's model state, ranging from 1 MB to 4 GB ..."],
  "question_type": "factual",
  "metadata": {"source_chunk_ids": ["2b2e6ae7-..."], "source_documents": ["llama3_paper.pdf"]}
}
```

`gold_contexts` holds the full text of each gold chunk. A retrieved chunk counts
as relevant when its word-level Jaccard similarity with a gold context is at
least 0.5 (`app/eval/metrics/retrieval_metrics.py`), so the eval keeps working
after a re-ingest changes chunk ids. `question_type` is `factual`, `reasoning`
or `analytical` in the golden sets and `multi-hop` in `multihop-v1`.

## multihop-v1

Each question asks for two facts that live in two different chunks: nine
compare the original Transformer with Llama 3, four join two sections of one
document, and two join unrelated documents. `metadata.kind` records which
(`comparison`, `within-document`, `conjunction`). The questions were written
with Claude Code from chunk pairs chosen by hand, and the build script checked
two things before writing the file:

- every fact's pattern matches **exactly one** chunk in the corpus, so a
  duplicate or near-duplicate chunk cannot hold the answer and be scored wrong
- **no single chunk** matches both halves of a question, so none of them can be
  answered from one chunk

Several first-choice facts failed the first check ("1.1°C" appears in 8 IPCC
chunks, "Tectonic" also in the contributor list) and were replaced. With two
gold contexts per question, `full_support@10` (both facts in the top 10) is the
metric that matters: `recall@10` gives a question with one of its two facts
found 0.5.
