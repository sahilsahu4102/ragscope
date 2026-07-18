# RAGScope — Versioned golden evaluation datasets

This directory holds versioned JSONL golden datasets for evaluation.

## Format

Each dataset is a `.jsonl` file where each line is a JSON object:

```json
{
  "question": "What is the statute of limitations for breach of contract?",
  "gold_answer": "The statute of limitations for breach of contract is typically 4-6 years...",
  "source_chunks": ["chunk_id_1", "chunk_id_2"],
  "query_type": "single-hop",
  "difficulty": "medium"
}
```

## Naming Convention

`{corpus_name}_v{version}_{sample_count}.jsonl`

Example: `legal_corpus_v1_300.jsonl`

## Dataset Generation

Datasets are generated using Ragas TestsetGenerator (Phase 3) and manually curated.
