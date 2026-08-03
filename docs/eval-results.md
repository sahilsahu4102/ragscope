# Reranker A/B — LLM-as-scorer vs cross-encoder

Phase 6 replaced the reranker with a cross-encoder and made that stage ~123x
faster. This is the evidence that the speedup did not cost retrieval quality.

**Variant A** — `OllamaReranker`: prompts `llama3.2:3b` to score each candidate
passage 0-10 in a single batched call.
**Variant B** — `CrossEncoderReranker`: `cross-encoder/ms-marco-MiniLM-L-6-v2`
(22M params), scoring query-passage pairs directly.

Both variants see identical retrieved candidates (hybrid dense + BM25 + RRF,
top_k=20). Only the reranker differs. Significance is a paired bootstrap over
per-sample scores.

## Results

### golden-v2 (28 questions, hand-reviewed) — headline

| Metric | A (Ollama LLM) | B (cross-encoder) | Delta | p | Significant |
|---|---|---|---|---|---|
| mrr | 0.158 | **0.893** | +467% | 0.000 | yes |
| ndcg@10 | 0.184 | **0.902** | +391% | 0.000 | yes |
| precision@10 | 0.032 | 0.093 | +189% | 0.000 | yes |
| recall@10 | 0.286 | **0.768** | +169% | 0.000 | yes |
| hit_rate | 0.571 | **0.929** | +63% | 0.002 | yes |
| context_precision | 0.029 | 0.046 | +62% | 0.000 | yes |
| answer_correctness | 0.204 | 0.248 | +21% | 0.032 | yes |
| context_recall | 0.732 | 0.839 | +15% | 0.070 | no |
| answer_relevance | 0.172 | 0.179 | +4% | 0.354 | no |
| faithfulness | 0.242 | 0.236 | -2% | 0.650 | no |

### golden-v1 (30 questions, raw LLM-generated) — replication

| Metric | A | B | Delta | p |
|---|---|---|---|---|
| mrr | 0.220 | 0.917 | +316% | 0.000 |
| ndcg@10 | 0.241 | 0.921 | +282% | 0.000 |
| recall@10 | 0.300 | 0.750 | +150% | 0.000 |
| hit_rate | 0.567 | 0.933 | +65% | 0.000 |
| answer_correctness | 0.209 | 0.236 | +12.5% | 0.216 |

Same direction, same magnitude, on two independently-built dataset versions.
`answer_correctness` only reaches significance on v2 — the v1 set contained
7 poorly-posed questions that added noise to answer grading.

## Validity check

The gap is large enough to warrant asking whether variant A was actually
working. `OllamaReranker._parse_batch_scores` defaults any unparsed line to
0.5, so a parse failure would give every chunk an identical score, make the
sort a no-op, and silently degrade variant A into "RRF with no reranking" —
inflating the result.

Tested directly on a real query:

- the LLM reranker emitted **10 distinct scores**, not a uniform 0.5
- it moved **4 of the top 5** results out of RRF order
- the cross-encoder also moved 4 of the top 5, and the two disagreed on their
  top-5 ordering

So both rerankers genuinely reorder. This is cross-encoder vs *working* LLM
reranking, not vs a no-op.

## Reading the absolute numbers

`precision@10` and `context_precision` look low in isolation. They are bounded
by how many gold contexts each sample has — an average of 1.4, so the ceiling
for precision@10 is about 0.14. B's 0.093 is roughly 66% of what is
achievable, not 9%.

Generation metrics (`faithfulness`, `answer_relevance`, `hallucination_score`)
are heuristic, computed without the LLM judge. They are useful for detecting
regression between variants, not as absolute quality scores.

## Latency

Measured after Phase 6, 5 queries per mode, questions drawn from golden-v2 so
they are answerable from the indexed corpus.

| Mode | p50 | Mean |
|---|---|---|
| Dense only | 8,522 ms | 8,638 ms |
| Hybrid (RRF) | 8,409 ms | 8,546 ms |
| Hybrid + rerank | 8,007 ms | 8,731 ms |
| Cache hit | **49 ms** | — |
| Cache miss | 9,035 ms | — |

Cache speedup: **185x**.

Two things this shows:

**Retrieval mode no longer moves the needle.** Dense, hybrid, and hybrid+rerank
land within noise of each other. Generation dominates end-to-end latency; the
retrieval and reranking stages are now a rounding error against it. Before
Phase 6 the same comparison spanned 47s to 117s.

**Earlier absolute figures were measured on unanswerable questions.** The
benchmark's original question list asked about BM25, RRF and semantic caching
— none of which appear in the indexed corpus (the Attention and Llama 3
papers). Retrieval returned near-misses and the model replied "I don't have
enough information", which is cheap to generate. That produced a misleadingly
fast ~3.9s figure. With answerable questions the model generates full grounded
answers and p50 is ~8s.

The Phase 6 before/after deltas are unaffected — both sides used the same
question set — but ~8s p50 is the honest absolute number for real queries.

## Vector index scaling

The corpus is 5,869 chunks — too small to tell you whether retrieval survives
growth. `app/scripts/hnsw_scale.py` measures the scaling curve by seeding a
scratch table with the real embeddings and expanding it with perturbed copies
(perturbed rather than random, because uniformly random vectors sit off the
manifold real embeddings occupy and ANN recall depends on that geometry).

Recall is measured against exact search on the same table:
`|ANN top-k INTERSECT exact top-k| / k`.

| Corpus | Exact (seq scan) | HNSW ef=40 | recall@10 | Speedup | Build |
|---|---|---|---|---|---|
| 5,869 | 16.0 ms | 2.9 ms | 1.000 | 5.4x | 2.9s |
| 25,000 | 55.7 ms | 3.4 ms | 1.000 | 16.5x | 40.3s |
| 100,000 | 373.5 ms | 4.6 ms | 0.995 | 81.9x | 350.2s |

Exact search scales linearly (16 -> 56 -> 374 ms). HNSW scales logarithmically
— 1.6x the latency for 17x the data — at effectively unchanged recall. Index
build cost grows superlinearly and is the real price: ~6 minutes at 100k.

This measures **index behaviour only**. Synthetic rows carry no meaningful
text, so retrieval *quality* numbers stay on the real corpus above.

### The index alone was not enough

Creating the HNSW index did not speed anything up, because Postgres kept
choosing a sequential scan over it:

```
Seq Scan on chunks c  (cost=0.00..456.36)  actual time=0.031..13.061
```

`random_page_cost` defaults to 4.0, which models the seek cost of a spinning
disk. On NVMe that overprices index scans badly. Measured on the live table:

| random_page_cost | Plan | Execution |
|---|---|---|
| 4.0 (default) | Seq Scan | 13.5 ms |
| 1.1 | Index Scan | **1.35 ms** |

Set in docker-compose.yml. Worth knowing that an ANN index can be present,
correct, and completely unused — `EXPLAIN ANALYZE` is the only way to tell.

### Effect on the live pipeline

Per-stage spans, warm, hybrid + rerank:

| Stage | Before (2,094 chunks, no index) | After (5,869 chunks, HNSW) |
|---|---|---|
| vector scan | ~15.5 ms | 3.7 ms |
| BM25 | 9.2 ms | 20.4 ms |
| rerank | 507 ms | 472 ms |
| retrieval total | 581 ms | 508 ms |

The vector scan got ~4x faster while the corpus grew 2.8x. BM25 moved the
other way, since it scores the whole corpus per query. Postgres FTS looked
like the obvious replacement; measuring it showed otherwise — see the next
section.

## Sparse retrieval: BM25 vs Postgres FTS

BM25 regressed as the corpus grew (9.2ms at 2,094 chunks -> 20.4ms at 5,869),
which looked like the in-process O(N) scorer starting to lose. A GIN-indexed
tsvector was added to replace it.

Measured, that was wrong (app/scripts/sparse_scale.py, same synthetic text
corpus sampled from real chunks, same queries):

| rows | postgres_fts p50 | bm25 p50 | bm25 index build |
|---|---|---|---|
| 5,000 | 15.1 ms | **6.4 ms** | 0.1s |
| 25,000 | 104.4 ms | **45.3 ms** | 0.4s |
| 100,000 | 423.2 ms | **188.8 ms** | 1.5s |

BM25 is ~2.2x faster at every size tested. There is no crossover in this range.

The cause is ranking rather than lookup. Question-shaped queries need OR
semantics — `plainto_tsquery` and `websearch_to_tsquery` both AND their terms,
and the question *"Why does the synchronous nature of Llama 3 16K-GPU training
make it less fault-tolerant"* matched **0** chunks under AND versus **923**
under OR. OR then matches ~16% of the corpus, so `ts_rank_cd` scores and sorts
~16k rows at 100k. The GIN index finds candidates fast; scoring them is the
cost.

So the default stayed `bm25`. `postgres_fts` is kept and wired for A/B because
its advantages are real but operational, not latency: no per-worker memory, no
rebuild stall after ingestion, correct across multiple workers.

Neither is acceptable at 100k — 189ms and 423ms both dominate the retrieval
budget, against a ~4.6ms HNSW dense side. The fix at that scale is a
purpose-built lexical index (pg_search / ParadeDB, or an external engine),
not tuning either of these.

## Semantic cache lookup

A cache hit measured ~49ms, and the suspected culprit was the `KEYS` glob —
an O(keyspace) call that blocks Redis. Measuring the lookup by cache size
showed the real distribution:

| entries | KEYS | MGET | parse+cosine | total |
|---|---|---|---|---|
| 50 | 0.5 ms | 1.1 ms | 4.7 ms | 6.3 ms |
| 250 | 1.0 ms | 4.0 ms | 21.4 ms | 26.4 ms |
| 1,000 | 2.5 ms | 19.3 ms | **102.7 ms** | 124.5 ms |

`KEYS` is 2.5ms of 124.5ms. The cost was JSON-decoding every cached embedding
— 1,000 x 768 floats — on every single query.

Fix: hold the embeddings in a process-local L2-normalised matrix and refetch
only when a Redis version counter moves. Steady state is one small GET for the
version, a matmul, and one GET to confirm the winning entry.

| entries | before | after |
|---|---|---|
| 50 | 6.3 ms | **0.22 ms** |
| 250 | 26.4 ms | **0.24 ms** |
| 1,000 | 124.5 ms | **0.12 ms** |

Lookup no longer scales with cache size.

Two correctness properties the version counter buys, both tested: a second
worker's writes are visible (it bumps the version, every process resyncs), and
a TTL-expired entry is never served — expiry does not bump the version, so the
winning key is confirmed with a GET before its answer is returned.

### What this does not fix

A cache hit is still ~38-58ms, because the embedding call dominates:

    embed         58.50 ms   (98.6% of a 59.30 ms hit)
    KEYS           0.34 ms
    MGET           0.24 ms
    parse+cosine   0.24 ms

Batching shows why: 8 texts cost 65ms and 32 cost 183ms, implying ~26-40ms of
fixed overhead and ~5ms per text. A single embedding is almost entirely HTTP
round-trip to Ollama, not inference.

Removing that means running the encoder in-process, which means either a
different model (re-embedding 5,869 chunks and changing the vector dimension)
or the same model unquantised (whose embeddings would not match the quantised
GGUF vectors already in the index). Against an ~8s end-to-end query dominated
by generation, saving ~50ms is under 1%, so it was not worth the migration
risk. Recorded rather than done.

## Regression check after the Phase 7 changes

HNSW indexing, the `random_page_cost` change, running sparse retrieval on its
own session, and the semantic-cache rewrite all touch the retrieval path. Re-run
on golden-v2, same config (hybrid + cross-encoder + bm25):

| metric | before Phase 7 | after | delta |
|---|---|---|---|
| ndcg@10 | 0.9022 | 0.9022 | 0.0000 |
| mrr | 0.8929 | 0.8929 | 0.0000 |
| recall@10 | 0.7679 | 0.7679 | 0.0000 |
| hit_rate | 0.9286 | 0.9286 | 0.0000 |
| context_precision | 0.0464 | 0.0464 | 0.0000 |
| context_recall | 0.8393 | 0.8393 | 0.0000 |
| answer_correctness | 0.2480 | 0.2472 | -0.0008 |

Retrieval is bit-identical, which is what recall@10 = 1.000 at this corpus size
predicts: HNSW returns exactly what exact search returns. The only movement is
answer_correctness, from generation nondeterminism.

## Configuration sweep — latency vs quality

Rather than guess which pipeline settings to run, 15 configurations were timed
(`app/scripts/latency_sweep.py`) and the five most promising were then scored
for quality on golden-v2. Raw data: `docs/latency-sweep-raw.csv`,
`docs/config-sweep-results.csv`.

TTFT is measured from the streaming endpoint. Answer length is recorded
alongside latency because a configuration can look fast purely by emitting a
shorter answer; ms/char separates throughput from verbosity.

| Config | TTFT p50 | Total p50 | ms/char | NDCG@10 | ans_corr | Verdict |
|---|---|---|---|---|---|---|
| lean1b (1b, tk3, 400ch, np256) | 655 ms | **3,120 ms** | 2.95 | 0.9022 | 0.2405 | Pareto-optimal |
| base1b (1b, tk5) | 901 ms | 3,833 ms | 2.56 | 0.9022 | 0.2323 | dominated by lean1b |
| lean3b (3b, tk3, 400ch, np256) | 708 ms | 4,834 ms | 6.10 | 0.9022 | **0.2809** | Pareto-optimal |
| baseline3b (3b, tk5) | 938 ms | 5,547 ms | 7.32 | 0.9022 | 0.2531 | **dominated by lean3b** |
| norerank3b | **498 ms** | 5,843 ms | 7.16 | 0.7515 | 0.2521 | Pareto-optimal (TTFT only) |

### The previous default was strictly worse than an available configuration

`baseline3b` is dominated by `lean3b` on every axis — faster TTFT (938 -> 708),
faster total (5,547 -> 4,834), higher answer correctness (0.2531 -> 0.2809),
and identical retrieval metrics. No tradeoff, so the defaults were changed.

Re-measured back-to-back on identical questions to confirm (absolute numbers
drift ~20% between sessions, so only paired comparisons are trustworthy):

| | TTFT p50 | Total p50 |
|---|---|---|
| old defaults (tk5, no truncation) | 1,078.9 ms | 7,520.1 ms |
| new defaults (tk3 + 400 chars) | **874.6 ms** | **5,672.2 ms** |

### Three results that contradicted the expectation

**top_k=3 costs nothing.** NDCG@10, MRR, recall@10 and hit_rate are *identical*
at top_k=3 and top_k=5. MRR is 0.89, so the answering chunk is almost always
rank 1 — chunks 4 and 5 were pure prefill cost.

**Prefill is not the TTFT bottleneck; retrieval is.** Truncating chunks to 400
characters moved TTFT 938ms -> 942ms, i.e. not at all. Removing the reranker
moved it 938ms -> 498ms. TTFT here is dominated by the ~500ms rerank stage,
not by prompt size.

**num_predict is not a lever.** The default cap is 2048 tokens but answers
average ~190 tokens, so the cap never binds. Capping at 256 produced *more*
text in one run and was slower. It was therefore left alone rather than
applied as part of the new default.

### Why the fastest TTFT was not chosen

`norerank3b` has the best TTFT on the board (498ms) and is the wrong choice: it
is the *slowest* overall (5,843ms) and drops NDCG@10 by 0.15. It wins only the
metric people quote.

This also reframes the earlier reranker A/B. No reranking at all scores 0.7515
(plain RRF ordering), while the Ollama LLM reranker scored 0.184 — meaning that
reranker was not merely slow, it was **worse than not reranking**. The fix was
not a faster reranker but abandoning a decoder for a task it is bad at.

### Caveat

`answer_correctness` differences of +/-0.04 at n=28 are within noise. The
identical retrieval metrics and the large gaps (0.9022 vs 0.7515) are the
trustworthy signals here.

## Limitations

- Latency n=5 per mode. The p95/p99 columns the benchmark prints are
  arithmetic artifacts at that sample size (`sorted[int(5*0.95)]` is just the
  maximum); n>=20 is needed before p95 means anything.
- n=28 (v2) and n=30 (v1); single run per variant
- Corpus is 2 papers / 2,094 chunks — small, and no HNSW index, so these
  numbers would not transfer to a 100k-chunk corpus unchanged
- The dataset was generated by the same model family under evaluation, a
  known source of bias, and only partially mitigated by hand review
- Generation metrics are heuristic, not judged

## Reproducing

```bash
# Regenerate the dataset (seeded chunk sampling)
curl -X POST localhost:8000/api/v1/datasets/generate \
  -H 'Content-Type: application/json' \
  -d '{"name":"golden-v1","num_questions":30,"version":"1.0"}'

# Apply the hand review
python backend/app/scripts/revise_golden.py

# Run the A/B
curl -X POST localhost:8000/api/v1/experiments \
  -H 'Content-Type: application/json' \
  -d '{"name":"rerank-backend-v2","dataset_id":"<id>",
       "config_a":{"reranker_backend":"ollama"},
       "config_b":{"reranker_backend":"cross_encoder"}}'
```
