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

## Multi-hop and entity-graph retrieval (Phase 8)

Every question in golden-v1 and golden-v2 is answered by one chunk. Phase 8
added two strategies for questions that need two. Each is selected with
`strategy` on `/query` and `/retrieve`, or `retrieval_strategy` in an
experiment config.

- **agentic**:
  1. One LLM call splits the question into sub-questions.
  2. Each sub-question runs the standard hybrid + rerank pipeline against
     itself, concurrently.
  3. The LLM sees the best evidence so far and either replies DONE or writes
     one follow-up search.
  4. The hops are merged.

  `query_transform="decompose"` is the same without step 3. Before this
  change, decompose kept the first sub-question and silently dropped the rest.
- **graph**: adds a third arm to the RRF fusion, a personalized PageRank over
  a chunk–entity graph. The graph has 5,869 chunks, 5,990 entities and 20,811
  edges. It is built in-process without an LLM and cached like the BM25 index.

### Measuring it: multihop-v1 and full_support@10

`eval-datasets/multihop-v1.jsonl` holds 18 two-hop questions. A build check
confirmed that every fact sits in exactly one chunk and that no chunk holds
both halves of a question (see `eval-datasets/README.md`).

**New metric.** `full_support@10` is 1 when both gold chunks are in the top 10.
`recall@10` gives a question with one of its two facts found 0.5, which
overstates an answer that needs both.

**Corrected metric.** `recall@10` now counts the distinct gold contexts it
matched. It used to count matching chunks, so a duplicate of one gold chunk
could stand in for the other, and the corpus holds 187 groups of byte-identical
chunks. The golden-v2 baseline did not move: recall@10 is 0.7679 under both
definitions.

### First design: RRF over the hops made ranking significantly worse

Measured with the experiments API (paired bootstrap; the config is otherwise
identical):

| Dataset | Metric | standard | agentic v1 | p |
|---|---|---|---|---|
| multihop-v1 (n=18) | ndcg@10 | 0.663 | 0.553 | 0.034 |
| multihop-v1 | full_support@10 | 0.556 | 0.444 | 0.216 |
| golden-v2 (n=28) | ndcg@10 | 0.902 | **0.765** | <0.001 |
| golden-v2 | mrr | 0.893 | **0.713** | <0.001 |

On golden-v2, recall@10 and hit rate did not change. The right chunks were
still retrieved; they were ranked lower.

The per-question results showed why. RRF rewards agreement, and sub-questions
about one topic agree on generic chunks. "Figure 1: The Transformer - model
architecture" sat at rank 2 or 3 for several sub-questions. Three appearances at
1/(60+2) beat the 1/(60+1) of the chunk that actually answered one
sub-question, so the generic chunk took the top slot.

### Second design: a merge that protects the original ranking

The original question's top 2 chunks keep their places. Each other hop
contributes only its single best chunk, in hop order. The rest of the original
ranking fills the remaining slots. Nothing accumulates across hops, so a hub
chunk cannot climb.

This was measured with `app/scripts/retrieval_ab.py`. It runs retrieval only,
with the same metric functions and the same paired bootstrap as the
experiments API. Full experiments with generation ran the 16 GB development
laptop out of memory. On the standard pipeline the script reproduces the
experiments API's numbers exactly (multihop-v1 0.6630, golden-v2 0.9022).

| Dataset | Metric | standard | agentic v2 | p |
|---|---|---|---|---|
| multihop-v1 (n=18) | ndcg@10 | 0.663 | 0.684 | 0.50 |
| multihop-v1 | recall@10 | 0.667 | 0.722 | 0.27 |
| multihop-v1 | full_support@10 | 0.556 (10/18) | 0.611 (11/18) | 0.72 |
| golden-v2 (n=28) | ndcg@10 / mrr | 0.902 / 0.893 | 0.902 / 0.893 | 1.0 |
| golden-v1 (n=30, held out) | ndcg@10 / mrr | 0.921 / 0.917 | 0.921 / 0.917 | 1.0 |

**Ablation on multihop-v1** (decompose vs agentic): ndcg@10 is 0.698 vs 0.659
(p = 0.21), and full_support@10 is 0.611 for both. The LLM follow-up step costs
a call and a retrieval and adds nothing measurable here.

**Graph arm** (experiments API):

- golden-v2 is bit-identical to standard (ndcg@10 0.9022).
- On multihop-v1, ndcg@10 goes from 0.663 to 0.669 (p = 0.074) and
  full_support@10 is unchanged.

The entity arm adds candidates, but the cross-encoder reorders the union back
to the same top 10.

**Latency** (retrieval only, median over the runs above, on a laptop short of
memory, so the absolute numbers run high):

| Strategy | Median retrieval |
|---|---|
| standard | 1.6–1.8 s |
| decompose | 8.7 s |
| agentic | 11.5–12.3 s |

On a warm, unloaded machine the same question took 583 ms standard and
6,571 ms agentic. Generation takes another ~5–8 s, so agentic roughly doubles
end-to-end time.

### What this shows and what it does not

- The regression was real and its cause is documented. The merge removes it on
  both single-hop sets, including golden-v1, which was not consulted while the
  merge was designed.
- The multi-hop gain is one question in 18 and not significant. The merge was
  designed after inspecting failures on multihop-v1 and golden-v2, so if
  anything that gain is optimistic. Other caveats:
  - the questions were written with Claude Code
  - each variant ran once
  - decomposition runs at temperature 0.3 on a 3B model
- Neither strategy is the default. Settling whether multi-hop retrieval helps
  needs a larger multi-hop set and a stronger decomposition model than 3B.

## Answers, latency and multi-hop (Phase 9)

Phase 8 left four open problems: answer scores that looked poor (correctness
~0.25, faithfulness ~0.24), ~8 s end to end, multi-hop questions that were
answered from one of their two facts, and statistics built on n=5 latency runs
and test sets of 18-30. Most of the first problem turned out to be the
measurement, so the measurement was fixed first and a new baseline was taken
before any system change.

### What the old numbers were measuring

| Defect | Effect |
|---|---|
| Faithfulness was token-set F1 between the answer and all 20 retrieved chunks joined | a grounded answer scored ~0.2, and "hallucination = 1 - faithfulness" reported ~80% for everything |
| Tokens compared raw ("GPU." vs "GPU") | correctness and faithfulness understated |
| The eval generated from all 20 retrieved chunks; `/query` sends 3 | answer scores and prompt sizes described a request production never makes |
| The eval retrieved `top_k=20`, which reranks 80 candidates; `/query` (`top_k=3`) reranks 12 | retrieval took ~3 s in the eval and ~0.5 s in production, and the eval's top 3 came from a better ranking than production's |
| `generation_max_chunk_chars=400` cut every chunk before the prompt | the answer was cut out of 10 of the 90 golden-v3 questions before the model saw it, to save ~190 ms |
| Latency from 5 queries; no p95 | not enough data for a percentile |

`app/scripts/answer_eval.py` now retrieves exactly as `/query` does, scores the
answer against the context the model actually saw, and grades it three ways:

- **Hosted judge** (`app/eval/answer_judge.py`, `gemini-3.5-flash-lite`, free
  tier): correctness against the reference and faithfulness against the
  context, as labels (correct/partial/incorrect, faithful/minor/unfaithful),
  four answers per request. Its first rubric credited "the context does not
  say" as correct when the context really lacked the answer. The rubric now
  states that this is incorrect (faithfulness credits it), and every run below
  was graded with the fixed rubric.
- **Fact recall**, deterministic: each question lists the facts a correct
  answer must state; `fact_recall_all` is 1 when the answer states all of them.
  On the 288 graded test answers below, the judge's "correct" and this check
  agree 94.8% of the time (89% under the old rubric).
- **Latency** per question: retrieval, time to first token (retrieval plus
  prompt processing) and total, as p50 and p95 with bootstrap 95% intervals.

Every mean has a bootstrap 95% interval, and before/after pairs use the
paired bootstrap from the experiments API.

### New sets, split before any run

`golden-v3` (28 dev, 62 test) and `multihop-v2` (16 dev, 34 test) were drafted
by Gemini and kept only if code-level checks passed: each fact appears in its
gold chunk and in no other chunk, no question contains its answer, and no
single chunk answers a two-passage question. Construction and hand review are
in `eval-datasets/README.md`. Prompts and retrieval were tuned on the dev sets
only; each configuration below ran once on the test sets.

### Before and after, held-out test sets

"Before" is the production configuration at the start of this phase:
verbose prompt, 400-character chunks, `llama3.2:3b` with 3 of its 29 layers on
the CPU. "After" changes only what is listed in the next section.

golden-v3-test (62 single-fact questions):

| | judge correct | judge faithful | all facts stated | answer in top 3 | total p50 | total p95 |
|---|---|---|---|---|---|---|
| before | 0.79 [0.68, 0.89] | 0.92 [0.86, 0.98] | 0.79 | 0.90 | 6,400 ms | 11,676 ms |
| after, standard | 0.86 [0.78, 0.94] | 0.86 [0.77, 0.94] | 0.85 | 0.90 | **1,664 ms** | **2,701 ms** |
| after, agentic | **0.90** [0.83, 0.97] | 0.86 [0.78, 0.93] | **0.90** | 0.94 | 3,929 ms | 5,817 ms |

multihop-v2-test (34 two-passage questions):

| | judge correct | judge faithful | all facts stated | both passages in top 3 | total p50 | total p95 |
|---|---|---|---|---|---|---|
| before | 0.31 [0.22, 0.40] | 0.88 [0.78, 0.97] | 0.03 | 0.03 | 7,314 ms | 10,352 ms |
| after, standard | 0.47 [0.41, 0.53] | 0.81 [0.66, 0.93] | 0.03 | 0.03 | 1,913 ms | 2,779 ms |
| after, agentic | **0.88** [0.81, 0.94] | 0.93 [0.84, 0.99] | **0.74** | **0.79** | 5,394 ms | 7,424 ms |

Paired against "before" (bootstrap p):

- Standard: total time 6.8 s to 1.7 s mean on golden-v3-test (p<0.001),
  7.1 s to 2.0 s on multihop-v2-test (p<0.001). Correctness +0.07 on
  golden-v3-test (p=0.14, not significant), +0.16 on multihop-v2-test (p<0.001).
- Agentic: correctness +0.11 on golden-v3-test (p=0.012) and +0.57 on
  multihop-v2-test (p<0.001); both passages reach the model in 27 of 34
  two-passage questions instead of 1 (p<0.001). It is still faster than the
  old standard path, 4.1 s and 5.4 s mean.
- Judge faithfulness: -0.07 standard (p=0.19), -0.06 agentic (p=0.20) on
  golden-v3-test; neither is significant, and the cause is below.

### What changed

**Generation (latency and answers).**

- The 3B model ran with 3 of 29 layers on the CPU: with the embedder loaded,
  Ollama had 2.5 GiB of VRAM and the f16 KV cache for a 4,096-token context
  needed about 450 MiB of it. `OLLAMA_KV_CACHE_TYPE=q8_0` (flash attention was
  already on) halves the cache to 238 MiB, and `ollama ps` now shows
  `100% GPU` with the context unchanged. The context stays at 4,096 because
  seven code paths call the same model, and a per-request size would make
  Ollama reload it whenever two of them alternate.
- The prompt asked for "a well-structured answer" and a closing list of
  sources, so answers ran 150-300 tokens. It now asks for the direct answer in
  one to three sentences with inline citations: p50 output 152 to 42 tokens.
- Chunks are no longer truncated (`generation_max_chunk_chars` 400 to 0), and
  `generation_num_predict` is 512 instead of 2,048.
- Retrieval itself was not changed. It got faster (p50 535 to 424 ms) once the
  generator stopped running layers on the CPU, which the reranker shares.

**Multi-hop (agentic and decompose strategies).**

- `/query` defaults to `top_k=3`, and the merge kept the original question's
  top 2 chunks first, so a two-part question had room for one hop's chunk and
  the second fact never reached the model. The merge now protects one chunk.
- The decomposition prompt asked for "2-4 sub-questions"; it now asks for one
  search per fact. Its first version included a concrete example (BERT's
  batch size), and llama3.2:3b returned that example verbatim as the
  sub-questions for 4 of 10 dev questions. The example now uses placeholders,
  and any sub-question that shares no content word with the question is
  dropped.
- A further rule ("split questions joined by *and*") made dev results worse
  (both passages present 0.69 to 0.56) and was reverted.

On multihop-v2-dev the decomposed hops found their own passage at rank 1 in
almost every case where the model split the question. The remaining misses are
questions it returned unsplit.

### What this shows and what it does not

- Most of the latency was the model running partly on the CPU and writing long
  answers. Retrieval was ~0.5 s throughout; the 3 s in earlier eval runs came
  from reranking 80 candidates that production never reranks.
- Agentic retrieval now helps on held-out data, significantly, on both sets.
  It adds ~2.3 s (one decomposition call, one follow-up call, three
  retrievals), so it is opt-in (`strategy=agentic`); the default stays
  `standard`.
- **Faithfulness did not improve.** Reading the golden-v3-test answers the
  judge marked unfaithful shows two patterns the concise prompt introduced:
  the instruction "if it answers only part, say which part is missing" makes
  the 3B model append a false "the exact figure is not specified" to complete
  answers, and with no answer in the context it sometimes guesses where the
  verbose prompt hedged. Because this was found on the test set, the prompt was
  not changed afterwards: a fix has to be tuned on dev and confirmed on a
  fresh test split.
- The merge now protects one chunk instead of two. Phase 8 measured the
  two-chunk merge on golden-v2 (no regression); the one-chunk merge has only
  been measured on golden-v3, where agentic put the answer in the top 3 for
  0.94 of test questions against 0.90 for standard.
- Caveats: the questions were drafted by an LLM; the judge is a small hosted
  model (checked against the deterministic fact recall, not against people);
  each configuration ran once, and generation and decomposition run at
  temperature 0.3; 62 and 34 questions give intervals of roughly ±0.1; one
  machine (RTX 3050 Laptop, 4 GB VRAM).

Per-question results, including the context each answer was generated from,
are in `eval-datasets/results/` (`baseline-*`, `after-*`, `after-agentic-*`).

## Public benchmarks (NanoBEIR, MultiHop-RAG)

The golden sets above are small and come from RAGScope's own corpus, so the
production retrieval pipeline also runs on two public benchmarks with published
results: `app/scripts/public_bench.py`. Each benchmark has its own `bench_*`
database, and the script refuses any other, so no public data reaches the main
corpus.

The first run (2026-11-04, the code as it stood) fell short of the published
numbers, and finding out why turned up two bugs and a better reranker. Every
change after that was decided on development data, and the held-out sets were
run once more at the end (2026-11-05):

| Split | Sets |
|---|---|
| Dev | NanoSciFact, NanoFiQA2018, NanoHotpotQA; MultiHop-RAG's last 250 non-null queries; golden-v1, golden-v2, golden-v3-dev, multihop-v1, multihop-v2-dev |
| Held-out | NanoMSMARCO, NanoNQ, NanoNFCorpus (the three with published numbers for the reranker); all 2,255 non-null MultiHop-RAG queries, which include the 250 dev queries, so results on the other 2,005 are given too (the first 250 for agentic retrieval); golden-v3-test, multihop-v2-test |

A change shipped only if it helped the public dev sets and did not hurt
RAGScope's own. The held-out sets did point at the problems first (the
2026-11-04 NanoNFCorpus gap is how the 800-character cut was found), but nothing
was chosen or tuned on them after that.

### What changed, and what was dropped

NanoBEIR dev, production NDCG@10 (SciFact / FiQA / HotpotQA). Each row is on
top of the kept rows above it:

| Change | NDCG@10 | Decision |
|---|---|---|
| Before (the 2026-11-04 code) | 0.723 / 0.483 / 0.879 | |
| 1. The reranker reads whole passages (up to 512 tokens), not the first 800 characters | 0.740 / 0.500 / 0.876 | kept |
| 2. BM25 tokenizes with Postgres's English lexemes, not `lower().split()` | 0.745 / 0.515 / 0.886 | kept |
| nomic-embed-text task prefixes (`search_query:` / `search_document:`) | 0.742 / 0.514 / 0.886 | dropped: no gain |
| Blend the reranker score with the first-stage rank | 0.779 / 0.523 / 0.904 | rejected: golden-v2 NDCG@10 -0.013, multihop-v1 -0.041 |
| 3. Reranker `cross-encoder/ettin-reranker-32m-v1` instead of `ms-marco-MiniLM-L-6-v2` | 0.754 / 0.555 / 0.936 | kept |

1. **The 800-character cut.** The cross-encoder scored only the first 800
   characters of each passage, on the theory that relevance shows early. That
   hid the end of 93% of NanoNFCorpus documents and 96% of MultiHop-RAG chunks.
   RAGScope's own chunks are at most 511 characters, so its own sets could not
   show it.
2. **BM25 tokenization.** `SparseRetriever` split on whitespace, so punctuation
   stayed attached ("broccoli?" never matched "broccoli") and nothing was
   stemmed. It now indexes `to_tsvector('english', ...)`, the lexemes Postgres
   full-text search already stores, each repeated once per position so term
   frequencies survive. BM25 alone on dev went from 0.576 / 0.231 / 0.671 to
   0.714 / 0.423 / 0.838.
3. **The reranker.** On MultiHop-RAG the rest of the gap to the papers was at
   the top of the ranking (MRR@10), and both papers rerank with
   bge-reranker-large (560M). Six cross-encoders were compared on the dev sets,
   each reranking the same candidates (production's 40 per query, captured
   once; there MiniLM reproduced its production scores to within one query):

   | Reranker | Params | CPU, 40 chunks | MultiHop-RAG dev MRR@10 / MAP@10 / Hits@10 / Hits@4 | NanoBEIR dev |
   |---|---|---|---|---|
   | ms-marco-MiniLM-L-6-v2 (before) | 23M | 2.1 s | 0.528 / 0.239 / 0.820 / 0.652 | 0.745 / 0.515 / 0.886 |
   | ettin-reranker-17m-v1 | 18M | 1.2 s | 0.635 / 0.308 / 0.904 / 0.780 | 0.770 / 0.565 / 0.942 |
   | **ettin-reranker-32m-v1** | 33M | 2.9 s | **0.667 / 0.329 / 0.908 / 0.784** | 0.754 / 0.555 / 0.936 |
   | ettin-reranker-68m-v1 | 69M | 9.0 s | 0.682 / 0.336 / 0.932 / 0.804 | 0.747 / 0.591 / 0.945 |
   | ettin-reranker-150m-v1 | 151M | not timed | 0.688 / 0.344 / 0.928 / 0.804 | 0.764 / 0.614 / 0.948 |
   | bge-reranker-large (the papers') | 560M | not timed | 0.658 / 0.315 / 0.884 / 0.764 | 0.766 / 0.526 / 0.947 |

   CPU times are medians over 20 MultiHop-RAG queries on 4 CPUs, run the way
   production runs them. 32m was picked over 17m, which lost recall on
   multihop-v1 (0.667 to 0.556, p = 0.03), and over 68m (3x the CPU time) and
   150m, which add +0.015 and +0.021 MRR@10. On RAGScope's dev sets, each
   run the way it is served, 32m raised golden-v3-dev NDCG@10 from 0.917 to
   1.000 (p = 0.004) and multihop-v2-dev full support from 0.63 to 1.00
   (p < 0.001), and left golden-v2 within noise (0.902 to 0.890, p = 0.71). On
   multihop-v1 (agentic) NDCG@10 rose from 0.701 to 0.779 (p = 0.018) while full
   support fell from 10 to 8 of 18 questions (p = 0.29). Passages stay capped
   at 512 tokens: uncapped, ettin scored no better on the long-document dev sets
   and ran 1.5x slower. The ettin-reranker-v1 models are distilled from
   mxbai-rerank-large-v2 on a mix that includes MS MARCO, NQ, HotpotQA, FiQA and
   FEVER training data (MiniLM's includes MS MARCO), and not MultiHop-RAG.
4. **Agentic merge: unchanged.** Ammann et al. rerank the union of all
   sub-query results against the original question. With the hops replayed from
   recorded sub-queries, so the LLM's randomness is held fixed, that union, a
   max-score variant and protecting the original question's top 2 were compared
   with the shipped merge (the original's top 1, then each hop's best). None beat
   it on every dev set under either reranker. With MiniLM the union lost
   MultiHop-RAG MAP@10 (0.270 to 0.242, p < 0.001).

### NanoBEIR (held-out): NDCG@10

NanoBEIR is a 50-query subset of each BEIR dataset (binary relevance), so the
95% intervals are about +/-0.1 and p-values are paired over the 50 queries.

| Mode | NanoNFCorpus | NanoMSMARCO | NanoNQ |
|---|---|---|---|
| BM25 (rank-bm25): before / after | 0.253 / **0.347** | 0.429 / 0.485 | 0.341 / **0.530** |
| BM25, published | 0.325 | 0.541 | 0.501 |
| Dense (nomic-embed-text, unchanged) | 0.321 | 0.557 | 0.659 |
| Hybrid (RRF): before / after | 0.316 / 0.365 | 0.569 / 0.562 | 0.546 / 0.648 |
| Production, before (MiniLM, 800-character cut) | 0.322 | 0.663 | 0.708 |
| Production, the two fixes with MiniLM (ablation) | 0.357 | 0.663 | 0.744 |
| **Production, after (the fixes and ettin-32m)** | 0.337 | **0.707** | **0.767** |
| Published: MiniLM reranking BM25's top 100 plus every relevant document BM25 missed | 0.393 | 0.669 | 0.760 |
| That protocol here: MiniLM | 0.393 | 0.669 | 0.760 |
| That protocol here: MiniLM with the 800-character cut | 0.380 | 0.669 | 0.736 |
| That protocol here: ettin-32m | 0.380 | 0.707 | 0.780 |

The published rows come from sentence-transformers' `CrossEncoderNanoBEIREvaluator`
as reported on the model card of a fine-tune of
`cross-encoder/ms-marco-MiniLM-L6-v2`
([ayushexel/reranker-ms-marco-MiniLM-L6-v2-gooaq-bce](https://huggingface.co/ayushexel/reranker-ms-marco-MiniLM-L6-v2-gooaq-bce)):
the cross-encoder row is its pre-training evaluation (the base model), and the
BM25 row is the evaluator's BM25 baseline. `public_bench.py reference` runs that
protocol: BM25's top 100 from the evaluator's own lists, plus every relevant
document it missed, on document text without titles.

- **The harness reproduces the published numbers.** With that protocol the
  MiniLM reranker scores 0.393 / 0.669 / 0.760, the published values to three
  decimals. With the old 800-character cut it loses 0.013 and 0.024 on the two
  sets with long documents and nothing on NanoMSMARCO, whose passages are short.
  The cut, not the wiring, caused the 2026-11-04 gap.
- **Production scores above the published reranker numbers on NanoMSMARCO
  (+0.038) and NanoNQ (+0.007) without being handed the relevant documents,
  but on 50 queries neither gap is significant:** paired by query with the
  reproduced MiniLM rows, p = 0.14 (95% interval -0.016 to +0.091) and
  p = 0.71 (-0.026 to +0.042). After the tokenizer fix BM25 is above the
  published BM25 on NanoNFCorpus (+0.022, p = 0.076) and NanoNQ (+0.029,
  p = 0.26) and below it on NanoMSMARCO (0.485 against 0.541, p = 0.046),
  paired with the evaluator's own BM25 lists. Against before: production
  +0.016 / +0.044 / +0.059 (p = 0.54 / 0.11 / 0.032), BM25
  +0.094 / +0.056 / +0.189 (p < 0.001 / 0.28 / < 0.001).
- **NanoNFCorpus stays below the published row** (0.337 against 0.393,
  p = 0.04). Its queries have 50 relevant documents on average (median 23.5)
  against about one for the other two, so a protocol that adds every one of
  them to the candidates gains most there; on that protocol ettin is no better
  than MiniLM on this medical set (0.380 against 0.393, p = 0.60). On this set the fixes with MiniLM
  (0.357) score above the shipped reranker (p = 0.34), and hybrid alone scores
  0.365.
- On NanoNQ the fixes give +0.037 and the reranker +0.023 more; on NanoNFCorpus
  the fixes give +0.036 (p < 0.001). On NanoMSMARCO it is all the reranker
  (+0.044): the passages are short, and BM25's gain there does not survive
  fusion.

### MultiHop-RAG (held-out)

MultiHop-RAG ([Tang and Yang, 2024](https://arxiv.org/abs/2401.15391)) has 609
news articles and 2,556 queries that each need evidence from 2-4 articles; the
2,255 non-null queries are scored. Articles are split into ~256-token sentence
windows (190 words, 60-word overlap), each prefixed with the article's title,
source and date, and scored with the logic of the benchmark's
`retrieval_evaluate.py` (a chunk is relevant when an evidence sentence, with
spaces and newlines removed, is a substring of it). "All evidence" is
RAGScope's `full_support@10`: every evidence sentence in the top 10.

| Run | Queries | MRR@10 | MAP@10 | Hits@10 | Hits@4 | All evidence | Time per query |
|---|---|---|---|---|---|---|---|
| Tang and Yang: voyage-02 + bge-reranker-large (their best) | | 0.586 | | 0.747 | 0.663 | | |
| Ammann et al.: bge-large-en-v1.5 | eval split | 0.464 | 0.217 | 0.781 | 0.611 | | 0.03 s |
| Ammann et al.: + bge-reranker-large | eval split | 0.574 | 0.274 | 0.781 | 0.687 | | 0.88 s |
| Ammann et al.: + LLM query decomposition | eval split | 0.498 | 0.238 | 0.810 | 0.655 | | 16.7 s |
| Ammann et al.: both (their best) | eval split | 0.635 | 0.322 | 0.872 | 0.763 | | 18.9 s |
| RAGScope BM25 alone, after the fix | 2,255 | 0.629 | 0.318 | 0.886 | 0.755 | 0.304 | 0.05 s |
| RAGScope production, before | 2,255 | 0.543 | 0.251 | 0.835 | 0.682 | 0.148 | 2.4 s |
| **RAGScope production, after** | 2,255 | **0.731** | **0.378** | **0.936** | **0.841** | **0.389** | 3.8 s |
| RAGScope production, after, without the dev queries | first 2,005 | 0.738 | 0.385 | 0.940 | 0.848 | 0.398 | |
| RAGScope agentic, before | first 250 | 0.569 | 0.296 | 0.860 | 0.704 | 0.272 | 10.2 s |
| **RAGScope agentic, after** | first 250 | **0.774** | **0.419** | **0.948** | **0.880** | **0.468** | 9.7 s |

The Tang and Yang rows are Table 5 of their paper; Ammann et al.'s
([2025](https://arxiv.org/abs/2507.00355)) are Table 1 and Table 4 (latency
over 250 queries on A100 GPUs). Full-set intervals are about +/-0.02
(production MRR@10 after: [0.715, 0.745]); on 250 queries about +/-0.05.
The full set's last 250 queries are the dev set the changes were chosen on.
Without them production scores higher, MRR@10 0.738 [0.723, 0.755]: the dev
queries are harder (0.668), so they pull the full-set numbers down rather than
up, and every published value stays outside the 2,005-query intervals.

- **Single-pass production retrieval now beats the best published
  configuration on every metric,** including Ammann et al.'s, which adds LLM
  query decomposition (18.9 s a query on an A100) to a 560M reranker:
  MRR@10 +0.096, MAP@10 +0.056, Hits@10 +0.064, Hits@4 +0.078, each outside
  the full-set interval. Against Tang and Yang's best: MRR@10 +0.145, Hits@10
  +0.189, Hits@4 +0.178.
- **Where it came from**, paired on the first 250 queries: the two fixes with
  MiniLM, MRR@10 0.541 to 0.578 (p = 0.04) and Hits@10 0.832 to 0.864
  (p = 0.15); then the reranker swap, MRR@10 0.578 to 0.774 and Hits@4 0.700 to 0.868
  (p < 0.001 for both). The before side is a rerun of the 2026-11-04 code:
  production matched that run exactly; agentic, which samples its
  sub-questions, gave MRR@10 0.563.
- **BM25 alone**, after the tokenizer fix, is level with Ammann et al.'s best:
  within 0.01 on MRR@10, MAP@10 and Hits@4 and 0.014 above on Hits@10, each
  published value inside BM25's 95% interval (Hits@10's at its lower edge,
  [0.872, 0.899]). It beats every other published row. MultiHop-RAG's questions name the outlets, people and
  products their evidence mentions, which is what lexical matching is for.
- **Agentic retrieval** (first 250 queries) adds MAP@10 +0.018 and all-evidence
  +0.052 over production on the same queries (p < 0.001 for both), not MRR@10
  (+0.001, p = 0.93), for ~6 s more a query. Against before: MRR@10 0.569 to 0.774, all evidence
  0.272 to 0.468.
- **MAP@10 is compared with Ammann et al. only.** Their values sit on the scale
  of the released `retrieval_evaluate.py`; Tang and Yang's (0.31-0.48) are higher
  relative to their MRR than that script produces on any run here, so how they
  were computed could not be reproduced.
- Same queries, corpus and scoring logic, but not the same chunker. Like
  RAGScope, the benchmark's own `simple_retrieval.py` attaches each article's
  title, source and date to every chunk and uses ~256-token chunks, but its
  chunks overlap by 25 tokens where RAGScope's overlap by ~80. Ammann et al. do
  not size their evaluation split or name their chunker.

### RAGScope's own held-out sets

The same comparison on golden-v3-test (62 single-fact questions, served
single-pass) and multihop-v2-test (34 two-passage questions, served agentic),
judged as in Phase 9 and paired against the Phase 9 results:

| Set | Metric | Phase 9 | After | p |
|---|---|---|---|---|
| golden-v3-test (62, single-pass) | Context recall | 0.903 | 0.952 | 0.20 |
| | Judged correctness | 0.863 | 0.879 | 0.69 |
| | Judged faithfulness | 0.855 | 0.863 | 0.87 |
| multihop-v2-test (34, agentic) | Both passages retrieved | 0.794 | 0.912 | 0.094 |
| | Judged correctness | 0.882 | 0.912 | 0.53 |
| | Judged faithfulness | 0.926 | 0.912 | 0.96 |
| | Every required fact in the answer | 0.735 | 0.824 | 0.20 |

Retrieval improved on both sets and correctness with it, none of it
significant at 34-62 questions: Phase 9 had left little room (two-passage
retrieval was already 27 of 34, now 31). Faithfulness on multihop-v2-test fell
by 0.015 (p = 0.96), inside the noise. A second run of the same code, made
for the latency figures below, retrieved both passages for 30 of 34 with
judged correctness 0.882: agentic retrieval samples its sub-questions, so
these move by a question or two between runs.

### Hardware and latency

The published latencies are on A100 GPUs. RAGScope runs on one laptop: an RTX
3050 (4 GB) for Ollama and 4 CPUs for the reranker. To fit one night, the
held-out runs above put the reranker on the GPU next to Ollama, so their
timings are not the deployed ones; offline, the GPU reproduced the CPU's dev
scores to within one query. The times in the tables come from separate runs
with the reranker on the CPU, as deployed: MultiHop-RAG production 3.8 s a
query at p50 (first 100 queries; 2.4 s before, when the reranker read 800
characters per chunk) and agentic 9.7 s (first 40). The CPU run's production
scores on its 100 queries equal the GPU run's on every metric. On RAGScope's
own corpus, whose chunks are short, end-to-end p50 did not move: 1.66 s to
1.67 s on golden-v3-test and 5.37 s to 5.17 s on multihop-v2-test.

Two held-out runs were cut off by Docker errors (a DNS failure reaching
Postgres in the agentic run, a lost container in the judged agentic run) and
were rerun; a cut-off run writes no result, so nothing was chosen between runs.

Per-mode results with intervals and per-query rows are in
`eval-datasets/results/public/` (tags: `base` before, `final` after, `minilm`
the ablation, `cpu` the latency runs; `*-reference.json` the published
protocol). The dev-stage evidence is in `eval-datasets/results/dev/`:
`reranker-compare-dev.json` (per-query scores for the six rerankers),
`reranker-uncapped-dev.json` (ettin without the 512-token cap) and
`golden-retrieval-*.json` (RAGScope's golden sets under each candidate fix).

## Limitations

The bullets below describe Phases 1-8. Phase 9 measures latency over 62 and 34
questions per configuration, with p95 intervals, and grades answers with a
hosted judge and a deterministic fact check.


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

# Phase 9: build and check the sets, then run and compare (inside the backend container)
python -m app.scripts.build_golden single --n 100 --out /eval-datasets/golden-v3
python -m app.scripts.build_golden multi --source /eval-datasets/golden-v3-test.jsonl --n 48 --out /eval-datasets/multihop-v2-test
python -m app.scripts.build_golden check /eval-datasets/golden-v3-test.jsonl /eval-datasets/multihop-v2-test.jsonl
python -m app.scripts.answer_eval run /eval-datasets/golden-v3-test.jsonl --label after --out /eval-datasets/results/after-golden-v3-test.jsonl
python -m app.scripts.answer_eval run /eval-datasets/multihop-v2-test.jsonl --label agentic --strategy agentic --out /eval-datasets/results/after-agentic-multihop-v2-test.jsonl
python -m app.scripts.answer_eval compare /eval-datasets/results/baseline-golden-v3-test.jsonl /eval-datasets/results/after-golden-v3-test.jsonl

# Public benchmarks (each in its own bench_* database; data is downloaded and cached on first run).
# The reranker is RERANKER_MODEL (default in app/config.py); an .env that pins it wins, so pass -e to choose.
docker compose exec postgres createdb -U ragscope bench_nanonq
docker compose run --rm -e POSTGRES_DB=bench_nanonq backend python -m app.scripts.public_bench nanobeir NanoNQ --modes bm25 dense hybrid production --tag final
docker compose run --rm -e POSTGRES_DB=bench_nanonq -e RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 backend python -m app.scripts.public_bench reference NanoNQ --tag minilm
docker compose exec postgres createdb -U ragscope bench_multihoprag
docker compose run --rm -e POSTGRES_DB=bench_multihoprag backend python -m app.scripts.public_bench multihop-rag --modes bm25 hybrid production --tag final
docker compose run --rm -e POSTGRES_DB=bench_multihoprag backend python -m app.scripts.public_bench multihop-rag --limit 250 --modes agentic --tag final
docker compose run --rm backend python -m app.scripts.public_bench compare /eval-datasets/results/public/NanoNQ-base-bm25-dense-hybrid-production.json /eval-datasets/results/public/NanoNQ-final-bm25-dense-hybrid-production.json

# Run the A/B
curl -X POST localhost:8000/api/v1/experiments \
  -H 'Content-Type: application/json' \
  -d '{"name":"rerank-backend-v2","dataset_id":"<id>",
       "config_a":{"reranker_backend":"ollama"},
       "config_b":{"reranker_backend":"cross_encoder"}}'
```
