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
| `multihop-v1.jsonl` | 18 | two-hop questions written from chosen chunk pairs, with automated checks (below) | multi-hop dev set: tuning only |
| `golden-v3-dev.jsonl` | 28 | drafted by Gemini, kept only if code-level fairness checks pass, then hand-reviewed (`app/scripts/build_golden.py`) | single-hop dev set: tuning only |
| `golden-v3-test.jsonl` | 62 | same build, other passages | single-hop test set: reported, never tuned on |
| `multihop-v2-test.jsonl` | 34 | two-passage questions joined from golden-v3-test facts, same checks, hand-curated | multi-hop test set: reported, never tuned on |

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

`answer_facts` (golden-v3, multihop-v2, and multihop-v1 since 2026-11-04) lists
the facts a correct answer must state. Each fact is a list of accepted
wordings: the first is copied from the passage, the rest are answer-side
rewordings (`"2.45%"` for `"2.450%"`). `answer_fact_recall` is the share of
facts an answer states, matched as whole terms after one normalization
(`4,000` = `4000`, `22.0` = `22`, `°C` = `degrees Celsius`, `®`/`™` dropped).

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

## golden-v3 (dev and test)

Built on 2026-11-04 with `python -m app.scripts.build_golden single --n 100`.
`gemini-3.5-flash-lite` drafted one question per passage, eight passages per
request, from paragraph chunks of 150 to 511 characters with no duplicate text
in the corpus. Code then kept a draft only if:

- each required fact's passage wording appears in the gold chunk,
- that wording appears in no other chunk (near-duplicates of the gold chunk,
  which the retrieval metrics already count as relevant, are allowed), so a
  retriever that finds the fact somewhere else is never scored wrong,
- the question does not contain any accepted wording of its answer.

Of 511 passages the writer saw, it skipped 279 as having no self-contained
fact, code rejected 138 (125 of them "fact also stated elsewhere"), and 94
passed (25 Llama 3, 19 Attention, 25 IPCC, 25 Apple). A hand review then
dropped 2 (a question about a figure caption, and one whose answer was a
complexity formula), corrected the facts of 4, and added answer-side
rewordings to 8. Every question was split by document into dev (30%) and test
(70%) before any system ran on it.

One defect surfaced only in the first graded run. The judge accepted 1.625%
for "the interest rate of the Notes due in 2026", against a reference of
2.450%, and it was right to: Apple's 10-K lists six notes due 2026 and three
due 2028, so the question names none of them. The fact checks test that an
answer's wording is unique, not that the question has one answer. The 2026
and 2028 note questions were dropped (one test, one dev, and the two
multihop-v2 questions built on the 2026 fact) before the before/after runs
reported in docs/eval-results.md, which use 28 dev and 62 test questions.

Re-check finished files after any hand edit; it runs the same checks:

```bash
python -m app.scripts.build_golden check /eval-datasets/golden-v3-dev.jsonl /eval-datasets/golden-v3-test.jsonl
```

## multihop-v2-test

Built with `build_golden multi --source golden-v3-test.jsonl --n 48`, so its
facts come only from test passages. Gemini was given pairs of single-hop
facts (half from one document, half from two) and asked for one question that
needs both. Code rejected drafts that contain an answer or that one chunk
could answer alone. Of 48 kept drafts, a hand review dropped 3 malformed ones
(two comparisons of unrelated quantities, one garbled question) and 9 more so
that no single fact appears in more than two questions: repeats are correlated
and would make the confidence intervals look tighter than they are.

36 questions remained: 27 within one document, 9 across two. Two more were
dropped with the ambiguous 2026 note question (above), leaving 34. `metadata.kind`
records the style the writer was asked for (23 conjunction, 9 bridge, 4
comparison), but most "bridge" drafts came back as two-part questions, so
treat the set as two-passage questions rather than chained reasoning.

## multihop-v1 and answer facts

multihop-v1's retrieval checks above used long, specific patterns. Its
`answer_facts`, added later for answer scoring, use the short wordings an
answer would contain ("0.1", "Cupertino", "128K tokens"), and many of those
appear in other chunks too: `build_golden check` flags 11 of the 18 rows. That
does not affect its retrieval metrics, and it is a tuning set only.

## results/

Per-question output of `app/scripts/answer_eval.py`, one JSON object per
question, including the formatted context the answer was generated from, so a
run can be re-graded (`answer_eval judge --all`) or audited. The top level
holds the Phase 9 test-set runs reported in `docs/eval-results.md`
(`baseline-*`, `after-*`, `after-agentic-*`). `dev/` holds the tuning runs:
`baseline-*` and `genfix-*` (new generator) on the dev sets,
`agentic-*`/`decompose-*` on multihop-v1 and golden-v3-dev (decomposer before
the relevance guard), `*-multihop-v2-dev` (after it), and `*3-multihop-v2-dev`
(a prompt rule that measured worse and was reverted).
