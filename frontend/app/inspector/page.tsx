"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { Search, X } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { ChunkScore, QueryTransform, RetrieveResponse } from "@/lib/types";
import { ms } from "@/lib/format";
import { EMPTY_FILTERS, FiltersPanel, toFilters, type FilterState } from "@/components/FiltersPanel";
import { Entry, EmptyState, ErrorState, Field, Figure, Loading, NumberInput, PageHeader, Select, Toggle } from "@/components/ui";

const TRANSFORMS: readonly { value: QueryTransform; label: string }[] = [
  { value: "none", label: "None" },
  { value: "rewrite", label: "Rewrite" },
  { value: "hyde", label: "HyDE" },
  { value: "decompose", label: "Decompose" },
];

/** The four scoring stages, in pipeline order. */
const STAGES = [
  { key: "dense_score" as const, label: "dense", hint: "pgvector cosine similarity, 0 to 1" },
  { key: "sparse_score" as const, label: "sparse", hint: "BM25 term overlap, unbounded and relative" },
  { key: "rrf_score" as const, label: "rrf", hint: "sum of 1 / (k + rank) across the retrievers" },
  { key: "rerank_score" as const, label: "rerank", hint: "cross-encoder relevance" },
];

type StageKey = (typeof STAGES)[number]["key"];

/**
 * Bars are normalised per column, not globally: the four stages live on incompatible scales (cosine 0 to 1,
 * raw BM25, RRF near 0.016). A bar shows standing within this result set; the printed number is the raw value.
 */
function columnMaxes(chunks: ChunkScore[]): Record<StageKey, number> {
  const out = {} as Record<StageKey, number>;
  for (const { key } of STAGES) {
    out[key] = chunks.reduce((m, c) => Math.max(m, c[key] ?? 0), 0);
  }
  return out;
}

function ScoreCell({ value, max }: { value: number | null; max: number }) {
  if (value == null) {
    return <div className="score none"><span className="val">n/a</span></div>;
  }
  const width = max > 0 ? Math.max(2, (value / max) * 100) : 2;
  return (
    <div className="score">
      <span className="bar" aria-hidden="true"><i style={{ width: `${width}%` }} /></span>
      <span className="val">{value < 0.001 && value > 0 ? value.toExponential(2) : value.toFixed(4)}</span>
    </div>
  );
}

function ChunkRow({ chunk, rank, maxes }: { chunk: ChunkScore; rank: number; maxes: Record<StageKey, number> }) {
  const [open, setOpen] = useState(false);
  return (
    <tr>
      <td className="num">{rank}</td>
      <td style={{ minWidth: 280, maxWidth: 460 }}>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <b>{chunk.document_name}</b>
          {chunk.element_type && <span className="chip">{chunk.element_type}</span>}
          {chunk.page_number != null && <span className="mono" style={{ color: "var(--ink-3)" }}>p. {chunk.page_number}</span>}
          <span className="mono" style={{ color: "var(--ink-3)" }} title={chunk.chunk_id}>{chunk.chunk_id.slice(0, 8)}</span>
        </div>
        <p
          style={{ marginTop: 6, fontSize: "0.875rem", color: "var(--ink-2)", ...(open ? {} : { display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" as const, overflow: "hidden" }) }}
        >
          {chunk.content}
        </p>
        <button type="button" className="btn btn-sm" style={{ marginTop: 8 }} aria-expanded={open} onClick={() => setOpen((o) => !o)}>
          {open ? "Collapse" : "Read the whole chunk"}
        </button>
      </td>
      {STAGES.map((s) => (
        <td key={s.key}><ScoreCell value={chunk[s.key]} max={maxes[s.key]} /></td>
      ))}
    </tr>
  );
}

export default function InspectorPage() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<RetrieveResponse | null>(null);
  const [ranQuery, setRanQuery] = useState<string | null>(null);
  const [ranAt, setRanAt] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorStatus, setErrorStatus] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);

  const [topK, setTopK] = useState("20");
  const [rrfK, setRrfK] = useState("60");
  const [useHybrid, setUseHybrid] = useState(true);
  const [useReranker, setUseReranker] = useState(false);
  const [transform, setTransform] = useState<QueryTransform>("none");
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS);

  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async () => {
    const q = question.trim();
    if (!q || busy) return;

    setBusy(true);
    setError(null);
    setErrorStatus(null);
    setResult(null);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const data = await api.retrieve(
        {
          question: q,
          top_k: Math.min(100, Math.max(1, Number(topK) || 20)),
          use_hybrid: useHybrid,
          use_reranker: useReranker,
          query_transform: transform,
          rrf_k: Math.min(200, Math.max(10, Number(rrfK) || 60)),
          filters: toFilters(filters),
        },
        controller.signal,
      );
      setResult(data);
      setRanQuery(q);
      setRanAt(Date.now());
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        setError("Cancelled.");
      } else if (e instanceof ApiError) {
        setError(e.message);
        setErrorStatus(e.status);
      } else {
        setError("Unexpected error");
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }, [question, busy, topK, rrfK, useHybrid, useReranker, transform, filters]);

  const maxes = useMemo(() => columnMaxes(result?.chunks ?? []), [result]);

  const mode = useHybrid
    ? useReranker ? "hybrid + rerank" : "hybrid (dense + BM25 + RRF)"
    : useReranker ? "dense + rerank" : "dense only";

  return (
    <>
      <PageHeader
        title="Retrieval Inspector"
        subtitle="Retrieval without generation. See exactly which chunks come back and what each stage scored them: the fastest way to tell a retrieval problem from a generation problem."
      />

      <Entry kind="Query" source="POST /api/v1/retrieve">
        <div className="stack">
          <div className="row" style={{ alignItems: "center" }}>
            <label className="field" style={{ flex: "1 1 320px" }}>
              <span className="lab"><Search aria-hidden="true" style={{ display: "inline", width: 14, height: 14, verticalAlign: -2 }} /> Query to inspect</span>
              <input className="input" value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === "Enter" && run()} placeholder="Ask in your own words" />
            </label>
            <div style={{ alignSelf: "flex-end" }}>
              {busy ? (
                <button type="button" className="btn btn-danger" onClick={() => abortRef.current?.abort()}><X aria-hidden="true" /> Cancel</button>
              ) : (
                <button type="button" className="btn btn-primary" onClick={run} disabled={!question.trim()}>Retrieve</button>
              )}
            </div>
          </div>

          <div className="row">
            <Field label="Top K" className="w-20"><NumberInput value={topK} onChange={setTopK} min={1} max={100} disabled={busy} /></Field>
            <Field label="RRF k" className="w-20" hint={useHybrid ? undefined : "hybrid off"}><NumberInput value={rrfK} onChange={setRrfK} min={10} max={200} disabled={busy || !useHybrid} /></Field>
            <Field label="Query transform" className="w-40"><Select value={transform} onChange={setTransform} options={TRANSFORMS} disabled={busy} /></Field>
            <div className="flex flex-wrap items-center gap-2" style={{ paddingBottom: 2 }}>
              <Toggle label="Hybrid + RRF" checked={useHybrid} onChange={setUseHybrid} disabled={busy} title="Fuse dense and BM25 rankings with reciprocal rank fusion" />
              <Toggle label="Reranker" checked={useReranker} onChange={setUseReranker} disabled={busy} title="Costs an extra model call; off by default here so you see the raw retrieval order" />
            </div>
          </div>

          <FiltersPanel value={filters} onChange={setFilters} disabled={busy} />
        </div>
      </Entry>

      {error && (
        <div style={{ marginBottom: 24 }}>
          <ErrorState message={error}>
            {errorStatus === 0 && <p>Start the stack with <code>docker compose up -d</code> and check <code>/readyz</code> on the API.</p>}
          </ErrorState>
        </div>
      )}

      {busy && <Loading label={`Retrieving, ${mode}${transform !== "none" ? `, ${transform}` : ""}`} />}

      {!busy && result && (
        <Entry kind="Retrieval" date={ranAt} title={<>“{ranQuery}”</>}>
          <dl className="deflist" style={{ marginBottom: 20 }}>
            <dt>Chunks</dt><dd>{result.chunks.length}</dd>
            <dt>Retrieval time</dt><dd>{ms(result.latency_ms)}</dd>
            <dt>Mode</dt><dd>{mode}</dd>
            {result.query_transformed && (<><dt>Transformed query</dt><dd style={{ fontFamily: "inherit", fontSize: "0.9375rem" }}>“{result.query_transformed}”</dd></>)}
          </dl>

          {result.chunks.length === 0 ? (
            <EmptyState title="No chunks matched" hint={`Nothing came back for “${ranQuery}”. Either no documents are ingested, or your metadata filters excluded every candidate.`} />
          ) : (
            <Figure
              caption={
                <>
                  <b>Retrieved chunks with their score at each stage.</b>{" "}
                  {STAGES.map((s) => `${s.label}: ${s.hint}`).join("; ")}. Each bar is scaled within its own column because the four scales are not comparable; the number is the raw score.
                </>
              }
            >
              <div className="tab-wrap">
                <table className="tab">
                  <caption className="sr-only">Retrieved chunks and their stage scores</caption>
                  <thead>
                    <tr>
                      <th className="num">Rank</th>
                      <th>Chunk</th>
                      {STAGES.map((s) => <th key={s.key}>{s.label}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {result.chunks.map((c, i) => (
                      <ChunkRow key={`${c.chunk_id}-${i}`} chunk={c} rank={i + 1} maxes={maxes} />
                    ))}
                  </tbody>
                </table>
              </div>
            </Figure>
          )}
        </Entry>
      )}

      {!busy && !result && !error && (
        <EmptyState title="Run a query to inspect retrieval" hint="Calls POST /api/v1/retrieve, the same pipeline the playground uses, but stopping before generation." />
      )}
    </>
  );
}
