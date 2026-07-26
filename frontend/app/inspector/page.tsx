"use client";

import clsx from "clsx";
import { useCallback, useMemo, useRef, useState } from "react";
import { ChevronDown, Loader2, Search, X } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { ChunkScore, QueryTransform, RetrieveResponse } from "@/lib/types";
import { ms } from "@/lib/format";
import {
  EMPTY_FILTERS,
  FiltersPanel,
  toFilters,
  type FilterState,
} from "@/components/FiltersPanel";
import {
  Card,
  EmptyState,
  ErrorState,
  Field,
  NumberInput,
  PageHeader,
  Select,
  Toggle,
} from "@/components/ui";

const TRANSFORMS: readonly { value: QueryTransform; label: string }[] = [
  { value: "none", label: "None" },
  { value: "rewrite", label: "Rewrite" },
  { value: "hyde", label: "HyDE" },
  { value: "decompose", label: "Decompose" },
];

/** The four scoring stages, in pipeline order. */
const STAGES = [
  {
    key: "dense_score" as const,
    label: "dense",
    color: "#4cd6fb",
    hint: "pgvector cosine similarity · 0–1",
  },
  {
    key: "sparse_score" as const,
    label: "sparse",
    color: "#00b4d8",
    hint: "BM25 term overlap · unbounded, relative",
  },
  {
    key: "rrf_score" as const,
    label: "rrf",
    color: "#a78bfa",
    hint: "Σ 1/(k + rank) across retrievers",
  },
  {
    key: "rerank_score" as const,
    label: "rerank",
    color: "#00d4aa",
    hint: "LLM cross-encoder relevance · 0–1",
  },
];

type StageKey = (typeof STAGES)[number]["key"];

/**
 * Bars are normalised per column, not globally — the four stages live on
 * incompatible scales (cosine 0–1 vs raw BM25 vs RRF ~0.016). Widths show
 * relative standing within this result set; the printed number is the raw value.
 */
function columnMaxes(chunks: ChunkScore[]): Record<StageKey, number> {
  const out = {} as Record<StageKey, number>;
  for (const { key } of STAGES) {
    out[key] = chunks.reduce((m, c) => Math.max(m, c[key] ?? 0), 0);
  }
  return out;
}

function ScoreBar({
  value,
  max,
  color,
  label,
}: {
  value: number | null;
  max: number;
  color: string;
  label: string;
}) {
  if (value == null) {
    return (
      <div className="flex items-center gap-2 opacity-30">
        <span className="w-14 shrink-0 font-mono text-[10px] text-on-surface-muted">
          {label}
        </span>
        <span className="font-mono text-[10px] text-on-surface-muted">—</span>
      </div>
    );
  }
  const width = max > 0 ? Math.max(2, (value / max) * 100) : 2;
  return (
    <div className="flex items-center gap-2">
      <span className="w-14 shrink-0 font-mono text-[10px] text-on-surface-muted">
        {label}
      </span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-background">
        <div
          className="h-full rounded-full"
          style={{ width: `${width}%`, backgroundColor: color }}
        />
      </div>
      <span className="w-16 shrink-0 text-right font-mono text-[10px] text-on-surface">
        {value < 0.001 && value > 0 ? value.toExponential(2) : value.toFixed(4)}
      </span>
    </div>
  );
}

function ChunkRow({
  chunk,
  rank,
  maxes,
}: {
  chunk: ChunkScore;
  rank: number;
  maxes: Record<StageKey, number>;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-t border-border first:border-t-0">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-start gap-3 px-4 py-3 text-left hover:bg-surface-container-high"
      >
        <span className="mt-0.5 w-6 shrink-0 font-mono text-xs text-on-surface-muted">
          #{rank}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="truncate text-xs text-on-surface">
              {chunk.document_name}
            </span>
            {chunk.element_type && (
              <span className="rounded bg-surface-container-high px-1.5 py-0.5 font-mono text-[9px] text-on-surface-muted">
                {chunk.element_type}
              </span>
            )}
            {chunk.page_number != null && (
              <span className="font-mono text-[10px] text-on-surface-muted">
                p.{chunk.page_number}
              </span>
            )}
            <span className="font-mono text-[9px] text-on-surface-muted opacity-60">
              {chunk.chunk_id.slice(0, 8)}
            </span>
          </div>
          <p
            className={clsx(
              "mt-1 text-[11px] leading-relaxed text-on-surface-muted",
              !open && "line-clamp-2",
            )}
          >
            {chunk.content}
          </p>
          <div className="mt-2 space-y-1">
            {STAGES.map((s) => (
              <ScoreBar
                key={s.key}
                label={s.label}
                value={chunk[s.key]}
                max={maxes[s.key]}
                color={s.color}
              />
            ))}
          </div>
        </div>
        <ChevronDown
          className={clsx(
            "mt-0.5 h-4 w-4 shrink-0 text-on-surface-muted transition-transform",
            open && "rotate-180",
          )}
        />
      </button>
    </div>
  );
}

export default function InspectorPage() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<RetrieveResponse | null>(null);
  const [ranQuery, setRanQuery] = useState<string | null>(null);
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

  const maxes = useMemo(
    () => columnMaxes(result?.chunks ?? []),
    [result],
  );

  const mode = useHybrid
    ? useReranker
      ? "hybrid + rerank"
      : "hybrid (dense + BM25 + RRF)"
    : useReranker
      ? "dense + rerank"
      : "dense only";

  return (
    <div>
      <PageHeader
        title="Retrieval Inspector"
        subtitle="Retrieval without generation. See exactly which chunks come back and what each stage scored them — the fastest way to tell a retrieval problem from a generation problem."
      />

      {/* ── Query + config ── */}
      <Card className="mb-6">
        <div className="flex items-center gap-3">
          <Search className="h-4 w-4 shrink-0 text-on-surface-muted" />
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder="Query to inspect…"
            className="flex-1 bg-transparent text-sm text-on-surface outline-none placeholder:text-on-surface-muted"
          />
          {busy ? (
            <button
              onClick={() => abortRef.current?.abort()}
              className="flex items-center gap-2 rounded-lg border border-border bg-surface-container px-4 py-2 text-sm text-on-surface hover:border-danger/50 hover:text-danger"
            >
              <X className="h-4 w-4" /> Cancel
            </button>
          ) : (
            <button
              onClick={run}
              disabled={!question.trim()}
              className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90 disabled:opacity-40"
            >
              Retrieve
            </button>
          )}
        </div>

        <div className="mt-4 flex flex-wrap items-end gap-x-4 gap-y-3 border-t border-border pt-4">
          <Field label="Top K" className="w-20">
            <NumberInput
              value={topK}
              onChange={setTopK}
              min={1}
              max={100}
              disabled={busy}
            />
          </Field>
          <Field label="RRF k" className="w-20" hint={useHybrid ? undefined : "hybrid off"}>
            <NumberInput
              value={rrfK}
              onChange={setRrfK}
              min={10}
              max={200}
              disabled={busy || !useHybrid}
            />
          </Field>
          <Field label="Query transform" className="w-40">
            <Select
              value={transform}
              onChange={setTransform}
              options={TRANSFORMS}
              disabled={busy}
            />
          </Field>
          <div className="flex flex-wrap items-center gap-2 pb-0.5">
            <Toggle
              label="Hybrid + RRF"
              checked={useHybrid}
              onChange={setUseHybrid}
              disabled={busy}
              title="Fuse dense and BM25 rankings with reciprocal rank fusion"
            />
            <Toggle
              label="Reranker"
              checked={useReranker}
              onChange={setUseReranker}
              disabled={busy}
              title="Costs an extra LLM call — off by default here so you see raw retrieval order"
            />
          </div>
        </div>

        <div className="mt-4">
          <FiltersPanel value={filters} onChange={setFilters} disabled={busy} />
        </div>
      </Card>

      {error && (
        <div className="mb-6">
          <ErrorState message={error} />
          {errorStatus === 0 && (
            <p className="mt-2 text-xs text-on-surface-muted">
              Start the stack with{" "}
              <span className="font-mono text-on-surface">make dev</span> and check{" "}
              <span className="font-mono text-on-surface">
                localhost:8000/readyz
              </span>
              .
            </p>
          )}
        </div>
      )}

      {busy && (
        <Card>
          <div className="flex items-center gap-3 text-sm text-on-surface-muted">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            Retrieving · {mode}
            {transform !== "none" && ` · ${transform}`}
          </div>
        </Card>
      )}

      {!busy && result && (
        <>
          <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-on-surface-muted">
            <span>
              <span className="font-mono text-primary">
                {result.chunks.length}
              </span>{" "}
              chunks
            </span>
            <span>
              <span className="font-mono text-on-surface">
                {ms(result.latency_ms)}
              </span>{" "}
              retrieval
            </span>
            <span className="font-mono">{mode}</span>
            {result.query_transformed && (
              <span className="italic">
                transformed: “{result.query_transformed}”
              </span>
            )}
          </div>

          {result.chunks.length === 0 ? (
            <EmptyState
              title="No chunks matched"
              hint={`Nothing came back for “${ranQuery}”. Either no documents are ingested, or your metadata filters excluded every candidate.`}
            />
          ) : (
            <>
              <Card className="mb-4">
                <div className="flex flex-wrap gap-x-6 gap-y-2">
                  {STAGES.map((s) => (
                    <div key={s.key} className="flex items-center gap-2">
                      <span
                        className="h-2 w-2 rounded-full"
                        style={{ backgroundColor: s.color }}
                      />
                      <span className="font-mono text-[11px] text-on-surface">
                        {s.label}
                      </span>
                      <span className="text-[10px] text-on-surface-muted">
                        {s.hint}
                      </span>
                    </div>
                  ))}
                </div>
                <p className="mt-3 text-[10px] text-on-surface-muted">
                  Bar length is normalised per stage — the four scales are not
                  comparable to each other. The number on the right is the raw
                  score.
                </p>
              </Card>

              <Card className="overflow-hidden p-0">
                {result.chunks.map((c, i) => (
                  <ChunkRow
                    key={`${c.chunk_id}-${i}`}
                    chunk={c}
                    rank={i + 1}
                    maxes={maxes}
                  />
                ))}
              </Card>
            </>
          )}
        </>
      )}

      {!busy && !result && !error && (
        <EmptyState
          title="Run a query to inspect retrieval"
          hint="Calls POST /api/v1/retrieve — the same pipeline the playground uses, but stopping before generation."
        />
      )}
    </div>
  );
}
