"use client";

import clsx from "clsx";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  Check,
  Copy,
  CornerDownLeft,
  Database,
  Eraser,
  Loader2,
  Send,
  ThumbsDown,
  ThumbsUp,
  X,
  Zap,
} from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { Citation, QueryResponse, QueryTransform } from "@/lib/types";
import { ms, num, pct } from "@/lib/format";
import {
  EMPTY_FILTERS,
  FiltersPanel,
  toFilters,
  type FilterState,
} from "@/components/FiltersPanel";
import {
  Card,
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

interface Turn {
  id: string;
  question: string;
  status: "pending" | "done" | "error" | "cancelled";
  response?: QueryResponse;
  error?: string;
  errorStatus?: number;
  startedAt: number;
  elapsedMs?: number;
  feedback?: 1 | -1;
  feedbackError?: string;
}

/**
 * Rebuild which `[N]` marker maps to which entry of `citations`.
 *
 * The backend walks the answer's `[N]` refs in order, skips out-of-range ones,
 * dedupes, and appends the matching chunk — so citation order is first-appearance
 * order. We replay that here. If the reconstruction doesn't line up we return
 * null and render the markers as plain text rather than mislabel a source.
 */
function buildCitationMap(
  answer: string,
  citations: Citation[],
  topK: number,
): Map<number, number> | null {
  const seen = new Set<number>();
  const order: number[] = [];
  for (const m of answer.matchAll(/\[(\d+)\]/g)) {
    const n = Number(m[1]);
    if (n >= 1 && n <= topK && !seen.has(n)) {
      seen.add(n);
      order.push(n);
    }
  }
  if (order.length !== citations.length) return null;
  return new Map(order.map((refNum, idx) => [refNum, idx]));
}

function AnswerText({
  answer,
  citationMap,
  activeCitation,
  onCiteClick,
}: {
  answer: string;
  citationMap: Map<number, number> | null;
  activeCitation: number | null;
  onCiteClick: (citationIndex: number) => void;
}) {
  const parts = answer.split(/(\[\d+\])/g);
  return (
    <div className="whitespace-pre-wrap text-sm leading-relaxed text-on-surface">
      {parts.map((part, i) => {
        const match = /^\[(\d+)\]$/.exec(part);
        if (!match) return <span key={i}>{part}</span>;

        const refNum = Number(match[1]);
        const citationIndex = citationMap?.get(refNum);
        if (citationIndex === undefined) {
          return (
            <span key={i} className="font-mono text-xs text-on-surface-muted">
              {part}
            </span>
          );
        }
        return (
          <button
            key={i}
            onClick={() => onCiteClick(citationIndex)}
            title="Show this source"
            className={clsx(
              "mx-0.5 rounded px-1 align-baseline font-mono text-[11px] transition-colors",
              activeCitation === citationIndex
                ? "bg-primary text-background"
                : "bg-primary/15 text-primary hover:bg-primary/30",
            )}
          >
            {refNum}
          </button>
        );
      })}
    </div>
  );
}

function CitationList({
  citations,
  active,
  onSelect,
}: {
  citations: Citation[];
  active: number | null;
  onSelect: (i: number | null) => void;
}) {
  if (citations.length === 0) {
    return (
      <div className="mt-3 rounded-lg border border-dashed border-border px-3 py-2 text-xs text-on-surface-muted">
        The model returned no <span className="font-mono">[N]</span> citations
        for this answer.
      </div>
    );
  }
  return (
    <div className="mt-3 space-y-1.5">
      <div className="text-[10px] uppercase tracking-widest text-on-surface-muted">
        Sources · {citations.length}
      </div>
      {citations.map((c, i) => {
        const open = active === i;
        return (
          <button
            key={`${c.chunk_id}-${i}`}
            onClick={() => onSelect(open ? null : i)}
            className={clsx(
              "block w-full rounded-lg border px-3 py-2 text-left transition-colors",
              open
                ? "border-primary/50 bg-primary/10"
                : "border-border bg-background/40 hover:border-border hover:bg-surface-container-high",
            )}
          >
            <div className="flex items-center gap-2">
              <span
                className={clsx(
                  "rounded px-1.5 font-mono text-[10px]",
                  open
                    ? "bg-primary text-background"
                    : "bg-primary/15 text-primary",
                )}
              >
                {i + 1}
              </span>
              <span className="truncate text-xs text-on-surface">
                {c.document_name}
              </span>
              {c.page_number != null && (
                <span className="shrink-0 font-mono text-[10px] text-on-surface-muted">
                  p.{c.page_number}
                </span>
              )}
              <span className="ml-auto shrink-0 font-mono text-[10px] text-on-surface-muted">
                {c.score.toFixed(3)}
              </span>
            </div>
            <p
              className={clsx(
                "mt-1 text-[11px] leading-relaxed text-on-surface-muted",
                !open && "line-clamp-2",
              )}
            >
              {c.content_snippet}
            </p>
          </button>
        );
      })}
    </div>
  );
}

function TraceId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(id);
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      }}
      title="Copy trace id"
      className="flex items-center gap-1 font-mono text-[10px] text-on-surface-muted hover:text-on-surface"
    >
      {copied ? (
        <Check className="h-3 w-3 text-primary" />
      ) : (
        <Copy className="h-3 w-3" />
      )}
      {id.slice(0, 16)}
    </button>
  );
}

export default function PlaygroundPage() {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [tick, setTick] = useState(0);

  // Pipeline config
  const [topK, setTopK] = useState("5");
  const [useHybrid, setUseHybrid] = useState(true);
  const [useReranker, setUseReranker] = useState(true);
  const [useCache, setUseCache] = useState(true);
  const [transform, setTransform] = useState<QueryTransform>("none");
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS);

  const [activeCitation, setActiveCitation] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Drive the live elapsed counter while a request is in flight.
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => setTick((n) => n + 1), 100);
    return () => clearInterval(t);
  }, [busy]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, busy]);

  const parsedTopK = Math.min(50, Math.max(1, Number(topK) || 5));

  const send = useCallback(async () => {
    const q = question.trim();
    if (!q || busy) return;

    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const startedAt = Date.now();
    setTurns((prev) => [...prev, { id, question: q, status: "pending", startedAt }]);
    setQuestion("");
    setBusy(true);

    const controller = new AbortController();
    abortRef.current = controller;

    const finish = (patch: Partial<Turn>) =>
      setTurns((prev) =>
        prev.map((t) =>
          t.id === id ? { ...t, elapsedMs: Date.now() - startedAt, ...patch } : t,
        ),
      );

    try {
      const response = await api.query(
        {
          question: q,
          top_k: parsedTopK,
          use_hybrid: useHybrid,
          use_reranker: useReranker,
          use_cache: useCache,
          query_transform: transform,
          filters: toFilters(filters),
        },
        controller.signal,
      );
      finish({ status: "done", response });
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        finish({ status: "cancelled" });
      } else if (e instanceof ApiError) {
        finish({ status: "error", error: e.message, errorStatus: e.status });
      } else {
        finish({ status: "error", error: "Unexpected error" });
      }
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }, [
    question,
    busy,
    parsedTopK,
    useHybrid,
    useReranker,
    useCache,
    transform,
    filters,
  ]);

  const rate = async (turn: Turn, rating: 1 | -1) => {
    if (!turn.response) return;
    setTurns((prev) =>
      prev.map((t) =>
        t.id === turn.id ? { ...t, feedback: rating, feedbackError: undefined } : t,
      ),
    );
    try {
      await api.submitFeedback({ trace_id: turn.response.trace_id, rating });
    } catch (e) {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turn.id
            ? {
                ...t,
                feedback: undefined,
                feedbackError:
                  e instanceof ApiError ? e.message : "Feedback failed",
              }
            : t,
        ),
      );
    }
  };

  return (
    <div>
      <PageHeader
        title="Playground"
        subtitle="Ask a question against your ingested documents. Every answer is grounded, cited, and traced — flip the pipeline switches to feel the difference."
        actions={
          turns.length > 0 && (
            <button
              onClick={() => setTurns([])}
              disabled={busy}
              className="flex items-center gap-2 rounded-lg border border-border bg-surface-container px-3 py-2 text-sm text-on-surface hover:border-primary/50 disabled:opacity-50"
            >
              <Eraser className="h-4 w-4" /> Clear
            </button>
          )
        }
      />

      {/* ── Pipeline config ── */}
      <Card className="mb-6">
        <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
          <Field label="Top K" className="w-20">
            <NumberInput
              value={topK}
              onChange={setTopK}
              min={1}
              max={50}
              disabled={busy}
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
              title="Fuse dense (pgvector) and sparse (BM25) results with reciprocal rank fusion"
            />
            <Toggle
              label="Reranker"
              checked={useReranker}
              onChange={setUseReranker}
              disabled={busy}
              title="Rerank candidates with the LLM cross-encoder before generation"
            />
            <Toggle
              label="Semantic cache"
              checked={useCache}
              onChange={setUseCache}
              disabled={busy}
              title="Return a cached answer when a past query is semantically close enough"
            />
          </div>
        </div>
        <div className="mt-4">
          <FiltersPanel value={filters} onChange={setFilters} disabled={busy} />
        </div>
      </Card>

      {/* ── Conversation ── */}
      {turns.length === 0 ? (
        <Card className="border-dashed">
          <div className="flex flex-col items-center gap-3 py-10 text-center">
            <Database className="h-8 w-8 text-on-surface-muted" />
            <div className="text-sm font-medium text-on-surface">
              Ask your first question
            </div>
            <p className="max-w-md text-xs text-on-surface-muted">
              This hits{" "}
              <span className="font-mono text-on-surface">POST /api/v1/query</span>{" "}
              — the real pipeline. You need at least one ingested document; upload
              one with{" "}
              <span className="font-mono text-on-surface">POST /api/v1/ingest</span>{" "}
              first.
            </p>
          </div>
        </Card>
      ) : (
        <div className="space-y-6">
          {turns.map((turn) => (
            <div key={turn.id}>
              {/* Question */}
              <div className="mb-3 flex justify-end">
                <div className="max-w-2xl rounded-xl rounded-br-sm border border-primary/40 bg-primary/10 px-4 py-2.5 text-sm text-on-surface">
                  {turn.question}
                </div>
              </div>

              {/* Answer */}
              <Card>
                {turn.status === "pending" ? (
                  <div className="flex items-center gap-3 text-sm text-on-surface-muted">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    <span>
                      Running pipeline
                      {useHybrid ? " · hybrid" : " · dense"}
                      {useReranker ? " · rerank" : ""}
                      {transform !== "none" ? ` · ${transform}` : ""}
                    </span>
                    <span className="ml-auto font-mono text-xs text-on-surface">
                      {((Date.now() - turn.startedAt) / 1000).toFixed(1)}s
                      <span className="hidden">{tick}</span>
                    </span>
                  </div>
                ) : turn.status === "cancelled" ? (
                  <div className="text-sm text-on-surface-muted">
                    Cancelled after {ms(turn.elapsedMs)}.
                  </div>
                ) : turn.status === "error" ? (
                  <div>
                    <ErrorState message={turn.error ?? "Query failed"} />
                    {turn.errorStatus === 404 && (
                      <p className="mt-3 text-xs text-on-surface-muted">
                        No chunks matched. Either nothing has been ingested yet, or
                        your metadata filters excluded everything — try clearing
                        them or lowering <span className="font-mono">min score</span>.
                      </p>
                    )}
                    {turn.errorStatus === 400 && (
                      <p className="mt-3 text-xs text-on-surface-muted">
                        Blocked by the input guardrails before reaching the
                        pipeline.
                      </p>
                    )}
                    {turn.errorStatus === 0 && (
                      <p className="mt-3 text-xs text-on-surface-muted">
                        Start the stack with{" "}
                        <span className="font-mono text-on-surface">make dev</span>{" "}
                        and confirm{" "}
                        <span className="font-mono text-on-surface">
                          localhost:8000/readyz
                        </span>{" "}
                        reports ready.
                      </p>
                    )}
                  </div>
                ) : (
                  turn.response && (
                    <div>
                      {/* Meta strip */}
                      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1 border-b border-border pb-3 text-[11px] text-on-surface-muted">
                        <span className="font-mono text-primary">
                          {ms(turn.response.latency_ms)}
                        </span>
                        {turn.response.tokens_used != null && (
                          <span className="font-mono">
                            {num(turn.response.tokens_used)} tokens
                          </span>
                        )}
                        {turn.response.cached && (
                          <span className="flex items-center gap-1 rounded bg-secondary/15 px-1.5 py-0.5 font-mono text-[10px] text-secondary">
                            <Zap className="h-3 w-3" />
                            cache hit
                            {turn.response.cache_similarity != null &&
                              ` · ${pct(turn.response.cache_similarity, 1)}`}
                          </span>
                        )}
                        <div className="ml-auto flex items-center gap-3">
                          <TraceId id={turn.response.trace_id} />
                          <Link
                            href="/traces"
                            className="text-[10px] text-on-surface-muted underline-offset-2 hover:text-primary hover:underline"
                          >
                            traces →
                          </Link>
                        </div>
                      </div>

                      <AnswerText
                        answer={turn.response.answer}
                        citationMap={buildCitationMap(
                          turn.response.answer,
                          turn.response.citations,
                          parsedTopK,
                        )}
                        activeCitation={
                          activeCitation?.startsWith(`${turn.id}:`)
                            ? Number(activeCitation.split(":")[1])
                            : null
                        }
                        onCiteClick={(i) =>
                          setActiveCitation((cur) =>
                            cur === `${turn.id}:${i}` ? null : `${turn.id}:${i}`,
                          )
                        }
                      />

                      <CitationList
                        citations={turn.response.citations}
                        active={
                          activeCitation?.startsWith(`${turn.id}:`)
                            ? Number(activeCitation.split(":")[1])
                            : null
                        }
                        onSelect={(i) =>
                          setActiveCitation(i === null ? null : `${turn.id}:${i}`)
                        }
                      />

                      {/* Feedback */}
                      <div className="mt-4 flex items-center gap-2 border-t border-border pt-3">
                        <span className="text-[11px] text-on-surface-muted">
                          Was this answer useful?
                        </span>
                        <button
                          onClick={() => rate(turn, 1)}
                          className={clsx(
                            "rounded p-1.5 transition-colors",
                            turn.feedback === 1
                              ? "bg-primary/20 text-primary"
                              : "text-on-surface-muted hover:bg-surface-container-high hover:text-on-surface",
                          )}
                          title="Thumbs up"
                        >
                          <ThumbsUp className="h-3.5 w-3.5" />
                        </button>
                        <button
                          onClick={() => rate(turn, -1)}
                          className={clsx(
                            "rounded p-1.5 transition-colors",
                            turn.feedback === -1
                              ? "bg-danger/20 text-danger"
                              : "text-on-surface-muted hover:bg-surface-container-high hover:text-on-surface",
                          )}
                          title="Thumbs down"
                        >
                          <ThumbsDown className="h-3.5 w-3.5" />
                        </button>
                        {turn.feedback && !turn.feedbackError && (
                          <span className="text-[11px] text-on-surface-muted">
                            Recorded.
                          </span>
                        )}
                        {turn.feedbackError && (
                          <span className="text-[11px] text-danger">
                            {turn.feedbackError}
                          </span>
                        )}
                      </div>
                    </div>
                  )
                )}
              </Card>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      )}

      {/* ── Composer ── */}
      <div className="sticky bottom-4 mt-6">
        <Card className="glass">
          <div className="flex items-end gap-3">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={2}
              placeholder="Ask a question about your ingested documents…"
              className="flex-1 resize-none bg-transparent text-sm text-on-surface outline-none placeholder:text-on-surface-muted"
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
                onClick={send}
                disabled={!question.trim()}
                className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90 disabled:opacity-40"
              >
                <Send className="h-4 w-4" /> Ask
              </button>
            )}
          </div>
          <div className="mt-2 flex items-center gap-1 text-[10px] text-on-surface-muted">
            <CornerDownLeft className="h-3 w-3" /> Enter to send · Shift+Enter for
            a new line
          </div>
        </Card>
      </div>
    </div>
  );
}
