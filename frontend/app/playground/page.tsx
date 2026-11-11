"use client";

import clsx from "clsx";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { Check, Copy, Eraser, Send, ThumbsDown, ThumbsUp, X } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { Citation, QueryResponse, QueryTransform } from "@/lib/types";
import { ms, num, pct } from "@/lib/format";
import { EMPTY_FILTERS, FiltersPanel, toFilters, type FilterState } from "@/components/FiltersPanel";
import { StateMark } from "@/components/Marks";
import { Entry, EmptyState, ErrorState, Field, NumberInput, PageHeader, Select, Toggle } from "@/components/ui";

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
 * The backend walks the answer's `[N]` refs in order, skips out-of-range ones, dedupes, and appends the
 * matching chunk, so citation order is first-appearance order. We replay that here. If the reconstruction
 * doesn't line up we return null and render the markers as plain text rather than mislabel a source.
 */
function buildCitationMap(answer: string, citations: Citation[], topK: number): Map<number, number> | null {
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
    <div className="answer">
      {parts.map((part, i) => {
        const match = /^\[(\d+)\]$/.exec(part);
        if (!match) return <span key={i}>{part}</span>;

        const refNum = Number(match[1]);
        const citationIndex = citationMap?.get(refNum);
        if (citationIndex === undefined) {
          return <span key={i} className="mono" style={{ color: "var(--ink-3)" }}>{part}</span>;
        }
        return (
          <button
            key={i}
            type="button"
            onClick={() => onCiteClick(citationIndex)}
            title="Show this source"
            aria-label={`Source ${refNum}`}
            aria-pressed={activeCitation === citationIndex}
            className={clsx("cite", activeCitation === citationIndex && "is-on")}
          >
            {refNum}
          </button>
        );
      })}
    </div>
  );
}

function CitationList({ citations, active, onSelect }: { citations: Citation[]; active: number | null; onSelect: (i: number | null) => void }) {
  if (citations.length === 0) {
    return (
      <p className="lede" style={{ marginTop: 16 }}>
        The model returned no <code>[N]</code> citations for this answer.
      </p>
    );
  }
  return (
    <div style={{ marginTop: 20 }}>
      <div className="lab" style={{ marginBottom: 4 }}>Sources, {citations.length}</div>
      <div style={{ borderTop: "2px solid var(--ink)" }}>
        {citations.map((c, i) => {
          const open = active === i;
          return (
            <button key={`${c.chunk_id}-${i}`} type="button" onClick={() => onSelect(open ? null : i)} aria-expanded={open} className="source">
              <span className="flex flex-wrap items-baseline gap-x-3">
                <span className={clsx("cite", open && "is-on")} style={{ margin: 0 }}>{i + 1}</span>
                <b>{c.document_name}</b>
                {c.page_number != null && <span className="mono" style={{ color: "var(--ink-3)" }}>p. {c.page_number}</span>}
                <span className="mono" style={{ marginLeft: "auto", color: "var(--ink-3)" }}>score {c.score.toFixed(3)}</span>
              </span>
              <span className={clsx("snip", !open && "clamp")} style={{ display: open ? "block" : undefined }}>{c.content_snippet}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function TraceId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={() => {
        navigator.clipboard?.writeText(id);
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      }}
      title="Copy trace id"
      className="btn btn-sm"
    >
      {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}
      <span className="mono">{id.slice(0, 16)}</span>
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
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, elapsedMs: Date.now() - startedAt, ...patch } : t)));

    try {
      const response = await api.query(
        { question: q, top_k: parsedTopK, use_hybrid: useHybrid, use_reranker: useReranker, use_cache: useCache, query_transform: transform, filters: toFilters(filters) },
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
  }, [question, busy, parsedTopK, useHybrid, useReranker, useCache, transform, filters]);

  const rate = async (turn: Turn, rating: 1 | -1) => {
    if (!turn.response) return;
    setTurns((prev) => prev.map((t) => (t.id === turn.id ? { ...t, feedback: rating, feedbackError: undefined } : t)));
    try {
      await api.submitFeedback({ trace_id: turn.response.trace_id, rating });
    } catch (e) {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === turn.id ? { ...t, feedback: undefined, feedbackError: e instanceof ApiError ? e.message : "Feedback failed" } : t,
        ),
      );
    }
  };

  return (
    <>
      <PageHeader
        title="Playground"
        subtitle="Ask a question against your ingested documents. Every answer is grounded, cited and traced; change the pipeline settings to see the difference."
        actions={
          turns.length > 0 ? (
            <button type="button" onClick={() => setTurns([])} disabled={busy} className="btn"><Eraser aria-hidden="true" /> Clear</button>
          ) : undefined
        }
      />

      <Entry kind="Pipeline" source="POST /api/v1/query">
        <div className="stack">
          <div className="row">
            <Field label="Top K" className="w-20"><NumberInput value={topK} onChange={setTopK} min={1} max={50} disabled={busy} /></Field>
            <Field label="Query transform" className="w-40"><Select value={transform} onChange={setTransform} options={TRANSFORMS} disabled={busy} /></Field>
            <div className="flex flex-wrap items-center gap-2" style={{ paddingBottom: 2 }}>
              <Toggle label="Hybrid + RRF" checked={useHybrid} onChange={setUseHybrid} disabled={busy} title="Fuse dense (pgvector) and sparse (BM25) results with reciprocal rank fusion" />
              <Toggle label="Reranker" checked={useReranker} onChange={setUseReranker} disabled={busy} title="Rerank candidates with the cross-encoder before generation" />
              <Toggle label="Semantic cache" checked={useCache} onChange={setUseCache} disabled={busy} title="Return a cached answer when a past query is semantically close enough" />
            </div>
          </div>
          <FiltersPanel value={filters} onChange={setFilters} disabled={busy} />
        </div>
      </Entry>

      {turns.length === 0 ? (
        <EmptyState
          title="Ask your first question"
          hint={<>This calls <code>POST /api/v1/query</code>, the real pipeline. You need at least one ingested document; add one on the <Link href="/documents" style={{ fontWeight: 700 }}>Documents</Link> page.</>}
        />
      ) : (
        <div>
          {turns.map((turn) => (
            <Entry key={turn.id} kind="Question" date={turn.startedAt} id={turn.response?.trace_id} title={turn.question}>
              {turn.status === "pending" ? (
                <div className="note is-wait" role="status">
                  <StateMark kind="wait" />
                  <p>
                    Running the pipeline{useHybrid ? ", hybrid" : ", dense"}{useReranker ? ", rerank" : ""}{transform !== "none" ? `, ${transform}` : ""}.{" "}
                    <span className="mono">{((Date.now() - turn.startedAt) / 1000).toFixed(1)} s</span>
                    <span className="sr-only">{tick}</span>
                  </p>
                </div>
              ) : turn.status === "cancelled" ? (
                <p className="lede">Cancelled after {ms(turn.elapsedMs)}.</p>
              ) : turn.status === "error" ? (
                <ErrorState message={turn.error ?? "Query failed"}>
                  {turn.errorStatus === 404 && <p>No chunks matched. Either nothing has been ingested yet, or your metadata filters excluded everything: try clearing them or lowering the minimum score.</p>}
                  {turn.errorStatus === 400 && <p>Blocked by the input guardrails before reaching the pipeline.</p>}
                  {turn.errorStatus === 0 && <p>Start the stack with <code>docker compose up -d</code> and confirm <code>/readyz</code> on the API reports ready.</p>}
                </ErrorState>
              ) : (
                turn.response && (
                  <div>
                    <div className="flex flex-wrap items-center gap-x-5 gap-y-2" style={{ marginBottom: 16, fontSize: "0.875rem", color: "var(--ink-2)" }}>
                      <span><b className="mono" style={{ color: "var(--ink)" }}>{ms(turn.response.latency_ms)}</b> to answer</span>
                      {turn.response.tokens_used != null && <span className="mono">{num(turn.response.tokens_used)} tokens</span>}
                      {turn.response.cached && (
                        <span className="chip is-on">
                          cache hit{turn.response.cache_similarity != null && `, ${pct(turn.response.cache_similarity, 1)} similar`}
                        </span>
                      )}
                      <span className="flex items-center gap-2" style={{ marginLeft: "auto" }}>
                        <TraceId id={turn.response.trace_id} />
                        <Link href="/traces" className="btn btn-sm">All traces</Link>
                      </span>
                    </div>

                    <AnswerText
                      answer={turn.response.answer}
                      citationMap={buildCitationMap(turn.response.answer, turn.response.citations, parsedTopK)}
                      activeCitation={activeCitation?.startsWith(`${turn.id}:`) ? Number(activeCitation.split(":")[1]) : null}
                      onCiteClick={(i) => setActiveCitation((cur) => (cur === `${turn.id}:${i}` ? null : `${turn.id}:${i}`))}
                    />

                    <CitationList
                      citations={turn.response.citations}
                      active={activeCitation?.startsWith(`${turn.id}:`) ? Number(activeCitation.split(":")[1]) : null}
                      onSelect={(i) => setActiveCitation(i === null ? null : `${turn.id}:${i}`)}
                    />

                    <div className="flex flex-wrap items-center gap-2" style={{ marginTop: 20 }}>
                      <span style={{ fontSize: "0.875rem", color: "var(--ink-2)" }}>Was this answer useful?</span>
                      <button type="button" className="btn btn-sm" aria-pressed={turn.feedback === 1} onClick={() => rate(turn, 1)} title="Thumbs up" aria-label="Thumbs up" style={turn.feedback === 1 ? { background: "var(--ink)", color: "var(--paper)" } : undefined}>
                        <ThumbsUp aria-hidden="true" />
                      </button>
                      <button type="button" className="btn btn-sm" aria-pressed={turn.feedback === -1} onClick={() => rate(turn, -1)} title="Thumbs down" aria-label="Thumbs down" style={turn.feedback === -1 ? { background: "var(--ink)", color: "var(--paper)" } : undefined}>
                        <ThumbsDown aria-hidden="true" />
                      </button>
                      {turn.feedback && !turn.feedbackError && <span style={{ fontSize: "0.875rem" }} role="status">Recorded.</span>}
                      {turn.feedbackError && <span style={{ fontSize: "0.875rem", color: "var(--red)", fontWeight: 600 }} role="alert">{turn.feedbackError}</span>}
                    </div>
                  </div>
                )
              )}
            </Entry>
          ))}
          <div ref={bottomRef} style={{ scrollMarginBottom: 170 }} />
        </div>
      )}

      <div className="composer">
        <div className="flex items-end gap-3">
          <label className="field" style={{ flex: 1 }}>
            <span className="sr-only">Your question</span>
            <textarea
              className="textarea"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              rows={2}
              placeholder="Ask a question about your ingested documents"
            />
          </label>
          {busy ? (
            <button type="button" onClick={() => abortRef.current?.abort()} className="btn btn-danger"><X aria-hidden="true" /> Cancel</button>
          ) : (
            <button type="button" onClick={send} disabled={!question.trim()} className="btn btn-primary"><Send aria-hidden="true" /> Ask</button>
          )}
        </div>
        <p style={{ marginTop: 6, fontSize: "0.75rem", color: "var(--ink-3)" }}>Enter to send, Shift+Enter for a new line.</p>
      </div>
    </>
  );
}
