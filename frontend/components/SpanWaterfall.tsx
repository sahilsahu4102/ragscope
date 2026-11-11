"use client";

import clsx from "clsx";
import { useMemo, useState } from "react";
import type { Span } from "@/lib/types";
import { ms, usd } from "@/lib/format";
import { KindMark, kindStyle } from "./Marks";

interface FlatSpan {
  span: Span;
  depth: number;
  offsetPct: number;
  widthPct: number;
}

function flatten(spans: Span[]): { rows: FlatSpan[]; totalMs: number } {
  if (spans.length === 0) return { rows: [], totalMs: 0 };

  const starts = spans
    .map((s) => (s.start_time ? Date.parse(s.start_time) : NaN))
    .filter((n) => !Number.isNaN(n));
  const ends = spans
    .map((s) => (s.end_time ? Date.parse(s.end_time) : NaN))
    .filter((n) => !Number.isNaN(n));
  const traceStart = starts.length ? Math.min(...starts) : 0;
  const traceEnd = ends.length ? Math.max(...ends) : traceStart + 1;
  const totalMs = Math.max(1, traceEnd - traceStart);

  const byId = new Map(spans.map((s) => [s.otel_span_id, s]));
  const children = new Map<string | null, Span[]>();
  for (const s of spans) {
    const parent = s.parent_span_id && byId.has(s.parent_span_id) ? s.parent_span_id : null;
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent)!.push(s);
  }
  for (const list of children.values()) {
    list.sort((a, b) => (a.start_time ? Date.parse(a.start_time) : 0) - (b.start_time ? Date.parse(b.start_time) : 0));
  }

  const rows: FlatSpan[] = [];
  const walk = (parent: string | null, depth: number) => {
    for (const s of children.get(parent) ?? []) {
      const start = s.start_time ? Date.parse(s.start_time) : traceStart;
      const dur = s.duration_ms ?? 0;
      rows.push({
        span: s,
        depth,
        offsetPct: ((start - traceStart) / totalMs) * 100,
        widthPct: Math.max(0.5, (dur / totalMs) * 100),
      });
      walk(s.otel_span_id, depth + 1);
    }
  };
  walk(null, 0);
  return { rows, totalMs };
}

export function waterfallTotal(spans: Span[]): number {
  return flatten(spans).totalMs;
}

export function SpanWaterfall({ spans }: { spans: Span[] }) {
  const { rows, totalMs } = useMemo(() => flatten(spans), [spans]);
  const [selected, setSelected] = useState<string | null>(null);

  if (rows.length === 0) {
    return <p className="lede">No spans in this trace.</p>;
  }

  const selectedSpan = rows.find((r) => r.span.id === selected)?.span;
  const kinds = Array.from(new Set(rows.map((r) => r.span.span_kind)));

  return (
    <div>
      <div className="flex flex-wrap gap-x-5 gap-y-1" style={{ marginBottom: 12, fontSize: "0.8125rem" }} aria-label="Span kinds in this trace">
        {kinds.map((k) => (
          <span key={k} className="state-line" style={{ fontWeight: 600 }}>
            <KindMark kind={k} />
            {k}
          </span>
        ))}
      </div>

      <div className="plot" style={{ padding: "10px 6px" }}>
        <div className="wf-row" style={{ cursor: "default", paddingBottom: 2 }} aria-hidden="true">
          <span />
          <div className="flex justify-between mono" style={{ color: "var(--ink-3)", fontSize: "0.6875rem" }}>
            <span>0 ms</span>
            <span>{ms(totalMs)}</span>
          </div>
        </div>
        {rows.map(({ span, depth, offsetPct, widthPct }) => {
          const { color } = kindStyle(span.span_kind);
          const on = span.id === selected;
          return (
            <button
              key={span.id}
              type="button"
              onClick={() => setSelected(on ? null : span.id)}
              aria-pressed={on}
              className={clsx("wf-row", on && "is-on", selected && !on && "is-dim")}
            >
              <span className="flex items-center gap-2" style={{ paddingLeft: depth * 14, minWidth: 0 }}>
                <KindMark kind={span.span_kind} />
                <span className="mono truncate" style={{ fontSize: "0.75rem" }}>{span.name}</span>
              </span>
              <span className="wf-track">
                <span
                  className="wf-bar"
                  style={{
                    left: `${offsetPct}%`,
                    width: `${widthPct}%`,
                    background: color,
                    outline: span.status === "error" ? "2px solid var(--red)" : undefined,
                  }}
                  title={`${span.name} · ${ms(span.duration_ms)}`}
                />
                {widthPct >= 18 ? (
                  <span className="wf-time" style={{ left: `${offsetPct}%`, color: "var(--paper)", paddingLeft: 6 }}>
                    {ms(span.duration_ms)}
                  </span>
                ) : (
                  <span className="wf-time" style={{ left: `clamp(0px, ${offsetPct + widthPct}%, calc(100% - 64px))` }}>
                    {ms(span.duration_ms)}
                  </span>
                )}
              </span>
            </button>
          );
        })}
      </div>

      {selectedSpan && (
        <div style={{ marginTop: 16 }}>
          <h3 className="state-line" style={{ gap: 8 }}>
            <KindMark kind={selectedSpan.span_kind} />
            <span className="mono">{selectedSpan.name}</span>
            <span className="chip">{selectedSpan.span_kind}</span>
          </h3>
          <dl className="deflist" style={{ marginTop: 10 }}>
            <dt>Duration</dt><dd>{ms(selectedSpan.duration_ms)}</dd>
            <dt>Tokens</dt><dd>{selectedSpan.tokens ?? "n/a"}</dd>
            <dt>Cost</dt><dd>{usd(selectedSpan.cost_usd)}</dd>
            <dt>Status</dt><dd>{selectedSpan.status}</dd>
          </dl>
          {selectedSpan.attributes && (
            <pre className="attrs" tabIndex={0} aria-label="Span attributes">
              {JSON.stringify(selectedSpan.attributes, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
