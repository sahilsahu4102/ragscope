"use client";

import { useMemo, useState } from "react";
import type { Span } from "@/lib/types";
import { SPAN_KIND_COLOR, ms, usd } from "@/lib/format";

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
    const parent =
      s.parent_span_id && byId.has(s.parent_span_id) ? s.parent_span_id : null;
    if (!children.has(parent)) children.set(parent, []);
    children.get(parent)!.push(s);
  }
  for (const list of children.values()) {
    list.sort((a, b) =>
      (a.start_time ? Date.parse(a.start_time) : 0) -
      (b.start_time ? Date.parse(b.start_time) : 0),
    );
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

export function SpanWaterfall({ spans }: { spans: Span[] }) {
  const { rows, totalMs } = useMemo(() => flatten(spans), [spans]);
  const [selected, setSelected] = useState<string | null>(null);

  if (rows.length === 0) {
    return (
      <div className="text-sm text-on-surface-muted">No spans in this trace.</div>
    );
  }

  const selectedSpan = rows.find((r) => r.span.id === selected)?.span;

  return (
    <div>
      <div className="mb-2 flex justify-between font-mono text-[10px] text-on-surface-muted">
        <span>0 ms</span>
        <span>total {ms(totalMs)}</span>
      </div>
      <div className="space-y-1">
        {rows.map(({ span, depth, offsetPct, widthPct }) => {
          const color = SPAN_KIND_COLOR[span.span_kind] ?? "#8fa0b0";
          const isSel = span.id === selected;
          return (
            <button
              key={span.id}
              onClick={() => setSelected(isSel ? null : span.id)}
              className="group grid w-full grid-cols-[minmax(180px,1fr)_2.5fr] items-center gap-3 rounded-md px-2 py-1 text-left hover:bg-surface-container-high"
            >
              <div
                className="flex items-center gap-2 truncate"
                style={{ paddingLeft: depth * 14 }}
              >
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className="truncate font-mono text-xs text-on-surface">
                  {span.name}
                </span>
                <span
                  className="shrink-0 rounded px-1 font-mono text-[9px]"
                  style={{ backgroundColor: `${color}22`, color }}
                >
                  {span.span_kind}
                </span>
              </div>
              <div className="relative h-5">
                <div
                  className="absolute top-0.5 flex h-4 items-center rounded"
                  style={{
                    left: `${offsetPct}%`,
                    width: `${widthPct}%`,
                    backgroundColor: color,
                    opacity: span.status === "error" ? 0.5 : 0.85,
                    minWidth: 2,
                  }}
                  title={`${span.name} · ${ms(span.duration_ms)}`}
                />
                <span
                  className="absolute top-0 whitespace-nowrap font-mono text-[10px] text-on-surface-muted"
                  style={{
                    left: `clamp(0px, ${offsetPct + widthPct}%, calc(100% - 60px))`,
                    paddingLeft: 4,
                  }}
                >
                  {ms(span.duration_ms)}
                </span>
              </div>
            </button>
          );
        })}
      </div>

      {selectedSpan && (
        <div className="mt-4 rounded-lg border border-border bg-surface-container/60 p-4">
          <div className="mb-2 flex items-center gap-2">
            <span className="font-mono text-sm text-on-surface">
              {selectedSpan.name}
            </span>
            <span
              className="rounded px-1.5 py-0.5 font-mono text-[10px]"
              style={{
                backgroundColor: `${SPAN_KIND_COLOR[selectedSpan.span_kind]}22`,
                color: SPAN_KIND_COLOR[selectedSpan.span_kind],
              }}
            >
              {selectedSpan.span_kind}
            </span>
          </div>
          <div className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-xs text-on-surface-muted sm:grid-cols-4">
            <div>
              <span className="text-on-surface-muted">duration</span>
              <div className="text-on-surface">{ms(selectedSpan.duration_ms)}</div>
            </div>
            <div>
              <span className="text-on-surface-muted">tokens</span>
              <div className="text-on-surface">{selectedSpan.tokens ?? "—"}</div>
            </div>
            <div>
              <span className="text-on-surface-muted">cost</span>
              <div className="text-on-surface">{usd(selectedSpan.cost_usd)}</div>
            </div>
            <div>
              <span className="text-on-surface-muted">status</span>
              <div className="text-on-surface">{selectedSpan.status}</div>
            </div>
          </div>
          {selectedSpan.attributes && (
            <pre className="mt-3 max-h-56 overflow-auto rounded bg-background/60 p-3 font-mono text-[11px] leading-relaxed text-on-surface-muted">
              {JSON.stringify(selectedSpan.attributes, null, 2)}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
