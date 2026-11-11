"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { RefreshCw } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { TraceDetail, TraceSummary } from "@/lib/types";
import { dateTime, ms, num, shortId, timeAgo, usd } from "@/lib/format";
import { SpanWaterfall, waterfallTotal } from "@/components/SpanWaterfall";
import { StateLine } from "@/components/Marks";
import { Entry, EmptyState, ErrorState, Figure, Loading, PageHeader } from "@/components/ui";

function TracesInner() {
  const params = useSearchParams();
  const wanted = params.get("trace");

  const [traces, setTraces] = useState<TraceSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [minLatency, setMinLatency] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [selected, setSelected] = useState<TraceDetail | null>(null);
  const [loadingTrace, setLoadingTrace] = useState(false);
  const detailRef = useRef<HTMLDivElement | null>(null);
  const opened = useRef(false);

  const load = useCallback(async () => {
    setError(null);
    setTraces(null);
    try {
      const data = await api.listTraces({
        limit: 100,
        minLatencyMs: minLatency ? Number(minLatency) : undefined,
        status: status || undefined,
      });
      setTraces(data);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load traces");
      setTraces([]);
    }
  }, [minLatency, status]);

  useEffect(() => {
    load();
  }, [load]);

  const openTrace = useCallback(async (id: string) => {
    setLoadingTrace(true);
    setSelected(null);
    try {
      setSelected(await api.getTrace(id));
      detailRef.current?.scrollIntoView({ block: "start" });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load trace");
    } finally {
      setLoadingTrace(false);
    }
  }, []);

  // Open the trace named in the address (?trace=...), once.
  useEffect(() => {
    if (wanted && !opened.current) {
      opened.current = true;
      openTrace(wanted);
    }
  }, [wanted, openTrace]);

  return (
    <>
      <PageHeader
        title="Trace Viewer"
        subtitle="Every query persists its OpenTelemetry span tree. Open a trace to see its timeline: where the milliseconds, tokens and cost went."
        actions={<button type="button" onClick={load} className="btn"><RefreshCw aria-hidden="true" /> Refresh</button>}
      />

      <div className="cols">
        <Entry kind="Traces" source="GET /api/v1/traces">
          <div className="row" style={{ marginBottom: 16 }}>
            <label className="field" style={{ width: 140 }}>
              <span className="lab">Min latency (ms)</span>
              <input type="number" className="input" value={minLatency} onChange={(e) => setMinLatency(e.target.value)} placeholder="0" />
            </label>
            <label className="field" style={{ width: 140 }}>
              <span className="lab">Status</span>
              <select className="select" value={status} onChange={(e) => setStatus(e.target.value)}>
                <option value="">All</option>
                <option value="ok">ok</option>
                <option value="error">error</option>
              </select>
            </label>
          </div>

          {error && <div style={{ marginBottom: 16 }}><ErrorState message={error} onRetry={load} /></div>}

          {traces === null ? (
            <Loading label="Loading traces" />
          ) : traces.length === 0 && !error ? (
            <EmptyState title="No traces match" hint="Ask a question in the Playground and its trace appears here, or loosen the filters above." />
          ) : (
            <div className="tab-wrap" style={{ maxHeight: "70vh", overflowY: "auto" }}>
              <table className="tab">
                <caption className="sr-only">Traces, newest first</caption>
                <thead>
                  <tr><th>Trace</th><th className="num">Duration</th><th className="num hide-sm">Spans</th><th>Recorded</th></tr>
                </thead>
                <tbody>
                  {traces.map((t) => (
                    <tr key={t.id} className={selected?.id === t.id ? "sel" : undefined} onClick={() => openTrace(t.id)} style={{ cursor: "pointer" }}>
                      <td>
                        <button type="button" className="row-btn" onClick={(e) => { e.stopPropagation(); openTrace(t.id); }} aria-pressed={selected?.id === t.id}>
                          <span className="mono">{t.name ?? shortId(t.otel_trace_id)}</span>
                          <span className="id">{shortId(t.otel_trace_id, 16)}</span>
                        </button>
                      </td>
                      <td className="num">{ms(t.total_duration_ms)}</td>
                      <td className="num hide-sm">{t.span_count}</td>
                      <td>
                        {dateTime(t.created_at)}
                        <span className="id">{timeAgo(t.created_at)}{t.status === "error" ? " · error" : ""}</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Entry>

        <div ref={detailRef} style={{ scrollMarginTop: 16 }}>
          {loadingTrace ? (
            <Entry kind="Trace"><Loading label="Loading the span tree" /></Entry>
          ) : selected ? (
            <Entry kind="Trace" date={selected.created_at} id={selected.id} title={<span className="mono">{selected.name ?? shortId(selected.otel_trace_id)}</span>}>
              <dl className="deflist" style={{ marginBottom: 20 }}>
                <dt>Duration</dt><dd>{ms(selected.total_duration_ms)}</dd>
                <dt>Spans</dt><dd>{selected.span_count}</dd>
                <dt>Tokens</dt><dd>{num(selected.total_tokens)}</dd>
                <dt>Cost</dt><dd>{usd(selected.total_cost_usd)}</dd>
                <dt>Status</dt><dd><StateLine kind={selected.status === "error" ? "down" : "ok"}>{selected.status}</StateLine></dd>
                <dt>OpenTelemetry id</dt><dd style={{ overflowWrap: "anywhere" }}>{selected.otel_trace_id}</dd>
              </dl>
              <Figure
                caption={
                  <>
                    <b>Span timeline.</b> {selected.spans.length} span{selected.spans.length === 1 ? "" : "s"} over {ms(waterfallTotal(selected.spans))}.
                    Each bar starts where its span started and lasts as long as it ran; indentation shows the parent span. Select a span to read its attributes.
                  </>
                }
              >
                <SpanWaterfall spans={selected.spans} />
              </Figure>
            </Entry>
          ) : (
            <Entry kind="Trace">
              <EmptyState title="No trace open" hint="Pick a trace in the list to render its span timeline." />
            </Entry>
          )}
        </div>
      </div>
    </>
  );
}

export default function TracesPage() {
  return (
    <Suspense fallback={<Loading label="Loading" />}>
      <TracesInner />
    </Suspense>
  );
}
