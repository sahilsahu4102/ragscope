"use client";

import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { TraceDetail, TraceSummary } from "@/lib/types";
import { ms, num, shortId, timeAgo, usd } from "@/lib/format";
import { SpanWaterfall } from "@/components/SpanWaterfall";
import { Badge, Card, EmptyState, ErrorState, Loading, PageHeader } from "@/components/ui";

export default function TracesPage() {
  const [traces, setTraces] = useState<TraceSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [minLatency, setMinLatency] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [selected, setSelected] = useState<TraceDetail | null>(null);
  const [loadingTrace, setLoadingTrace] = useState(false);

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

  const openTrace = async (id: string) => {
    setLoadingTrace(true);
    setSelected(null);
    try {
      setSelected(await api.getTrace(id));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load trace");
    } finally {
      setLoadingTrace(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="Trace Viewer"
        subtitle="Every query persists its OpenTelemetry span tree. Click a trace to see the waterfall — where the milliseconds, tokens, and cost actually go."
        actions={
          <button
            onClick={load}
            className="flex items-center gap-2 rounded-lg border border-border bg-surface-container px-3 py-2 text-sm text-on-surface hover:border-primary/50"
          >
            <RefreshCw className="h-4 w-4" /> Refresh
          </button>
        }
      />

      <Card className="mb-6">
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-xs text-on-surface-muted">
            Min latency (ms)
            <input
              type="number"
              value={minLatency}
              onChange={(e) => setMinLatency(e.target.value)}
              placeholder="0"
              className="mt-1 block w-32 rounded-md border border-border bg-background px-2 py-1.5 font-mono text-sm text-on-surface"
            />
          </label>
          <label className="text-xs text-on-surface-muted">
            Status
            <select
              value={status}
              onChange={(e) => setStatus(e.target.value)}
              className="mt-1 block w-32 rounded-md border border-border bg-background px-2 py-1.5 text-sm text-on-surface"
            >
              <option value="">All</option>
              <option value="ok">ok</option>
              <option value="error">error</option>
            </select>
          </label>
          <button
            onClick={load}
            className="rounded-lg bg-primary/20 px-4 py-2 text-sm font-medium text-primary hover:bg-primary/30"
          >
            Apply
          </button>
        </div>
      </Card>

      {error && (
        <div className="mb-6">
          <ErrorState message={error} />
        </div>
      )}

      {traces === null ? (
        <Loading label="Loading traces…" />
      ) : traces.length === 0 && !error ? (
        <EmptyState
          title="No traces yet"
          hint="Run a query at POST /api/v1/query and it will appear here with its full span tree."
        />
      ) : (
        <div className="grid gap-6 lg:grid-cols-[1fr_1.3fr]">
          <Card className="overflow-hidden p-0">
            <div className="max-h-[70vh] overflow-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-surface-container text-left text-xs text-on-surface-muted">
                  <tr>
                    <th className="px-4 py-3 font-medium">Trace</th>
                    <th className="px-3 py-3 font-medium">Duration</th>
                    <th className="px-3 py-3 font-medium">Spans</th>
                    <th className="px-3 py-3 font-medium">Cost</th>
                    <th className="px-4 py-3 font-medium">Age</th>
                  </tr>
                </thead>
                <tbody>
                  {traces.map((t) => (
                    <tr
                      key={t.id}
                      onClick={() => openTrace(t.id)}
                      className={`cursor-pointer border-t border-border hover:bg-surface-container-high ${
                        selected?.id === t.id ? "bg-primary/10" : ""
                      }`}
                    >
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs text-on-surface">
                            {t.name ?? shortId(t.otel_trace_id)}
                          </span>
                          {t.status === "error" && (
                            <Badge color="#ffb4ab">error</Badge>
                          )}
                        </div>
                        <div className="font-mono text-[10px] text-on-surface-muted">
                          {shortId(t.otel_trace_id, 16)}
                        </div>
                      </td>
                      <td className="px-3 py-3 font-mono text-xs text-on-surface">
                        {ms(t.total_duration_ms)}
                      </td>
                      <td className="px-3 py-3 font-mono text-xs text-on-surface-muted">
                        {t.span_count}
                      </td>
                      <td className="px-3 py-3 font-mono text-xs text-on-surface-muted">
                        {usd(t.total_cost_usd)}
                      </td>
                      <td className="px-4 py-3 text-xs text-on-surface-muted">
                        {timeAgo(t.created_at)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card>
            {loadingTrace ? (
              <Loading label="Loading span tree…" />
            ) : selected ? (
              <div>
                <div className="mb-4 flex flex-wrap items-center gap-x-6 gap-y-1">
                  <div>
                    <div className="text-xs text-on-surface-muted">Trace</div>
                    <div className="font-mono text-sm text-on-surface">
                      {selected.name ?? shortId(selected.otel_trace_id)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-on-surface-muted">Duration</div>
                    <div className="font-mono text-sm text-primary">
                      {ms(selected.total_duration_ms)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-on-surface-muted">Tokens</div>
                    <div className="font-mono text-sm text-on-surface">
                      {num(selected.total_tokens)}
                    </div>
                  </div>
                  <div>
                    <div className="text-xs text-on-surface-muted">Cost</div>
                    <div className="font-mono text-sm text-on-surface">
                      {usd(selected.total_cost_usd)}
                    </div>
                  </div>
                </div>
                <SpanWaterfall spans={selected.spans} />
              </div>
            ) : (
              <EmptyState
                title="Select a trace"
                hint="Pick a trace on the left to render its span waterfall."
              />
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
