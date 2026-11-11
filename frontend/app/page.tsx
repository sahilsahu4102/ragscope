"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { ArrowRight } from "lucide-react";
import { api, ROOT_URL } from "@/lib/api";
import type { CacheAnalytics, Dataset, Experiment, IngestedDocument, TraceSummary } from "@/lib/types";
import { ageDays, dateTime, ms, num, pct, shortId, timeAgo } from "@/lib/format";
import { useStackContext } from "@/components/StackContext";
import { StateLine, StateMark, type StateKind } from "@/components/Marks";
import { Entry, EmptyState, ErrorState, Loading, PageHeader } from "@/components/ui";

type Slot<T> = { data: T | null; error: string | null };
const empty = <T,>(): Slot<T> => ({ data: null, error: null });

const EXP_STATE: Record<string, StateKind> = { completed: "ok", running: "wait", failed: "down" };

function checkState(v: string | undefined): { kind: StateKind; word: string } {
  if (v === "ok") return { kind: "ok", word: "reachable" };
  if (v === "error") return { kind: "down", word: "not reachable" };
  return { kind: "wait", word: "not reported" };
}

export default function OverviewPage() {
  const stack = useStackContext();
  const [docs, setDocs] = useState<Slot<IngestedDocument[]>>(empty());
  const [traces, setTraces] = useState<Slot<TraceSummary[]>>(empty());
  const [exps, setExps] = useState<Slot<Experiment[]>>(empty());
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [cache, setCache] = useState<Slot<CacheAnalytics>>(empty());

  const load = useCallback(async () => {
    const [d, t, e, ds, c] = await Promise.allSettled([
      api.listDocuments(),
      api.listTraces({ limit: 20 }),
      api.listExperiments(),
      api.listDatasets(),
      api.cache(90),
    ]);
    const err = (r: PromiseRejectedResult) => (r.reason instanceof Error ? r.reason.message : "Request failed");
    setDocs(d.status === "fulfilled" ? { data: d.value, error: null } : { data: null, error: err(d) });
    setTraces(t.status === "fulfilled" ? { data: t.value, error: null } : { data: null, error: err(t) });
    setExps(e.status === "fulfilled" ? { data: e.value, error: null } : { data: null, error: err(e) });
    setCache(c.status === "fulfilled" ? { data: c.value, error: null } : { data: null, error: err(c) });
    if (ds.status === "fulfilled") setDatasets(ds.value);
  }, []);

  // Load on arrival, and again whenever the API comes back from down or starting.
  const reachable = stack.status === "ready" || stack.status === "degraded";
  useEffect(() => {
    if (reachable) load();
  }, [reachable, load]);

  // Before the API answers, nothing is being read yet: say so instead of showing a spinner that never ends.
  const pending = (label: string) =>
    reachable ? <Loading label={label} /> : stack.status === "down" ? <p className="lede">Not loaded: the API is unreachable.</p> : <Loading label="Waiting for the API" />;

  const completed = docs.data?.filter((x) => x.status === "completed") ?? [];
  const chunks = completed.reduce((n, x) => n + x.chunk_count, 0);
  const inFlight = docs.data?.filter((x) => x.status === "pending" || x.status === "processing").length ?? 0;
  const failed = docs.data?.filter((x) => x.status === "failed").length ?? 0;

  const latestTraces = [...(traces.data ?? [])].sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 5);
  const latestExps = [...(exps.data ?? [])].sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 5);
  const dsName = (id: string) => datasets.find((d) => d.id === id)?.name;

  return (
    <>
      <PageHeader
        title="Overview"
        subtitle={<>Everything here is read from the API at <code>{ROOT_URL}</code> when this page loads. Nothing is typed in by hand.</>}
      />

      <div className="cols">
        <div>
          <Entry kind="Stack" source="GET /readyz" actions={stack.checkedAt ? <span>checked {dateTime(stack.checkedAt)}</span> : null}>
            {stack.status === "down" ? (
              <ErrorState message={`Could not reach the API at ${ROOT_URL}`} onRetry={stack.retry}>
                <p>Start the stack with <code>docker compose up -d</code>, or check <code>NEXT_PUBLIC_API_URL</code>.</p>
              </ErrorState>
            ) : stack.status === "checking" || stack.status === "waking" ? (
              <div className="note is-wait" role="status">
                <StateMark kind="wait" />
                <div>
                  <b>{stack.status === "waking" ? "The API is waking up" : "Checking the API"}</b>
                  {stack.status === "waking" && <p>A host that sleeps when idle can take about a minute to answer its first request.</p>}
                </div>
              </div>
            ) : (
              <dl className="deflist">
                <dt>API</dt>
                <dd><StateLine kind="ok">answering</StateLine></dd>
                <dt>Database</dt>
                <dd><StateLine kind={checkState(stack.checks.postgres).kind}>{checkState(stack.checks.postgres).word}</StateLine></dd>
                <dt>Cache</dt>
                <dd><StateLine kind={checkState(stack.checks.redis).kind}>{checkState(stack.checks.redis).word}</StateLine></dd>
                <dt>Semantic cache</dt>
                <dd>
                  {cache.data
                    ? `${num(cache.data.hits)} hits, ${num(cache.data.misses)} misses, ${pct(cache.data.hit_rate)} hit rate`
                    : cache.error ?? "loading"}
                </dd>
              </dl>
            )}
          </Entry>

          <Entry kind="Ingested" source="GET /api/v1/documents" actions={<Link href="/documents" className="btn btn-sm">Documents <ArrowRight aria-hidden="true" /></Link>}>
            {docs.error ? (
              <ErrorState message="Could not load documents" onRetry={load}><p>{docs.error}</p></ErrorState>
            ) : docs.data === null ? (
              pending("Reading the document list")
            ) : docs.data.length === 0 ? (
              <EmptyState title="No documents ingested yet" hint="Upload one on the Documents page; the playground and inspector have nothing to retrieve from until a document finishes processing." />
            ) : (
              <dl className="deflist">
                <dt>Ready</dt>
                <dd>{completed.length} document{completed.length === 1 ? "" : "s"}, {num(chunks)} embedded chunks</dd>
                {inFlight > 0 && (<><dt>In progress</dt><dd><StateLine kind="wait">{inFlight} being processed</StateLine></dd></>)}
                {failed > 0 && (<><dt>Failed</dt><dd><StateLine kind="down">{failed} failed to ingest</StateLine></dd></>)}
              </dl>
            )}
          </Entry>
        </div>

        <div>
          <Entry kind="Latest traces" source="GET /api/v1/traces" actions={<Link href="/traces" className="btn btn-sm">All traces <ArrowRight aria-hidden="true" /></Link>}>
            {traces.error ? (
              <ErrorState message="Could not load traces" onRetry={load}><p>{traces.error}</p></ErrorState>
            ) : traces.data === null ? (
              pending("Reading the trace list")
            ) : latestTraces.length === 0 ? (
              <EmptyState title="No traces yet" hint="Ask a question in the Playground and its span tree will appear here." />
            ) : (
              <>
                {ageDays(latestTraces[0].created_at) >= 7 && (
                  <p className="lede" style={{ marginBottom: 12 }}>
                    The newest trace is {ageDays(latestTraces[0].created_at)} days old ({dateTime(latestTraces[0].created_at)}).
                  </p>
                )}
                <div className="tab-wrap">
                  <table className="tab">
                    <thead><tr><th>Trace</th><th className="num">Duration</th><th className="num hide-sm">Spans</th><th>Recorded</th></tr></thead>
                    <tbody>
                      {latestTraces.map((t) => (
                        <tr key={t.id}>
                          <td>
                            <Link href={`/traces?trace=${t.id}`}>{t.name ?? shortId(t.otel_trace_id)}</Link>
                            <span className="id">{shortId(t.otel_trace_id, 16)}</span>
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
              </>
            )}
          </Entry>

          <Entry kind="Latest experiments" source="GET /api/v1/experiments" actions={<Link href="/experiments" className="btn btn-sm">All experiments <ArrowRight aria-hidden="true" /></Link>}>
            {exps.error ? (
              <ErrorState message="Could not load experiments" onRetry={load}><p>{exps.error}</p></ErrorState>
            ) : exps.data === null ? (
              pending("Reading the experiment list")
            ) : latestExps.length === 0 ? (
              <EmptyState title="No experiments yet" hint="Run an A/B comparison on a dataset from the Experiments page." />
            ) : (
              <div className="tab-wrap">
                <table className="tab">
                  <thead><tr><th>Experiment</th><th>Status</th><th>Recorded</th></tr></thead>
                  <tbody>
                    {latestExps.map((e) => (
                      <tr key={e.id}>
                        <td>
                          <Link href={`/experiments?exp=${e.id}`}>{e.name}</Link>
                          {dsName(e.dataset_id) && <span className="id">dataset {dsName(e.dataset_id)}</span>}
                        </td>
                        <td><StateLine kind={EXP_STATE[e.status] ?? "off"}>{e.status}</StateLine></td>
                        <td>
                          {dateTime(e.created_at)}
                          <span className="id">{timeAgo(e.created_at)}</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Entry>
        </div>
      </div>
    </>
  );
}
