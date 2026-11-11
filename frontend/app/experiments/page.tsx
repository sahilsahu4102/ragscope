"use client";

import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Play, RefreshCw } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, ApiError } from "@/lib/api";
import type { Dataset, Experiment, MetricDelta } from "@/lib/types";
import { dateTime, timeAgo } from "@/lib/format";
import { StateLine, type StateKind } from "@/components/Marks";
import { Chip, Entry, EmptyState, ErrorState, Figure, Loading, PageHeader } from "@/components/ui";

const PRESETS: Record<string, { a: Record<string, unknown>; b: Record<string, unknown> }> = {
  "Reranker: on vs off": {
    a: { use_reranker: true, use_hybrid: true },
    b: { use_reranker: false, use_hybrid: true },
  },
  "Hybrid vs dense-only": {
    a: { use_hybrid: true, use_reranker: false },
    b: { use_hybrid: false, use_reranker: false },
  },
  "HyDE vs no transform": {
    a: { query_transform: "hyde", use_hybrid: true },
    b: { query_transform: "none", use_hybrid: true },
  },
};

const STATE: Record<string, StateKind> = { completed: "ok", running: "wait", failed: "down" };
const SEEN_KEY = "ragscope.seenExperiments";

function readSeen(): Set<string> | null {
  try {
    const raw = localStorage.getItem(SEEN_KEY);
    return raw ? new Set(JSON.parse(raw) as string[]) : null;
  } catch {
    return null;
  }
}

function ExperimentsInner() {
  const params = useSearchParams();
  const wanted = params.get("exp");

  const [experiments, setExperiments] = useState<Experiment[] | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Experiment | null>(null);
  const [seen, setSeen] = useState<Set<string>>(new Set());
  const opened = useRef(false);

  const [name, setName] = useState("Reranker: on vs off");
  const [datasetId, setDatasetId] = useState("");
  const [preset, setPreset] = useState("Reranker: on vs off");
  const [running, setRunning] = useState(false);

  const markSeen = useCallback((id: string) => {
    setSeen((prev) => {
      const next = new Set(prev).add(id);
      try { localStorage.setItem(SEEN_KEY, JSON.stringify([...next])); } catch { /* storage unavailable */ }
      return next;
    });
  }, []);

  const select = useCallback((e: Experiment) => {
    setSelected(e);
    markSeen(e.id);
  }, [markSeen]);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [exps, ds] = await Promise.all([api.listExperiments(), api.listDatasets().catch(() => [])]);
      setExperiments(exps);
      setDatasets(ds);
      setDatasetId((cur) => cur || (ds.length ? ds[0].id : ""));
      // A change you have not looked at keeps a marker. The first visit marks nothing: the history is not news.
      const stored = readSeen();
      if (stored === null) {
        const all = new Set(exps.map((x) => x.id));
        try { localStorage.setItem(SEEN_KEY, JSON.stringify([...all])); } catch { /* storage unavailable */ }
        setSeen(all);
      } else {
        setSeen(stored);
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load experiments");
      setExperiments([]);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (wanted && experiments && !opened.current) {
      const found = experiments.find((x) => x.id === wanted);
      if (found) {
        opened.current = true;
        select(found);
      }
    }
  }, [wanted, experiments, select]);

  const run = async () => {
    if (!datasetId) {
      setError("Select a dataset first (POST /api/v1/datasets creates one).");
      return;
    }
    setRunning(true);
    setError(null);
    try {
      const cfg = PRESETS[preset];
      const exp = await api.createExperiment({ name, dataset_id: datasetId, config_a: cfg.a, config_b: cfg.b });
      select(exp);
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Experiment run failed");
    } finally {
      setRunning(false);
    }
  };

  const sorted = [...(experiments ?? [])].sort((a, b) => b.created_at.localeCompare(a.created_at));

  return (
    <>
      <PageHeader
        title="A/B Experiments"
        subtitle="Compare two pipeline configurations on the same golden dataset. Each difference carries a paired-bootstrap significance test, not just a difference of means."
        actions={<button type="button" onClick={load} className="btn"><RefreshCw aria-hidden="true" /> Refresh</button>}
      />

      {error && <div style={{ marginBottom: 24 }}><ErrorState message={error} onRetry={load} /></div>}

      <div className="cols">
        <div>
          <Entry kind="New experiment" source="POST /api/v1/experiments">
            <div className="stack">
              <label className="field">
                <span className="lab">Name</span>
                <input className="input" value={name} onChange={(e) => setName(e.target.value)} />
              </label>
              <div className="row">
                <label className="field" style={{ flex: "1 1 180px" }}>
                  <span className="lab">Dataset</span>
                  <select className="select" value={datasetId} onChange={(e) => setDatasetId(e.target.value)}>
                    {datasets.length === 0 && <option value="">No datasets</option>}
                    {datasets.map((d) => (
                      <option key={d.id} value={d.id}>{d.name} ({d.sample_count} samples)</option>
                    ))}
                  </select>
                </label>
                <label className="field" style={{ flex: "1 1 180px" }}>
                  <span className="lab">Comparison</span>
                  <select className="select" value={preset} onChange={(e) => { setPreset(e.target.value); setName(e.target.value); }}>
                    {Object.keys(PRESETS).map((p) => <option key={p}>{p}</option>)}
                  </select>
                </label>
              </div>
              <p className="mono" style={{ color: "var(--ink-2)", overflowWrap: "anywhere" }}>
                A: {JSON.stringify(PRESETS[preset].a)}<br />B: {JSON.stringify(PRESETS[preset].b)}
              </p>
              <div>
                <button type="button" className="btn btn-primary" onClick={run} disabled={running}>
                  <Play aria-hidden="true" /> {running ? "Running both variants" : "Run experiment"}
                </button>
              </div>
              {running && (
                <p className="lede" role="status">Both variants run over every sample, so this can take several minutes. The page will show the result when it finishes.</p>
              )}
            </div>
          </Entry>

          <Entry kind="History" source="GET /api/v1/experiments">
            {experiments === null ? (
              <Loading label="Loading experiments" />
            ) : sorted.length === 0 ? (
              <EmptyState title="No experiments yet" hint="Run one above." />
            ) : (
              <div className="tab-wrap">
                <table className="tab">
                  <caption className="sr-only">Experiments, newest first</caption>
                  <thead><tr><th>Experiment</th><th>Status</th></tr></thead>
                  <tbody>
                    {sorted.map((e) => (
                      <tr key={e.id} className={selected?.id === e.id ? "sel" : undefined} onClick={() => select(e)} style={{ cursor: "pointer" }}>
                        <td>
                          <button type="button" className="row-btn" aria-pressed={selected?.id === e.id} onClick={(ev) => { ev.stopPropagation(); select(e); }}>
                            {e.name}
                            <span className="id">{dateTime(e.created_at)} · {timeAgo(e.created_at)}</span>
                          </button>
                        </td>
                        <td>
                          <StateLine kind={STATE[e.status] ?? "off"}>{e.status}</StateLine>
                          {e.status === "completed" && !seen.has(e.id) && <> <Chip tone="on">new</Chip></>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Entry>
        </div>

        {selected ? (
          <ExperimentResult experiment={selected} dataset={datasets.find((d) => d.id === selected.dataset_id)} />
        ) : (
          <Entry kind="Experiment">
            <EmptyState title="No experiment open" hint="Pick one from the history to read its figure and the table behind it." />
          </Entry>
        )}
      </div>
    </>
  );
}

const fmt = (v: number | undefined) => (v == null ? "n/a" : Number.isFinite(v) ? v.toFixed(4) : String(v));
const WINNER: Record<MetricDelta["winner"], string> = { A: "A ahead", B: "B ahead", tie: "tie" };

function ExperimentResult({ experiment, dataset }: { experiment: Experiment; dataset?: Dataset }) {
  const entries = Object.entries(experiment.deltas ?? {});
  const n = entries[0]?.[1]?.n;

  const head = (
    <dl className="deflist" style={{ marginBottom: 20 }}>
      <dt>Status</dt><dd><StateLine kind={STATE[experiment.status] ?? "off"}>{experiment.status}</StateLine></dd>
      <dt>Dataset</dt><dd>{dataset ? `${dataset.name}, ${dataset.sample_count} samples` : experiment.dataset_id}</dd>
      <dt>Variant A</dt><dd style={{ overflowWrap: "anywhere" }}>{JSON.stringify(experiment.config_a)}</dd>
      <dt>Variant B</dt><dd style={{ overflowWrap: "anywhere" }}>{JSON.stringify(experiment.config_b)}</dd>
      {experiment.completed_at && (<><dt>Finished</dt><dd>{dateTime(experiment.completed_at)}</dd></>)}
    </dl>
  );

  if (experiment.status === "failed") {
    return (
      <Entry kind="Experiment" date={experiment.created_at} id={experiment.id} title={experiment.name}>
        {head}
        <ErrorState message={experiment.error_message ?? "The experiment failed"} />
      </Entry>
    );
  }
  if (entries.length === 0) {
    return (
      <Entry kind="Experiment" date={experiment.created_at} id={experiment.id} title={experiment.name}>
        {head}
        <EmptyState title="No comparable metrics" hint={experiment.status === "running" ? "This experiment is still running." : "This experiment produced no comparable metrics."} />
      </Entry>
    );
  }

  const chartData = entries.map(([metric, d]) => ({ metric, A: Number(fmt(d.a)), B: Number(fmt(d.b)) }));
  const significant = entries.filter(([, d]) => d.significant).length;

  return (
    <Entry kind="Experiment" date={experiment.created_at} id={experiment.id} title={experiment.name}>
      {head}
      <Figure
        caption={
          <>
            <b>Variant A against variant B</b>
            {n != null ? <> on {n} samples</> : null}
            {dataset ? <> of the dataset {dataset.name}</> : null}. Solid bars are the mean of each metric for A, hatched bars for B, drawn on the same axes.
            {" "}{significant} of {entries.length} differences {significant === 1 ? "is" : "are"} significant by a paired bootstrap; the table gives each p-value.
          </>
        }
      >
        <div className="plot" style={{ height: 280 }} role="img" aria-label={`Bar chart of ${entries.length} metrics for variant A and variant B`}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 8, left: -16 }}>
              <defs>
                <pattern id="hatchB" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <rect width="6" height="6" fill="#fbf9f3" />
                  <rect width="3" height="6" fill="#2b5fb0" />
                </pattern>
              </defs>
              <CartesianGrid stroke="#cfd8ea" />
              <XAxis dataKey="metric" tick={{ fill: "#3a425a", fontSize: 11 }} interval={0} angle={-20} textAnchor="end" height={54} />
              <YAxis tick={{ fill: "#3a425a", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "var(--sheet)", border: "1.5px solid var(--ink)", borderRadius: 0, fontSize: 12 }} />
              <Bar dataKey="A" fill="#14182b" />
              <Bar dataKey="B" fill="url(#hatchB)" stroke="#2b5fb0" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Figure>

      <div className="tab-wrap" style={{ marginTop: 24 }}>
        <table className="tab">
          <caption className="sr-only">Metric values for variants A and B</caption>
          <thead>
            <tr><th>Metric</th><th className="num">A</th><th className="num">B</th><th className="num hide-sm">Difference</th><th className="num hide-sm">Change</th><th className="hide-sm">Verdict</th></tr>
          </thead>
          <tbody>
            {entries.map(([metric, d]) => (
              <tr key={metric}>
                <td className="mono">
                  {metric}
                  <span className="id only-sm">{WINNER[d.winner]}, {d.significant ? `significant, p = ${d.p_value}` : "not significant"}</span>
                </td>
                <td className="num">{fmt(d.a)}</td>
                <td className="num">{fmt(d.b)}</td>
                <td className="num hide-sm">{d.delta > 0 ? "+" : ""}{d.delta}</td>
                <td className="num hide-sm">{d.pct_change == null ? "n/a" : `${d.pct_change}%`}</td>
                <td className="hide-sm">
                  {WINNER[d.winner]}
                  <span className="id">{d.significant ? `significant, p = ${d.p_value}` : "not significant"}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Entry>
  );
}

export default function ExperimentsPage() {
  return (
    <Suspense fallback={<Loading label="Loading" />}>
      <ExperimentsInner />
    </Suspense>
  );
}
