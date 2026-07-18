"use client";

import { useEffect, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  Minus,
  Play,
  RefreshCw,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "@/lib/api";
import type { Dataset, Experiment, MetricDelta } from "@/lib/types";
import { timeAgo } from "@/lib/format";
import {
  Badge,
  Card,
  EmptyState,
  ErrorState,
  Loading,
  PageHeader,
} from "@/components/ui";

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

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[] | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Experiment | null>(null);

  // create form
  const [name, setName] = useState("Reranker: on vs off");
  const [datasetId, setDatasetId] = useState("");
  const [preset, setPreset] = useState("Reranker: on vs off");
  const [running, setRunning] = useState(false);

  const load = async () => {
    setError(null);
    try {
      const [exps, ds] = await Promise.all([
        api.listExperiments(),
        api.listDatasets().catch(() => []),
      ]);
      setExperiments(exps);
      setDatasets(ds);
      if (ds.length && !datasetId) setDatasetId(ds[0].id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load experiments");
      setExperiments([]);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const run = async () => {
    if (!datasetId) {
      setError("Create/select a dataset first (POST /api/v1/datasets).");
      return;
    }
    setRunning(true);
    setError(null);
    try {
      const cfg = PRESETS[preset];
      const exp = await api.createExperiment({
        name,
        dataset_id: datasetId,
        config_a: cfg.a,
        config_b: cfg.b,
      });
      setSelected(exp);
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Experiment run failed");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div>
      <PageHeader
        title="A/B Experiments"
        subtitle="Compare two pipeline configurations on the same golden dataset. Deltas carry a paired-bootstrap significance test — not just a difference of means."
        actions={
          <button
            onClick={load}
            className="flex items-center gap-2 rounded-lg border border-border bg-surface-container px-3 py-2 text-sm text-on-surface hover:border-primary/50"
          >
            <RefreshCw className="h-4 w-4" /> Refresh
          </button>
        }
      />

      {error && (
        <div className="mb-6">
          <ErrorState message={error} />
        </div>
      )}

      <Card className="mb-6">
        <h3 className="mb-4 font-semibold text-on-surface">New experiment</h3>
        <div className="grid gap-4 md:grid-cols-4">
          <label className="text-xs text-on-surface-muted md:col-span-2">
            Name
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="mt-1 block w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm text-on-surface"
            />
          </label>
          <label className="text-xs text-on-surface-muted">
            Dataset
            <select
              value={datasetId}
              onChange={(e) => setDatasetId(e.target.value)}
              className="mt-1 block w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm text-on-surface"
            >
              {datasets.length === 0 && <option value="">No datasets</option>}
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.sample_count})
                </option>
              ))}
            </select>
          </label>
          <label className="text-xs text-on-surface-muted">
            Comparison
            <select
              value={preset}
              onChange={(e) => {
                setPreset(e.target.value);
                setName(e.target.value);
              }}
              className="mt-1 block w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm text-on-surface"
            >
              {Object.keys(PRESETS).map((p) => (
                <option key={p}>{p}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="mt-4 flex items-center gap-3">
          <button
            onClick={run}
            disabled={running}
            className="flex items-center gap-2 rounded-lg bg-primary/20 px-4 py-2 text-sm font-medium text-primary hover:bg-primary/30 disabled:opacity-50"
          >
            <Play className="h-4 w-4" />
            {running ? "Running both variants…" : "Run experiment"}
          </button>
          <span className="font-mono text-xs text-on-surface-muted">
            A: {JSON.stringify(PRESETS[preset].a)} · B:{" "}
            {JSON.stringify(PRESETS[preset].b)}
          </span>
        </div>
      </Card>

      <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
        <Card className="p-0">
          <div className="border-b border-border px-4 py-3 text-xs font-medium uppercase tracking-wider text-on-surface-muted">
            History
          </div>
          {experiments === null ? (
            <div className="p-5">
              <Loading />
            </div>
          ) : experiments.length === 0 ? (
            <div className="p-5">
              <EmptyState title="No experiments yet" hint="Run one above." />
            </div>
          ) : (
            <ul className="divide-y divide-border">
              {experiments.map((e) => (
                <li key={e.id}>
                  <button
                    onClick={() => setSelected(e)}
                    className={`flex w-full items-center justify-between px-4 py-3 text-left hover:bg-surface-container-high ${
                      selected?.id === e.id ? "bg-primary/10" : ""
                    }`}
                  >
                    <div>
                      <div className="text-sm text-on-surface">{e.name}</div>
                      <div className="text-[11px] text-on-surface-muted">
                        {timeAgo(e.created_at)}
                      </div>
                    </div>
                    <Badge
                      color={
                        e.status === "completed"
                          ? "#00d4aa"
                          : e.status === "failed"
                            ? "#ffb4ab"
                            : "#ffb77a"
                      }
                    >
                      {e.status}
                    </Badge>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <Card>
          {selected ? (
            <ExperimentResult experiment={selected} />
          ) : (
            <EmptyState
              title="Select an experiment"
              hint="Pick one from history to see the metric deltas and comparison chart."
            />
          )}
        </Card>
      </div>
    </div>
  );
}

function DeltaArrow({ delta }: { delta: MetricDelta }) {
  if (delta.winner === "tie")
    return <Minus className="inline h-3.5 w-3.5 text-on-surface-muted" />;
  const improved = delta.winner === "B";
  const Icon = improved ? ArrowUp : ArrowDown;
  return (
    <Icon
      className={`inline h-3.5 w-3.5 ${improved ? "text-primary" : "text-danger"}`}
    />
  );
}

function ExperimentResult({ experiment }: { experiment: Experiment }) {
  const deltas = experiment.deltas ?? {};
  const entries = Object.entries(deltas);

  if (experiment.status === "failed") {
    return <ErrorState message={experiment.error_message ?? "Experiment failed"} />;
  }
  if (entries.length === 0) {
    return (
      <EmptyState
        title="No metrics"
        hint="This experiment produced no comparable metrics."
      />
    );
  }

  const chartData = entries.map(([metric, d]) => ({
    metric: metric.replace("@", "@​"),
    A: Number(d.a?.toFixed?.(4) ?? d.a),
    B: Number(d.b?.toFixed?.(4) ?? d.b),
  }));

  return (
    <div>
      <h3 className="mb-1 font-semibold text-on-surface">{experiment.name}</h3>
      <p className="mb-4 text-xs text-on-surface-muted">
        Variant A vs Variant B · {entries[0]?.[1]?.n ?? "?"} samples
      </p>

      <div className="mb-6 h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 8, right: 8, bottom: 8, left: -16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#26313f" />
            <XAxis
              dataKey="metric"
              tick={{ fill: "#8fa0b0", fontSize: 10 }}
              interval={0}
              angle={-20}
              textAnchor="end"
              height={50}
            />
            <YAxis tick={{ fill: "#8fa0b0", fontSize: 10 }} />
            <Tooltip
              contentStyle={{
                background: "#141b26",
                border: "1px solid #26313f",
                borderRadius: 8,
                fontSize: 12,
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="A" fill="#00b4d8" radius={[3, 3, 0, 0]} />
            <Bar dataKey="B" fill="#00d4aa" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-on-surface-muted">
            <tr>
              <th className="py-2 pr-4 font-medium">Metric</th>
              <th className="py-2 pr-4 font-medium">A</th>
              <th className="py-2 pr-4 font-medium">B</th>
              <th className="py-2 pr-4 font-medium">Δ</th>
              <th className="py-2 pr-4 font-medium">%</th>
              <th className="py-2 font-medium">Sig.</th>
            </tr>
          </thead>
          <tbody className="font-mono text-xs">
            {entries.map(([metric, d]) => (
              <tr key={metric} className="border-t border-border">
                <td className="py-2 pr-4 text-on-surface">{metric}</td>
                <td className="py-2 pr-4 text-on-surface-muted">
                  {d.a?.toFixed?.(4) ?? d.a}
                </td>
                <td className="py-2 pr-4 text-on-surface-muted">
                  {d.b?.toFixed?.(4) ?? d.b}
                </td>
                <td className="py-2 pr-4">
                  <DeltaArrow delta={d} />{" "}
                  <span
                    className={
                      d.winner === "B"
                        ? "text-primary"
                        : d.winner === "A"
                          ? "text-danger"
                          : "text-on-surface-muted"
                    }
                  >
                    {d.delta > 0 ? "+" : ""}
                    {d.delta}
                  </span>
                </td>
                <td className="py-2 pr-4 text-on-surface-muted">
                  {d.pct_change == null ? "—" : `${d.pct_change}%`}
                </td>
                <td className="py-2">
                  {d.significant ? (
                    <Badge color="#00d4aa">p={d.p_value}</Badge>
                  ) : (
                    <span className="text-on-surface-muted">ns</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
