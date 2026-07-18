"use client";

import { useCallback, useEffect, useState } from "react";
import { Coins, Database, Gauge, Timer } from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "@/lib/api";
import type {
  CacheAnalytics,
  CostAnalytics,
  LatencyAnalytics,
  ThroughputAnalytics,
} from "@/lib/types";
import { ms, num, pct, usd } from "@/lib/format";
import { Card, ErrorState, Loading, PageHeader } from "@/components/ui";
import { MetricCard } from "@/components/MetricCard";

const tooltipStyle = {
  background: "#141b26",
  border: "1px solid #26313f",
  borderRadius: 8,
  fontSize: 12,
};
const axisTick = { fill: "#8fa0b0", fontSize: 10 };

export default function AnalyticsPage() {
  const [days, setDays] = useState(7);
  const [latency, setLatency] = useState<LatencyAnalytics | null>(null);
  const [cost, setCost] = useState<CostAnalytics | null>(null);
  const [cache, setCache] = useState<CacheAnalytics | null>(null);
  const [throughput, setThroughput] = useState<ThroughputAnalytics | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [l, c, ca, t] = await Promise.all([
        api.latency(days),
        api.cost(days),
        api.cache(days),
        api.throughput(days),
      ]);
      setLatency(l);
      setCost(c);
      setCache(ca);
      setThroughput(t);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load analytics");
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div>
      <PageHeader
        title="Cost & Latency Analytics"
        subtitle="Aggregated from the persisted query history: latency percentiles, cost per model, cache efficiency, and throughput."
        actions={
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="rounded-lg border border-border bg-surface-container px-3 py-2 text-sm text-on-surface"
          >
            <option value={1}>Last 24h</option>
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
        }
      />

      {error && (
        <div className="mb-6">
          <ErrorState message={error} />
        </div>
      )}

      {loading ? (
        <Loading label="Loading analytics…" />
      ) : (
        <div className="space-y-6">
          {/* KPI row */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="p95 Latency"
              value={ms(latency?.overall.p95 ?? null)}
              sublabel={`${num(latency?.overall.count ?? 0)} queries`}
              accent="secondary"
              icon={<Timer className="h-4 w-4" />}
            />
            <MetricCard
              label="Total Cost"
              value={usd(cost?.total_cost_usd ?? 0)}
              sublabel={`${num(cost?.total_tokens ?? 0)} tokens`}
              accent="primary"
              icon={<Coins className="h-4 w-4" />}
            />
            <MetricCard
              label="Cache Hit Rate"
              value={pct(cache?.hit_rate ?? 0)}
              sublabel={`${num(cache?.hits ?? 0)} hits / ${num(cache?.misses ?? 0)} miss`}
              accent="primary"
              icon={<Database className="h-4 w-4" />}
            />
            <MetricCard
              label="Throughput"
              value={`${throughput?.total_queries ?? 0}`}
              sublabel={`peak ${throughput?.peak_qps ?? 0} qps`}
              accent="muted"
              icon={<Gauge className="h-4 w-4" />}
            />
          </div>

          {/* Latency percentiles */}
          <Card>
            <h3 className="mb-4 font-semibold text-on-surface">
              Latency percentiles (p50 / p95 / p99)
            </h3>
            {latency && latency.series.length > 0 ? (
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={latency.series}
                    margin={{ top: 8, right: 8, bottom: 8, left: -16 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#26313f" />
                    <XAxis dataKey="bucket" tick={axisTick} />
                    <YAxis tick={axisTick} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Line type="monotone" dataKey="p50" stroke="#4cd6fb" dot={false} />
                    <Line type="monotone" dataKey="p95" stroke="#00d4aa" dot={false} />
                    <Line type="monotone" dataKey="p99" stroke="#ffb77a" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <NoData />
            )}
          </Card>

          <div className="grid gap-6 lg:grid-cols-2">
            {/* Cost over time */}
            <Card>
              <h3 className="mb-4 font-semibold text-on-surface">Cost over time</h3>
              {cost && cost.series.length > 0 ? (
                <div className="h-56 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart
                      data={cost.series}
                      margin={{ top: 8, right: 8, bottom: 8, left: -16 }}
                    >
                      <defs>
                        <linearGradient id="costFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#00d4aa" stopOpacity={0.5} />
                          <stop offset="100%" stopColor="#00d4aa" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#26313f" />
                      <XAxis dataKey="bucket" tick={axisTick} />
                      <YAxis tick={axisTick} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Area
                        type="monotone"
                        dataKey="cost_usd"
                        stroke="#00d4aa"
                        fill="url(#costFill)"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <NoData />
              )}
            </Card>

            {/* Throughput */}
            <Card>
              <h3 className="mb-4 font-semibold text-on-surface">
                Throughput (queries / day)
              </h3>
              {throughput && throughput.series.length > 0 ? (
                <div className="h-56 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={throughput.series}
                      margin={{ top: 8, right: 8, bottom: 8, left: -16 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#26313f" />
                      <XAxis dataKey="bucket" tick={axisTick} />
                      <YAxis tick={axisTick} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Bar dataKey="count" fill="#00b4d8" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <NoData />
              )}
            </Card>
          </div>

          {/* Cost by model */}
          <Card>
            <h3 className="mb-4 font-semibold text-on-surface">Cost by model</h3>
            {cost && cost.by_model.length > 0 ? (
              <table className="w-full text-sm">
                <thead className="text-left text-xs text-on-surface-muted">
                  <tr>
                    <th className="py-2 font-medium">Model</th>
                    <th className="py-2 font-medium">Queries</th>
                    <th className="py-2 font-medium">Tokens</th>
                    <th className="py-2 font-medium">Cost</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-xs">
                  {cost.by_model.map((m) => (
                    <tr key={m.model} className="border-t border-border">
                      <td className="py-2 text-on-surface">{m.model}</td>
                      <td className="py-2 text-on-surface-muted">{num(m.queries)}</td>
                      <td className="py-2 text-on-surface-muted">{num(m.tokens)}</td>
                      <td className="py-2 text-primary">{usd(m.cost_usd)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <NoData />
            )}
            {cache && (
              <p className="mt-4 text-xs text-on-surface-muted">
                Estimated cache savings:{" "}
                <span className="font-mono text-primary">
                  {usd(cache.estimated_savings_usd)}
                </span>{" "}
                across {num(cache.hits)} hits (threshold {cache.threshold}).
              </p>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}

function NoData() {
  return (
    <div className="flex h-40 items-center justify-center text-sm text-on-surface-muted">
      No data in this window yet — run some queries.
    </div>
  );
}
