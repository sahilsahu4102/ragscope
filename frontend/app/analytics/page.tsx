"use client";

import { useCallback, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, ApiError } from "@/lib/api";
import type { CacheAnalytics, CostAnalytics, LatencyAnalytics, ThroughputAnalytics } from "@/lib/types";
import { ageDays, dateTime, ms, num, pct, usd } from "@/lib/format";
import { StateMark } from "@/components/Marks";
import { Entry, ErrorState, Figure, Loading, PageHeader } from "@/components/ui";

const WINDOWS = [
  { value: 1, label: "Last 24 hours" },
  { value: 7, label: "Last 7 days" },
  { value: 30, label: "Last 30 days" },
  { value: 90, label: "Last 90 days" },
];

const tooltipStyle = { background: "#fbf9f3", border: "1.5px solid #14182b", borderRadius: 0, fontSize: 12 };
const tick = { fill: "#3a425a", fontSize: 11 };
const GRID = "#cfd8ea";

export default function AnalyticsPage() {
  const [days, setDays] = useState(7);
  const [latency, setLatency] = useState<LatencyAnalytics | null>(null);
  const [cost, setCost] = useState<CostAnalytics | null>(null);
  const [cache, setCache] = useState<CacheAnalytics | null>(null);
  const [throughput, setThroughput] = useState<ThroughputAnalytics | null>(null);
  const [newest, setNewest] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [l, c, ca, t] = await Promise.all([api.latency(days), api.cost(days), api.cache(days), api.throughput(days)]);
      setLatency(l);
      setCost(c);
      setCache(ca);
      setThroughput(t);
      if (l.overall.count === 0) {
        // Say why it is empty: how old is the newest trace the API holds?
        const latest = await api.listTraces({ limit: 1 }).catch(() => []);
        setNewest(latest[0]?.created_at ?? null);
      } else {
        setNewest(null);
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load analytics");
    } finally {
      setLoading(false);
    }
  }, [days]);

  useEffect(() => {
    load();
  }, [load]);

  const windowLabel = WINDOWS.find((w) => w.value === days)?.label.toLowerCase() ?? `last ${days} days`;
  const count = latency?.overall.count ?? 0;

  return (
    <>
      <PageHeader
        title="Cost & Latency Analytics"
        subtitle="Aggregated from the persisted query history: latency percentiles, cost per model, cache efficiency and throughput. The API reports at most the last 90 days."
        actions={
          <label className="field" style={{ width: 200 }}>
            <span className="sr-only">Time window</span>
            <select className="select" value={days} onChange={(e) => setDays(Number(e.target.value))}>
              {WINDOWS.map((w) => <option key={w.value} value={w.value}>{w.label}</option>)}
            </select>
          </label>
        }
      />

      {error && <div style={{ marginBottom: 24 }}><ErrorState message={error} onRetry={load} /></div>}

      {loading ? (
        <Loading label="Loading analytics" />
      ) : (
        <>
          <Entry kind="Summary" source={`GET /api/v1/analytics, ${windowLabel}`}>
            {count === 0 && (
              <div className="note is-wait" style={{ marginBottom: 20 }}>
                <StateMark kind="wait" />
                <div>
                  <b>No queries in the {windowLabel}</b>
                  <p>
                    {newest
                      ? <>The newest trace in the database is {ageDays(newest)} days old ({dateTime(newest)}), which is {ageDays(newest) > 90 ? "outside the 90-day window the API reports on" : "outside this window"}. Ask a question in the Playground to produce fresh data.</>
                      : <>No traces exist yet. Ask a question in the Playground to produce data.</>}
                  </p>
                </div>
              </div>
            )}
            <div className="cols-even cols" style={{ gap: "8px 40px" }}>
              <dl className="deflist">
                <dt>Queries</dt><dd>{num(count)}</dd>
                <dt>Latency p50</dt><dd>{ms(latency?.overall.p50 ?? null)}</dd>
                <dt>Latency p95</dt><dd>{ms(latency?.overall.p95 ?? null)}</dd>
                <dt>Latency p99</dt><dd>{ms(latency?.overall.p99 ?? null)}</dd>
                <dt>Throughput</dt><dd>{num(throughput?.total_queries ?? 0)} {throughput?.total_queries === 1 ? "query" : "queries"}, peak {throughput?.peak_qps ?? 0} qps</dd>
              </dl>
              <dl className="deflist">
                <dt>Total cost</dt><dd>{usd(cost?.total_cost_usd ?? 0)}</dd>
                <dt>Tokens</dt><dd>{num(cost?.total_tokens ?? 0)}</dd>
                <dt>Cost per query</dt><dd>{usd(cost?.avg_cost_per_query ?? 0)}</dd>
                <dt>Cache</dt><dd>{num(cache?.hits ?? 0)} hits, {num(cache?.misses ?? 0)} misses, {pct(cache?.hit_rate ?? 0)}</dd>
                <dt>Cache savings</dt><dd>{usd(cache?.estimated_savings_usd ?? 0)} estimated</dd>
              </dl>
            </div>
          </Entry>

          <Entry kind="Latency" source={`GET /api/v1/analytics/latency?days=${days}`}>
            {latency && latency.series.length > 0 ? (
              <Figure
                caption={<><b>Latency percentiles per day</b> over the {windowLabel}, from {num(count)} {count === 1 ? "query" : "queries"}. The solid line is p50, the dashed line p95 and the dotted line p99.</>}
              >
                <div className="plot" style={{ height: 280 }} role="img" aria-label="Line chart of p50, p95 and p99 latency per day">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={latency.series} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
                      <CartesianGrid stroke={GRID} />
                      <XAxis dataKey="bucket" tick={tick} />
                      <YAxis tick={tick} />
                      <Tooltip contentStyle={tooltipStyle} />
                      <Line type="monotone" dataKey="p50" stroke="#14182b" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="p95" stroke="#2b5fb0" strokeWidth={2} strokeDasharray="7 4" dot={false} />
                      <Line type="monotone" dataKey="p99" stroke="#8a5a10" strokeWidth={2} strokeDasharray="2 4" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Figure>
            ) : (
              <p className="lede">No latency data in the {windowLabel}.</p>
            )}
          </Entry>

          <div className="cols-even cols">
            <Entry kind="Cost over time" source={`GET /api/v1/analytics/cost?days=${days}`}>
              {cost && cost.series.length > 0 ? (
                <Figure caption={<><b>Cost per day</b> over the {windowLabel}, in US dollars, {num(cost.total_tokens)} tokens in total.</>}>
                  <div className="plot" style={{ height: 240 }} role="img" aria-label="Line chart of cost per day">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={cost.series} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
                        <CartesianGrid stroke={GRID} />
                        <XAxis dataKey="bucket" tick={tick} />
                        <YAxis tick={tick} />
                        <Tooltip contentStyle={tooltipStyle} />
                        <Line type="monotone" dataKey="cost_usd" stroke="#14182b" strokeWidth={2} dot={{ r: 3, fill: "#14182b" }} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Figure>
              ) : (
                <p className="lede">No cost data in the {windowLabel}.</p>
              )}
            </Entry>

            <Entry kind="Throughput" source={`GET /api/v1/analytics/throughput?days=${days}`}>
              {throughput && throughput.series.length > 0 ? (
                <Figure caption={<><b>Queries per day</b> over the {windowLabel}; {num(throughput.total_queries)} in total, average {throughput.avg_qps} qps.</>}>
                  <div className="plot" style={{ height: 240 }} role="img" aria-label="Bar chart of queries per day">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={throughput.series} margin={{ top: 8, right: 12, bottom: 8, left: -8 }}>
                        <CartesianGrid stroke={GRID} />
                        <XAxis dataKey="bucket" tick={tick} />
                        <YAxis tick={tick} />
                        <Tooltip contentStyle={tooltipStyle} />
                        <Bar dataKey="count" fill="#14182b" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Figure>
              ) : (
                <p className="lede">No throughput data in the {windowLabel}.</p>
              )}
            </Entry>
          </div>

          <Entry kind="Cost by model" source={`GET /api/v1/analytics/cost?days=${days}`}>
            {cost && cost.by_model.length > 0 ? (
              <div className="tab-wrap">
                <table className="tab">
                  <caption className="sr-only">Cost by model</caption>
                  <thead><tr><th>Model</th><th className="num">Queries</th><th className="num">Tokens</th><th className="num">Cost</th></tr></thead>
                  <tbody>
                    {cost.by_model.map((m) => (
                      <tr key={m.model}>
                        <td className="mono">{m.model}</td>
                        <td className="num">{num(m.queries)}</td>
                        <td className="num">{num(m.tokens)}</td>
                        <td className="num">{usd(m.cost_usd)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="lede">No per-model cost in the {windowLabel}.</p>
            )}
            {cache && (
              <p className="lede" style={{ marginTop: 16 }}>
                The semantic cache returns a stored answer when a past query is at least {cache.threshold} similar; entries live {Math.round(cache.ttl_seconds / 60)} minutes.
              </p>
            )}
          </Entry>
        </>
      )}
    </>
  );
}
