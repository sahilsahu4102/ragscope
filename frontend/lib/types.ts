// RAGScope — shared API types (mirror backend Pydantic schemas)

export type SpanKind =
  | "CHAIN"
  | "RETRIEVER"
  | "RERANKER"
  | "LLM"
  | "EMBEDDING"
  | "TOOL"
  | "AGENT"
  | "GUARDRAIL"
  | "EVALUATOR"
  | "PROMPT";

export interface Span {
  id: string;
  otel_span_id: string;
  parent_span_id: string | null;
  span_kind: SpanKind;
  name: string;
  status: string;
  start_time: string | null;
  end_time: string | null;
  duration_ms: number | null;
  tokens: number | null;
  cost_usd: number | null;
  attributes: Record<string, unknown> | null;
}

export interface TraceSummary {
  id: string;
  otel_trace_id: string;
  query_id: string | null;
  name: string | null;
  status: string;
  total_duration_ms: number | null;
  total_tokens: number | null;
  total_cost_usd: number | null;
  span_count: number;
  created_at: string;
}

export interface TraceDetail extends TraceSummary {
  spans: Span[];
}

export interface MetricDelta {
  a: number;
  b: number;
  delta: number;
  pct_change: number | null;
  winner: "A" | "B" | "tie";
  significant?: boolean;
  p_value?: number | null;
  n?: number;
}

export interface Experiment {
  id: string;
  name: string;
  description: string | null;
  dataset_id: string;
  status: string;
  config_a: Record<string, unknown> | null;
  config_b: Record<string, unknown> | null;
  run_a_id: string | null;
  run_b_id: string | null;
  metrics_a: Record<string, number> | null;
  metrics_b: Record<string, number> | null;
  deltas: Record<string, MetricDelta> | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface Dataset {
  id: string;
  name: string;
  version: string;
  description: string | null;
  sample_count: number;
  created_at: string;
}

export interface Percentiles {
  p50: number | null;
  p95: number | null;
  p99: number | null;
  count: number;
}

export interface LatencyPoint {
  bucket: string;
  p50: number | null;
  p95: number | null;
  p99: number | null;
  count: number;
}

export interface LatencyAnalytics {
  overall: Percentiles;
  series: LatencyPoint[];
}

export interface ModelCost {
  model: string;
  queries: number;
  tokens: number;
  cost_usd: number;
}

export interface CostPoint {
  bucket: string;
  cost_usd: number;
  queries: number;
}

export interface CostAnalytics {
  total_cost_usd: number;
  total_tokens: number;
  avg_cost_per_query: number;
  by_model: ModelCost[];
  series: CostPoint[];
}

export interface CacheAnalytics {
  entries: number;
  hits: number;
  misses: number;
  hit_rate: number;
  threshold: number;
  ttl_seconds: number;
  estimated_savings_usd: number;
}

export interface ThroughputPoint {
  bucket: string;
  count: number;
  qps: number;
}

export interface ThroughputAnalytics {
  total_queries: number;
  avg_qps: number;
  peak_qps: number;
  series: ThroughputPoint[];
}
