// RAGScope — typed API client (fetch wrapper with error handling)

import type {
  CacheAnalytics,
  CostAnalytics,
  Dataset,
  Experiment,
  LatencyAnalytics,
  ThroughputAnalytics,
  TraceDetail,
  TraceSummary,
} from "./types";

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch {
    throw new ApiError(
      "Could not reach the RAGScope API. Is the backend running?",
      0,
    );
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore non-JSON error bodies */
    }
    throw new ApiError(detail, res.status);
  }
  return (await res.json()) as T;
}

export const api = {
  // ── Traces ──
  listTraces: (params?: {
    limit?: number;
    minLatencyMs?: number;
    status?: string;
  }) => {
    const q = new URLSearchParams();
    if (params?.limit) q.set("limit", String(params.limit));
    if (params?.minLatencyMs != null)
      q.set("min_latency_ms", String(params.minLatencyMs));
    if (params?.status) q.set("status", params.status);
    const qs = q.toString();
    return request<TraceSummary[]>(`/traces${qs ? `?${qs}` : ""}`);
  },
  getTrace: (id: string) => request<TraceDetail>(`/traces/${id}`),

  // ── Experiments ──
  listExperiments: () => request<Experiment[]>("/experiments"),
  getExperiment: (id: string) => request<Experiment>(`/experiments/${id}`),
  createExperiment: (payload: {
    name: string;
    description?: string;
    dataset_id: string;
    config_a: Record<string, unknown>;
    config_b: Record<string, unknown>;
    use_llm_judge?: boolean;
  }) =>
    request<Experiment>("/experiments", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  // ── Datasets ──
  listDatasets: () => request<Dataset[]>("/datasets"),

  // ── Analytics ──
  latency: (days = 7) =>
    request<LatencyAnalytics>(`/analytics/latency?days=${days}`),
  cost: (days = 7) => request<CostAnalytics>(`/analytics/cost?days=${days}`),
  cache: (days = 7) => request<CacheAnalytics>(`/analytics/cache?days=${days}`),
  throughput: (days = 7) =>
    request<ThroughputAnalytics>(`/analytics/throughput?days=${days}`),

  // ── Feedback ──
  submitFeedback: (payload: {
    query_id?: string;
    trace_id?: string;
    rating: number;
    correction?: string;
  }) =>
    request<{ id: string; query_id: string; message: string }>("/feedback", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};
