// RAGScope — typed API client (fetch wrapper with error handling)

import type {
  CacheAnalytics,
  CostAnalytics,
  Dataset,
  DocumentDeleteResponse,
  Experiment,
  IngestedDocument,
  IngestResponse,
  LatencyAnalytics,
  QueryRequest,
  QueryResponse,
  RetrieveRequest,
  RetrieveResponse,
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
    // FormData must set its own multipart Content-Type (it carries the
    // boundary), so only force JSON when we're not uploading a file.
    const isForm =
      typeof FormData !== "undefined" && init?.body instanceof FormData;
    res = await fetch(`${BASE_URL}${path}`, {
      ...init,
      headers: isForm
        ? { ...(init?.headers ?? {}) }
        : { "Content-Type": "application/json", ...(init?.headers ?? {}) },
      cache: "no-store",
    });
  } catch (e) {
    // A caller-triggered abort is not a connectivity failure — let it through
    // so the UI can distinguish "cancelled" from "backend is down".
    if (e instanceof DOMException && e.name === "AbortError") throw e;
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
  // ── Documents / ingestion ──
  listDocuments: (status?: string) =>
    request<IngestedDocument[]>(
      `/documents${status ? `?status=${encodeURIComponent(status)}` : ""}`,
    ),
  deleteDocument: (id: string) =>
    request<DocumentDeleteResponse>(`/documents/${id}`, { method: "DELETE" }),
  /** Upload a file for ingestion. Returns immediately with a Celery job id. */
  ingestDocument: (file: File, signal?: AbortSignal) => {
    const form = new FormData();
    form.append("file", file);
    return request<IngestResponse>("/ingest", {
      method: "POST",
      body: form,
      signal,
    });
  },

  // ── RAG pipeline ──
  /** Full RAG pipeline: retrieve → generate → grounded answer with citations. */
  query: (payload: QueryRequest, signal?: AbortSignal) =>
    request<QueryResponse>("/query", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    }),
  /** Retrieval only — returns chunks with per-stage scores, no generation. */
  retrieve: (payload: RetrieveRequest, signal?: AbortSignal) =>
    request<RetrieveResponse>("/retrieve", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    }),

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
