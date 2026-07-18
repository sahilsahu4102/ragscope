// RAGScope — small formatting helpers

export function ms(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value < 1000) return `${value.toFixed(0)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

export function usd(value: number | null | undefined): string {
  if (value == null) return "—";
  if (value === 0) return "$0.00";
  if (value < 0.01) return `$${value.toFixed(6)}`;
  return `$${value.toFixed(4)}`;
}

export function num(value: number | null | undefined): string {
  if (value == null) return "—";
  return value.toLocaleString();
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function shortId(id: string, len = 8): string {
  return id.length > len ? id.slice(0, len) : id;
}

export function timeAgo(iso: string): string {
  const then = new Date(iso).getTime();
  const secs = Math.floor((Date.now() - then) / 1000);
  if (secs < 60) return `${secs}s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)}m ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)}h ago`;
  return `${Math.floor(secs / 86400)}d ago`;
}

export const SPAN_KIND_COLOR: Record<string, string> = {
  CHAIN: "#8fa0b0",
  RETRIEVER: "#00b4d8",
  RERANKER: "#a78bfa",
  LLM: "#00d4aa",
  EMBEDDING: "#4cd6fb",
  TOOL: "#ffb77a",
  GUARDRAIL: "#ffb4ab",
  EVALUATOR: "#f0abfc",
  PROMPT: "#94a3b8",
  AGENT: "#fbbf24",
};
