import Link from "next/link";
import {
  ArrowRight,
  BarChart3,
  FlaskConical,
  GitBranch,
  MessageSquare,
  Microscope,
  Waypoints,
} from "lucide-react";
import { Card, PageHeader } from "@/components/ui";

const FEATURES = [
  {
    href: "/playground",
    icon: MessageSquare,
    title: "Playground",
    body: "Ask questions against your ingested documents. Grounded answers with clickable citations, live latency and token counts, and thumbs up/down feedback.",
  },
  {
    href: "/inspector",
    icon: Microscope,
    title: "Retrieval Inspector",
    body: "Retrieval without generation — every chunk with its dense, BM25, RRF and rerank score, plus the staged metadata filters applied.",
  },
  {
    href: "/traces",
    icon: Waypoints,
    title: "Trace Viewer",
    body: "Span-waterfall timeline of every query — CHAIN → RETRIEVER → RERANKER → LLM, with per-span duration, tokens, and cost.",
  },
  {
    href: "/experiments",
    icon: FlaskConical,
    title: "A/B Experiments",
    body: "Run the same golden set through two configs (reranker on vs off) and get per-metric deltas with bootstrap significance.",
  },
  {
    href: "/analytics",
    icon: BarChart3,
    title: "Cost & Latency Analytics",
    body: "p50/p95/p99 latency, cost per query by model, cache hit rate and estimated savings, throughput over time.",
  },
];

export default function OverviewPage() {
  return (
    <div>
      <PageHeader
        title="Observability & Experiments"
        subtitle="Phase 4 turns RAGScope's OpenInference spans into a queryable trace store, an A/B experiment framework, and live cost/latency analytics."
      />

      <Card className="mb-8 border-primary/30 bg-gradient-to-br from-primary/10 to-transparent">
        <div className="flex items-center gap-2 text-primary">
          <GitBranch className="h-4 w-4" />
          <span className="font-mono text-xs uppercase tracking-widest">
            v0.5.0 · Phase 4
          </span>
        </div>
        <h2 className="mt-3 text-xl font-semibold text-on-surface">
          Every query is traced, every config is measurable.
        </h2>
        <p className="mt-2 max-w-3xl text-sm text-on-surface-muted">
          The pipeline emits an OpenTelemetry span tree from the first request.
          Phase 4 persists those spans to Postgres, computes analytics from the
          query history, and adds an experiment runner that compares two
          pipeline variants on a golden dataset with statistical rigor.
        </p>
      </Card>

      <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-5">
        {FEATURES.map(({ href, icon: Icon, title, body }) => (
          <Link key={href} href={href}>
            <Card className="group h-full transition-colors hover:border-primary/50">
              <Icon className="h-6 w-6 text-primary" />
              <h3 className="mt-3 flex items-center gap-1 font-semibold text-on-surface">
                {title}
                <ArrowRight className="h-4 w-4 opacity-0 transition-opacity group-hover:opacity-100" />
              </h3>
              <p className="mt-2 text-sm text-on-surface-muted">{body}</p>
            </Card>
          </Link>
        ))}
      </div>

      <div className="mt-8 grid gap-4 md:grid-cols-2">
        <Card>
          <h3 className="font-semibold text-on-surface">Span kinds</h3>
          <p className="mt-1 text-xs text-on-surface-muted">
            OpenInference conventions, color-coded across the app.
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {[
              ["CHAIN", "#8fa0b0"],
              ["RETRIEVER", "#00b4d8"],
              ["RERANKER", "#a78bfa"],
              ["LLM", "#00d4aa"],
              ["EMBEDDING", "#4cd6fb"],
            ].map(([kind, color]) => (
              <span
                key={kind}
                className="rounded-md px-2 py-1 font-mono text-xs"
                style={{ backgroundColor: `${color}22`, color }}
              >
                {kind}
              </span>
            ))}
          </div>
        </Card>
        <Card>
          <h3 className="font-semibold text-on-surface">API surface</h3>
          <p className="mt-1 text-xs text-on-surface-muted">
            Endpoints this dashboard drives, mounted under <code>/api/v1</code>.
          </p>
          <ul className="mt-4 space-y-1 font-mono text-xs text-on-surface-muted">
            <li>POST /query · /retrieve</li>
            <li>GET /traces · /traces/{"{id}"}</li>
            <li>POST/GET /experiments</li>
            <li>POST/GET /feedback</li>
            <li>GET /analytics/latency · /cost · /cache · /throughput</li>
          </ul>
        </Card>
      </div>
    </div>
  );
}
