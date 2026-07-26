"use client";

import clsx from "clsx";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  FileText,
  Loader2,
  RefreshCw,
  Trash2,
  Upload,
} from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { DocumentStatus, IngestedDocument } from "@/lib/types";
import { bytes, num, timeAgo } from "@/lib/format";
import { Card, EmptyState, ErrorState, Loading, PageHeader } from "@/components/ui";

const ACCEPT = ".pdf,.txt,.md";
/** Mirrors the allow-list in the backend's ingest router. */
const ACCEPTED_EXT = [".pdf", ".txt", ".md"];

/** Local state for a file between "dropped" and "row appears in the table". */
interface Upload {
  key: string;
  name: string;
  size: number;
  state: "uploading" | "queued" | "rejected";
  error?: string;
  documentId?: string;
}

const STATUS_STYLE: Record<
  DocumentStatus,
  { label: string; className: string; icon: typeof Clock }
> = {
  pending: {
    label: "queued",
    className: "bg-surface-container-high text-on-surface-muted",
    icon: Clock,
  },
  processing: {
    label: "processing",
    className: "bg-secondary/15 text-secondary",
    icon: Loader2,
  },
  completed: {
    label: "completed",
    className: "bg-primary/15 text-primary",
    icon: CheckCircle2,
  },
  failed: {
    label: "failed",
    className: "bg-danger/15 text-danger",
    icon: AlertTriangle,
  },
};

function StatusPill({ status }: { status: DocumentStatus }) {
  const s = STATUS_STYLE[status] ?? STATUS_STYLE.pending;
  const Icon = s.icon;
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-mono text-[10px]",
        s.className,
      )}
    >
      <Icon
        className={clsx("h-3 w-3", status === "processing" && "animate-spin")}
      />
      {s.label}
    </span>
  );
}

export default function DocumentsPage() {
  const [docs, setDocs] = useState<IngestedDocument[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [dragging, setDragging] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api.listDocuments();
      setDocs(data);
      setError(null);
      // Drop local upload cards once the real row shows up in the table.
      setUploads((prev) =>
        prev.filter(
          (u) =>
            u.state === "rejected" ||
            !u.documentId ||
            !data.some((d) => d.id === u.documentId),
        ),
      );
      return data;
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load documents");
      setDocs([]);
      return [];
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Poll while anything is still in flight — ingestion is a background job.
  const inFlight =
    (docs?.some((d) => d.status === "pending" || d.status === "processing") ??
      false) || uploads.some((u) => u.state !== "rejected");

  useEffect(() => {
    if (!inFlight) return;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [inFlight, load]);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      const list = Array.from(files);
      for (const file of list) {
        const key = `${file.name}-${Date.now()}-${Math.random()}`;
        const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();

        if (!ACCEPTED_EXT.includes(ext)) {
          setUploads((p) => [
            ...p,
            {
              key,
              name: file.name,
              size: file.size,
              state: "rejected",
              error: `Unsupported file type "${ext}". Allowed: ${ACCEPTED_EXT.join(", ")}`,
            },
          ]);
          continue;
        }

        setUploads((p) => [
          ...p,
          { key, name: file.name, size: file.size, state: "uploading" },
        ]);

        try {
          const res = await api.ingestDocument(file);
          setUploads((p) =>
            p.map((u) =>
              u.key === key
                ? { ...u, state: "queued", documentId: res.document_id }
                : u,
            ),
          );
        } catch (e) {
          setUploads((p) =>
            p.map((u) =>
              u.key === key
                ? {
                    ...u,
                    state: "rejected",
                    error: e instanceof ApiError ? e.message : "Upload failed",
                  }
                : u,
            ),
          );
        }
      }
      load();
    },
    [load],
  );

  const remove = async (doc: IngestedDocument) => {
    if (
      !confirm(
        `Delete "${doc.filename}" and its ${doc.chunk_count} chunks? This cannot be undone.`,
      )
    )
      return;
    setDeleting(doc.id);
    try {
      await api.deleteDocument(doc.id);
      await load();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Delete failed");
    } finally {
      setDeleting(null);
    }
  };

  const completed = docs?.filter((d) => d.status === "completed") ?? [];
  const totalChunks = completed.reduce((n, d) => n + d.chunk_count, 0);

  return (
    <div>
      <PageHeader
        title="Documents"
        subtitle="Upload PDFs, text or markdown. Parsing, chunking and embedding run as a background job — the table below reflects live status."
        actions={
          <button
            onClick={load}
            className="flex items-center gap-2 rounded-lg border border-border bg-surface-container px-3 py-2 text-sm text-on-surface hover:border-primary/50"
          >
            <RefreshCw className="h-4 w-4" /> Refresh
          </button>
        }
      />

      {/* ── Drop zone ── */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files?.length) upload(e.dataTransfer.files);
        }}
        onClick={() => fileRef.current?.click()}
        className={clsx(
          "mb-6 cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors",
          dragging
            ? "border-primary bg-primary/10"
            : "border-border bg-surface-container/30 hover:border-primary/50",
        )}
      >
        <input
          ref={fileRef}
          type="file"
          accept={ACCEPT}
          multiple
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) upload(e.target.files);
            e.target.value = "";
          }}
        />
        <Upload
          className={clsx(
            "mx-auto h-8 w-8",
            dragging ? "text-primary" : "text-on-surface-muted",
          )}
        />
        <div className="mt-3 text-sm font-medium text-on-surface">
          Drop files here, or click to browse
        </div>
        <div className="mt-1 text-xs text-on-surface-muted">
          PDF, TXT, MD · multiple files supported
        </div>
      </div>

      {/* ── In-flight uploads ── */}
      {uploads.length > 0 && (
        <div className="mb-6 space-y-2">
          {uploads.map((u) => (
            <div
              key={u.key}
              className={clsx(
                "flex items-center gap-3 rounded-lg border px-4 py-2.5 text-sm",
                u.state === "rejected"
                  ? "border-danger/40 bg-danger/10"
                  : "border-border bg-surface-container/40",
              )}
            >
              {u.state === "uploading" ? (
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
              ) : u.state === "queued" ? (
                <Clock className="h-4 w-4 text-secondary" />
              ) : (
                <AlertTriangle className="h-4 w-4 text-danger" />
              )}
              <span className="truncate text-on-surface">{u.name}</span>
              <span className="font-mono text-[11px] text-on-surface-muted">
                {bytes(u.size)}
              </span>
              <span className="ml-auto text-xs text-on-surface-muted">
                {u.state === "uploading"
                  ? "Uploading…"
                  : u.state === "queued"
                    ? "Queued for ingestion"
                    : u.error}
              </span>
              {u.state === "rejected" && (
                <button
                  onClick={() =>
                    setUploads((p) => p.filter((x) => x.key !== u.key))
                  }
                  className="text-on-surface-muted hover:text-on-surface"
                >
                  ✕
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {error && (
        <div className="mb-6">
          <ErrorState message={error} />
        </div>
      )}

      {/* ── Corpus summary ── */}
      {completed.length > 0 && (
        <div className="mb-4 flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-on-surface-muted">
          <span>
            <span className="font-mono text-primary">{completed.length}</span>{" "}
            document{completed.length === 1 ? "" : "s"} ready
          </span>
          <span>
            <span className="font-mono text-on-surface">
              {num(totalChunks)}
            </span>{" "}
            embedded chunks
          </span>
          <Link
            href="/playground"
            className="underline-offset-2 hover:text-primary hover:underline"
          >
            ask a question →
          </Link>
        </div>
      )}

      {/* ── Document table ── */}
      {docs === null ? (
        <Loading label="Loading documents…" />
      ) : docs.length === 0 && !error ? (
        <EmptyState
          title="No documents ingested yet"
          hint="Upload one above. Until at least one document finishes processing, the playground and inspector have nothing to retrieve from."
        />
      ) : (
        <Card className="overflow-hidden p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-surface-container text-left text-xs text-on-surface-muted">
                <tr>
                  <th className="px-4 py-3 font-medium">Document</th>
                  <th className="px-3 py-3 font-medium">Status</th>
                  <th className="px-3 py-3 font-medium">Pages</th>
                  <th className="px-3 py-3 font-medium">Chunks</th>
                  <th className="px-3 py-3 font-medium">Size</th>
                  <th className="px-3 py-3 font-medium">Age</th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody>
                {docs.map((d) => (
                  <tr key={d.id} className="border-t border-border">
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <FileText className="h-4 w-4 shrink-0 text-on-surface-muted" />
                        <span className="text-on-surface">{d.filename}</span>
                      </div>
                      <div className="mt-0.5 pl-6 font-mono text-[10px] text-on-surface-muted">
                        {d.id}
                      </div>
                      {d.error_message && (
                        <div className="mt-1 pl-6 text-[11px] text-danger">
                          {d.error_message}
                        </div>
                      )}
                    </td>
                    <td className="px-3 py-3">
                      <StatusPill status={d.status} />
                    </td>
                    <td className="px-3 py-3 font-mono text-xs text-on-surface-muted">
                      {d.page_count ?? "—"}
                    </td>
                    <td className="px-3 py-3 font-mono text-xs text-on-surface">
                      {d.chunk_count ? num(d.chunk_count) : "—"}
                    </td>
                    <td className="px-3 py-3 font-mono text-xs text-on-surface-muted">
                      {bytes(d.file_size_bytes)}
                    </td>
                    <td className="px-3 py-3 text-xs text-on-surface-muted">
                      {timeAgo(d.created_at)}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => remove(d)}
                        disabled={deleting === d.id}
                        title="Delete document and its chunks"
                        className="rounded p-1.5 text-on-surface-muted transition-colors hover:bg-danger/15 hover:text-danger disabled:opacity-40"
                      >
                        {deleting === d.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Trash2 className="h-4 w-4" />
                        )}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
