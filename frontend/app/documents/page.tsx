"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { RefreshCw, Trash2, Upload as UploadIcon } from "lucide-react";
import { api, ApiError } from "@/lib/api";
import type { DocumentStatus, IngestedDocument } from "@/lib/types";
import { bytes, dateTime, num, shortId, timeAgo } from "@/lib/format";
import { StateLine, StateMark, type StateKind } from "@/components/Marks";
import { Entry, EmptyState, ErrorState, Loading, PageHeader } from "@/components/ui";

const ACCEPT = ".pdf,.txt,.md";
/** Mirrors the allow-list in the backend's ingest router. */
const ACCEPTED_EXT = [".pdf", ".txt", ".md"];

/** Local state for a file between "chosen" and "row appears in the table". */
interface Upload {
  key: string;
  name: string;
  size: number;
  state: "uploading" | "queued" | "rejected";
  error?: string;
  documentId?: string;
}

const STATUS: Record<DocumentStatus, { word: string; kind: StateKind }> = {
  pending: { word: "queued", kind: "wait" },
  processing: { word: "processing", kind: "wait" },
  completed: { word: "completed", kind: "ok" },
  failed: { word: "failed", kind: "down" },
};

export default function DocumentsPage() {
  const [docs, setDocs] = useState<IngestedDocument[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploads, setUploads] = useState<Upload[]>([]);
  const [dragging, setDragging] = useState(false);
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await api.listDocuments();
      setDocs(data);
      setError(null);
      // Drop local upload rows once the real row shows up in the table.
      setUploads((prev) =>
        prev.filter((u) => u.state === "rejected" || !u.documentId || !data.some((d) => d.id === u.documentId)),
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

  // Poll while anything is still in flight: ingestion is a background job.
  const inFlight =
    (docs?.some((d) => d.status === "pending" || d.status === "processing") ?? false) ||
    uploads.some((u) => u.state !== "rejected");

  useEffect(() => {
    if (!inFlight) return;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [inFlight, load]);

  const upload = useCallback(
    async (files: FileList | File[]) => {
      for (const file of Array.from(files)) {
        const key = `${file.name}-${Date.now()}-${Math.random()}`;
        const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();

        if (!ACCEPTED_EXT.includes(ext)) {
          setUploads((p) => [
            ...p,
            { key, name: file.name, size: file.size, state: "rejected", error: `Unsupported file type "${ext}". Allowed: ${ACCEPTED_EXT.join(", ")}` },
          ]);
          continue;
        }

        setUploads((p) => [...p, { key, name: file.name, size: file.size, state: "uploading" }]);

        try {
          const res = await api.ingestDocument(file);
          setUploads((p) => p.map((u) => (u.key === key ? { ...u, state: "queued", documentId: res.document_id } : u)));
        } catch (e) {
          setUploads((p) =>
            p.map((u) => (u.key === key ? { ...u, state: "rejected", error: e instanceof ApiError ? e.message : "Upload failed" } : u)),
          );
        }
      }
      load();
    },
    [load],
  );

  const remove = async (doc: IngestedDocument) => {
    setDeleting(doc.id);
    try {
      await api.deleteDocument(doc.id);
      setConfirmId(null);
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
    <>
      <PageHeader
        title="Documents"
        subtitle="Upload PDFs, text or markdown. Parsing, chunking and embedding run as a background job, and the table shows its live status."
        actions={<button type="button" onClick={load} className="btn"><RefreshCw aria-hidden="true" /> Refresh</button>}
      />

      <Entry kind="Ingest" source="POST /api/v1/ingest">
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            if (e.dataTransfer.files?.length) upload(e.dataTransfer.files);
          }}
          style={{
            border: `2px dashed ${dragging ? "var(--ink)" : "var(--ink-3)"}`,
            background: dragging ? "var(--sunk)" : "var(--sheet)",
            padding: "24px 20px",
            display: "flex", flexWrap: "wrap", alignItems: "center", gap: 16,
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept={ACCEPT}
            multiple
            className="sr-only"
            tabIndex={-1}
            aria-label="Choose files to ingest"
            onChange={(e) => {
              if (e.target.files?.length) upload(e.target.files);
              e.target.value = "";
            }}
          />
          <button type="button" className="btn btn-primary" onClick={() => fileRef.current?.click()}>
            <UploadIcon aria-hidden="true" /> Choose files
          </button>
          <p className="lede" style={{ margin: 0 }}>
            or drop them here. PDF, TXT or MD; several at once is fine.
          </p>
        </div>

        {uploads.length > 0 && (
          <ul className="stack" style={{ listStyle: "none", padding: 0, margin: "16px 0 0" }}>
            {uploads.map((u) => (
              <li key={u.key} className={u.state === "rejected" ? "note is-error" : "note"}>
                <StateMark kind={u.state === "rejected" ? "down" : "wait"} />
                <div>
                  <b>{u.name}</b> <span className="mono" style={{ color: "var(--ink-3)" }}>{bytes(u.size)}</span>
                  <p>
                    {u.state === "uploading" ? "Uploading" : u.state === "queued" ? "Queued for ingestion" : u.error}
                  </p>
                  {u.state === "rejected" && (
                    <button type="button" className="btn btn-sm" onClick={() => setUploads((p) => p.filter((x) => x.key !== u.key))}>
                      Dismiss
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </Entry>

      {error && <div style={{ marginBottom: 24 }}><ErrorState message={error} onRetry={load} /></div>}

      <Entry
        kind="Documents"
        source="GET /api/v1/documents"
        actions={completed.length > 0 ? <Link href="/playground" className="btn btn-sm">Ask a question</Link> : undefined}
      >
        {docs === null ? (
          <Loading label="Loading documents" />
        ) : docs.length === 0 && !error ? (
          <EmptyState title="No documents ingested yet" hint="Upload one above. Until at least one document finishes processing, the playground and inspector have nothing to retrieve from." />
        ) : (
          <>
            {completed.length > 0 && (
              <p className="lede" style={{ marginBottom: 16 }}>
                {completed.length} document{completed.length === 1 ? "" : "s"} ready, {num(totalChunks)} embedded chunks.
              </p>
            )}
            <div className="tab-wrap">
              <table className="tab">
                <caption className="sr-only">Ingested documents</caption>
                <thead>
                  <tr>
                    <th>Document</th><th>Status</th><th className="num hide-sm">Pages</th><th className="num">Chunks</th><th className="num hide-sm">Size</th><th className="hide-sm">Added</th><th><span className="sr-only">Actions</span></th>
                  </tr>
                </thead>
                <tbody>
                  {docs.flatMap((d) => {
                    const s = STATUS[d.status] ?? STATUS.pending;
                    const rows = [
                      <tr key={d.id}>
                        <td>
                          {d.filename}
                          <span className="id" title={d.id}>{shortId(d.id, 8)}</span>
                          <span className="id only-sm">added {timeAgo(d.created_at)}</span>
                          {d.error_message && <span className="id" style={{ color: "var(--red)" }}>{d.error_message}</span>}
                        </td>
                        <td><StateLine kind={s.kind}>{s.word}</StateLine></td>
                        <td className="num hide-sm">{d.page_count ?? "n/a"}</td>
                        <td className="num">{d.chunk_count ? num(d.chunk_count) : "n/a"}</td>
                        <td className="num hide-sm">{bytes(d.file_size_bytes)}</td>
                        <td className="hide-sm">{dateTime(d.created_at)}<span className="id">{timeAgo(d.created_at)}</span></td>
                        <td>
                          <button type="button" className="btn btn-sm btn-danger" onClick={() => setConfirmId(d.id)} aria-label={`Delete ${d.filename}`} title="Delete document and its chunks">
                            <Trash2 aria-hidden="true" />
                          </button>
                        </td>
                      </tr>,
                    ];
                    if (confirmId === d.id) {
                      rows.push(
                        <tr key={`${d.id}-confirm`} className="confirm">
                          <td colSpan={7}>
                            <div className="flex flex-wrap items-center justify-between gap-3" role="alertdialog" aria-label={`Confirm deleting ${d.filename}`}>
                              <b>Delete “{d.filename}” and its {d.chunk_count} chunks? This cannot be undone.</b>
                              <span className="flex gap-2">
                                <button type="button" className="btn btn-sm btn-danger" disabled={deleting === d.id} onClick={() => remove(d)}>
                                  {deleting === d.id ? "Deleting" : "Delete"}
                                </button>
                                <button type="button" className="btn btn-sm" autoFocus onClick={() => setConfirmId(null)}>Cancel</button>
                              </span>
                            </div>
                          </td>
                        </tr>,
                      );
                    }
                    return rows;
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </Entry>
    </>
  );
}
