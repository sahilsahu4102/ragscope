"use client";

import { useEffect, useState } from "react";
import { ChevronDown, FilterX } from "lucide-react";
import { api } from "@/lib/api";
import type { IngestedDocument, RetrievalFilters } from "@/lib/types";
import { Field, NumberInput, TextInput } from "@/components/ui";

/**
 * Form-shaped mirror of `RetrievalFilters`. Everything is a string here because
 * that is what inputs hold; `toFilters()` converts to the API shape and drops
 * blanks so we never send an empty filter that narrows nothing.
 */
export interface FilterState {
  documentIds: string[];
  elementTypes: string[];
  dateFrom: string;
  dateTo: string;
  minScore: string;
  minTokens: string;
  maxTokens: string;
}

export const EMPTY_FILTERS: FilterState = {
  documentIds: [],
  elementTypes: [],
  dateFrom: "",
  dateTo: "",
  minScore: "",
  minTokens: "",
  maxTokens: "",
};

/** Element types the PDF parser actually emits, plus the ones chunkers can set. */
const ELEMENT_TYPES = [
  "title",
  "heading",
  "paragraph",
  "table",
  "list_item",
  "image_caption",
  "code",
] as const;

function numOrUndef(v: string): number | undefined {
  if (v.trim() === "") return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

/** Build the API payload. Returns undefined when no filter is actually set. */
export function toFilters(s: FilterState): RetrievalFilters | undefined {
  const f: RetrievalFilters = {};

  if (s.documentIds.length) f.document_ids = s.documentIds;
  if (s.elementTypes.length) f.element_types = s.elementTypes;
  if (s.dateFrom) f.date_from = s.dateFrom;
  if (s.dateTo) f.date_to = s.dateTo;

  const minScore = numOrUndef(s.minScore);
  const minTokens = numOrUndef(s.minTokens);
  const maxTokens = numOrUndef(s.maxTokens);
  if (minScore !== undefined) f.min_score = minScore;
  if (minTokens !== undefined) f.min_tokens = minTokens;
  if (maxTokens !== undefined) f.max_tokens = maxTokens;

  return Object.keys(f).length ? f : undefined;
}

export function activeFilterCount(s: FilterState): number {
  const f = toFilters(s);
  return f ? Object.keys(f).length : 0;
}

export function FiltersPanel({
  value,
  onChange,
  disabled,
}: {
  value: FilterState;
  onChange: (v: FilterState) => void;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [docs, setDocs] = useState<IngestedDocument[] | null>(null);
  const count = activeFilterCount(value);
  const set = <K extends keyof FilterState>(key: K, v: FilterState[K]) =>
    onChange({ ...value, [key]: v });

  // Load the corpus lazily — only once the panel is actually opened.
  useEffect(() => {
    if (!open || docs !== null) return;
    api
      .listDocuments("completed")
      .then(setDocs)
      .catch(() => setDocs([]));
  }, [open, docs]);

  const toggleIn = (key: "elementTypes" | "documentIds", t: string) =>
    set(
      key,
      value[key].includes(t)
        ? value[key].filter((x) => x !== t)
        : [...value[key], t],
    );

  return (
    <div style={{ border: "1.5px solid var(--ink)", background: "var(--sheet)" }}>
      <div className="flex items-center">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          aria-expanded={open}
          className="flex flex-1 items-center gap-2"
          style={{ minHeight: 40, padding: "0 12px", background: "transparent", border: 0, fontWeight: 650, textAlign: "left" }}
        >
          <ChevronDown aria-hidden="true" style={{ width: 16, height: 16, transform: open ? "rotate(180deg)" : undefined }} />
          Metadata filters
          {count > 0 && <span className="chip is-on">{count} active</span>}
        </button>
        {count > 0 && (
          <button type="button" disabled={disabled} onClick={() => onChange(EMPTY_FILTERS)} title="Clear all filters" className="btn btn-sm" style={{ margin: 4 }}>
            <FilterX aria-hidden="true" /> Clear
          </button>
        )}
      </div>

      {open && (
        <div className="stack" style={{ borderTop: "1px solid var(--rule)", padding: 16, gap: 20 }}>
          <div>
            <div className="lab">
              Element types <span style={{ fontWeight: 400, color: "var(--ink-3)" }}>stage 1, indexed pre-filter</span>
            </div>
            <div className="flex flex-wrap gap-2" style={{ marginTop: 8 }}>
              {ELEMENT_TYPES.map((t) => {
                const on = value.elementTypes.includes(t);
                return (
                  <button key={t} type="button" disabled={disabled} aria-pressed={on} onClick={() => toggleIn("elementTypes", t)} className="toggle mono" style={{ minHeight: 30 }}>
                    {t}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <div className="lab">
              Documents{" "}
              <span style={{ fontWeight: 400, color: "var(--ink-3)" }}>
                {value.documentIds.length === 0 ? "all documents" : `${value.documentIds.length} selected`}
              </span>
            </div>
            {docs === null ? (
              <p className="lede" style={{ marginTop: 8 }}>Loading the corpus</p>
            ) : docs.length === 0 ? (
              <p className="lede" style={{ marginTop: 8 }}>No completed documents to filter by.</p>
            ) : (
              <div className="flex flex-wrap gap-2" style={{ marginTop: 8, maxHeight: 140, overflowY: "auto" }}>
                {docs.map((d) => {
                  const on = value.documentIds.includes(d.id);
                  return (
                    <button
                      key={d.id}
                      type="button"
                      disabled={disabled}
                      aria-pressed={on}
                      title={`${d.id} · ${d.chunk_count} chunks`}
                      onClick={() => toggleIn("documentIds", d.id)}
                      className="toggle"
                      style={{ minHeight: 30, maxWidth: 260 }}
                    >
                      <span className="truncate">{d.filename}</span>
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          <div className="row">
            <Field label="Date from"><TextInput type="date" value={value.dateFrom} onChange={(v) => set("dateFrom", v)} disabled={disabled} /></Field>
            <Field label="Date to"><TextInput type="date" value={value.dateTo} onChange={(v) => set("dateTo", v)} disabled={disabled} /></Field>
          </div>

          <div>
            <div className="lab">
              Post-filters <span style={{ fontWeight: 400, color: "var(--ink-3)" }}>stage 3, applied to the result set</span>
            </div>
            <div className="row" style={{ marginTop: 8 }}>
              <Field label="Min score"><NumberInput value={value.minScore} onChange={(v) => set("minScore", v)} min={0} max={1} step={0.05} placeholder="0.0" disabled={disabled} /></Field>
              <Field label="Min tokens"><NumberInput value={value.minTokens} onChange={(v) => set("minTokens", v)} min={1} placeholder="any" disabled={disabled} /></Field>
              <Field label="Max tokens"><NumberInput value={value.maxTokens} onChange={(v) => set("maxTokens", v)} min={1} placeholder="any" disabled={disabled} /></Field>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
