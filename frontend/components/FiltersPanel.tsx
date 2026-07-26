"use client";

import clsx from "clsx";
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
    <div className="rounded-lg border border-border bg-background/40">
      <div className="flex items-center">
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex flex-1 items-center gap-2 px-3 py-2 text-xs text-on-surface-muted hover:text-on-surface"
        >
          <ChevronDown
            className={clsx(
              "h-3.5 w-3.5 transition-transform",
              open && "rotate-180",
            )}
          />
          Metadata filters
          {count > 0 && (
            <span className="rounded bg-primary/20 px-1.5 py-0.5 font-mono text-[10px] text-primary">
              {count} active
            </span>
          )}
        </button>
        {count > 0 && (
          <button
            type="button"
            disabled={disabled}
            onClick={() => onChange(EMPTY_FILTERS)}
            title="Clear all filters"
            className="flex items-center gap-1 px-3 py-2 text-[11px] text-on-surface-muted hover:text-danger disabled:opacity-50"
          >
            <FilterX className="h-3.5 w-3.5" /> Clear
          </button>
        )}
      </div>

      {open && (
        <div className="space-y-4 border-t border-border px-3 py-3">
          <div>
            <div className="text-xs text-on-surface-muted">
              Element types
              <span className="ml-2 text-[10px] opacity-70">
                Stage 1 · indexed pre-filter
              </span>
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {ELEMENT_TYPES.map((t) => {
                const on = value.elementTypes.includes(t);
                return (
                  <button
                    key={t}
                    type="button"
                    disabled={disabled}
                    onClick={() => toggleIn("elementTypes", t)}
                    className={clsx(
                      "rounded border px-2 py-1 font-mono text-[10px] transition-colors disabled:opacity-50",
                      on
                        ? "border-secondary/50 bg-secondary/15 text-secondary"
                        : "border-border bg-surface-container text-on-surface-muted hover:text-on-surface",
                    )}
                  >
                    {t}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <div className="text-xs text-on-surface-muted">
              Documents
              <span className="ml-2 text-[10px] opacity-70">
                {value.documentIds.length === 0
                  ? "all documents"
                  : `${value.documentIds.length} selected`}
              </span>
            </div>
            {docs === null ? (
              <div className="mt-2 text-[11px] text-on-surface-muted">
                Loading corpus…
              </div>
            ) : docs.length === 0 ? (
              <div className="mt-2 text-[11px] text-on-surface-muted">
                No completed documents to filter by.
              </div>
            ) : (
              <div className="mt-2 flex max-h-32 flex-wrap gap-1.5 overflow-y-auto">
                {docs.map((d) => {
                  const on = value.documentIds.includes(d.id);
                  return (
                    <button
                      key={d.id}
                      type="button"
                      disabled={disabled}
                      title={`${d.id} · ${d.chunk_count} chunks`}
                      onClick={() => toggleIn("documentIds", d.id)}
                      className={clsx(
                        "max-w-[16rem] truncate rounded border px-2 py-1 text-[10px] transition-colors disabled:opacity-50",
                        on
                          ? "border-primary/50 bg-primary/15 text-primary"
                          : "border-border bg-surface-container text-on-surface-muted hover:text-on-surface",
                      )}
                    >
                      {d.filename}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Date from">
                <TextInput
                  type="date"
                  value={value.dateFrom}
                  onChange={(v) => set("dateFrom", v)}
                  disabled={disabled}
                />
              </Field>
              <Field label="Date to">
                <TextInput
                  type="date"
                  value={value.dateTo}
                  onChange={(v) => set("dateTo", v)}
                  disabled={disabled}
                />
              </Field>
            </div>
          </div>

          <div>
            <div className="mb-2 text-xs text-on-surface-muted">
              Post-filters
              <span className="ml-2 text-[10px] opacity-70">
                Stage 3 · applied to the result set
              </span>
            </div>
            <div className="grid grid-cols-3 gap-3">
              <Field label="Min score">
                <NumberInput
                  value={value.minScore}
                  onChange={(v) => set("minScore", v)}
                  min={0}
                  max={1}
                  step={0.05}
                  placeholder="0.0"
                  disabled={disabled}
                />
              </Field>
              <Field label="Min tokens">
                <NumberInput
                  value={value.minTokens}
                  onChange={(v) => set("minTokens", v)}
                  min={1}
                  placeholder="—"
                  disabled={disabled}
                />
              </Field>
              <Field label="Max tokens">
                <NumberInput
                  value={value.maxTokens}
                  onChange={(v) => set("maxTokens", v)}
                  min={1}
                  placeholder="—"
                  disabled={disabled}
                />
              </Field>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
