/* The drawn marks. Shape and fill carry the meaning, never colour alone, and each mark is paired with a word
   wherever it appears. State: solid square = on, ring = off or empty, dashed ring = on its way, struck = down.
   Span kinds each get their own shape. */

import type { ReactNode } from "react";

const Wrap = ({ children }: { children: ReactNode }) => (
  <span className="mk" aria-hidden="true">
    <svg viewBox="0 0 14 14" focusable="false">{children}</svg>
  </span>
);

export type StateKind = "ok" | "off" | "wait" | "down";

export function StateMark({ kind }: { kind: StateKind }) {
  switch (kind) {
    case "ok":
      return <Wrap><rect x="1.5" y="1.5" width="11" height="11" fill="currentColor" /></Wrap>;
    case "off":
      return <Wrap><rect x="2.5" y="2.5" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="2" /></Wrap>;
    case "wait":
      return <Wrap><rect x="2.5" y="2.5" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray="3 2" /></Wrap>;
    case "down":
      return (
        <Wrap>
          <rect x="2.5" y="2.5" width="9" height="9" fill="none" stroke="currentColor" strokeWidth="2" />
          <path d="M1 13 13 1" stroke="currentColor" strokeWidth="2" />
        </Wrap>
      );
  }
}

export function StateLine({ kind, children }: { kind: StateKind; children: ReactNode }) {
  return (
    <span className="state-line">
      <StateMark kind={kind} />
      {children}
    </span>
  );
}

type Shape = "square" | "circle" | "diamond" | "triangle" | "inverted" | "ring";

export const KIND_STYLE: Record<string, { color: string; shape: Shape }> = {
  CHAIN: { color: "#566075", shape: "square" },
  RETRIEVER: { color: "#2b5fb0", shape: "circle" },
  RERANKER: { color: "#6b3f73", shape: "diamond" },
  LLM: { color: "#2f6b4f", shape: "triangle" },
  EMBEDDING: { color: "#8a5a10", shape: "inverted" },
};
const FALLBACK = { color: "#566075", shape: "ring" as Shape };

export const kindStyle = (kind: string) => KIND_STYLE[kind] ?? FALLBACK;

export function KindMark({ kind }: { kind: string }) {
  const { color, shape } = kindStyle(kind);
  return (
    <span className="mk" aria-hidden="true" style={{ color, width: 12, height: 12 }}>
      <svg viewBox="0 0 14 14" focusable="false">
        {shape === "square" && <rect x="1.5" y="1.5" width="11" height="11" fill="currentColor" />}
        {shape === "circle" && <circle cx="7" cy="7" r="5.8" fill="currentColor" />}
        {shape === "diamond" && <path d="M7 .8 13.2 7 7 13.2 .8 7z" fill="currentColor" />}
        {shape === "triangle" && <path d="M7 1.2 13 12.8H1z" fill="currentColor" />}
        {shape === "inverted" && <path d="M7 12.8 1 1.2h12z" fill="currentColor" />}
        {shape === "ring" && <circle cx="7" cy="7" r="4.8" fill="none" stroke="currentColor" strokeWidth="2" />}
      </svg>
    </span>
  );
}
