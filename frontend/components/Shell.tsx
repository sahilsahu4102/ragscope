"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  FileStack,
  FlaskConical,
  LayoutDashboard,
  MessageSquare,
  Microscope,
  Waypoints,
} from "lucide-react";
import type { ReactNode } from "react";
import { StackProvider, useStackContext } from "./StackContext";
import { StateLine, type StateKind } from "./Marks";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/documents", label: "Documents", icon: FileStack },
  { href: "/playground", label: "Playground", icon: MessageSquare },
  { href: "/inspector", label: "Retrieval Inspector", icon: Microscope },
  { href: "/traces", label: "Trace Viewer", icon: Waypoints },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/analytics", label: "Analytics", icon: BarChart3 },
];

const STATE: Record<string, { kind: StateKind; word: string }> = {
  checking: { kind: "wait", word: "Checking the API" },
  waking: { kind: "wait", word: "API waking up" },
  ready: { kind: "ok", word: "API ready" },
  degraded: { kind: "off", word: "API degraded" },
  down: { kind: "down", word: "API unreachable" },
};

function RailStatus() {
  const { status } = useStackContext();
  const s = STATE[status];
  return (
    <div className="rail-foot">
      <span role="status">
        <StateLine kind={s.kind}>{s.word}</StateLine>
      </span>
    </div>
  );
}

function Rail() {
  const pathname = usePathname();
  return (
    <aside className="rail">
      <Link href="/" className="plate">
        <b>RAGScope</b>
        <span>Evaluation and observability for RAG</span>
      </Link>
      <nav className="nav" aria-label="Main">
        {NAV.map(({ href, label, icon: Icon }) => (
          <Link key={href} href={href} aria-current={pathname === href ? "page" : undefined}>
            <Icon aria-hidden="true" />
            {label}
          </Link>
        ))}
      </nav>
      <RailStatus />
    </aside>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  return (
    <StackProvider>
      <div className="shell">
        <a className="skip" href="#main">Skip to main content</a>
        <Rail />
        <main className="main" id="main">
          <div className="page">{children}</div>
        </main>
      </div>
    </StackProvider>
  );
}
