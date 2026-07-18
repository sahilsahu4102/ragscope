"use client";

import clsx from "clsx";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  FlaskConical,
  LayoutDashboard,
  MessageSquare,
  Microscope,
  Waypoints,
} from "lucide-react";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/traces", label: "Trace Viewer", icon: Waypoints, badge: "P4" },
  { href: "/experiments", label: "Experiments", icon: FlaskConical, badge: "P4" },
  { href: "/analytics", label: "Analytics", icon: BarChart3, badge: "P4" },
];

const EXTERNAL = [
  { href: "/chat.html", label: "Playground", icon: MessageSquare },
  { href: "/inspector.html", label: "Retrieval Inspector", icon: Microscope },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex w-64 flex-col border-r border-border bg-surface-container/40">
      <div className="flex items-center gap-2 px-5 py-5">
        <Activity className="h-6 w-6 text-primary" />
        <div>
          <div className="font-mono text-lg font-bold tracking-tight text-on-surface">
            RAGScope
          </div>
          <div className="text-[10px] uppercase tracking-widest text-on-surface-muted">
            Observability
          </div>
        </div>
      </div>

      <nav className="flex-1 space-y-1 px-3 py-2">
        {NAV.map(({ href, label, icon: Icon, badge }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                active
                  ? "bg-primary/15 text-primary"
                  : "text-on-surface-muted hover:bg-surface-container-high hover:text-on-surface",
              )}
            >
              <Icon className="h-4 w-4" />
              <span className="flex-1">{label}</span>
              {badge && (
                <span className="rounded bg-primary/20 px-1.5 py-0.5 font-mono text-[10px] text-primary">
                  {badge}
                </span>
              )}
            </Link>
          );
        })}

        <div className="px-3 pb-1 pt-5 text-[10px] uppercase tracking-widest text-on-surface-muted">
          Design mockups
        </div>
        {EXTERNAL.map(({ href, label, icon: Icon }) => (
          <a
            key={href}
            href={href}
            className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm text-on-surface-muted transition-colors hover:bg-surface-container-high hover:text-on-surface"
          >
            <Icon className="h-4 w-4" />
            <span>{label}</span>
          </a>
        ))}
      </nav>

      <div className="border-t border-border px-5 py-4 text-[11px] text-on-surface-muted">
        <div className="font-mono">v0.5.0 · Phase 4</div>
        <div className="mt-1">Self-hosted RAG platform</div>
      </div>
    </aside>
  );
}
