import clsx from "clsx";
import { AlertTriangle, Inbox, Loader2 } from "lucide-react";
import type { ReactNode } from "react";

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-on-surface">
          {title}
        </h1>
        {subtitle && (
          <p className="mt-1 max-w-2xl text-sm text-on-surface-muted">
            {subtitle}
          </p>
        )}
      </div>
      {actions}
    </div>
  );
}

export function Card({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={clsx(
        "rounded-xl border border-border bg-surface-container/40 p-5",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function Loading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-border bg-surface-container/40 p-8 text-on-surface-muted">
      <Loader2 className="h-5 w-5 animate-spin text-primary" />
      <span className="text-sm">{label}</span>
    </div>
  );
}

export function EmptyState({
  title,
  hint,
}: {
  title: string;
  hint?: string;
}) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border bg-surface-container/20 p-10 text-center">
      <Inbox className="h-8 w-8 text-on-surface-muted" />
      <div className="text-sm font-medium text-on-surface">{title}</div>
      {hint && <div className="max-w-md text-xs text-on-surface-muted">{hint}</div>}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 rounded-xl border border-danger/40 bg-danger/10 p-5 text-danger">
      <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0" />
      <div className="text-sm">{message}</div>
    </div>
  );
}

export function Badge({
  children,
  color,
}: {
  children: ReactNode;
  color?: string;
}) {
  return (
    <span
      className="rounded px-1.5 py-0.5 font-mono text-[10px] font-medium"
      style={{
        backgroundColor: color ? `${color}22` : "#26313f",
        color: color ?? "#8fa0b0",
      }}
    >
      {children}
    </span>
  );
}
