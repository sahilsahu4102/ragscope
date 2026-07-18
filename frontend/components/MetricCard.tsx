import clsx from "clsx";
import type { ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: ReactNode;
  sublabel?: ReactNode;
  accent?: "primary" | "secondary" | "warning" | "danger" | "muted";
  icon?: ReactNode;
}

const ACCENT: Record<NonNullable<MetricCardProps["accent"]>, string> = {
  primary: "text-primary",
  secondary: "text-secondary",
  warning: "text-warning",
  danger: "text-danger",
  muted: "text-on-surface",
};

export function MetricCard({
  label,
  value,
  sublabel,
  accent = "primary",
  icon,
}: MetricCardProps) {
  return (
    <div className="rounded-xl border border-border bg-surface-container/50 p-4">
      <div className="flex items-center justify-between">
        <div className="text-xs uppercase tracking-wider text-on-surface-muted">
          {label}
        </div>
        {icon && <div className="text-on-surface-muted">{icon}</div>}
      </div>
      <div className={clsx("mt-2 font-mono text-2xl font-bold", ACCENT[accent])}>
        {value}
      </div>
      {sublabel && (
        <div className="mt-1 text-xs text-on-surface-muted">{sublabel}</div>
      )}
    </div>
  );
}
