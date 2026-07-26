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

/** Labelled wrapper for a form control. */
export function Field({
  label,
  hint,
  children,
  className,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={clsx("block", className)}>
      <span className="text-xs text-on-surface-muted">{label}</span>
      {children}
      {hint && (
        <span className="mt-1 block text-[10px] text-on-surface-muted">
          {hint}
        </span>
      )}
    </label>
  );
}

const INPUT_CLASS =
  "mt-1 block w-full rounded-md border border-border bg-background px-2 py-1.5 font-mono text-sm text-on-surface outline-none focus:border-primary/60 disabled:opacity-50";

export function NumberInput({
  value,
  onChange,
  min,
  max,
  step,
  placeholder,
  disabled,
}: {
  value: string;
  onChange: (v: string) => void;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  disabled?: boolean;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className={INPUT_CLASS}
    />
  );
}

export function TextInput({
  value,
  onChange,
  placeholder,
  disabled,
  type = "text",
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  disabled?: boolean;
  type?: string;
}) {
  return (
    <input
      type={type}
      value={value}
      placeholder={placeholder}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className={INPUT_CLASS}
    />
  );
}

export function Select<T extends string>({
  value,
  onChange,
  options,
  disabled,
}: {
  value: T;
  onChange: (v: T) => void;
  options: readonly { value: T; label: string }[];
  disabled?: boolean;
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value as T)}
      className={clsx(INPUT_CLASS, "font-sans")}
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

/** Pill-style on/off switch — used for hybrid / reranker / cache toggles. */
export function Toggle({
  label,
  checked,
  onChange,
  disabled,
  title,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      aria-pressed={checked}
      onClick={() => onChange(!checked)}
      className={clsx(
        "flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs transition-colors disabled:opacity-50",
        checked
          ? "border-primary/50 bg-primary/15 text-primary"
          : "border-border bg-surface-container text-on-surface-muted hover:border-border hover:text-on-surface",
      )}
    >
      <span
        className={clsx(
          "h-1.5 w-1.5 rounded-full",
          checked ? "bg-primary" : "bg-on-surface-muted/50",
        )}
      />
      {label}
    </button>
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
