import clsx from "clsx";
import type { ReactNode } from "react";
import { dateTime, shortId } from "@/lib/format";
import { StateMark } from "./Marks";

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <div className="page-head">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <h1>{title}</h1>
        {actions}
      </div>
      {subtitle && <p className="lede">{subtitle}</p>}
    </div>
  );
}

/**
 * A notebook entry: a header strip with the record kind, its real date and short id (or the endpoint it was
 * read from), the body, and a witness footer naming the full id. Nothing here is typed in by hand.
 */
export function Entry({
  kind,
  date,
  id,
  source,
  actions,
  title,
  children,
  className,
}: {
  kind: string;
  date?: string | number | Date | null;
  id?: string;
  /** For entries that summarise live state rather than one record, e.g. "GET /readyz". */
  source?: string;
  actions?: ReactNode;
  title?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={clsx("entry", className)}>
      <header className="entry-head">
        <span className="kind">{kind}</span>
        {date != null && <time dateTime={new Date(date).toISOString()}>{dateTime(date)}</time>}
        {id && <code title={id}>{shortId(id, 8)}</code>}
        {source && <code>{source}</code>}
        {actions && <span className="actions">{actions}</span>}
      </header>
      <div className="entry-body">
        {title && <h2 className="entry-title">{title}</h2>}
        {children}
      </div>
      {id && <footer className="witness">Recorded by the RAGScope API. {kind} id <code>{id}</code></footer>}
    </section>
  );
}

/** A figure with a caption written from the data it shows. */
export function Figure({ children, caption }: { children: ReactNode; caption: ReactNode }) {
  return (
    <figure className="figure">
      {children}
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="note is-wait" role="status">
      <StateMark kind="wait" />
      <p>{label}…</p>
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: ReactNode }) {
  return (
    <div className="empty">
      <b>{title}</b>
      {hint && <p>{hint}</p>}
    </div>
  );
}

export function ErrorState({
  message,
  onRetry,
  children,
}: {
  message: string;
  onRetry?: () => void;
  children?: ReactNode;
}) {
  return (
    <div className="note is-error" role="alert">
      <StateMark kind="down" />
      <div>
        <b>{message}</b>
        {children}
        {onRetry && (
          <div>
            <button type="button" className="btn btn-sm" onClick={onRetry}>
              Try again
            </button>
          </div>
        )}
      </div>
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
    <label className={clsx("field", className)}>
      <span className="lab">{label}</span>
      {children}
      {hint && <span className="hint">{hint}</span>}
    </label>
  );
}

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
      className="input"
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
      className="input"
    />
  );
}

export function Select<T extends string>({
  value,
  onChange,
  options,
  disabled,
  label,
}: {
  value: T;
  onChange: (v: T) => void;
  options: readonly { value: T; label: string }[];
  disabled?: boolean;
  label?: string;
}) {
  return (
    <select
      value={value}
      disabled={disabled}
      aria-label={label}
      onChange={(e) => onChange(e.target.value as T)}
      className="select"
    >
      {options.map((o) => (
        <option key={o.value} value={o.value}>
          {o.label}
        </option>
      ))}
    </select>
  );
}

/** On/off switch for the pipeline options: a drawn mark plus the label, pressed = filled. */
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
      className="toggle"
    >
      <StateMark kind={checked ? "ok" : "off"} />
      {label}
      <span className="sr-only">{checked ? "on" : "off"}</span>
    </button>
  );
}

export function Chip({ children, tone }: { children: ReactNode; tone?: "on" | "red" }) {
  return <span className={clsx("chip", tone === "on" && "is-on", tone === "red" && "is-red")}>{children}</span>;
}
