// RAGScope: small formatting helpers. A missing value is written "n/a", never invented.

export function ms(value: number | null | undefined): string {
  if (value == null) return "n/a";
  if (value < 1000) return `${value.toFixed(0)} ms`;
  return `${(value / 1000).toFixed(2)} s`;
}

export function usd(value: number | null | undefined): string {
  if (value == null) return "n/a";
  if (value === 0) return "$0.00";
  if (value < 0.01) return `$${value.toFixed(6)}`;
  return `$${value.toFixed(4)}`;
}

export function num(value: number | null | undefined): string {
  if (value == null) return "n/a";
  return value.toLocaleString("en-US");
}

export function pct(value: number | null | undefined, digits = 1): string {
  if (value == null) return "n/a";
  return `${(value * 100).toFixed(digits)}%`;
}

export function bytes(value: number | null | undefined): string {
  if (value == null) return "n/a";
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(0)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

export function shortId(id: string, len = 8): string {
  return id.length > len ? id.slice(0, len) : id;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "14 Aug 2026, 14:03" in the viewer's time zone. */
export function dateTime(iso: string | number | Date | null | undefined): string {
  if (iso == null) return "n/a";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "n/a";
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}, ${hh}:${mm}`;
}

export function timeAgo(iso: string): string {
  const then = new Date(iso).getTime();
  const secs = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (secs < 60) return `${secs} s ago`;
  if (secs < 3600) return `${Math.floor(secs / 60)} min ago`;
  if (secs < 86400) return `${Math.floor(secs / 3600)} h ago`;
  const days = Math.floor(secs / 86400);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

/** Whole days between an ISO time and now. */
export function ageDays(iso: string): number {
  return Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 86400000));
}
