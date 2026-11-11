"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ROOT_URL } from "./api";

export type StackStatus = "checking" | "waking" | "ready" | "degraded" | "down";

export interface StackState {
  status: StackStatus;
  /** Per-dependency result from /readyz, e.g. { postgres: "ok", redis: "error" }. */
  checks: Record<string, string>;
  checkedAt: number | null;
  retry: () => void;
}

/**
 * Polls the API's /readyz. "waking" means the first answer is taking long (a sleeping host); "down" means
 * nothing answered. The result is only what the API reported, never a guess.
 */
export function useStack(intervalMs = 20000): StackState {
  const [status, setStatus] = useState<StackStatus>("checking");
  const [checks, setChecks] = useState<Record<string, string>>({});
  const [checkedAt, setCheckedAt] = useState<number | null>(null);
  const first = useRef(true);

  const check = useCallback(async () => {
    const slow = first.current
      ? setTimeout(() => setStatus((s) => (s === "checking" ? "waking" : s)), 4000)
      : null;
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 90000);
      const res = await fetch(`${ROOT_URL}/readyz`, { cache: "no-store", signal: controller.signal });
      clearTimeout(timer);
      const body = (await res.json()) as { status?: string; checks?: Record<string, string> };
      setChecks(body.checks ?? {});
      setStatus(body.status === "ready" ? "ready" : "degraded");
    } catch {
      setChecks({});
      setStatus("down");
    } finally {
      if (slow) clearTimeout(slow);
      first.current = false;
      setCheckedAt(Date.now());
    }
  }, []);

  useEffect(() => {
    check();
    const t = setInterval(check, intervalMs);
    return () => clearInterval(t);
  }, [check, intervalMs]);

  return {
    status,
    checks,
    checkedAt,
    retry: () => {
      setStatus("checking");
      check();
    },
  };
}
