"use client";

import { createContext, useContext, type ReactNode } from "react";
import { useStack, type StackState } from "@/lib/useStack";

const Ctx = createContext<StackState | null>(null);

/** One readiness poller for the whole app, shared by the rail and the Overview. */
export function StackProvider({ children }: { children: ReactNode }) {
  const stack = useStack();
  return <Ctx.Provider value={stack}>{children}</Ctx.Provider>;
}

export function useStackContext(): StackState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useStackContext must be used inside StackProvider");
  return v;
}
