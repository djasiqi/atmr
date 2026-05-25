import { useMemo } from "react";
import { useActiveCompanyContextId } from "../hooks";

export function parseCompanyNumericId(contextId: string | null | undefined): number | null {
  if (!contextId?.startsWith("company:")) return null;
  const parsed = Number.parseInt(contextId.slice("company:".length), 10);
  return Number.isFinite(parsed) ? parsed : null;
}

export function useCompanyNumericId(): number | null {
  const contextId = useActiveCompanyContextId();
  return useMemo(() => parseCompanyNumericId(contextId), [contextId]);
}
