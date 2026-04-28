import type { CompanyDispatchMission } from "../api/contracts";

export function isDispatchCompleted(m: CompanyDispatchMission | { status: string }): boolean {
  const s = (m.status ?? "").toLowerCase();
  return s === "completed" || s === "return_completed";
}

export function isDispatchCancelled(m: CompanyDispatchMission | { status: string }): boolean {
  const s = (m.status ?? "").toLowerCase();
  return s === "cancelled" || s === "canceled";
}
