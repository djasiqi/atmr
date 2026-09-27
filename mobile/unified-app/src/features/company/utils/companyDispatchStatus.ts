import type { CompanyDispatchMission } from "../api/contracts";

export function isDispatchCompleted(m: CompanyDispatchMission | { status: string }): boolean {
  const s = (m.status ?? "").toLowerCase();
  return s === "completed" || s === "return_completed";
}

export function isDispatchCancelled(m: CompanyDispatchMission | { status: string }): boolean {
  const s = (m.status ?? "").toLowerCase();
  return s === "cancelled" || s === "canceled";
}

const MANUAL_COMPLETE_STATUSES = new Set(["accepted", "assigned", "in_progress", "en_route"]);

/** Même règle que le bouton web « Valider la course ». */
export function canManualCompleteRide(m: { status?: string | null }): boolean {
  return MANUAL_COMPLETE_STATUSES.has((m.status ?? "").toLowerCase());
}
