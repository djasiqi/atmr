import type { CompanyDispatchMission } from "../api/contracts";
import { isDispatchCancelled, isDispatchCompleted } from "./companyDispatchStatus";

function isActiveDispatchLikeStatus(s: string): boolean {
  return s === "assigned" || s === "en_route" || s === "in_progress" || s === "arrived";
}

/**
 * Aligne l’affichage sur l’intention de chaque pastille, même si le repli d’API ou
 * un cache apporte des statuts incohérents avec `?status=…`.
 * (L’API mobile résume parfois plusieurs `BookingStatus` en `status: "assigned"`.)
 */
export function missionMatchesDispatchListFilter(
  m: CompanyDispatchMission,
  chip: string
): boolean {
  const c = (chip || "all").trim().toLowerCase();
  if (!c || c === "all" || c === "tout" || c === "total" || c === "any") {
    return true;
  }
  const s = (m.status ?? "").toLowerCase();

  if (c === "pending" || c === "en_attente" || c === "en-attente" || c === "awaiting") {
    return s === "pending" || s === "accepted" || s === "proposed";
  }
  if (c === "completed" || c === "termines" || c === "termine" || c === "done") {
    return isDispatchCompleted(m);
  }
  if (
    c === "cancelled" ||
    c === "canceled" ||
    c === "annules" ||
    c === "annulés" ||
    c === "annule"
  ) {
    return isDispatchCancelled(m);
  }
  if (
    c === "in_flight" ||
    c === "en_course" ||
    c === "en-route" ||
    c === "active" ||
    c === "on_trip"
  ) {
    if (isDispatchCompleted(m) || isDispatchCancelled(m)) return false;
    if (s === "pending" || s === "accepted" || s === "proposed") return false;
    if (s !== "en_route" && s !== "in_progress") return false;
    return m.driver_id != null;
  }
  if (c === "assigned" || c === "assignes" || c === "affecte" || c === "affectes") {
    if (isDispatchCompleted(m) || isDispatchCancelled(m)) return false;
    if (s === "pending" || s === "accepted" || s === "proposed") return false;
    return m.driver_id != null && isActiveDispatchLikeStatus(s);
  }
  if (c === "unassigned" || c === "non_affecte") {
    return true;
  }
  if (c === "urgent") {
    return true;
  }
  return true;
}

export function filterMissionsByDispatchListChip(
  list: CompanyDispatchMission[],
  chip: string
): CompanyDispatchMission[] {
  if (!list.length) return list;
  return list.filter((m) => missionMatchesDispatchListFilter(m, chip));
}
