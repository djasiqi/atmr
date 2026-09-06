/** DRIVER-COLD-02 — géométrie réservée du hub chauffeur (pas de swap brutal). */

export { DRIVER_DASHBOARD_STATUS_AREA_HEIGHT } from "./driverHubStatusModel";

export const DRIVER_DASHBOARD_MAP_HEIGHT = 142;
export const DRIVER_DASHBOARD_MISSION_SLOT_MIN = 248;
export const DRIVER_DASHBOARD_HEADER_MIN = 56;
export const DRIVER_DASHBOARD_AVATAR = 46;
/** Plus de bande StatusArea séparée : ligne 3 du header = GPS ou alerte. */
export const DRIVER_DASHBOARD_HEADER_TO_STATUS_GAP = 0;
/** Ligne GPS/statut → haut de mission : 24–32 px visuels. */
export const DRIVER_DASHBOARD_STATUS_TO_MISSION_GAP = 24;

export type DriverDashboardPrimarySlot = "pending" | "mission" | "idle";

export function resolveDriverDashboardPrimarySlot(input: {
  pending: boolean;
  hasActiveMission: boolean;
}): DriverDashboardPrimarySlot {
  if (input.pending) return "pending";
  if (input.hasActiveMission) return "mission";
  return "idle";
}
