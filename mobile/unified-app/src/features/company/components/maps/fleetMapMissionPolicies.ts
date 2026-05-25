import type { FleetRouteEmphasisLevel } from "./fleetMapMissionVisual";

/** Garde-fous mission-first — lisibilité cockpit > exhaustivité visuelle. */
export const FLEET_MISSION_MAP_POLICY = {
  routeCapCompact: 2,
  routeCapExpanded: 2,
  maxVisibleMissionsCompact: 6,
  maxVisibleMissionsExpanded: 12,
  maxRouteStrokePx: 4,
  minRouteStrokePx: 2,
  maxGlowExtraPx: 4,
  maxSimultaneousEtaBadges: 1,
  cameraAnimationMs: 620,
  cameraEaseMs: 720,
  etaSmoothHoldMs: 45_000,
  etaMinChangeMinutes: 2,
  priorityDecayMs: 520,
  routeEnterMs: 380,
  routeExitMs: 420,
  overlayPositionHoldMs: 12_000,
  maxConcurrentMarkerPulse: 2,
  passiveRouteOpacity: 0.38,
  secondaryRouteOpacity: 0.62,
  urgentRouteOpacity: 0.85,
  activeRouteOpacity: 0.85,
  focusRegionPadding: 1.55,
  minFocusDelta: 0.02,
} as const;

/** Gradient d’emphase progressive (niveau 1 → 4). */
export const FLEET_ROUTE_EMPHASIS_OPACITY: Record<FleetRouteEmphasisLevel, number> = {
  1: FLEET_MISSION_MAP_POLICY.activeRouteOpacity,
  2: FLEET_MISSION_MAP_POLICY.urgentRouteOpacity,
  3: FLEET_MISSION_MAP_POLICY.secondaryRouteOpacity,
  4: FLEET_MISSION_MAP_POLICY.passiveRouteOpacity,
};

export const FLEET_ROUTE_EMPHASIS_STROKE: Record<FleetRouteEmphasisLevel, number> = {
  1: FLEET_MISSION_MAP_POLICY.maxRouteStrokePx,
  2: 4,
  3: 3,
  4: FLEET_MISSION_MAP_POLICY.minRouteStrokePx,
};

export const FLEET_MISSION_LIFECYCLE_LEGEND: {
  phase: string;
  label: string;
  color: string;
  dash?: boolean;
}[] = [
  { phase: "assigned", label: "Assignée", color: "#94A3B8", dash: true },
  { phase: "en_route_pickup", label: "Vers prise en charge", color: "#00796B" },
  { phase: "patient_on_board", label: "Patient à bord", color: "#3B82F6" },
  { phase: "arrived", label: "Arrivée", color: "#94A3B8", dash: true },
  { phase: "delayed", label: "Retard", color: "#EF4444" },
];

export function applyPriorityDecayMultiplier(
  emphasis: FleetRouteEmphasisLevel,
  recentlyUnfocused: boolean,
  nowMs: number,
  unfocusedAtMs: number | null
): number {
  if (!recentlyUnfocused || unfocusedAtMs == null) return 1;
  const elapsed = nowMs - unfocusedAtMs;
  if (elapsed >= FLEET_MISSION_MAP_POLICY.priorityDecayMs) return 1;
  const t = elapsed / FLEET_MISSION_MAP_POLICY.priorityDecayMs;
  const floor = emphasis === 4 ? 0.72 : emphasis === 3 ? 0.78 : 0.85;
  return floor + (1 - floor) * t;
}

export function shouldShowEtaBadge(options: {
  isSelected: boolean;
  emphasis: FleetRouteEmphasisLevel;
  badgeIndex: number;
}): boolean {
  if (options.badgeIndex >= FLEET_MISSION_MAP_POLICY.maxSimultaneousEtaBadges) return false;
  return options.isSelected || options.emphasis <= 2;
}

export function clampRouteStroke(width: number): number {
  return Math.min(
    FLEET_MISSION_MAP_POLICY.maxRouteStrokePx,
    Math.max(FLEET_MISSION_MAP_POLICY.minRouteStrokePx, width)
  );
}
