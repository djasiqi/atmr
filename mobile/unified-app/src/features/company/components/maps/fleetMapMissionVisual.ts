import type { CompanyDispatchMission } from "../../api/contracts";
import {
  formatEtaLabel,
  formatMissionTime,
  isMissionDelayed,
  missionHasDefinedPickupTime,
} from "../../dashboard/companyDashboardMissionUi";
import type { FleetDriverMapItem } from "./fleetMapTypes";
import {
  applyPriorityDecayMultiplier,
  clampRouteStroke,
  FLEET_MISSION_MAP_POLICY,
  FLEET_ROUTE_EMPHASIS_OPACITY,
  FLEET_ROUTE_EMPHASIS_STROKE,
} from "./fleetMapMissionPolicies";
import { classifySemanticRoute } from "../../dashboard/cockpit/semanticRouteSystem";
import { computeMapDensityPolicy } from "../../dashboard/cockpit/mapDensityGovernor";
import {
  resolveFleetMissionLegPlans,
  resolveFleetMissionRouteFocusLeg,
  resolveFleetOverlayRouteDrawPoints,
  type FleetDirectionsPlan,
  type FleetRouteLegId,
  type FleetRoutedPathState,
} from "./fleetMapDirections";
import { FLEET_MAP_COLORS } from "./mapStatusTheme";

/** Phases lifecycle mission-first (lecture instantanée carte). */
export type FleetMissionLifecyclePhase =
  | "assigned"
  | "en_route_pickup"
  | "patient_on_board"
  | "arrived"
  | "delayed";

/** Hiérarchie visuelle des routes (1 = dominante). */
export type FleetRouteEmphasisLevel = 1 | 2 | 3 | 4;

export type FleetMissionRouteStyle = {
  color: string;
  glowColor: string;
  strokeWidth: number;
  glowWidth: number;
  opacity: number;
  lineDashPattern: number[] | null;
  zIndex: number;
};

export type FleetMissionAnchorRole = "pickup" | "dropoff" | "driver" | "urgent" | "active";

export type FleetMissionAnchorStyle = {
  role: FleetMissionAnchorRole;
  fill: string;
  stroke: string;
  radius: number;
  opacity: number;
  zIndex: number;
};

export type FleetMissionOverlay = {
  missionId: number;
  driverId: number | null;
  lifecyclePhase: FleetMissionLifecyclePhase;
  emphasisLevel: FleetRouteEmphasisLevel;
  isSelected: boolean;
  isUrgent: boolean;
  pickup: { latitude: number; longitude: number } | null;
  dropoff: { latitude: number; longitude: number } | null;
  driverPosition: { latitude: number; longitude: number } | null;
  points: { latitude: number; longitude: number }[];
  /** Requête Google Directions (itinéraire routier). */
  directionsPlan: FleetDirectionsPlan | null;
  /** Plans par segment quand la mission est sélectionnée (chauffeur→pickup / pickup→dropoff). */
  legDirectionsPlans?: Partial<Record<FleetRouteLegId, FleetDirectionsPlan>>;
  routeFocusLeg?: FleetRouteLegId | null;
  routeStyle: FleetMissionRouteStyle;
  pickupAnchor: FleetMissionAnchorStyle | null;
  dropoffAnchor: FleetMissionAnchorStyle | null;
  etaLabel: string | null;
  etaBadgeLabel: string | null;
  zIndex: number;
  /** Opacité finale après decay / breathing (0–1). */
  displayOpacity: number;
  showEtaBadge: boolean;
};

export type FleetMissionRouteLegRender = {
  leg: FleetRouteLegId;
  coordinates: { latitude: number; longitude: number }[];
  style: FleetMissionRouteStyle;
  zIndex: number;
};

const PICKUP_COLOR = "#00796B";
const DROPOFF_COLOR = "#3B82F6";
const DELAY_COLOR = "#EF4444";
const ARRIVED_MUTED = "rgba(100, 116, 139, 0.55)";

const etaSmoothCache = new Map<number, { label: string; atMs: number }>();

export function resolveMissionLifecyclePhase(
  mission: CompanyDispatchMission
): FleetMissionLifecyclePhase {
  if (isMissionDelayed(mission)) return "delayed";
  if (mission.status === "arrived") return "arrived";
  if (mission.status === "in_progress") return "patient_on_board";
  if (mission.status === "en_route") return "en_route_pickup";
  return "assigned";
}

export function resolveRouteEmphasisLevel(options: {
  isSelected: boolean;
  isUrgent: boolean;
  mission: CompanyDispatchMission;
  rankIndex: number;
}): FleetRouteEmphasisLevel {
  if (options.isSelected) return 1;
  if (options.isUrgent) return 2;
  if (options.rankIndex <= 1) return 3;
  return 4;
}

export function resolveMissionRouteStyle(
  phase: FleetMissionLifecyclePhase,
  emphasis: FleetRouteEmphasisLevel,
  opacityMultiplier = 1
): FleetMissionRouteStyle {
  const passive = emphasis === 4;
  const baseOpacity = FLEET_ROUTE_EMPHASIS_OPACITY[emphasis] * opacityMultiplier;
  const strokeWidth = clampRouteStroke(FLEET_ROUTE_EMPHASIS_STROKE[emphasis]);
  const glowWidth = strokeWidth + FLEET_MISSION_MAP_POLICY.maxGlowExtraPx;

  switch (phase) {
    case "assigned":
      return {
        color: passive ? "rgba(148, 163, 184, 0.7)" : PICKUP_COLOR,
        glowColor: "rgba(0, 121, 107, 0.12)",
        strokeWidth,
        glowWidth,
        opacity: baseOpacity * 0.85,
        lineDashPattern: [8, 10],
        zIndex: 20 + (4 - emphasis),
      };
    case "en_route_pickup":
      return {
        color: PICKUP_COLOR,
        glowColor: "rgba(0, 121, 107, 0.12)",
        strokeWidth,
        glowWidth,
        opacity: Math.min(FLEET_MISSION_MAP_POLICY.activeRouteOpacity, baseOpacity),
        lineDashPattern: null,
        zIndex: 30 + (4 - emphasis),
      };
    case "patient_on_board":
      return {
        color: DROPOFF_COLOR,
        glowColor: "rgba(59, 130, 246, 0.12)",
        strokeWidth,
        glowWidth,
        opacity: Math.min(FLEET_MISSION_MAP_POLICY.activeRouteOpacity, baseOpacity),
        lineDashPattern: null,
        zIndex: 40 + (4 - emphasis),
      };
    case "arrived":
      return {
        color: ARRIVED_MUTED,
        glowColor: "rgba(100, 116, 139, 0.1)",
        strokeWidth: Math.max(2, strokeWidth - 1),
        glowWidth: glowWidth - 2,
        opacity: baseOpacity * 0.7,
        lineDashPattern: [4, 8],
        zIndex: 15,
      };
    case "delayed":
      return {
        color: DELAY_COLOR,
        glowColor: "rgba(239, 68, 68, 0.10)",
        strokeWidth,
        glowWidth,
        opacity: Math.min(FLEET_MISSION_MAP_POLICY.urgentRouteOpacity, baseOpacity),
        lineDashPattern: null,
        zIndex: 50 + (4 - emphasis),
      };
    default:
      return {
        color: FLEET_MAP_COLORS.route,
        glowColor: "rgba(52, 152, 219, 0.15)",
        strokeWidth,
        glowWidth,
        opacity: baseOpacity,
        lineDashPattern: null,
        zIndex: 20,
      };
  }
}

function resolvePickupAnchor(
  phase: FleetMissionLifecyclePhase,
  emphasis: FleetRouteEmphasisLevel,
  isSelected: boolean
): FleetMissionAnchorStyle {
  const delayed = phase === "delayed";
  return {
    role: delayed ? "urgent" : isSelected ? "active" : "pickup",
    fill: delayed ? DELAY_COLOR : PICKUP_COLOR,
    stroke: "#FFFFFF",
    radius: isSelected ? 9 : emphasis <= 2 ? 8 : 6,
    opacity: emphasis === 4 ? 0.55 : 1,
    zIndex: isSelected ? 80 : 50,
  };
}

export function resolveMissionRouteLegStyle(
  baseStyle: FleetMissionRouteStyle,
  leg: FleetRouteLegId,
  focusLeg: FleetRouteLegId
): FleetMissionRouteStyle {
  if (leg === focusLeg) return baseStyle;
  return {
    ...baseStyle,
    color: "rgba(148, 163, 184, 0.72)",
    glowColor: "rgba(148, 163, 184, 0.08)",
    strokeWidth: Math.max(FLEET_MISSION_MAP_POLICY.minRouteStrokePx, baseStyle.strokeWidth - 1),
    glowWidth: Math.max(baseStyle.strokeWidth, baseStyle.glowWidth - 2),
    opacity: Math.min(baseStyle.opacity, FLEET_MISSION_MAP_POLICY.passiveRouteOpacity),
    lineDashPattern: [6, 10],
    zIndex: baseStyle.zIndex - 2,
  };
}

export function resolveFleetMissionRouteLegRenders(
  overlay: FleetMissionOverlay,
  routedPathsByKey: ReadonlyMap<string, { latitude: number; longitude: number }[]>,
  routedStateByKey: ReadonlyMap<string, FleetRoutedPathState>
): FleetMissionRouteLegRender[] {
  const useSplitLegs =
    overlay.isSelected &&
    overlay.routeFocusLeg != null &&
    overlay.legDirectionsPlans != null &&
    Object.keys(overlay.legDirectionsPlans).length > 0;

  if (!useSplitLegs) {
    const points = resolveFleetOverlayRouteDrawPoints(overlay, routedPathsByKey, routedStateByKey);
    if (points.length < 2) return [];
    return [
      {
        leg: "to_dropoff",
        coordinates: points,
        style: overlay.routeStyle,
        zIndex: overlay.zIndex,
      },
    ];
  }

  const focusLeg = overlay.routeFocusLeg!;
  const legs: FleetRouteLegId[] = ["to_pickup", "to_dropoff"];
  const renders: FleetMissionRouteLegRender[] = [];

  for (const leg of legs) {
    if (!overlay.legDirectionsPlans?.[leg]) continue;
    const points = resolveFleetOverlayRouteDrawPoints(
      overlay,
      routedPathsByKey,
      routedStateByKey,
      leg
    );
    if (points.length < 2) continue;
    renders.push({
      leg,
      coordinates: points,
      style: resolveMissionRouteLegStyle(overlay.routeStyle, leg, focusLeg),
      zIndex: overlay.zIndex + (leg === focusLeg ? 2 : 0),
    });
  }

  return renders.sort((a, b) => a.zIndex - b.zIndex);
}

function resolveDropoffAnchor(
  phase: FleetMissionLifecyclePhase,
  emphasis: FleetRouteEmphasisLevel,
  isSelected: boolean
): FleetMissionAnchorStyle {
  const muted = phase === "arrived";
  return {
    role: isSelected ? "active" : "dropoff",
    fill: muted ? "#94A3B8" : DROPOFF_COLOR,
    stroke: "#FFFFFF",
    radius: isSelected ? 10 : emphasis <= 2 ? 8 : 7,
    opacity: emphasis === 4 ? 0.5 : muted ? 0.75 : 1,
    zIndex: isSelected ? 85 : 55,
  };
}

/** Libellé court pour pastille ETA sur la carte (évite la troncature « ~6 m »). */
export function formatMapMissionEtaBadge(
  mission: CompanyDispatchMission,
  nowMs = Date.now()
): string | null {
  if (!missionHasDefinedPickupTime(mission.scheduled_at)) {
    return null;
  }
  const delay = Number(mission.assignment_pickup_delay_minutes);
  if (Number.isFinite(delay) && delay > 0) {
    return `+${Math.round(delay)} min`;
  }

  const duration = Number(mission.route_duration_min);
  if (Number.isFinite(duration) && duration > 0) {
    return `~${Math.round(duration)} min`;
  }

  const eta = formatEtaLabel(mission);
  if (!eta) return null;
  if (eta === "Imminent" || eta === "En route") return eta;
  return eta.split("·")[0]?.trim() ?? eta;
}

/** ETA stabilisé (anti-jitter) pour badges carte. */
export function formatStableMissionEtaBadge(
  mission: CompanyDispatchMission,
  nowMs = Date.now()
): string | null {
  const raw = formatMapMissionEtaBadge(mission, nowMs);
  if (!raw) return null;

  const cached = etaSmoothCache.get(mission.mission_id);
  if (cached && nowMs - cached.atMs < FLEET_MISSION_MAP_POLICY.etaSmoothHoldMs) {
    const nextNumeric = extractMinutes(raw);
    const cachedNumeric = extractMinutes(cached.label);
    if (
      nextNumeric != null &&
      cachedNumeric != null &&
      Math.abs(nextNumeric - cachedNumeric) <= FLEET_MISSION_MAP_POLICY.etaMinChangeMinutes
    ) {
      return cached.label;
    }
  }

  etaSmoothCache.set(mission.mission_id, { label: raw, atMs: nowMs });
  if (etaSmoothCache.size > 48) {
    const oldest = [...etaSmoothCache.entries()].sort((a, b) => a[1].atMs - b[1].atMs)[0];
    if (oldest) etaSmoothCache.delete(oldest[0]);
  }
  return raw;
}

function extractMinutes(label: string): number | null {
  const m = label.match(/(\d+)\s*min/i);
  return m ? Number(m[1]) : null;
}

/** Badge route enrichi : temps relatif + heure cible si disponible. */
export function formatRouteEtaBadge(mission: CompanyDispatchMission, nowMs = Date.now()): string | null {
  const delay = Number(mission.assignment_pickup_delay_minutes);
  if (Number.isFinite(delay) && delay > 0) {
    return `+${Math.round(delay)} min`;
  }

  const duration = Number(mission.route_duration_min);
  if (Number.isFinite(duration) && duration > 0) {
    const target = mission.scheduled_at ? formatMissionTime(mission.scheduled_at) : null;
    return target ? `~${Math.round(duration)} min · ${target}` : `~${Math.round(duration)} min`;
  }

  const eta = formatEtaLabel(mission);
  if (!eta) return null;

  if (mission.scheduled_at && (mission.status === "assigned" || mission.status === "accepted")) {
    const scheduled = formatMissionTime(mission.scheduled_at);
    return `${eta} · ${scheduled}`;
  }

  return eta;
}

function isMissionInProgress(status: CompanyDispatchMission["status"]): boolean {
  return status === "in_progress" || status === "arrived";
}

function isMissionTransit(status: CompanyDispatchMission["status"]): boolean {
  return status === "en_route" || status === "assigned" || status === "accepted" || status === "proposed";
}

function buildRoutePoints(
  driver: FleetDriverMapItem | null,
  mission: CompanyDispatchMission
): { latitude: number; longitude: number }[] {
  const points: { latitude: number; longitude: number }[] = [];
  const push = (p: { latitude: number; longitude: number }) => {
    const prev = points[points.length - 1];
    if (!prev || prev.latitude !== p.latitude || prev.longitude !== p.longitude) {
      points.push(p);
    }
  };

  if (driver) {
    push({ latitude: driver.latitude, longitude: driver.longitude });
  }

  const pickup =
    mission.pickup_lat != null && mission.pickup_lon != null
      ? { latitude: mission.pickup_lat, longitude: mission.pickup_lon }
      : null;
  const dropoff =
    mission.dropoff_lat != null && mission.dropoff_lon != null
      ? { latitude: mission.dropoff_lat, longitude: mission.dropoff_lon }
      : null;

  if (isMissionInProgress(mission.status)) {
    if (dropoff) push(dropoff);
  } else if (isMissionTransit(mission.status)) {
    if (pickup) push(pickup);
    if (dropoff) push(dropoff);
  } else {
    if (pickup) push(pickup);
    else if (dropoff) push(dropoff);
  }

  if (points.length < 2 && pickup) push(pickup);
  if (points.length < 2 && dropoff) push(dropoff);

  return points;
}

function missionDisplayScore(
  mission: CompanyDispatchMission,
  driver: FleetDriverMapItem | null,
  isSelected: boolean
): number {
  let score = 0;
  if (isSelected) score += 10_000;
  if (driver) {
    const st = driver.enrichment.operationalStatus;
    if (st === "incident") score += 5000;
    if (st === "delayed") score += 4500;
    if (st === "on_mission") score += 2000;
  }
  if (isMissionDelayed(mission)) score += 4200;
  if (mission.status === "in_progress") score += 3800;
  if (mission.status === "en_route") score += 3600;
  if (mission.status === "assigned") score += 3200;
  return score;
}

/** Fusionne missions dispatch + missions liées aux chauffeurs visibles sur la carte. */
export function collectMapMissions(
  missions: CompanyDispatchMission[],
  driversById: Map<number, FleetDriverMapItem>
): CompanyDispatchMission[] {
  const byId = new Map<number, CompanyDispatchMission>();
  for (const m of missions) byId.set(m.mission_id, m);
  for (const driver of driversById.values()) {
    const linked = driver.enrichment.linkedMission;
    if (linked) byId.set(linked.mission_id, linked);
  }
  return [...byId.values()];
}

export type BuildMissionOverlayOptions = {
  missions: CompanyDispatchMission[];
  driversById: Map<number, FleetDriverMapItem>;
  selectedMissionId: number | null;
  maxVisible?: number;
  /** Limite stricte anti-spaghetti (défaut 3) — hors missions liées à un chauffeur affiché. */
  routeCap?: number;
  /** Mission récemment désélectionnée → decay progressif d’opacité. */
  decayUnfocusedAt?: Map<number, number>;
  nowMs?: number;
};

export function buildFleetMissionOverlays({
  missions,
  driversById,
  selectedMissionId,
  maxVisible = FLEET_MISSION_MAP_POLICY.maxVisibleMissionsCompact,
  routeCap = FLEET_MISSION_MAP_POLICY.routeCapCompact,
  decayUnfocusedAt,
  nowMs = Date.now(),
}: BuildMissionOverlayOptions): FleetMissionOverlay[] {
  const allMissions = collectMapMissions(missions, driversById);
  const inFlight = allMissions.filter(
    (m) =>
      m.status !== "completed" &&
      m.status !== "cancelled" &&
      (m.pickup_lat != null || m.dropoff_lat != null || m.driver_id != null)
  );

  const ranked = [...inFlight].sort((a, b) => {
    const sa = missionDisplayScore(
      a,
      a.driver_id != null ? driversById.get(a.driver_id) ?? null : null,
      a.mission_id === selectedMissionId
    );
    const sb = missionDisplayScore(
      b,
      b.driver_id != null ? driversById.get(b.driver_id) ?? null : null,
      b.mission_id === selectedMissionId
    );
    return sb - sa;
  });

  const density = computeMapDensityPolicy({ driverCount: driversById.size });
  const visible = ranked.slice(0, maxVisible);
  const overlays: FleetMissionOverlay[] = [];

  visible.forEach((mission, rankIndex) => {
    const driver =
      mission.driver_id != null ? driversById.get(mission.driver_id) ?? null : null;
    const points = buildRoutePoints(driver, mission);
    if (points.length < 2) return;

    const isSelected = selectedMissionId === mission.mission_id;
    const isUrgent =
      isMissionDelayed(mission) ||
      driver?.enrichment.operationalStatus === "incident" ||
      driver?.enrichment.operationalStatus === "delayed";
    const lifecyclePhase = resolveMissionLifecyclePhase(mission);
    const emphasisLevel = resolveRouteEmphasisLevel({
      isSelected,
      isUrgent,
      mission,
      rankIndex,
    });

    const hasVisibleDriver =
      mission.driver_id != null && driversById.has(mission.driver_id);
    if (!isSelected && !isUrgent && !hasVisibleDriver && rankIndex >= routeCap) return;

    const unfocusedAt = decayUnfocusedAt?.get(mission.mission_id) ?? null;
    const decayMultiplier = applyPriorityDecayMultiplier(
      emphasisLevel,
      !isSelected && unfocusedAt != null,
      nowMs,
      unfocusedAt
    );
    let routeStyle = resolveMissionRouteStyle(lifecyclePhase, emphasisLevel, decayMultiplier);
    const semantic = classifySemanticRoute({
      isSelected,
      isCritical: isUrgent,
      isSearchResult: false,
      recentlyInteracted: isSelected,
      attentionLevel: isUrgent ? "WARNING" : "INFO",
      density,
      maxVisibleRoutes: routeCap,
      routeIndex: rankIndex,
    });
    if (!semantic.visible) return;
    routeStyle = {
      ...routeStyle,
      opacity: routeStyle.opacity * semantic.opacity,
      strokeWidth: semantic.strokeWidth,
    };
    const pickup =
      mission.pickup_lat != null && mission.pickup_lon != null
        ? { latitude: mission.pickup_lat, longitude: mission.pickup_lon }
        : null;
    const dropoff =
      mission.dropoff_lat != null && mission.dropoff_lon != null
        ? { latitude: mission.dropoff_lat, longitude: mission.dropoff_lon }
        : null;

    const driverPosition = driver
      ? { latitude: driver.latitude, longitude: driver.longitude }
      : null;

    const legPlans = resolveFleetMissionLegPlans({
      driverPosition,
      pickup,
      dropoff,
      lifecyclePhase,
      isSelected,
    });

    overlays.push({
      missionId: mission.mission_id,
      driverId: mission.driver_id ?? null,
      lifecyclePhase,
      emphasisLevel,
      isSelected,
      isUrgent,
      pickup,
      dropoff,
      driverPosition,
      points,
      directionsPlan: legPlans.combined,
      legDirectionsPlans: isSelected ? legPlans.legs : undefined,
      routeFocusLeg: isSelected ? resolveFleetMissionRouteFocusLeg(lifecyclePhase) : null,
      routeStyle,
      pickupAnchor: pickup ? resolvePickupAnchor(lifecyclePhase, emphasisLevel, isSelected) : null,
      dropoffAnchor: dropoff ? resolveDropoffAnchor(lifecyclePhase, emphasisLevel, isSelected) : null,
      etaLabel: formatEtaLabel(mission),
      etaBadgeLabel: formatStableMissionEtaBadge(mission, nowMs),
      zIndex: routeStyle.zIndex + (isSelected ? 100 : 0),
      displayOpacity: routeStyle.opacity,
      showEtaBadge: false,
    });
  });

  const sorted = overlays.sort((a, b) => a.zIndex - b.zIndex);
  let etaBadgeCount = 0;
  for (const overlay of sorted) {
    if (
      !overlay.isSelected &&
      overlay.emphasisLevel <= 2 &&
      etaBadgeCount < FLEET_MISSION_MAP_POLICY.maxSimultaneousEtaBadges
    ) {
      if (overlay.etaBadgeLabel) {
        overlay.showEtaBadge = true;
        etaBadgeCount += 1;
      }
    }
  }
  return sorted;
}

export function collectMissionFocusPositions(
  overlay: FleetMissionOverlay
): { latitude: number; longitude: number }[] {
  const points: { latitude: number; longitude: number }[] = [];
  if (overlay.driverPosition) points.push(overlay.driverPosition);
  if (overlay.pickup) points.push(overlay.pickup);
  if (overlay.dropoff) points.push(overlay.dropoff);
  return points;
}

export function computeMissionOverlayFocusRegion(
  overlay: FleetMissionOverlay,
  verticalBias = 0
): { latitude: number; longitude: number; latitudeDelta: number; longitudeDelta: number } | null {
  const points = collectMissionFocusPositions(overlay);
  if (points.length < 2) return null;

  const latitudes = points.map((p) => p.latitude);
  const longitudes = points.map((p) => p.longitude);
  const minLat = Math.min(...latitudes);
  const maxLat = Math.max(...latitudes);
  const minLng = Math.min(...longitudes);
  const maxLng = Math.max(...longitudes);
  const latitudeDelta = Math.max(
    FLEET_MISSION_MAP_POLICY.minFocusDelta,
    (maxLat - minLat) * FLEET_MISSION_MAP_POLICY.focusRegionPadding || 0.03
  );
  const longitudeDelta = Math.max(
    FLEET_MISSION_MAP_POLICY.minFocusDelta,
    (maxLng - minLng) * FLEET_MISSION_MAP_POLICY.focusRegionPadding || 0.03
  );

  return {
    latitude: (minLat + maxLat) / 2 - latitudeDelta * Math.max(0, verticalBias) * 0.18,
    longitude: (minLng + maxLng) / 2,
    latitudeDelta,
    longitudeDelta,
  };
}
