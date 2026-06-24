import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";

import type { DashboardLiveOverlay } from "../../dashboard/companyDashboardViewModel";

import { resolveDriverDisplayName } from "../../utils/companyDriverMapStatus";
import { isFleetDriverConstrained } from "./fleetMapStatusContract";

import {

  conciseRouteSegment,

  formatEtaLabel,

  isMissionDelayed,

} from "../../dashboard/companyDashboardMissionUi";

import { projectCoordinate } from "./fleetMapGeo";
import { countClusterFleetMarkers } from "./fleetMapDevInstrumentation";

import type { FleetOperationalStatus } from "./mapStatusTheme";

import type {

  FleetActiveRoute,

  FleetDriverMapItem,

  FleetMapFiltersState,

  FleetMapMarker,

} from "./fleetMapTypes";

import { FLEET_MAP_COLORS } from "./mapStatusTheme";



/** ~500 m à Genève — regroupe les chauffeurs dans la même zone visible. */
const CLUSTER_CELL_DEG = 0.006;

export function isMissionInFlight(status: CompanyDispatchMission["status"] | undefined): boolean {
  return (
    status === "assigned" ||
    status === "accepted" ||
    status === "en_route" ||
    status === "arrived" ||
    status === "in_progress"
  );
}

export type FleetMissionIndexMaps = {
  byDriverId: Map<number, CompanyDispatchMission>;
  byMissionId: Map<number, CompanyDispatchMission>;
};

export function buildFleetMissionIndexMaps(
  missions: CompanyDispatchMission[]
): FleetMissionIndexMaps {
  return {
    byDriverId: missionByDriverId(missions),
    byMissionId: missionById(missions),
  };
}

function missionByDriverId(

  missions: CompanyDispatchMission[]

): Map<number, CompanyDispatchMission> {

  const map = new Map<number, CompanyDispatchMission>();

  for (const m of missions) {

    const id = m.driver_id;

    if (id == null || !Number.isFinite(id)) continue;

    const existing = map.get(id);

    if (!existing) {

      map.set(id, m);

      continue;

    }

    if (isMissionInFlight(m.status) && !isMissionInFlight(existing.status)) {

      map.set(id, m);

    }

  }

  return map;

}



function missionById(missions: CompanyDispatchMission[]): Map<number, CompanyDispatchMission> {

  const map = new Map<number, CompanyDispatchMission>();

  for (const m of missions) map.set(m.mission_id, m);

  return map;

}



function mapMissionVehicleType(driverType: string | null | undefined): string | null {

  if (!driverType) return null;

  const t = driverType.toUpperCase();

  if (t === "EMERGENCY") return "urgence";

  if (t === "REGULAR") return "berline";

  return t.toLowerCase();

}



function formatDistanceKm(km: number | null | undefined): string | null {

  if (km == null || !Number.isFinite(km) || km <= 0) return null;

  return `${km < 10 ? km.toFixed(1) : Math.round(km)} km`;

}



function resolveMissionDestination(mission: CompanyDispatchMission): {

  latitude: number;

  longitude: number;

} | null {

  const inProgress =

    mission.status === "in_progress" || mission.status === "arrived" || mission.status === "en_route";

  if (inProgress && mission.dropoff_lat != null && mission.dropoff_lon != null) {

    return { latitude: mission.dropoff_lat, longitude: mission.dropoff_lon };

  }

  if (mission.pickup_lat != null && mission.pickup_lon != null) {

    return { latitude: mission.pickup_lat, longitude: mission.pickup_lon };

  }

  if (mission.dropoff_lat != null && mission.dropoff_lon != null) {

    return { latitude: mission.dropoff_lat, longitude: mission.dropoff_lon };

  }

  return null;

}



function isMissionInProgress(status: CompanyDispatchMission["status"]): boolean {
  return status === "in_progress" || status === "arrived";
}

function isMissionTransit(status: CompanyDispatchMission["status"]): boolean {
  return status === "en_route" || status === "assigned" || status === "accepted" || status === "proposed";
}

function missionPriorityStatusScore(mission: CompanyDispatchMission | null): number {
  if (!mission) return 0;
  if (mission.status === "in_progress") return 500;
  if (mission.status === "arrived") return 480;
  if (mission.status === "en_route") return 460;
  if (mission.status === "assigned") return 440;
  if (mission.status === "accepted") return 420;
  if (mission.status === "proposed") return 400;
  if (mission.status === "pending") return 360;
  return 0;
}

function operationalStatusScore(driver: FleetDriverMapItem): number {
  const st = driver.enrichment.operationalStatus;
  if (st === "incident") return 320;
  if (st === "constrained") return 280;
  if (st === "delayed") return 300;
  if (st === "busy") return 260;
  if (st === "assigned") return 240;
  if (st === "emergency") return 310;
  if (st === "available") return 200;
  if (st === "break") return 150;
  return 100;
}

/** Chauffeur principal cockpit : mission active > urgence > première option disponible. */
export function pickPrimaryFleetDriver(drivers: FleetDriverMapItem[]): FleetDriverMapItem | null {
  if (drivers.length === 0) return null;
  const ranked = [...drivers].sort((a, b) => {
    const missionDelta =
      missionPriorityStatusScore(b.enrichment.linkedMission) -
      missionPriorityStatusScore(a.enrichment.linkedMission);
    if (missionDelta !== 0) return missionDelta;
    const statusDelta = operationalStatusScore(b) - operationalStatusScore(a);
    if (statusDelta !== 0) return statusDelta;
    return a.driver_id - b.driver_id;
  });
  return ranked[0] ?? null;
}

export { isFleetDriverConstrained } from "./fleetMapStatusContract";

/** Compteur « localisés » carte — parité web (hors offline / offline_unknown). */
export function isFleetDriverLocated(driver: CompanyDriverLiveLocation): boolean {
  const lat = Number(driver.latitude);
  const lon = Number(driver.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return false;
  const tracking = String(driver.tracking_display_status ?? "").toLowerCase();
  if (tracking === "offline_unknown") return false;
  const status = String(driver.location_status ?? "").toLowerCase();
  if (status === "offline") return false;
  return true;
}

const CONSTRAINT_REASON_LABELS: Record<string, string> = {
  battery_optimized: "Optimisation batterie active",
  permission_fg_denied: "Permission localisation refusée",
  permission_bg_denied: "Localisation arrière-plan refusée",
  gps_provider_disabled: "GPS désactivé sur l'appareil",
  fgs_not_running: "Service avant-plan inactif",
  fix_stale: "Dernière position trop ancienne",
};

/** Libellé exploitant pour device_health.constraint_reason (O5). */
export function formatFleetConstraintReason(driver: CompanyDriverLiveLocation): string {
  const raw = driver.device_health?.constraint_reason;
  if (raw == null || String(raw).trim() === "") return "Raison inconnue";
  const key = String(raw).trim().toLowerCase();
  return CONSTRAINT_REASON_LABELS[key] ?? key.replace(/_/g, " ");
}

function isDriverEmergency(driver: CompanyDriverLiveLocation): boolean {
  const status = String(driver.status ?? "").toLowerCase();
  if (status === "emergency") return true;
  const emergencyMode = (driver as { emergency_mode?: boolean }).emergency_mode;
  return emergencyMode === true;
}

export function resolveFleetOperationalStatus(
  driver: CompanyDriverLiveLocation,
  linkedMission: CompanyDispatchMission | null
): FleetOperationalStatus {
  if (driver.location_status === "last_known") return "last_known";

  const activeMission =
    linkedMission && isMissionInFlight(linkedMission.status) ? linkedMission : null;

  if (!activeMission && isFleetDriverConstrained(driver)) return "constrained";

  if (isDriverEmergency(driver)) return "emergency";

  const delayMin = Number(activeMission?.assignment_pickup_delay_minutes);
  if (activeMission && Number.isFinite(delayMin) && delayMin >= 20) return "incident";
  if (activeMission && isMissionDelayed(activeMission)) return "delayed";

  if (activeMission) {
    const backendStatus = String(driver.status ?? "").toLowerCase();
    if (backendStatus === "assigned" || activeMission.status === "assigned") return "assigned";
    return "busy";
  }

  const backendStatus = String(driver.status ?? "").toLowerCase();
  if (backendStatus === "offline") return "offline";
  if (backendStatus === "assigned") return "assigned";
  if (backendStatus === "busy") return "busy";

  const speed = Number(driver.speed);
  if (
    driver.location_status === "live" &&
    Number.isFinite(speed) &&
    speed < 0.4 &&
    driver.mission_id == null &&
    !activeMission
  ) {
    return "break";
  }

  if (backendStatus === "available") return "available";
  return "available";
}



export function enrichFleetDriver(

  driver: CompanyDriverLiveLocation,

  missions: CompanyDispatchMission[],

  organizationName?: string | null,

  index?: FleetMissionIndexMaps

): FleetDriverMapItem {

  const byDriver = index?.byDriverId ?? missionByDriverId(missions);

  const byMission = index?.byMissionId ?? missionById(missions);

  const linkedRaw =
    (driver.mission_id != null ? byMission.get(driver.mission_id) : null) ??
    byDriver.get(driver.driver_id) ??
    null;

  const linked =
    linkedRaw && isMissionInFlight(linkedRaw.status) ? linkedRaw : null;

  const operationalStatus = resolveFleetOperationalStatus(driver, linked);

  const delayMinutes = Number(linked?.assignment_pickup_delay_minutes);

  const etaLabel = linked ? formatEtaLabel(linked) : null;

  const displayName = resolveDriverDisplayName(driver, {
    missionDriverName: linked?.driver_name ?? linked?.partner_company_name,
    organizationName,
  });



  return {

    ...driver,

    driver_name: displayName,

    full_name: displayName,

    enrichment: {

      operationalStatus,

      linkedMission: linked,

      delayMinutes: Number.isFinite(delayMinutes) ? delayMinutes : null,

      vehicleType: mapMissionVehicleType(linked?.driver_type),

      licensePlate: null,

      currentAddress: linked ? conciseRouteSegment(linked.pickup_label) : null,

      destinationAddress: linked ? conciseRouteSegment(linked.dropoff_label) : null,

      etaLabel,

      distanceLabel: formatDistanceKm(linked?.route_distance_km),

      phone: null,

    },

  };

}



export function enrichFleetDrivers(

  drivers: CompanyDriverLiveLocation[],

  missions: CompanyDispatchMission[],

  organizationName?: string | null

): FleetDriverMapItem[] {
  const started = Date.now();
  const index = buildFleetMissionIndexMaps(missions);
  const enriched = drivers.map((d) => enrichFleetDriver(d, missions, organizationName, index));
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const { recordFleetEnrichDuration } = require("../../../../core/observability/perfInstrumentation") as {
      recordFleetEnrichDuration: (ms: number) => void;
    };
    recordFleetEnrichDuration(Date.now() - started);
  } catch {
    // optional perf instrumentation
  }
  return enriched;
}

function fleetDriverLiveSignature(driver: CompanyDriverLiveLocation): string {
  return [
    driver.driver_id,
    driver.latitude,
    driver.longitude,
    driver.timestamp ?? "",
    driver.mission_id ?? "",
    driver.location_status ?? "",
    driver.speed ?? "",
    driver.heading ?? "",
  ].join("|");
}

/** Mise à jour incrémentale : un seul chauffeur modifié → O(1) au lieu de O(N). */
export function patchEnrichedFleetDrivers(
  previous: FleetDriverMapItem[],
  drivers: CompanyDriverLiveLocation[],
  missions: CompanyDispatchMission[],
  organizationName?: string | null | undefined,
  index?: FleetMissionIndexMaps
): FleetDriverMapItem[] | null {
  if (previous.length !== drivers.length) return null;
  const missionIndex = index ?? buildFleetMissionIndexMaps(missions);
  let changedIndex = -1;
  for (let i = 0; i < drivers.length; i += 1) {
    if (previous[i]?.driver_id !== drivers[i]?.driver_id) return null;
    if (fleetDriverLiveSignature(previous[i]!) !== fleetDriverLiveSignature(drivers[i]!)) {
      if (changedIndex >= 0) return null;
      changedIndex = i;
    }
  }
  if (changedIndex < 0) return previous;
  const next = previous.slice();
  next[changedIndex] = enrichFleetDriver(
    drivers[changedIndex]!,
    missions,
    organizationName,
    missionIndex
  );
  return next;
}




export function countActiveFleetFilters(filters: FleetMapFiltersState): number {

  let n = 0;

  if (filters.status !== "all") n += 1;

  if (filters.vehicleType !== "all") n += 1;

  if (filters.driverId != null) n += 1;

  if (filters.driverSearch.trim().length > 0) n += 1;

  if (filters.withMissionOnly) n += 1;

  if (filters.withoutMissionOnly) n += 1;

  return n;

}



export function filterFleetDrivers(

  drivers: FleetDriverMapItem[],

  filters: FleetMapFiltersState

): FleetDriverMapItem[] {

  const search = filters.driverSearch.trim().toLowerCase();

  return drivers.filter((d) => {

    const { operationalStatus, linkedMission } = d.enrichment;

    if (filters.driverId != null && d.driver_id !== filters.driverId) return false;

    if (search.length > 0) {

      const name = resolveDriverDisplayName(d).toLowerCase();

      if (!name.includes(search) && !String(d.driver_id).includes(search)) return false;

    }

    if (filters.withMissionOnly && !linkedMission && d.mission_id == null) return false;

    if (filters.withoutMissionOnly && (linkedMission || d.mission_id != null)) return false;

    if (filters.vehicleType !== "all" && d.enrichment.vehicleType !== filters.vehicleType) {

      return false;

    }

    switch (filters.status) {

      case "all":

        return true;

      case "available":

        return operationalStatus === "available";

      case "busy":

        return operationalStatus === "busy";

      case "assigned":

        return operationalStatus === "assigned";

      case "break":

        return operationalStatus === "break";

      case "delayed":

        return operationalStatus === "delayed" || operationalStatus === "incident";

      case "urgent":

        return operationalStatus === "delayed" || operationalStatus === "incident";

      default:

        return true;

    }

  });

}



export function computeFleetClusterCentroid(drivers: FleetDriverMapItem[]): {
  latitude: number;
  longitude: number;
} {
  if (drivers.length === 0) {
    return { latitude: 0, longitude: 0 };
  }
  const latitude = drivers.reduce((sum, driver) => sum + driver.latitude, 0) / drivers.length;
  const longitude = drivers.reduce((sum, driver) => sum + driver.longitude, 0) / drivers.length;
  return { latitude, longitude };
}

export function clusterSharesDriverIds(
  drivers: FleetDriverMapItem[],
  focusDriverIds: ReadonlySet<number>
): boolean {
  return drivers.some((driver) => focusDriverIds.has(driver.driver_id));
}

export function clusterFleetMarkers(

  drivers: FleetDriverMapItem[],

  cellDeg = CLUSTER_CELL_DEG

): FleetMapMarker[] {

  countClusterFleetMarkers();

  if (drivers.length === 0) return [];

  const buckets = new Map<string, FleetDriverMapItem[]>();

  for (const d of drivers) {

    const key = `${Math.floor(d.latitude / cellDeg)}:${Math.floor(d.longitude / cellDeg)}`;

    const list = buckets.get(key) ?? [];

    list.push(d);

    buckets.set(key, list);

  }

  const markers: FleetMapMarker[] = [];

  for (const [clusterKey, list] of buckets) {

    if (list.length === 1) {

      markers.push({ kind: "driver", driver: list[0] });

      continue;

    }

    const latitude = list.reduce((s, x) => s + x.latitude, 0) / list.length;

    const longitude = list.reduce((s, x) => s + x.longitude, 0) / list.length;

    markers.push({

      kind: "cluster",

      clusterKey,

      latitude,

      longitude,

      count: list.length,

      drivers: list,

    });

  }

  return markers.sort((a, b) => {

    const prio = (m: FleetMapMarker) => {

      if (m.kind === "cluster") return 0;

      return m.driver.enrichment.operationalStatus === "incident" ||

        m.driver.enrichment.operationalStatus === "delayed"

        ? 2

        : 1;

    };

    return prio(b) - prio(a);

  });

}



/** Taille de cellule clustering selon le zoom (delta latitude région). */

export function resolveClusterCellDeg(latitudeDelta: number | undefined): number {

  const delta = latitudeDelta ?? 0.12;

  if (delta > 0.2) return 0.022;

  if (delta > 0.1) return 0.014;

  if (delta > 0.05) return 0.009;

  return CLUSTER_CELL_DEG;

}



export function findUrgentFleetDriver(drivers: FleetDriverMapItem[]): FleetDriverMapItem | null {

  const ranked = [...drivers].sort((a, b) => {

    const pa =

      a.enrichment.operationalStatus === "incident"

        ? 100

        : a.enrichment.operationalStatus === "delayed"

          ? 90

          : 0;

    const pb =

      b.enrichment.operationalStatus === "incident"

        ? 100

        : b.enrichment.operationalStatus === "delayed"

          ? 90

          : 0;

    return pb - pa;

  });

  const top = ranked[0];

  if (!top) return null;

  if (top.enrichment.operationalStatus === "incident" || top.enrichment.operationalStatus === "delayed") {

    return top;

  }

  return null;

}



export function buildFleetActiveRoute(driver: FleetDriverMapItem | null): FleetActiveRoute | null {

  const mission = driver?.enrichment.linkedMission;

  if (!driver || !mission) return null;



  const points: { latitude: number; longitude: number }[] = [];
  const pushPoint = (point: { latitude: number; longitude: number }) => {
    const previous = points[points.length - 1];
    if (!previous || previous.latitude !== point.latitude || previous.longitude !== point.longitude) {
      points.push(point);
    }
  };

  pushPoint({ latitude: driver.latitude, longitude: driver.longitude });

  const pickup =
    mission.pickup_lat != null && mission.pickup_lon != null
      ? { latitude: mission.pickup_lat, longitude: mission.pickup_lon }
      : null;
  const dropoff =
    mission.dropoff_lat != null && mission.dropoff_lon != null
      ? { latitude: mission.dropoff_lat, longitude: mission.dropoff_lon }
      : null;

  if (isMissionInProgress(mission.status)) {
    if (dropoff) pushPoint(dropoff);
  } else if (isMissionTransit(mission.status)) {
    if (pickup) pushPoint(pickup);
    if (dropoff) pushPoint(dropoff);
  } else {
    const dest = resolveMissionDestination(mission);
    if (dest) pushPoint(dest);
  }

  if (points.length < 2) {
    const dest = resolveMissionDestination(mission);
    if (dest) pushPoint(dest);
  }

  if (points.length < 2 && driver.heading != null && Number.isFinite(driver.heading)) {
    pushPoint(projectCoordinate(driver.latitude, driver.longitude, driver.heading, 0.5));
  }



  if (points.length < 2) return null;



  return {

    missionId: mission.mission_id,

    points,

    color: FLEET_MAP_COLORS.routeActive,

    etaLabel: driver.enrichment.etaLabel,

  };

}



/** Zoom doux sur un chauffeur sélectionné (cockpit). */

export function computeFleetFocusRegion(
  driver: CompanyDriverLiveLocation,
  options?: { verticalBias?: number }
) {
  const delta = 0.024;
  const bias = options?.verticalBias ?? 0;

  return {

    latitude: driver.latitude - delta * bias,

    longitude: driver.longitude,

    latitudeDelta: delta,

    longitudeDelta: delta,

  };

}



export function computeFleetRegion(

  drivers: CompanyDriverLiveLocation[],

  padding = 1.8

) {

  if (drivers.length === 0) {

    return {

      latitude: 46.2044,

      longitude: 6.1432,

      latitudeDelta: 0.12,

      longitudeDelta: 0.12,

    };

  }

  const latitudes = drivers.map((d) => d.latitude);

  const longitudes = drivers.map((d) => d.longitude);

  const minLat = Math.min(...latitudes);

  const maxLat = Math.max(...latitudes);

  const minLng = Math.min(...longitudes);

  const maxLng = Math.max(...longitudes);

  return {

    latitude: (minLat + maxLat) / 2,

    longitude: (minLng + maxLng) / 2,

    latitudeDelta: Math.max(0.025, (maxLat - minLat) * padding || 0.06),

    longitudeDelta: Math.max(0.025, (maxLng - minLng) * padding || 0.06),

  };

}



/** KPI overlay carte — chauffeurs déjà enrichis (évite double enrichissement). */

export function buildFleetLiveOverlayFromEnriched(

  enriched: FleetDriverMapItem[],

  missions: CompanyDispatchMission[]

): DashboardLiveOverlay {

  let activeDrivers = 0;

  let driverDelayed = 0;

  for (const d of enriched) {

    const st = d.enrichment.operationalStatus;

    if (st !== "offline") activeDrivers += 1;

    if (st === "delayed" || st === "incident") driverDelayed += 1;

  }



  let missionsInProgress = 0;

  let missionDelayed = 0;

  for (const m of missions) {

    if (m.status === "en_route" || m.status === "in_progress") missionsInProgress += 1;

    if (m.status !== "completed" && m.status !== "cancelled" && isMissionDelayed(m)) {

      missionDelayed += 1;

    }

  }



  return {

    activeDrivers,

    missionsInProgress,

    delayedMissions: Math.max(driverDelayed, missionDelayed),

  };

}



/** KPI overlay carte — même logique que le dashboard, source données réelles. */

export function buildFleetLiveOverlay(

  drivers: CompanyDriverLiveLocation[],

  missions: CompanyDispatchMission[]

): DashboardLiveOverlay {

  return buildFleetLiveOverlayFromEnriched(enrichFleetDrivers(drivers, missions), missions);

}



export function listFleetDriverOptions(drivers: FleetDriverMapItem[]): { id: number; label: string }[] {

  return drivers

    .map((d) => ({ id: d.driver_id, label: resolveDriverDisplayName(d) }))

    .sort((a, b) => a.label.localeCompare(b.label, "fr"));

}

/** Suggestions recherche carte (nom ou id chauffeur). */
export function matchFleetDriversByQuery(
  drivers: FleetDriverMapItem[],
  query: string,
  limit = 12
): FleetDriverMapItem[] {
  const search = query.trim().toLowerCase();
  if (!search) {
    return [...drivers]
      .sort((a, b) =>
        resolveDriverDisplayName(a).localeCompare(resolveDriverDisplayName(b), "fr")
      )
      .slice(0, limit);
  }
  return drivers
    .filter((d) => {
      const name = resolveDriverDisplayName(d).toLowerCase();
      return name.includes(search) || String(d.driver_id).includes(search);
    })
    .sort((a, b) =>
      resolveDriverDisplayName(a).localeCompare(resolveDriverDisplayName(b), "fr")
    )
    .slice(0, limit);
}


