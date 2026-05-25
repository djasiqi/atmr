import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";
import type { FleetOperationalStatus } from "./mapStatusTheme";

export type FleetMapStatusFilter =
  | "all"
  | "available"
  | "on_mission"
  | "break"
  | "delayed"
  | "urgent";

export type FleetMapLayerType = "standard" | "satellite" | "terrain";
export type FleetMapTrafficEnabled = boolean;

export type FleetMapLayersState = {
  mapType: FleetMapLayerType;
  traffic: FleetMapTrafficEnabled;
  /** Réservé heatmap — non activé tant que les données ne sont pas exposées. */
  heatmapMode: "off" | "delays" | "demand";
  mission: FleetMapMissionLayers;
};

export const DEFAULT_FLEET_MAP_MISSION_LAYERS: FleetMapMissionLayers = {
  missionRoutes: true,
  compactRoutes: true,
  focusActive: true,
  autoSimplify: true,
};

export const DEFAULT_FLEET_MAP_LAYERS: FleetMapLayersState = {
  mapType: "standard",
  traffic: false,
  heatmapMode: "off",
  mission: DEFAULT_FLEET_MAP_MISSION_LAYERS,
};

export type FleetVehicleFilter = "all" | "berline" | "van" | "vsl" | "urgence";

export type FleetMapFiltersState = {
  status: FleetMapStatusFilter;
  vehicleType: FleetVehicleFilter;
  driverSearch: string;
  driverId: number | null;
  withMissionOnly: boolean;
  withoutMissionOnly: boolean;
};

export const DEFAULT_FLEET_MAP_FILTERS: FleetMapFiltersState = {
  status: "all",
  vehicleType: "all",
  driverSearch: "",
  driverId: null,
  withMissionOnly: false,
  withoutMissionOnly: false,
};

export type FleetDriverEnrichment = {
  operationalStatus: FleetOperationalStatus;
  linkedMission: CompanyDispatchMission | null;
  delayMinutes: number | null;
  vehicleType: string | null;
  licensePlate: string | null;
  currentAddress: string | null;
  destinationAddress: string | null;
  etaLabel: string | null;
  distanceLabel: string | null;
  phone: string | null;
};

export type FleetDriverMapItem = CompanyDriverLiveLocation & {
  enrichment: FleetDriverEnrichment;
};

export type FleetMapDriverMarker = {
  kind: "driver";
  driver: FleetDriverMapItem;
};

export type FleetMapClusterMarker = {
  kind: "cluster";
  clusterKey: string;
  latitude: number;
  longitude: number;
  count: number;
  drivers: FleetDriverMapItem[];
};

export type FleetMapMarker = FleetMapDriverMarker | FleetMapClusterMarker;

/** Pin cluster conservé en transparence après sélection d’un chauffeur dans le regroupement. */
export type FleetPinnedClusterFocus = {
  drivers: FleetDriverMapItem[];
  latitude: number;
  longitude: number;
  count: number;
};

/** Opacité des marqueurs non focalisés (chauffeur ou cluster). */
export const FLEET_MAP_MARKER_DIMMED_OPACITY = 0.45;

/** Segments de trajet actif (coords réelles uniquement — pas de géocodage mock). */
export type FleetActiveRoute = {
  missionId: number;
  points: { latitude: number; longitude: number }[];
  color: string;
  etaLabel: string | null;
};

export type {
  FleetMissionAnchorStyle,
  FleetMissionLifecyclePhase,
  FleetMissionOverlay,
  FleetMissionRouteStyle,
  FleetRouteEmphasisLevel,
} from "./fleetMapMissionVisual";

export type FleetMapRecenterMode = "all" | "selected" | "mission" | "urgent" | "user";

import type { CameraPolicy } from "../../dashboard/cockpit/cameraPolicyManager";

/** Politique carte pilotée par CockpitOrchestrator. */
export type CockpitMapPolicy = {
  maxVisibleRoutes: number;
  globalVectorMode: boolean;
  showImminentDepartures: boolean;
  showPassiveDrivers: boolean;
  showActiveRoute: boolean;
  routeFadeMs: number;
  allowDecorativeGlow: boolean;
  simplifyMarkers: boolean;
  cameraPolicy?: CameraPolicy;
};

export const DEFAULT_COCKPIT_MAP_POLICY: CockpitMapPolicy = {
  maxVisibleRoutes: 2,
  globalVectorMode: false,
  showImminentDepartures: true,
  showPassiveDrivers: true,
  showActiveRoute: true,
  routeFadeMs: 220,
  allowDecorativeGlow: true,
  simplifyMarkers: false,
};

export type MapSignalsSnapshot = {
  filtersOpen: boolean;
  layersOpen: boolean;
  searchActive: boolean;
  selectedDriverId: number | null;
  selectedMissionId: number | null;
};

export function areMapSignalsEqual(a: MapSignalsSnapshot, b: MapSignalsSnapshot): boolean {
  return (
    a.filtersOpen === b.filtersOpen &&
    a.layersOpen === b.layersOpen &&
    a.searchActive === b.searchActive &&
    a.selectedDriverId === b.selectedDriverId &&
    a.selectedMissionId === b.selectedMissionId
  );
}

/** Couches mission-first (densité / narration). */
export type FleetMapMissionLayers = {
  /** Afficher routes + ancres pickup/destination. */
  missionRoutes: boolean;
  /** Limiter aux missions prioritaires (anti-spaghetti). */
  compactRoutes: boolean;
  /** Mise en avant de la mission active (cohérence UX). */
  focusActive: boolean;
  /** Réduction automatique de densité sous forte charge. */
  autoSimplify: boolean;
};
