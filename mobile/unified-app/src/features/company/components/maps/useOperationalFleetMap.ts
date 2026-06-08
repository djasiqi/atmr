import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSession } from "../../../../core/sessionProvider";
import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";
import { buildFleetDelayHeatmapPoints } from "./fleetMapGeo";
import {
  buildFleetActiveRoute,
  clusterFleetMarkers,
  computeFleetFocusRegion,
  computeFleetRegion,
  countActiveFleetFilters,
  buildFleetMissionIndexMaps,
  computeFleetClusterCentroid,
  enrichFleetDrivers,
  patchEnrichedFleetDrivers,
  filterFleetDrivers,
  findUrgentFleetDriver,
  pickPrimaryFleetDriver,
  resolveClusterCellDeg,
} from "./fleetMapLogic";
import { FLEET_MISSION_MAP_POLICY } from "./fleetMapMissionPolicies";
import {
  buildFleetMissionOverlays,
  computeMissionOverlayFocusRegion,
  type FleetMissionOverlay,
} from "./fleetMapMissionVisual";
import {
  DEFAULT_COCKPIT_MAP_POLICY,
  DEFAULT_FLEET_MAP_FILTERS,
  DEFAULT_FLEET_MAP_LAYERS,
  type CockpitMapPolicy,
  type FleetActiveRoute,
  type FleetDriverMapItem,
  type FleetMapFiltersState,
  type FleetMapLayersState,
  type FleetMapRecenterMode,
  areMapSignalsEqual,
  type FleetPinnedClusterFocus,
  type MapSignalsSnapshot,
} from "./fleetMapTypes";
import type { CameraPolicy } from "../../dashboard/cockpit/cameraPolicyManager";
import { buildDriversBoundsSignature } from "./fleetMapFitPadding";
import type { ImminentDeparturesResult } from "../../dashboard/cockpit/imminentDepartures";
import { buildImminentDepartures } from "../../dashboard/cockpit/imminentDepartures";

export type FleetMapController = {
  recenter: (mode?: FleetMapRecenterMode) => void;
};

type Options = {
  drivers: CompanyDriverLiveLocation[];
  missions?: CompanyDispatchMission[];
  clusteringEnabled?: boolean;
  /** Décale le chauffeur vers le haut de l’écran quand la sheet masque le bas. */
  driverFocusVerticalBias?: number;
  onDriverSheetChange?: (open: boolean) => void;
  onSelectedDriverIdChange?: (driverId: number | null) => void;
  /** Cockpit : ouvre la sheet sur le chauffeur prioritaire au premier chargement. */
  autoFocusDriverOnMount?: boolean;
  /** Cockpit : sélectionne la mission prioritaire (route + ancres) sans ouvrir la sheet. */
  autoFocusMissionOnMount?: boolean;
  initialSelectedDriverId?: number | null;
  /** Cockpit immersif : garde toujours un chauffeur actif pour éviter une carte "vide". */
  alwaysSelectOperationalDriver?: boolean;
  cockpitMapPolicy?: Partial<CockpitMapPolicy>;
  onMapSignalsChange?: (signals: MapSignalsSnapshot) => void;
  /** Cockpit : sélection chauffeur sans sheet modale — tableau inline uniquement. */
  inlineDriverSelection?: boolean;
  /** Synchronise la sélection depuis le parent (ex. désélection via panneau ops). */
  syncSelectedDriverId?: number | null;
};

function computeMissionRouteFocusRegion(
  driver: FleetDriverMapItem,
  overlay: FleetMissionOverlay | null,
  verticalBias = 0
): { latitude: number; longitude: number; latitudeDelta: number; longitudeDelta: number } | null {
  if (overlay) {
    return computeMissionOverlayFocusRegion(overlay, verticalBias);
  }

  const mission = driver.enrichment.linkedMission;
  if (!mission) return null;

  const points: Array<{ latitude: number; longitude: number }> = [
    { latitude: driver.latitude, longitude: driver.longitude },
  ];

  if (mission.pickup_lat != null && mission.pickup_lon != null) {
    points.push({ latitude: mission.pickup_lat, longitude: mission.pickup_lon });
  }
  if (mission.dropoff_lat != null && mission.dropoff_lon != null) {
    points.push({ latitude: mission.dropoff_lat, longitude: mission.dropoff_lon });
  }
  if (points.length < 2) return null;

  const latitudes = points.map((point) => point.latitude);
  const longitudes = points.map((point) => point.longitude);
  const minLat = Math.min(...latitudes);
  const maxLat = Math.max(...latitudes);
  const minLng = Math.min(...longitudes);
  const maxLng = Math.max(...longitudes);
  const latitudeDelta = Math.max(0.02, (maxLat - minLat) * 1.65 || 0.03);
  const longitudeDelta = Math.max(0.02, (maxLng - minLng) * 1.65 || 0.03);

  return {
    latitude: (minLat + maxLat) / 2 - latitudeDelta * Math.max(0, verticalBias) * 0.2,
    longitude: (minLng + maxLng) / 2,
    latitudeDelta,
    longitudeDelta,
  };
}

export function useOperationalFleetMap({
  drivers,
  missions = [],
  clusteringEnabled = true,
  driverFocusVerticalBias = 0,
  onDriverSheetChange,
  onSelectedDriverIdChange,
  autoFocusDriverOnMount = false,
  autoFocusMissionOnMount = false,
  initialSelectedDriverId = null,
  alwaysSelectOperationalDriver = false,
  cockpitMapPolicy: cockpitMapPolicyPartial,
  onMapSignalsChange,
  inlineDriverSelection = false,
  syncSelectedDriverId,
}: Options) {
  const cockpitMapPolicy: CockpitMapPolicy = {
    ...DEFAULT_COCKPIT_MAP_POLICY,
    ...cockpitMapPolicyPartial,
  };
  const { activeContext } = useSession();
  const organizationName = activeContext?.organization_name ?? null;
  const [selectedDriverId, setSelectedDriverId] = useState<number | null>(null);

  useEffect(() => {
    if (syncSelectedDriverId === undefined) return;
    setSelectedDriverId(syncSelectedDriverId);
    if (syncSelectedDriverId == null) {
      setSheetDriver(null);
      setDriverSheetOpen(false);
      setPinnedClusterFocus(null);
      setSelectedMissionId(null);
    }
  }, [syncSelectedDriverId]);
  const [selectedMissionId, setSelectedMissionId] = useState<number | null>(null);
  const [sheetDriver, setSheetDriver] = useState<FleetDriverMapItem | null>(null);
  const [clusterPreview, setClusterPreview] = useState<FleetDriverMapItem[] | null>(null);
  const [pinnedClusterFocus, setPinnedClusterFocus] = useState<FleetPinnedClusterFocus | null>(null);
  const [filters, setFilters] = useState<FleetMapFiltersState>(DEFAULT_FLEET_MAP_FILTERS);
  const [layers, setLayers] = useState<FleetMapLayersState>(DEFAULT_FLEET_MAP_LAYERS);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [layersOpen, setLayersOpen] = useState(false);
  const [searchOpen, setSearchOpen] = useState(false);
  const [driverSheetOpen, setDriverSheetOpen] = useState(false);
  const recenterTokenRef = useRef(0);
  const [recenterToken, setRecenterToken] = useState(0);
  const [recenterMode, setRecenterMode] = useState<FleetMapRecenterMode>("all");
  const decayUnfocusedAtRef = useRef<Map<number, number>>(new Map());
  const previousMissionIdRef = useRef<number | null>(null);
  const [visualDecayTick, setVisualDecayTick] = useState(0);
  const etaAnchorMemoryRef = useRef<
    Map<number, { latitude: number; longitude: number; atMs: number }>
  >(new Map());

  useEffect(() => {
    const prev = previousMissionIdRef.current;
    if (prev != null && prev !== selectedMissionId) {
      decayUnfocusedAtRef.current.set(prev, Date.now());
    }
    previousMissionIdRef.current = selectedMissionId;
  }, [selectedMissionId]);

  useEffect(() => {
    const now = Date.now();
    const hasActiveDecay = [...decayUnfocusedAtRef.current.values()].some(
      (t) => now - t < FLEET_MISSION_MAP_POLICY.priorityDecayMs
    );
    if (!hasActiveDecay) return;
    const timer = setInterval(() => {
      setVisualDecayTick((n) => n + 1);
      const cutoff = Date.now() - FLEET_MISSION_MAP_POLICY.priorityDecayMs;
      for (const [id, t] of decayUnfocusedAtRef.current) {
        if (t < cutoff) decayUnfocusedAtRef.current.delete(id);
      }
    }, 90);
    return () => clearInterval(timer);
  }, [selectedMissionId, visualDecayTick]);

  const missionIndex = useMemo(() => buildFleetMissionIndexMaps(missions), [missions]);
  const enrichedCacheRef = useRef<{
    drivers: CompanyDriverLiveLocation[];
    missions: CompanyDispatchMission[];
    enriched: ReturnType<typeof enrichFleetDrivers>;
  } | null>(null);

  const enriched = useMemo(() => {
    const prev = enrichedCacheRef.current;
    if (prev && prev.missions === missions) {
      const patched = patchEnrichedFleetDrivers(
        prev.enriched,
        drivers,
        missions,
        organizationName,
        missionIndex
      );
      if (patched) {
        enrichedCacheRef.current = { drivers, missions, enriched: patched };
        return patched;
      }
    }
    const next = enrichFleetDrivers(drivers, missions, organizationName);
    enrichedCacheRef.current = { drivers, missions, enriched: next };
    return next;
  }, [drivers, missions, organizationName, missionIndex]);
  const filtered = useMemo(() => filterFleetDrivers(enriched, filters), [enriched, filters]);
  const activeFilterCount = useMemo(() => countActiveFleetFilters(filters), [filters]);

  const selectedDriver = useMemo(() => {
    if (sheetDriver && sheetDriver.driver_id === selectedDriverId) return sheetDriver;
    if (selectedDriverId == null) return null;
    return (
      enriched.find((d) => d.driver_id === selectedDriverId) ??
      filtered.find((d) => d.driver_id === selectedDriverId) ??
      null
    );
  }, [enriched, filtered, selectedDriverId, sheetDriver]);

  const primaryDriver = useMemo(() => pickPrimaryFleetDriver(enriched), [enriched]);

  const peekDriver = useMemo(() => selectedDriver ?? primaryDriver, [primaryDriver, selectedDriver]);

  const driversById = useMemo(() => {
    const map = new Map<number, FleetDriverMapItem>();
    for (const d of enriched) map.set(d.driver_id, d);
    return map;
  }, [enriched]);

  const missionsById = useMemo(() => {
    const map = new Map<number, CompanyDispatchMission>();
    for (const m of missions) map.set(m.mission_id, m);
    for (const d of enriched) {
      const linked = d.enrichment.linkedMission;
      if (linked) map.set(linked.mission_id, linked);
    }
    return map;
  }, [enriched, missions]);

  const routeCap = useMemo(() => {
    if (cockpitMapPolicy.globalVectorMode) return 0;
    return Math.min(
      cockpitMapPolicy.maxVisibleRoutes,
      layers.mission?.compactRoutes !== false
        ? FLEET_MISSION_MAP_POLICY.routeCapCompact
        : FLEET_MISSION_MAP_POLICY.routeCapExpanded
    );
  }, [cockpitMapPolicy.globalVectorMode, cockpitMapPolicy.maxVisibleRoutes, layers.mission?.compactRoutes]);

  const missionOverlays = useMemo(
    () =>
      buildFleetMissionOverlays({
        missions,
        driversById,
        selectedMissionId,
        maxVisible:
          layers.mission?.compactRoutes !== false
            ? FLEET_MISSION_MAP_POLICY.maxVisibleMissionsCompact
            : FLEET_MISSION_MAP_POLICY.maxVisibleMissionsExpanded,
        routeCap,
        decayUnfocusedAt: decayUnfocusedAtRef.current,
      }),
    [driversById, layers.mission?.compactRoutes, missions, routeCap, selectedMissionId, visualDecayTick]
  );

  const imminentDepartures: ImminentDeparturesResult = useMemo(
    () => (cockpitMapPolicy.showImminentDepartures ? buildImminentDepartures(missions) : { individual: [], clustered: [] }),
    [cockpitMapPolicy.showImminentDepartures, missions]
  );

  const stableEtaAnchors = useMemo(() => {
    const now = Date.now();
    for (const overlay of missionOverlays) {
      if (overlay.points.length === 0) continue;
      const mid = overlay.points[Math.floor(overlay.points.length / 2)] ?? overlay.points[0];
      const existing = etaAnchorMemoryRef.current.get(overlay.missionId);
      if (
        !existing ||
        now - existing.atMs > FLEET_MISSION_MAP_POLICY.overlayPositionHoldMs ||
        overlay.isSelected
      ) {
        etaAnchorMemoryRef.current.set(overlay.missionId, {
          latitude: mid.latitude,
          longitude: mid.longitude,
          atMs: now,
        });
      }
    }
    return etaAnchorMemoryRef.current;
  }, [missionOverlays]);

  const selectedMissionOverlay = useMemo(
    () => missionOverlays.find((o) => o.missionId === selectedMissionId) ?? null,
    [missionOverlays, selectedMissionId]
  );

  const selectedMission = useMemo(
    () => (selectedMissionId != null ? missions.find((m) => m.mission_id === selectedMissionId) ?? null : null),
    [missions, selectedMissionId]
  );

  const visibleMissionOverlays = useMemo(() => {
    if (cockpitMapPolicy.globalVectorMode) return [];
    if (layers.mission?.missionRoutes === false) return [];
    if (!cockpitMapPolicy.showActiveRoute) return [];

    if (selectedMissionId != null) {
      return missionOverlays.filter((overlay) => overlay.missionId === selectedMissionId);
    }
    const linkedMissionId = selectedDriver?.enrichment.linkedMission?.mission_id;
    if (linkedMissionId != null) {
      return missionOverlays.filter((overlay) => overlay.missionId === linkedMissionId);
    }
    return [];
  }, [
    cockpitMapPolicy.globalVectorMode,
    cockpitMapPolicy.showActiveRoute,
    layers.mission?.missionRoutes,
    missionOverlays,
    selectedDriver,
    selectedMissionId,
  ]);

  const activeRoute = useMemo((): FleetActiveRoute | null => {
    if (selectedMissionId != null && selectedMissionOverlay) {
      return {
        missionId: selectedMissionOverlay.missionId,
        points: selectedMissionOverlay.points,
        color: selectedMissionOverlay.routeStyle.color,
        etaLabel: selectedMissionOverlay.etaBadgeLabel ?? selectedMissionOverlay.etaLabel,
      };
    }
    if (selectedDriverId != null && selectedDriver) {
      return buildFleetActiveRoute(selectedDriver);
    }
    return null;
  }, [selectedDriver, selectedDriverId, selectedMissionId, selectedMissionOverlay]);

  const urgentDriver = useMemo(() => findUrgentFleetDriver(enriched), [enriched]);

  const resolveOperationalFallbackDriver = useCallback(() => {
    return pickPrimaryFleetDriver(enriched);
  }, [enriched]);

  const recenterRegion = useMemo(() => {
    void recenterToken;
    if ((recenterMode === "selected" || recenterMode === "mission") && selectedDriver) {
      const missionRouteRegion = computeMissionRouteFocusRegion(
        selectedDriver,
        selectedMissionOverlay,
        driverFocusVerticalBias
      );
      if (missionRouteRegion) return missionRouteRegion;
      return computeFleetFocusRegion(selectedDriver, {
        verticalBias: driverFocusVerticalBias,
      });
    }
    if (recenterMode === "mission" && selectedMissionOverlay && !selectedDriver) {
      return computeMissionOverlayFocusRegion(selectedMissionOverlay, driverFocusVerticalBias);
    }
    if (recenterMode === "urgent" && urgentDriver) {
      return computeFleetRegion([urgentDriver], 2.2);
    }
    return null;
  }, [
    driverFocusVerticalBias,
    recenterMode,
    recenterToken,
    selectedDriver,
    selectedMissionOverlay,
    urgentDriver,
  ]);

  const clusterCellDeg = useMemo(
    () => resolveClusterCellDeg(recenterRegion?.latitudeDelta),
    [recenterRegion?.latitudeDelta]
  );

  const markers = useMemo(() => {
    if (!clusteringEnabled || filtered.length <= 1) {
      return filtered.map((driver) => ({ kind: "driver" as const, driver }));
    }
    return clusterFleetMarkers(filtered, clusterCellDeg);
  }, [clusterCellDeg, clusteringEnabled, filtered]);

  const heatmapPoints = useMemo(() => {
    if (layers.heatmapMode !== "delays") return [];
    return buildFleetDelayHeatmapPoints(enriched);
  }, [enriched, layers.heatmapMode]);

  const onMapSignalsChangeRef = useRef(onMapSignalsChange);
  onMapSignalsChangeRef.current = onMapSignalsChange;
  const lastMapSignalsRef = useRef<MapSignalsSnapshot | null>(null);

  useEffect(() => {
    const snapshot: MapSignalsSnapshot = {
      filtersOpen,
      layersOpen,
      searchActive: searchOpen || filters.driverSearch.trim().length > 0,
      selectedDriverId,
      selectedMissionId,
    };
    const prev = lastMapSignalsRef.current;
    if (prev != null && areMapSignalsEqual(prev, snapshot)) return;
    lastMapSignalsRef.current = snapshot;
    onMapSignalsChangeRef.current?.(snapshot);
  }, [filters.driverSearch, filtersOpen, layersOpen, searchOpen, selectedDriverId, selectedMissionId]);

  const selectDriver = useCallback(
    (driverId: number | null) => {
      setSelectedDriverId(driverId);
      setClusterPreview(null);
      if (!inlineDriverSelection) {
        setDriverSheetOpen(driverId != null);
      } else {
        setDriverSheetOpen(false);
      }
      if (driverId == null) setSelectedMissionId(null);
    },
    [inlineDriverSelection]
  );

  const clearDriverMapFocus = useCallback(() => {
    setSelectedDriverId(null);
    setSelectedMissionId(null);
    setSheetDriver(null);
    setPinnedClusterFocus(null);
    setClusterPreview(null);
  }, []);

  const focusMission = useCallback(
    (missionId: number) => {
      const mission = missions.find((m) => m.mission_id === missionId);
      if (!mission) return;

      setSelectedMissionId(missionId);

      if (mission.driver_id != null) {
        const driver =
          enriched.find((d) => d.driver_id === mission.driver_id) ??
          filtered.find((d) => d.driver_id === mission.driver_id);
        if (driver) {
          setSheetDriver(driver);
          setSelectedDriverId(driver.driver_id);
          onSelectedDriverIdChange?.(driver.driver_id);
        }
      }

      setClusterPreview(null);
      if (!inlineDriverSelection) {
        setDriverSheetOpen(true);
        onDriverSheetChange?.(true);
      }
      setRecenterMode("mission");
      recenterTokenRef.current += 1;
      setRecenterToken(recenterTokenRef.current);
    },
    [enriched, filtered, inlineDriverSelection, missions, onDriverSheetChange, onSelectedDriverIdChange]
  );

  const onDriverMarkerPress = useCallback(
    (driver: FleetDriverMapItem) => {
      setClusterPreview(null);
      if (inlineDriverSelection) {
        const togglingOff = selectedDriverId === driver.driver_id;
        const nextId = togglingOff ? null : driver.driver_id;
        if (togglingOff) {
          clearDriverMapFocus();
        } else {
          setSheetDriver(driver);
          setSelectedDriverId(driver.driver_id);
          setSelectedMissionId(
            driver.enrichment.linkedMission?.mission_id ?? driver.mission_id ?? null
          );
          setPinnedClusterFocus(null);
        }
        setDriverSheetOpen(false);
        onSelectedDriverIdChange?.(nextId);
        if (nextId != null) {
          setRecenterMode("selected");
          recenterTokenRef.current += 1;
          setRecenterToken(recenterTokenRef.current);
        }
        return;
      }
      setPinnedClusterFocus(null);
      setSheetDriver(driver);
      setSelectedDriverId(driver.driver_id);
      setSelectedMissionId(driver.enrichment.linkedMission?.mission_id ?? driver.mission_id ?? null);
      setDriverSheetOpen(true);
      onDriverSheetChange?.(true);
      onSelectedDriverIdChange?.(driver.driver_id);
      setRecenterMode("mission");
      recenterTokenRef.current += 1;
      setRecenterToken(recenterTokenRef.current);
    },
    [clearDriverMapFocus, inlineDriverSelection, onDriverSheetChange, onSelectedDriverIdChange, selectedDriverId]
  );

  const onClusterPress = useCallback((clusterDrivers: FleetDriverMapItem[]) => {
    if (clusterDrivers.length === 1) {
      setPinnedClusterFocus(null);
      onDriverMarkerPress(clusterDrivers[0]);
      return;
    }
    setPinnedClusterFocus(null);
    setClusterPreview(clusterDrivers);
    setDriverSheetOpen(false);
    setSelectedDriverId(null);
    onSelectedDriverIdChange?.(null);
    setRecenterMode("all");
    recenterTokenRef.current += 1;
    setRecenterToken(recenterTokenRef.current);
  }, [onDriverMarkerPress]);

  const selectDriverFromCluster = useCallback(
    (driver: FleetDriverMapItem, clusterDrivers: FleetDriverMapItem[]) => {
      setClusterPreview(null);
      if (clusterDrivers.length > 1) {
        const { latitude, longitude } = computeFleetClusterCentroid(clusterDrivers);
        setPinnedClusterFocus({
          drivers: clusterDrivers,
          latitude,
          longitude,
          count: clusterDrivers.length,
        });
      } else {
        setPinnedClusterFocus(null);
      }
      setSheetDriver(driver);
      setSelectedDriverId(driver.driver_id);
      setSelectedMissionId(
        driver.enrichment.linkedMission?.mission_id ?? driver.mission_id ?? null
      );
      setDriverSheetOpen(false);
      onSelectedDriverIdChange?.(driver.driver_id);
      setRecenterMode("selected");
      recenterTokenRef.current += 1;
      setRecenterToken(recenterTokenRef.current);
    },
    [onSelectedDriverIdChange]
  );

  const recenter = useCallback((mode: FleetMapRecenterMode = "all") => {
    setRecenterMode(mode);
    recenterTokenRef.current += 1;
    setRecenterToken(recenterTokenRef.current);
  }, []);

  const lastCameraPolicyRef = useRef<CameraPolicy | undefined>(undefined);
  useEffect(() => {
    const policy = cockpitMapPolicy.cameraPolicy;
    if (!policy || policy === "user_gesture_preserve") return;
    if (lastCameraPolicyRef.current === policy) return;
    lastCameraPolicyRef.current = policy;

    if (policy === "global_soft_fit") {
      recenter("all");
      return;
    }
    if (policy === "driver_focus_fit") {
      recenter(selectedDriverId != null ? "selected" : "mission");
      return;
    }
    if (policy === "incident_soft_attention") {
      recenter(urgentDriver ? "urgent" : "all");
    }
  }, [cockpitMapPolicy.cameraPolicy, recenter, selectedDriverId, urgentDriver]);

  const didInitialRecenterRef = useRef(false);
  useEffect(() => {
    if (didInitialRecenterRef.current || enriched.length === 0) return;
    didInitialRecenterRef.current = true;
    setRecenterMode("all");
    recenterTokenRef.current += 1;
    setRecenterToken(recenterTokenRef.current);
  }, [enriched.length]);

  const driversBoundsSignature = useMemo(
    () =>
      buildDriversBoundsSignature(
        filtered.map((d) => ({
          driver_id: d.driver_id,
          latitude: d.latitude,
          longitude: d.longitude,
        }))
      ),
    [filtered]
  );
  const lastAutoFitBoundsRef = useRef("");
  useEffect(() => {
    if (filtered.length === 0) return;
    if (selectedDriverId != null) return;
    if (cockpitMapPolicy.cameraPolicy === "user_gesture_preserve") return;
    if (driversBoundsSignature === lastAutoFitBoundsRef.current) return;
    const isFirstFit = lastAutoFitBoundsRef.current === "";
    lastAutoFitBoundsRef.current = driversBoundsSignature;
    if (isFirstFit) return;
    setRecenterMode("all");
    recenterTokenRef.current += 1;
    setRecenterToken(recenterTokenRef.current);
  }, [
    cockpitMapPolicy.cameraPolicy,
    driversBoundsSignature,
    filtered.length,
    selectedDriverId,
  ]);

  const missionHasMapCoords = useCallback((mission: CompanyDispatchMission) => {
    const pickup =
      mission.pickup_lat != null &&
      mission.pickup_lon != null &&
      Number.isFinite(mission.pickup_lat) &&
      Number.isFinite(mission.pickup_lon);
    const dropoff =
      mission.dropoff_lat != null &&
      mission.dropoff_lon != null &&
      Number.isFinite(mission.dropoff_lat) &&
      Number.isFinite(mission.dropoff_lon);
    return pickup || dropoff;
  }, []);

  const didAutoFocusMissionRef = useRef(false);
  useEffect(() => {
    if (!autoFocusMissionOnMount || didAutoFocusMissionRef.current || enriched.length === 0) return;
    if (selectedMissionId != null) return;

    const fromSession =
      initialSelectedDriverId != null
        ? enriched.find((d) => d.driver_id === initialSelectedDriverId)
        : null;
    const target =
      fromSession ??
      findUrgentFleetDriver(enriched) ??
      enriched.find((d) => d.enrichment.linkedMission != null) ??
      null;
    const mission = target?.enrichment.linkedMission ?? null;
    if (!mission || !missionHasMapCoords(mission)) return;

    didAutoFocusMissionRef.current = true;
    setSelectedMissionId(mission.mission_id);
    if (target) {
      setSelectedDriverId(target.driver_id);
      onSelectedDriverIdChange?.(target.driver_id);
    }
    setRecenterMode("mission");
    recenterTokenRef.current += 1;
    setRecenterToken(recenterTokenRef.current);
  }, [
    autoFocusMissionOnMount,
    enriched,
    initialSelectedDriverId,
    missionHasMapCoords,
    onSelectedDriverIdChange,
    selectedMissionId,
  ]);

  const didAutoFocusDriverRef = useRef(false);
  useEffect(() => {
    if (!autoFocusDriverOnMount || didAutoFocusDriverRef.current || enriched.length === 0) return;
    didAutoFocusDriverRef.current = true;
    const fromSession =
      initialSelectedDriverId != null
        ? enriched.find((d) => d.driver_id === initialSelectedDriverId)
        : null;
    const target =
      fromSession ??
      findUrgentFleetDriver(enriched) ??
      enriched.find((d) => d.enrichment.linkedMission != null) ??
      enriched[0];
    if (target) onDriverMarkerPress(target);
  }, [autoFocusDriverOnMount, enriched, initialSelectedDriverId, onDriverMarkerPress]);

  useEffect(() => {
    if (!alwaysSelectOperationalDriver || enriched.length === 0) return;
    if (selectedDriverId != null && enriched.some((d) => d.driver_id === selectedDriverId)) return;
    const target = resolveOperationalFallbackDriver();
    if (!target) return;
    setSheetDriver(target);
    setSelectedDriverId(target.driver_id);
    if (!inlineDriverSelection) {
      setDriverSheetOpen(true);
      onDriverSheetChange?.(true);
    }
    onSelectedDriverIdChange?.(target.driver_id);
    setRecenterMode("selected");
    recenterTokenRef.current += 1;
    setRecenterToken(recenterTokenRef.current);
  }, [
    alwaysSelectOperationalDriver,
    inlineDriverSelection,
    enriched,
    onDriverSheetChange,
    onSelectedDriverIdChange,
    resolveOperationalFallbackDriver,
    selectedDriverId,
  ]);

  const closeDriverSheet = useCallback(() => {
    setDriverSheetOpen(false);
    clearDriverMapFocus();
    onDriverSheetChange?.(false);
    onSelectedDriverIdChange?.(null);
  }, [clearDriverMapFocus, onDriverSheetChange, onSelectedDriverIdChange]);

  const openDriverSheetFromPeek = useCallback(() => {
    const target = peekDriver;
    if (!target) return;
    setSheetDriver(target);
    setSelectedDriverId(target.driver_id);
    setClusterPreview(null);
    if (!inlineDriverSelection) {
      setDriverSheetOpen(true);
      onDriverSheetChange?.(true);
    }
    onSelectedDriverIdChange?.(target.driver_id);
    setRecenterMode("selected");
    recenterTokenRef.current += 1;
    setRecenterToken(recenterTokenRef.current);
  }, [inlineDriverSelection, onDriverSheetChange, onSelectedDriverIdChange, peekDriver]);

  const applySearchContext = useCallback(
    (query: string, result?: { type: "driver"; driverId: number } | { type: "client"; missionId: number }) => {
      setFilters((prev) => ({ ...prev, driverSearch: query }));
      if (!result) return;
      if (result.type === "driver") {
        const driver = enriched.find((d) => d.driver_id === result.driverId);
        if (driver) onDriverMarkerPress(driver);
        return;
      }
      focusMission(result.missionId);
    },
    [enriched, focusMission, onDriverMarkerPress]
  );

  return {
    enriched,
    filtered,
    markers,
    heatmapPoints,
    selectedDriver,
    primaryDriver,
    peekDriver,
    selectedDriverId,
    selectedMissionId,
    selectedMission,
    missionOverlays: visibleMissionOverlays,
    missionsById,
    stableEtaAnchors,
    selectedMissionOverlay,
    focusMission,
    activeRoute,
    urgentDriver,
    filters,
    setFilters,
    layers,
    setLayers,
    activeFilterCount,
    filtersOpen,
    setFiltersOpen,
    layersOpen,
    setLayersOpen,
    searchOpen,
    setSearchOpen,
    driverSheetOpen,
    clusterPreview,
    setClusterPreview,
    pinnedClusterFocus,
    selectDriverFromCluster,
    selectDriver,
    onDriverMarkerPress,
    onClusterPress,
    recenter,
    recenterRegion,
    recenterToken,
    recenterMode,
    closeDriverSheet,
    openDriverSheetFromPeek,
    driverOptions: enriched,
    cockpitMapPolicy,
    imminentDepartures,
    applySearchContext,
    mapSignals: {
      filtersOpen,
      layersOpen,
      searchActive: searchOpen || filters.driverSearch.trim().length > 0,
      selectedDriverId,
      selectedMissionId,
    } satisfies MapSignalsSnapshot,
  };
}
