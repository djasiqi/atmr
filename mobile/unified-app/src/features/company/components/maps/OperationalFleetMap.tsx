import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import {

  Modal,

  Platform,

  StyleSheet,

  View,

  type StyleProp,

  type ViewStyle,

} from "react-native";

import { SafeAreaView } from "react-native-safe-area-context";

import { AppText } from "../../../../design/ui/AppText";
import { useAppViewport } from "../../../../design/responsive/useAppViewport";

import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";
import { useCompanyRealtimeStatus } from "../../hooks";
import { EnterpriseDriversMap, type FleetMapCameraControl } from "../EnterpriseDriversMap";

import { MapFloatingControls } from "./MapFloatingControls";

import { MapLegend } from "./MapLegend";

import { MapEmptyState } from "./MapEmptyState";

import { DriverBottomSheet } from "./DriverBottomSheet";

import { MapDriverSearchSheet } from "./MapDriverSearchSheet";
import { MapFiltersSheet } from "./MapFiltersSheet";

import { MapLayersSheet } from "./MapLayersSheet";

import { ClusterDriversSheet } from "./ClusterDriversSheet";

import { useOperationalFleetMap } from "./useOperationalFleetMap";
import { useCompanyMapNativeOverlayGate } from "./companyMapNativeOverlayGate";
import { shouldEnableFleetClustering } from "./fleetMapOverlayMount";

import { FleetMapErrorBoundary } from "./FleetMapErrorBoundary";
import { ALL_LIRIE_DRIVER_MARKER_MODULES } from "./fleetLirieDriverMarkerModules";
import { prefetchLirieDriverMarkerAssets } from "./fleetLirieDriverMarkerAssets";

import { FLEET_MAP_COLORS } from "./mapStatusTheme";
import { computeFleetMapFitEdgePadding } from "./fleetMapFitPadding";
import type { CockpitMapPolicy, MapSignalsSnapshot } from "./fleetMapTypes";



export type OperationalFleetMapProps = {

  drivers: CompanyDriverLiveLocation[];

  missions?: CompanyDispatchMission[];

  /** Roster flotte résolu — gate badge N/T (évite 0/0 ou T partiel). */
  rosterResolved?: boolean;

  mapHeight?: number;

  /** `cockpit` : carte plein écran (canvas principal). `inline` : carte dans une page scrollable. */
  layout?: "inline" | "cockpit";

  cameraInsets?: { top: number; right: number; bottom: number; left: number };

  controlsTop?: number;

  driverSheetBottom?: number;

  containerStyle?: StyleProp<ViewStyle>;

  showLegend?: boolean;

  showControls?: boolean;

  enableFullscreen?: boolean;

  /** Cockpit : `false` = carte dans un cadre (ex. moitié écran), `true` = canvas plein écran. */
  cockpitImmersive?: boolean;

  cockpitExpanded?: boolean;

  onToggleCockpitExpand?: () => void;

  onDriverSheetChange?: (open: boolean) => void;

  onViewMission?: (missionId: number) => void;

  onMessage?: () => void;

  cameraVerticalBias?: number;

  simplifyClustering?: boolean;

  driverSheetSnap?: "collapsed" | "medium" | "expanded";

  onDriverSheetSnapChange?: (snap: "collapsed" | "medium" | "expanded") => void;

  onSelectedDriverIdChange?: (driverId: number | null) => void;

  autoFocusDriverOnMount?: boolean;

  /** Cockpit : trace la mission prioritaire au chargement sans ouvrir la sheet. */
  autoFocusMissionOnMount?: boolean;

  initialSelectedDriverId?: number | null;

  cockpitMapPolicy?: Partial<CockpitMapPolicy>;

  onMapSignalsChange?: (signals: MapSignalsSnapshot) => void;

  /** Synchronise la sélection chauffeur depuis le cockpit (désélection panneau ops). */
  syncSelectedDriverId?: number | null;

  /** Focus explicite depuis la feuille « Suivi en direct ». */
  focusDriverRequest?: { driverId: number; nonce: number } | null;

  /** iOS : diffère markers/routes tant que socket instable (reconnexion cockpit). */
  nativeOverlaysEnabled?: boolean;

  /** Cockpit : tableau « Prochaines courses » développé ou réduit. */
  upcomingTableExpanded?: boolean;

  onUpcomingTableExpandedChange?: (expanded: boolean) => void;

  /** false = cockpit monté hors focus : geler le travail visuel carte. */
  visualWorkEnabled?: boolean;

};



type FleetMapSurfaceProps = {

  mapHeight: number;

  isFullscreen: boolean;

  isCockpit: boolean;

  cockpitImmersive?: boolean;

  fleet: ReturnType<typeof useOperationalFleetMap>;

  showLegend: boolean;

  showControls: boolean;

  enableFullscreen: boolean;

  showEmpty: boolean;

  filteredOut: boolean;

  cameraInsets?: { top: number; right: number; bottom: number; left: number };

  cameraVerticalBias?: number;

  controlsTop?: number;

  driverSheetBottom?: number;

  containerStyle?: StyleProp<ViewStyle>;

  missions?: CompanyDispatchMission[];

  onViewMission?: (missionId: number) => void;

  onMessage?: () => void;

  onRecenter: () => void;

  onToggleFullscreen: () => void;

  driverSheetSnap?: "collapsed" | "medium" | "expanded";

  onDriverSheetSnapChange?: (snap: "collapsed" | "medium" | "expanded") => void;

  onSelectedDriverIdChange?: (driverId: number | null) => void;

  nativeOverlaysEnabled?: boolean;

  upcomingTableExpanded?: boolean;

  onUpcomingTableExpandedChange?: (expanded: boolean) => void;

  socketConnected?: boolean;

  cameraControlRef?: import("react").MutableRefObject<FleetMapCameraControl | null>;

  onUserCameraGesture?: () => void;

};



function FleetMapSurface({

  mapHeight,

  isFullscreen,

  isCockpit,

  cockpitImmersive = false,

  fleet,

  showLegend,

  showControls,

  enableFullscreen,

  showEmpty,

  filteredOut,

  cameraInsets,

  cameraVerticalBias = 0,

  controlsTop,

  driverSheetBottom,

  containerStyle,

  missions = [],

  onViewMission,

  onMessage,

  onRecenter,

  onToggleFullscreen,

  driverSheetSnap = "collapsed",

  onDriverSheetSnapChange,

  nativeOverlaysEnabled = true,

  upcomingTableExpanded = true,

  onUpcomingTableExpandedChange,

  socketConnected = true,

  cameraControlRef,

  onUserCameraGesture,

}: FleetMapSurfaceProps) {

  const fitEdgePadding = useMemo(() => {
    const defaultInsets = { top: 52, right: 52, bottom: 52, left: 52 };
    return computeFleetMapFitEdgePadding(cameraInsets ?? defaultInsets, {
      immersive: isCockpit && cockpitImmersive,
    });
  }, [cameraInsets, isCockpit, cockpitImmersive]);

  return (

    <View

      style={[

        s.root,

        !isCockpit && { height: mapHeight },

        (isFullscreen || (isCockpit && cockpitImmersive)) && s.rootFullscreen,

        isCockpit && cockpitImmersive && s.rootCockpit,

        isCockpit && !cockpitImmersive && s.rootCockpitSplit,

        containerStyle,

      ]}

      accessibilityLabel={isFullscreen ? "Carte flotte plein écran" : "Carte opérationnelle flotte"}

    >

      <FleetMapErrorBoundary>

      <EnterpriseDriversMap

        drivers={fleet.spatialDrivers}

        markers={fleet.markers}

        mapHeight={mapHeight}

        selectedDriverId={fleet.selectedDriverId}

        layers={fleet.layers}

        activeRoute={fleet.activeRoute}

        missionOverlays={fleet.missionOverlays}

        missionsById={fleet.missionsById}

        stableEtaAnchors={fleet.stableEtaAnchors}

        selectedMissionId={fleet.selectedMissionId}

        heatmapPoints={fleet.heatmapPoints}

        recenterRegion={fleet.recenterRegion}

        recenterMode={fleet.recenterMode}

        recenterToken={fleet.recenterToken}

        cameraInsets={cameraInsets}

        onDriverPress={fleet.onDriverMarkerPress}

        onClusterPress={fleet.onClusterPress}

        pinnedClusterFocus={fleet.pinnedClusterFocus}

        imminentDepartures={[
          ...fleet.imminentDepartures.individual,
          ...fleet.imminentDepartures.clustered,
        ]}

        globalVectorMode={fleet.cockpitMapPolicy.globalVectorMode}

        containerStyle={s.mapCard}

        logoClipFill={isCockpit && cockpitImmersive}

        fitEdgePadding={fitEdgePadding}

        nativeOverlaysEnabled={nativeOverlaysEnabled}

        onUserCameraGesture={onUserCameraGesture}

        cameraControlRef={cameraControlRef}

      />

      </FleetMapErrorBoundary>

      {!isCockpit ? (
        fleet.rosterResolved ? (
          <View
            style={s.liveBadge}
            accessibilityRole="text"
            accessibilityLabel={`${fleet.liveCount} sur ${fleet.totalCount} en direct`}
          >
            <View
              style={[
                s.liveBadgeDot,
                { backgroundColor: fleet.liveCount > 0 ? "#00796B" : "#91A3A0" },
              ]}
            />
            <AppText variant="caption" style={s.liveBadgeText}>
              {fleet.liveCount}/{fleet.totalCount} en direct
            </AppText>
            {fleet.filteredDisplayedCount !== fleet.totalCount ? (
              <AppText variant="caption" style={s.liveBadgeFilterHint}>
                · {fleet.filteredDisplayedCount} affiché
                {fleet.filteredDisplayedCount > 1 ? "s" : ""}
              </AppText>
            ) : null}
          </View>
        ) : (
          <View style={s.liveBadge} accessibilityLabel="Localisation en cours">
            <AppText variant="caption" style={s.liveBadgeText}>
              Localisation…
            </AppText>
          </View>
        )
      ) : null}

      {fleet.showNoGpsBanner && !isCockpit ? (
        <View style={[s.noGpsBanner, isCockpit && s.noGpsBannerCockpit]} pointerEvents="none">
          <AppText variant="label" style={s.noGpsTitle}>
            {socketConnected ? "Aucun GPS récent" : "Temps réel indisponible"}
          </AppText>
          <AppText variant="caption" style={s.noGpsDetail}>
            {socketConnected
              ? "Aucune position fraîche. Vérifiez l'app chauffeur et le GPS."
              : "Données issues du dernier chargement. Vérifiez la connexion réseau."}
          </AppText>
        </View>
      ) : null}

      {showLegend ? <MapLegend /> : null}



      {showControls ? (

        <MapFloatingControls

          activeFilterCount={fleet.activeFilterCount}

          searchActive={fleet.mapSignals.searchActive}

          onRecenter={onRecenter}

          onOpenSearch={() => {
            // La recherche doit partir d'une carte neutre (sans itinéraire sélectionné).
            fleet.selectDriver(null);
            fleet.setClusterPreview(null);
            fleet.setFiltersOpen(false);
            fleet.setLayersOpen(false);
            fleet.setSearchOpen(true);
          }}

          onOpenFilters={() => fleet.setFiltersOpen(true)}

          onOpenLayers={() => fleet.setLayersOpen(true)}

          stackTop={controlsTop}

          isFullscreen={isFullscreen}

          onToggleFullscreen={enableFullscreen ? onToggleFullscreen : undefined}

        />

      ) : null}



      {showEmpty ? <MapEmptyState filteredOut={filteredOut} /> : null}



      <DriverBottomSheet

        visible={isCockpit || fleet.driverSheetOpen || Boolean(fleet.peekDriver)}

        driver={fleet.selectedDriver}
        selectedMission={fleet.selectedMission}

        peekDriver={isCockpit && !fleet.selectedDriver ? fleet.peekDriver : null}
        upcomingMissions={isCockpit ? missions : undefined}
        availableDrivers={fleet.enriched}

        onClose={fleet.closeDriverSheet}
        onOpenFromPeek={fleet.openDriverSheetFromPeek}
        dismissible

        onViewMission={onViewMission}

        onMissionFocus={fleet.focusMission}

        onMessage={onMessage}

        presentation="overlay"

        inlineTableOnly={isCockpit}

        clusterDrivers={isCockpit ? fleet.clusterPreview : null}

        onCloseCluster={() => fleet.setClusterPreview(null)}

        onSelectClusterDriver={(d) => {
          const cluster = fleet.clusterPreview;
          if (cluster && cluster.length > 1) {
            fleet.selectDriverFromCluster(d, cluster);
            return;
          }
          fleet.setClusterPreview(null);
          fleet.onDriverMarkerPress(d);
        }}

        bottomOffset={driverSheetBottom}

        snapLevel={driverSheetSnap}

        onSnapChange={onDriverSheetSnapChange}

        onRecenter={() => fleet.recenter("selected")}

        upcomingTableExpanded={upcomingTableExpanded}

        onUpcomingTableExpandedChange={onUpcomingTableExpandedChange}

      />



      <MapDriverSearchSheet
        visible={fleet.searchOpen}
        drivers={fleet.enriched}
        query={fleet.filters.driverSearch}
        onChangeQuery={(driverSearch) => fleet.setFilters((prev) => ({ ...prev, driverSearch }))}
        onSelectDriver={(driver) => {
          fleet.setSearchOpen(false);
          fleet.onDriverMarkerPress(driver);
        }}
        onClose={() => fleet.setSearchOpen(false)}
      />

      <MapFiltersSheet

        visible={fleet.filtersOpen}

        filters={fleet.filters}

        drivers={fleet.enriched}

        onChange={fleet.setFilters}

        onClose={() => fleet.setFiltersOpen(false)}

      />



      <MapLayersSheet

        visible={fleet.layersOpen}

        layers={fleet.layers}

        onChange={fleet.setLayers}

        onClose={() => fleet.setLayersOpen(false)}

      />



      <ClusterDriversSheet

        visible={
          !isCockpit && fleet.clusterPreview != null && fleet.clusterPreview.length > 1
        }

        drivers={fleet.clusterPreview ?? []}

        onClose={() => fleet.setClusterPreview(null)}

        onSelectDriver={(d) => {
          const cluster = fleet.clusterPreview;
          if (cluster && cluster.length > 1) {
            fleet.selectDriverFromCluster(d, cluster);
            return;
          }
          fleet.setClusterPreview(null);
          fleet.onDriverMarkerPress(d);
        }}

      />

    </View>

  );

}



export function OperationalFleetMap({

  drivers,

  missions = [],

  rosterResolved = false,

  mapHeight = 248,

  layout = "inline",

  cameraInsets,

  controlsTop,

  driverSheetBottom,

  containerStyle,

  showLegend = true,

  showControls = true,

  enableFullscreen = true,

  cockpitImmersive = true,

  cockpitExpanded = false,

  onToggleCockpitExpand,

  onDriverSheetChange,

  onViewMission,

  onMessage,

  cameraVerticalBias = 0,

  simplifyClustering = false,

  driverSheetSnap = "collapsed",

  onDriverSheetSnapChange,
  onSelectedDriverIdChange,
  autoFocusDriverOnMount = false,
  autoFocusMissionOnMount = false,
  initialSelectedDriverId = null,
  cockpitMapPolicy,
  onMapSignalsChange,
  syncSelectedDriverId,
  focusDriverRequest = null,
  nativeOverlaysEnabled: nativeOverlaysEnabledProp,
  upcomingTableExpanded = true,
  onUpcomingTableExpandedChange,
  visualWorkEnabled = true,
}: OperationalFleetMapProps) {
  const realtime = useCompanyRealtimeStatus();
  const gatedNativeOverlays = useCompanyMapNativeOverlayGate(realtime.transportStatus);
  const nativeOverlaysEnabled = nativeOverlaysEnabledProp ?? gatedNativeOverlays;

  const isCockpit = layout === "cockpit";

  const cockpitExpandable = isCockpit && typeof onToggleCockpitExpand === "function";

  /** iOS : clustering off (évite ClusterMarker data-URI New Arch) ; sinon simplifyClustering. */
  const clusteringEnabled = shouldEnableFleetClustering(simplifyClustering);

  const fleetMapCameraControlRef = useRef<FleetMapCameraControl | null>(null);

  const fleet = useOperationalFleetMap({

    drivers,

    missions,

    rosterResolved,

    clusteringEnabled,

    driverFocusVerticalBias:
      isCockpit && cockpitImmersive ? Math.max(0.38, Math.min(0.42, cameraVerticalBias || 0.4)) : 0,

    onDriverSheetChange,
    onSelectedDriverIdChange,
    autoFocusDriverOnMount: isCockpit && autoFocusDriverOnMount,
    autoFocusMissionOnMount: isCockpit && autoFocusMissionOnMount,
    initialSelectedDriverId,
    cockpitMapPolicy,
    onMapSignalsChange,
    inlineDriverSelection: isCockpit,
    syncSelectedDriverId: isCockpit ? syncSelectedDriverId : undefined,
    focusDriverRequest: isCockpit ? focusDriverRequest : null,
    onBeforeExplicitRecenter: () => fleetMapCameraControlRef.current?.clearUserCameraLock(),
    visualWorkEnabled,

  });

  useEffect(() => {
    void prefetchLirieDriverMarkerAssets(ALL_LIRIE_DRIVER_MARKER_MODULES);
  }, []);

  const [fullscreen, setFullscreen] = useState(false);

  const { usableHeight } = useAppViewport();

  const showEmpty = fleet.filtered.length === 0;

  const filteredOut = showEmpty && drivers.length > 0;



  const fullscreenMapHeight = Math.max(320, usableHeight);



  const handleRecenter = useCallback(() => {

    fleet.recenter("all");

  }, [fleet]);



  const handleToggleFullscreen = useCallback(() => {

    setFullscreen((open) => !open);

  }, []);



  const closeFullscreen = useCallback(() => {

    setFullscreen(false);

  }, []);



  useEffect(() => {

    if (!fullscreen) return;

    const t = setTimeout(() => fleet.recenter("all"), 120);

    return () => clearTimeout(t);

  }, [fullscreen, fleet]);



  const surfaceProps: Omit<FleetMapSurfaceProps, "mapHeight" | "isFullscreen" | "isCockpit"> = {

    fleet,

    showLegend,

    showControls,

    enableFullscreen: cockpitExpandable ? true : isCockpit ? false : enableFullscreen,

    cockpitImmersive: isCockpit ? cockpitImmersive : false,

    showEmpty,

    filteredOut,

    isCockpit,

    cameraInsets,

    cameraVerticalBias,

    controlsTop,

    driverSheetBottom,

    containerStyle: fullscreen || isCockpit ? undefined : containerStyle,

    missions,

    onViewMission,

    onMessage,

    onRecenter: handleRecenter,

    onToggleFullscreen: cockpitExpandable
      ? () => onToggleCockpitExpand?.()
      : handleToggleFullscreen,

    driverSheetSnap,

    onDriverSheetSnapChange,
    onSelectedDriverIdChange,
    nativeOverlaysEnabled,

    upcomingTableExpanded,

    onUpcomingTableExpandedChange,

    socketConnected: realtime.transportStatus === "healthy",

    cameraControlRef: fleetMapCameraControlRef,

    onUserCameraGesture: fleet.notifyUserCameraGesture,

  };



  if (isCockpit) {

    return (

      <FleetMapSurface
        {...surfaceProps}
        mapHeight={mapHeight}
        isFullscreen={cockpitExpanded}
        isCockpit
      />

    );

  }



  let modalHost: ReactNode = null;

  if (fullscreen) {

    modalHost = (

      <Modal

        visible

        animationType="slide"

        presentationStyle="fullScreen"

        onRequestClose={closeFullscreen}

        statusBarTranslucent

      >

        <SafeAreaView style={s.modalSafe} edges={["top", "bottom", "left", "right"]}>

          <FleetMapSurface

            {...surfaceProps}

            mapHeight={fullscreenMapHeight}

            isFullscreen

            isCockpit={false}

            containerStyle={s.modalMap}

          />

        </SafeAreaView>

      </Modal>

    );

  }



  return (

    <>

      {fullscreen ? (

        <View

          style={[s.inlinePlaceholder, { height: mapHeight }, containerStyle]}

          accessibilityLabel="Carte en plein écran"

        />

      ) : (

        <FleetMapSurface {...surfaceProps} mapHeight={mapHeight} isFullscreen={false} isCockpit={false} />

      )}

      {modalHost}

    </>

  );

}



const s = StyleSheet.create({

  root: {

    overflow: "hidden",

    borderRadius: 22,

    backgroundColor: FLEET_MAP_COLORS.fabBg,

    position: "relative",

  },

  rootFullscreen: {

    flex: 1,

    width: "100%",

    borderRadius: 0,

    maxHeight: undefined,

  },

  rootCockpit: {

    ...StyleSheet.absoluteFillObject,

    zIndex: 0,

    borderRadius: 0,

    borderWidth: 0,

    backgroundColor: "transparent",

  },

  rootCockpitSplit: {

    position: "absolute",

    top: 0,

    left: 0,

    right: 0,

    bottom: 0,

    width: "100%",

    height: "100%",

    borderRadius: 22,

    overflow: "hidden",

    zIndex: 0,

    ...Platform.select({

      web: { isolation: "isolate" as const },

      default: {},

    }),

  },

  inlinePlaceholder: {

    borderRadius: 22,

    backgroundColor: FLEET_MAP_COLORS.fabBg,

    borderWidth: 1,

    borderColor: "rgba(145, 165, 157, 0.35)",

  },

  liveBadge: {
    position: "absolute",
    top: 10,
    left: 12,
    zIndex: 110,
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 6,
    backgroundColor: "rgba(255,255,255,0.94)",
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "#E2E8F0",
  },
  liveBadgeDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  liveBadgeText: {
    color: FLEET_MAP_COLORS.text,
    fontWeight: "600",
  },
  liveBadgeFilterHint: {
    color: FLEET_MAP_COLORS.textMuted,
  },

  noGpsBannerCockpit: {
    top: 118,
  },

  noGpsBanner: {

    position: "absolute",

    top: 44,

    alignSelf: "center",

    maxWidth: "92%",

    paddingHorizontal: 16,

    paddingVertical: 8,

    backgroundColor: "#FFFFFF",

    borderWidth: 2,

    borderColor: "#E2E8F0",

    borderRadius: 10,

    zIndex: 100,

    ...Platform.select({

      ios: { shadowColor: "#000", shadowOpacity: 0.08, shadowRadius: 3, shadowOffset: { width: 0, height: 1 } },

      android: { elevation: 2 },

      default: {},

    }),

  },

  noGpsTitle: {

    color: "#334155",

    textAlign: "center",

  },

  noGpsDetail: {

    color: "#64748B",

    textAlign: "center",

    marginTop: 4,

    lineHeight: 16,

  },

  modalSafe: {

    flex: 1,

    backgroundColor: "#FFFFFF",

  },

  modalMap: {

    flex: 1,

    height: undefined,

    maxHeight: undefined,

  },

  mapCard: {

    borderWidth: 0,

    borderRadius: 0,

    flex: 1,

    backgroundColor: "transparent",

    ...Platform.select({

      web: { boxShadow: "none" },

      default: { elevation: 0, shadowOpacity: 0 },

    }),

  },

});


