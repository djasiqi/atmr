import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Platform, StyleSheet, View } from "react-native";
import { useIsFocused } from "@react-navigation/native";
import { useAppViewport } from "../../../../design/responsive/useAppViewport";
import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";
import {
  computeCockpitOperationalCounts,
  orchestrationToMapPolicy,
  useCockpitRuntime,
  type CockpitOrchestrationDecision,
} from "../../dashboard/cockpit";
import {
  resolveCockpitVisualWork,
  shouldFreezeCockpitMapData,
} from "../../dashboard/cockpitVisualWork";
import { areMapSignalsEqual, type MapSignalsSnapshot } from "../maps/fleetMapTypes";
import { EnterpriseHeader } from "../EnterpriseHeader";
import { useCompanyMapNativeOverlayGate } from "../maps/companyMapNativeOverlayGate";
import { OperationalFleetMap } from "../maps/OperationalFleetMap";
import { fleetGlassPanel } from "../maps/fleetMapUiTokens";
import {
  computeDynamicCameraInsets,
  computeFleetCockpitLayout,
} from "./companyFleetCockpitLayout";
import { CockpitConnectivityBanner } from "./CockpitConnectivityBanner";
import { CockpitLiveCoverageIsland } from "./CockpitLiveCoverageIsland";
import { CockpitStateDebugger } from "./CockpitStateDebugger";
import { CockpitTrustBanner } from "./CockpitTrustBanner";
import { resolveCockpitConnectivityBanner } from "./resolveCockpitBanner";
import { DashboardFloatingStatusBar } from "./DashboardFloatingStatusBar";
import { computeGpsCoverageCounts } from "./liveGpsCoverage";
import type { DashboardOpsFeedProps } from "./DashboardOpsFeed";
import { DashboardOpsSheet } from "./DashboardOpsSheet";
import { DelayedMissionCard } from "./DelayedMissionCard";
import { E } from "../../theme/enterpriseOpsTheme";
import {
  beginTapFeedback,
  endTapLocal,
  recordScreenRender,
} from "../../../../core/observability/perfResponsiveness";
import { QuickActionsBar } from "./QuickActionsBar";
import { UrgencyBottomSheet } from "./UrgencyBottomSheet";

export type CompanyFleetCockpitProps = {
  drivers: CompanyDriverLiveLocation[];
  missions: CompanyDispatchMission[];
  /** Snapshot flotte résolu — gate badge N/T. */
  rosterResolved?: boolean;
  date: string;
  headerMode: string | null;
  /** Transport Socket.IO (`healthy`, `reconnecting`, …). */
  realtimeStatus: string;
  realtimeDataFreshness?: string;
  realtimeLastEventAt?: string | null;
  refreshing?: boolean;
  onRefresh?: () => void;
  onOpenDatePicker: () => void;
  onViewMission?: (missionId: number) => void;
  onMessage?: () => void;
  opsFeed: DashboardOpsFeedProps;
};

function CockpitMapBlockInner({
  layout,
  cameraInsets,
  cameraVerticalBias,
  drivers,
  missions,
  rosterResolved = false,
  simplifyClustering,
  onDriverSheetChange,
  onViewMission,
  onMessage,
  driverSheetSnap,
  onDriverSheetSnapChange,
  onSelectedDriverIdChange,
  autoFocusDriverOnMount,
  autoFocusMissionOnMount = true,
  initialSelectedDriverId,
  cockpitMapPolicy,
  onMapSignalsChange,
  syncSelectedDriverId,
  focusDriverRequest = null,
  nativeOverlaysEnabled,
  upcomingTableExpanded,
  onUpcomingTableExpandedChange,
  visualWorkEnabled,
  suppressConnectivityBanner = false,
}: {
  layout: ReturnType<typeof computeFleetCockpitLayout>;
  cameraInsets: { top: number; right: number; bottom: number; left: number };
  cameraVerticalBias: number;
  drivers: CompanyDriverLiveLocation[];
  missions: CompanyDispatchMission[];
  rosterResolved?: boolean;
  simplifyClustering?: boolean;
  onDriverSheetChange: (open: boolean) => void;
  onViewMission?: (missionId: number) => void;
  onMessage?: () => void;
  driverSheetSnap: "collapsed" | "medium" | "expanded";
  onDriverSheetSnapChange: (snap: "collapsed" | "medium" | "expanded") => void;
  onSelectedDriverIdChange: (driverId: number | null) => void;
  autoFocusDriverOnMount: boolean;
  autoFocusMissionOnMount?: boolean;
  initialSelectedDriverId: number | null;
  cockpitMapPolicy: CockpitOrchestrationDecision;
  onMapSignalsChange: (signals: MapSignalsSnapshot) => void;
  syncSelectedDriverId?: number | null;
  focusDriverRequest?: { driverId: number; nonce: number } | null;
  nativeOverlaysEnabled: boolean;
  upcomingTableExpanded: boolean;
  onUpcomingTableExpandedChange: (expanded: boolean) => void;
  visualWorkEnabled: boolean;
}) {
  const mapPolicy = orchestrationToMapPolicy(cockpitMapPolicy);
  return (
    <View style={[s.mapZone, s.mapZoneImmersive]}>
      <OperationalFleetMap
        layout="cockpit"
        cockpitImmersive
        cockpitExpanded
        visualWorkEnabled={visualWorkEnabled}
        drivers={drivers}
        missions={missions}
        rosterResolved={rosterResolved}
        mapHeight={layout.mapHeight}
        cameraInsets={cameraInsets}
        cameraVerticalBias={cameraVerticalBias}
        controlsTop={layout.controlsTop}
        driverSheetBottom={layout.driverSheetBottom}
        simplifyClustering={simplifyClustering}
        showLegend={false}
        enableFullscreen={false}
        onDriverSheetChange={onDriverSheetChange}
        onViewMission={onViewMission}
        onMessage={onMessage}
        driverSheetSnap={driverSheetSnap}
        onDriverSheetSnapChange={onDriverSheetSnapChange}
        onSelectedDriverIdChange={onSelectedDriverIdChange}
        autoFocusDriverOnMount={autoFocusDriverOnMount}
        autoFocusMissionOnMount={autoFocusMissionOnMount}
        initialSelectedDriverId={initialSelectedDriverId}
        cockpitMapPolicy={mapPolicy}
        onMapSignalsChange={onMapSignalsChange}
        syncSelectedDriverId={syncSelectedDriverId}
        focusDriverRequest={focusDriverRequest}
        nativeOverlaysEnabled={nativeOverlaysEnabled}
        upcomingTableExpanded={upcomingTableExpanded}
        onUpcomingTableExpandedChange={onUpcomingTableExpandedChange}
      />
    </View>
  );
}

type CockpitMapBlockProps = Parameters<typeof CockpitMapBlockInner>[0];

function areCockpitMapBlockPropsEqual(
  prev: CockpitMapBlockProps,
  next: CockpitMapBlockProps
): boolean {
  const freezeData = shouldFreezeCockpitMapData(
    prev.visualWorkEnabled,
    next.visualWorkEnabled
  );
  return (
    (freezeData || (prev.drivers === next.drivers && prev.missions === next.missions)) &&
    prev.visualWorkEnabled === next.visualWorkEnabled &&
    prev.layout === next.layout &&
    prev.cameraInsets === next.cameraInsets &&
    prev.cameraVerticalBias === next.cameraVerticalBias &&
    prev.rosterResolved === next.rosterResolved &&
    prev.simplifyClustering === next.simplifyClustering &&
    prev.onDriverSheetChange === next.onDriverSheetChange &&
    prev.onViewMission === next.onViewMission &&
    prev.onMessage === next.onMessage &&
    prev.driverSheetSnap === next.driverSheetSnap &&
    prev.onDriverSheetSnapChange === next.onDriverSheetSnapChange &&
    prev.onSelectedDriverIdChange === next.onSelectedDriverIdChange &&
    prev.autoFocusDriverOnMount === next.autoFocusDriverOnMount &&
    prev.autoFocusMissionOnMount === next.autoFocusMissionOnMount &&
    prev.initialSelectedDriverId === next.initialSelectedDriverId &&
    prev.cockpitMapPolicy === next.cockpitMapPolicy &&
    prev.onMapSignalsChange === next.onMapSignalsChange &&
    prev.syncSelectedDriverId === next.syncSelectedDriverId &&
    prev.focusDriverRequest === next.focusDriverRequest &&
    prev.nativeOverlaysEnabled === next.nativeOverlaysEnabled &&
    prev.upcomingTableExpanded === next.upcomingTableExpanded &&
    prev.onUpcomingTableExpandedChange === next.onUpcomingTableExpandedChange
  );
}

const CockpitMapBlock = memo(CockpitMapBlockInner, areCockpitMapBlockPropsEqual);

export function CompanyFleetCockpit({
  drivers,
  missions,
  rosterResolved = false,
  date,
  headerMode,
  realtimeStatus,
  realtimeDataFreshness,
  realtimeLastEventAt,
  refreshing = false,
  onRefresh,
  onOpenDatePicker,
  onViewMission,
  onMessage,
  opsFeed,
}: CompanyFleetCockpitProps) {
  const isScreenFocused = useIsFocused();
  const { visualWorkEnabled, shouldRecordScreenRender } =
    resolveCockpitVisualWork(isScreenFocused);
  if (shouldRecordScreenRender) {
    recordScreenRender("company.dashboard");
  }
  void refreshing;
  void onRefresh;

  const { topInset, bottomInset, usableHeight, safeLeft, safeRight } = useAppViewport();
  const nativeOverlaysEnabled = useCompanyMapNativeOverlayGate(realtimeStatus);
  const [opsSheetOpen, setOpsSheetOpen] = useState(false);
  const [focusDriverRequest, setFocusDriverRequest] = useState<{
    driverId: number;
    nonce: number;
  } | null>(null);
  const [urgencySheetOpen, setUrgencySheetOpen] = useState(false);
  const [driverSheetOpen, setDriverSheetOpen] = useState(false);
  const [upcomingTableExpanded, setUpcomingTableExpanded] = useState(true);
  const [selectedDriverId, setSelectedDriverId] = useState<number | null>(null);
  const [chromeStackHeight, setChromeStackHeight] = useState(0);
  const [mapSignals, setMapSignals] = useState<MapSignalsSnapshot>({
    filtersOpen: false,
    layersOpen: false,
    searchActive: false,
    selectedDriverId: null,
    selectedMissionId: null,
  });

  const parseStatValue = useCallback((key: (typeof opsFeed.stats)[0]["key"]) => {
    const raw = opsFeed.stats.find((s) => s.key === key)?.value ?? "0";
    const n = Number.parseInt(String(raw).replace(/\D/g, ""), 10);
    return Number.isFinite(n) ? n : 0;
  }, [opsFeed]);

  const operationalCounts = useMemo(
    () =>
      computeCockpitOperationalCounts({
        missions,
        drivers,
        opsFeed,
      }),
    [missions, drivers, opsFeed]
  );

  const { delayedCount, urgentCount, criticalEtaCount } = operationalCounts;
  const inProgress = parseStatValue("in_progress");

  const unassignedCount = missions.filter((m) => m.driver_id == null && m.status !== "completed" && m.status !== "cancelled").length;

  const cockpit = useCockpitRuntime({
    realtimeStatus,
    realtimeDataFreshness,
    realtimeLastEventAt,
    urgentCount,
    delayedCount,
    criticalEtaCount,
    activeDriverCount: drivers.length,
    inProgressMissionCount: inProgress,
    driverSheetOpen,
    opsSheetOpen,
    filtersOpen: mapSignals.filtersOpen,
    layersOpen: mapSignals.layersOpen,
    searchActive: mapSignals.searchActive,
    selectedDriverId: mapSignals.selectedDriverId ?? selectedDriverId,
    selectedMissionId: mapSignals.selectedMissionId ?? opsFeed.delayedMission?.missionId ?? null,
    unassignedCount,
    missions,
  });

  const { uiState, orchestration } = cockpit;

  const sessionDriverAppliedRef = useRef(false);
  useEffect(() => {
    if (!cockpit.sessionHydrated || cockpit.initialSelectedDriverId == null) return;
    if (sessionDriverAppliedRef.current) return;
    sessionDriverAppliedRef.current = true;
    setSelectedDriverId(cockpit.initialSelectedDriverId);
  }, [cockpit.sessionHydrated, cockpit.initialSelectedDriverId]);

  const dispatchRef = useRef(cockpit.dispatch);
  dispatchRef.current = cockpit.dispatch;
  const enqueueFrameEventRef = useRef(cockpit.enqueueFrameEvent);
  enqueueFrameEventRef.current = cockpit.enqueueFrameEvent;
  const mapSignalsEdgeRef = useRef({
    searchActive: false,
    filtersOpen: false,
    layersOpen: false,
    selectedDriverId: null as number | null,
  });

  const onMapSignalsChange = useCallback((signals: MapSignalsSnapshot) => {
    setMapSignals((prev) => (areMapSignalsEqual(prev, signals) ? prev : signals));

    const edge = mapSignalsEdgeRef.current;
    if (signals.selectedDriverId !== edge.selectedDriverId) {
      if (signals.selectedDriverId != null) {
        setSelectedDriverId(signals.selectedDriverId);
        dispatchRef.current({ type: "DRIVER_SELECT" });
      } else if (edge.selectedDriverId != null) {
        dispatchRef.current({ type: "DRIVER_CLEAR" });
      }
      edge.selectedDriverId = signals.selectedDriverId;
    }

    if (signals.searchActive && !edge.searchActive) {
      enqueueFrameEventRef.current({
        type: "SEARCH_OPENED",
        atMs: Date.now(),
        source: "map",
        coalescable: false,
      });
    }
    edge.searchActive = signals.searchActive;

    if (signals.filtersOpen && !edge.filtersOpen) {
      enqueueFrameEventRef.current({
        type: "FILTERS_OPENED",
        atMs: Date.now(),
        source: "map",
        coalescable: false,
      });
    }
    edge.filtersOpen = signals.filtersOpen;
    edge.layersOpen = signals.layersOpen;
  }, []);

  const layout = useMemo(
    () => computeFleetCockpitLayout(usableHeight, topInset, bottomInset, true),
    [usableHeight, topInset, bottomInset]
  );

  const chromeAwareLayout = useMemo(() => {
    if (chromeStackHeight <= 0) return layout;
    const stackedTop = layout.topBarTop + chromeStackHeight;
    return {
      ...layout,
      topBarHeight: stackedTop,
      controlsTop: Math.max(layout.controlsTop, stackedTop + 12),
    };
  }, [layout, chromeStackHeight]);

  const cameraInsets = useMemo(
    () =>
      computeDynamicCameraInsets(chromeAwareLayout, {
        driverSheetOpen: false,
        cockpitOpsPanel: upcomingTableExpanded,
        safeRight,
        safeLeft,
      }),
    [chromeAwareLayout, upcomingTableExpanded, safeRight, safeLeft]
  );

  const onDriverSheetChange = useCallback(
    (open: boolean) => {
      setDriverSheetOpen(open);
      if (!open) {
        setSelectedDriverId(null);
        cockpit.dispatch({ type: "DRIVER_CLEAR" });
      }
      cockpit.touchInteraction();
    },
    [cockpit]
  );

  const onSelectedDriverIdChange = useCallback(
    (driverId: number | null) => {
      setSelectedDriverId(driverId);
      if (driverId != null) {
        cockpit.dispatch({ type: "DRIVER_SELECT" });
      } else {
        cockpit.dispatch({ type: "DRIVER_CLEAR" });
      }
    },
    [cockpit]
  );

  const onPressStat = useCallback(
    (key: (typeof opsFeed.stats)[0]["key"]) => {
      const tapId = beginTapFeedback("cockpit.stat", "company.dashboard");
      cockpit.touchInteraction();
      opsFeed.onPressStat?.(key);
      endTapLocal(tapId);
    },
    [cockpit, opsFeed]
  );

  const onPressDelayed = useCallback(() => {
    cockpit.touchInteraction();
    if (opsFeed.onPressDelayed) {
      opsFeed.onPressDelayed();
      return;
    }
    setUrgencySheetOpen(true);
  }, [cockpit, opsFeed]);

  const onLivePillOpen = useCallback(() => {
    const tapId = beginTapFeedback("cockpit.live", "company.dashboard");
    cockpit.touchInteraction();
    endTapLocal(tapId);
  }, [cockpit]);

  const handleOpenDatePicker = useCallback(() => {
    const tapId = beginTapFeedback("cockpit.date", "company.dashboard");
    onOpenDatePicker();
    endTapLocal(tapId);
  }, [onOpenDatePicker]);

  const handleMessage = useCallback(() => {
    const tapId = beginTapFeedback("cockpit.inbox", "company.dashboard");
    onMessage?.();
    endTapLocal(tapId);
  }, [onMessage]);

  const onSelectLiveCoverageDriver = useCallback(
    (driverId: number) => {
      setSelectedDriverId(driverId);
      setFocusDriverRequest({ driverId, nonce: Date.now() });
      cockpit.dispatch({ type: "DRIVER_SELECT" });
    },
    [cockpit]
  );

  const liveCoverageDataFreshness =
    uiState.liveStatus === "connected" &&
    (realtimeDataFreshness === "fresh" ||
      realtimeDataFreshness === "idle" ||
      realtimeDataFreshness === "stale")
      ? realtimeDataFreshness
      : undefined;

  const liveCoverageIsland = uiState.overlays.livePill ? (
    <CockpitLiveCoverageIsland
      drivers={drivers}
      rosterResolved={rosterResolved}
      visualWorkEnabled={visualWorkEnabled}
      variant={uiState.overlays.topBar ? "header" : "default"}
      status={uiState.liveStatus}
      realtimeSocketExpected={uiState.realtimeSocketExpected}
      activityHint={uiState.liveActivityHint}
      dataFreshness={liveCoverageDataFreshness}
      animationIntensity={uiState.animationIntensity}
      onOpen={onLivePillOpen}
      onSelectDriver={onSelectLiveCoverageDriver}
      bottomOffset={layout.livePillBottom}
    />
  ) : null;

  const showOverlays = uiState.overlays.statusPills;
  const horizontalOverlayInset = Math.max(16, Math.max(safeLeft, safeRight) + 10);
  const gpsCoverage = useMemo(
    () => computeGpsCoverageCounts(drivers),
    [drivers]
  );
  const showNoGpsBanner =
    rosterResolved && gpsCoverage.totalCount > 0 && gpsCoverage.liveCount === 0;
  const connectivityCopy = resolveCockpitConnectivityBanner({
    showNoGps: showNoGpsBanner,
    socketConnected: uiState.liveStatus === "connected",
    realtimeOffline: uiState.liveStatus === "offline",
  });
  const governanceTrust =
    uiState.trustMessage && uiState.trustMessage !== "Temps réel indisponible"
      ? uiState.trustMessage
      : null;

  return (
    <View style={s.root} accessibilityLabel="Cockpit dispatch live">
      <CockpitMapBlock
        layout={chromeAwareLayout}
        cameraInsets={cameraInsets}
        cameraVerticalBias={uiState.cameraVerticalBias}
        drivers={drivers}
        missions={missions}
        rosterResolved={rosterResolved}
        simplifyClustering={uiState.simplifyClustering}
        onDriverSheetChange={onDriverSheetChange}
        onViewMission={onViewMission}
        onMessage={handleMessage}
        driverSheetSnap={cockpit.driverSheetSnap}
        onDriverSheetSnapChange={cockpit.setDriverSheetSnap}
        onSelectedDriverIdChange={onSelectedDriverIdChange}
        autoFocusDriverOnMount={false}
        autoFocusMissionOnMount={false}
        initialSelectedDriverId={cockpit.initialSelectedDriverId}
        cockpitMapPolicy={orchestration}
        onMapSignalsChange={onMapSignalsChange}
        syncSelectedDriverId={selectedDriverId}
        focusDriverRequest={focusDriverRequest}
        nativeOverlaysEnabled={nativeOverlaysEnabled}
        upcomingTableExpanded={upcomingTableExpanded}
        onUpcomingTableExpandedChange={setUpcomingTableExpanded}
        visualWorkEnabled={visualWorkEnabled}
      />

      {uiState.overlays.topBar ? (
        <View
          style={[s.chromeStack, { top: layout.topBarTop, left: horizontalOverlayInset, right: horizontalOverlayInset }]}
          pointerEvents="box-none"
          onLayout={(event) => {
            const next = Math.round(event.nativeEvent.layout.height);
            setChromeStackHeight((prev) => (prev === next ? prev : next));
          }}
        >
          <View style={s.topBarShell}>
            <EnterpriseHeader
              variant="floating"
              metaDetail="networkOnly"
              date={date}
              mode={headerMode}
              realtimeStatus={realtimeStatus}
              liveStatusPill={uiState.overlays.topBar ? liveCoverageIsland : null}
              topSafeAreaPx={0}
              showModeChip={false}
              onOpenDatePicker={handleOpenDatePicker}
            />
          </View>
          {showOverlays && uiState.overlays.statusPills ? (
            <DashboardFloatingStatusBar
              placement="stack"
              stats={opsFeed.stats}
              onPressStat={onPressStat}
            />
          ) : null}
          {connectivityCopy ? <CockpitConnectivityBanner copy={connectivityCopy} /> : null}
          {governanceTrust &&
          uiState.mode !== "focus_urgent" &&
          uiState.overlays.trustBanner !== false ? (
            <CockpitTrustBanner message={governanceTrust} />
          ) : null}
        </View>
      ) : null}

      {showOverlays && uiState.overlays.quickTools ? (
        <View style={[s.quickActionsWrap, { bottom: layout.quickActionsBottom }]}>
          <QuickActionsBar
            actions={opsFeed.quickActions}
            onPressAction={(key) => {
              cockpit.touchInteraction();
              opsFeed.onQuickAction(key);
            }}
            compact
          />
        </View>
      ) : null}

      {showOverlays && uiState.overlays.delayedBanner && opsFeed.delayedMission ? (
        <View style={[s.delayedWrap, { bottom: layout.delayedBannerBottom }]}>
          <DelayedMissionCard mission={opsFeed.delayedMission} onPress={onPressDelayed} />
        </View>
      ) : null}

      {!uiState.overlays.topBar ? liveCoverageIsland : null}

      {cockpit.flags.cockpitDebugger ? (
        <CockpitStateDebugger uiState={uiState} orchestration={orchestration} />
      ) : null}

      <DashboardOpsSheet
        visible={opsSheetOpen}
        onClose={() => setOpsSheetOpen(false)}
        {...opsFeed}
      />

      <UrgencyBottomSheet
        visible={urgencySheetOpen}
        mission={opsFeed.delayedMission}
        onClose={() => setUrgencySheetOpen(false)}
        onViewMission={
          opsFeed.delayedMission && opsFeed.onPressMission
            ? () => {
                setUrgencySheetOpen(false);
                opsFeed.onPressMission!(opsFeed.delayedMission!.missionId);
              }
            : undefined
        }
        onMessage={handleMessage}
      />
    </View>
  );
}

const s = StyleSheet.create({
  root: {
    flex: 1,
    position: "relative",
    backgroundColor: E.BG,
  },
  mapZone: { zIndex: 0 },
  mapZoneImmersive: {
    ...StyleSheet.absoluteFillObject,
    paddingHorizontal: 0,
    zIndex: 0,
  },
  chromeStack: {
    position: "absolute",
    left: 0,
    right: 0,
    zIndex: 25,
    elevation: 25,
    flexDirection: "column",
    alignItems: "stretch",
    gap: 8,
  },
  topBarShell: {
    overflow: "visible",
    backgroundColor: "transparent",
    borderWidth: 0,
    borderColor: "transparent",
    ...Platform.select({
      ios: {
        shadowOpacity: 0,
      },
      android: { elevation: 0 },
      web: {
        boxShadow: "none",
        backdropFilter: "none",
      },
      default: {},
    }),
  },
  quickActionsWrap: {
    position: "absolute",
    left: 16,
    right: 16,
    zIndex: 32,
    paddingHorizontal: 2,
    paddingVertical: 2,
    borderRadius: 999,
    ...fleetGlassPanel({
      borderRadius: 999,
      paddingHorizontal: 8,
      paddingVertical: 4,
    }),
  },
  delayedWrap: {
    position: "absolute",
    left: 16,
    right: 16,
    zIndex: 31,
  },
});
