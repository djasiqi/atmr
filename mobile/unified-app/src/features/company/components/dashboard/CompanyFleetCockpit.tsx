import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Platform, StyleSheet, View } from "react-native";
import { useAppViewport } from "../../../../design/responsive/useAppViewport";
import type { CompanyDispatchMission, CompanyDriverLiveLocation } from "../../api/contracts";
import {
  computeCockpitOperationalCounts,
  orchestrationToMapPolicy,
  useCockpitRuntime,
  type CockpitOrchestrationDecision,
} from "../../dashboard/cockpit";
import { areMapSignalsEqual, type MapSignalsSnapshot } from "../maps/fleetMapTypes";
import { EnterpriseHeader } from "../EnterpriseHeader";
import { useCompanyMapNativeOverlayGate } from "../maps/companyMapNativeOverlayGate";
import { OperationalFleetMap } from "../maps/OperationalFleetMap";
import { fleetGlassPanel } from "../maps/fleetMapUiTokens";
import {
  computeDynamicCameraInsets,
  computeFleetCockpitLayout,
} from "./companyFleetCockpitLayout";
import { CockpitStateDebugger } from "./CockpitStateDebugger";
import { CockpitTrustBanner } from "./CockpitTrustBanner";
import { DashboardFloatingStatusBar } from "./DashboardFloatingStatusBar";
import type { DashboardOpsFeedProps } from "./DashboardOpsFeed";
import { DashboardOpsSheet } from "./DashboardOpsSheet";
import { DelayedMissionCard } from "./DelayedMissionCard";
import { LiveSystemStatusPill } from "./LiveSystemStatusPill";
import { E } from "../../theme/enterpriseOpsTheme";
import { QuickActionsBar } from "./QuickActionsBar";
import { UrgencyBottomSheet } from "./UrgencyBottomSheet";

export type CompanyFleetCockpitProps = {
  drivers: CompanyDriverLiveLocation[];
  missions: CompanyDispatchMission[];
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

function CockpitMapBlock({
  layout,
  cameraInsets,
  cameraVerticalBias,
  drivers,
  missions,
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
  nativeOverlaysEnabled,
  upcomingTableExpanded,
  onUpcomingTableExpandedChange,
}: {
  layout: ReturnType<typeof computeFleetCockpitLayout>;
  cameraInsets: { top: number; right: number; bottom: number; left: number };
  cameraVerticalBias: number;
  drivers: CompanyDriverLiveLocation[];
  missions: CompanyDispatchMission[];
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
  nativeOverlaysEnabled: boolean;
  upcomingTableExpanded: boolean;
  onUpcomingTableExpandedChange: (expanded: boolean) => void;
}) {
  const mapPolicy = orchestrationToMapPolicy(cockpitMapPolicy);
  return (
    <View style={[s.mapZone, s.mapZoneImmersive]}>
      <OperationalFleetMap
        layout="cockpit"
        cockpitImmersive
        cockpitExpanded
        drivers={drivers}
        missions={missions}
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
        nativeOverlaysEnabled={nativeOverlaysEnabled}
        upcomingTableExpanded={upcomingTableExpanded}
        onUpcomingTableExpandedChange={onUpcomingTableExpandedChange}
      />
    </View>
  );
}

export function CompanyFleetCockpit({
  drivers,
  missions,
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
  void refreshing;
  void onRefresh;

  const { topInset, bottomInset, usableHeight, safeLeft, safeRight } = useAppViewport();
  const nativeOverlaysEnabled = useCompanyMapNativeOverlayGate(realtimeStatus);
  const [opsSheetOpen, setOpsSheetOpen] = useState(false);
  const [urgencySheetOpen, setUrgencySheetOpen] = useState(false);
  const [driverSheetOpen, setDriverSheetOpen] = useState(false);
  const [upcomingTableExpanded, setUpcomingTableExpanded] = useState(true);
  const [selectedDriverId, setSelectedDriverId] = useState<number | null>(null);
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

  const cameraInsets = useMemo(
    () =>
      computeDynamicCameraInsets(layout, {
        driverSheetOpen: false,
        cockpitOpsPanel: upcomingTableExpanded,
        safeRight,
        safeLeft,
      }),
    [layout, upcomingTableExpanded, safeRight, safeLeft]
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
      cockpit.touchInteraction();
      opsFeed.onPressStat?.(key);
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

  const onLivePillPress = useCallback(() => {
    cockpit.touchInteraction();
    if (uiState.mode === "focus_urgent" || opsFeed.delayedMission) {
      onPressDelayed();
      return;
    }
    setOpsSheetOpen(true);
  }, [cockpit, uiState.mode, opsFeed.delayedMission, onPressDelayed]);

  const showOverlays = uiState.overlays.statusPills;
  const topStatusPillsOffset = layout.topBarTop + 52;
  const horizontalOverlayInset = Math.max(16, Math.max(safeLeft, safeRight) + 10);

  return (
    <View style={s.root} accessibilityLabel="Cockpit dispatch live">
      <CockpitMapBlock
        layout={layout}
        cameraInsets={cameraInsets}
        cameraVerticalBias={uiState.cameraVerticalBias}
        drivers={drivers}
        missions={missions}
        simplifyClustering={uiState.simplifyClustering}
        onDriverSheetChange={onDriverSheetChange}
        onViewMission={onViewMission}
        onMessage={onMessage}
        driverSheetSnap={cockpit.driverSheetSnap}
        onDriverSheetSnapChange={cockpit.setDriverSheetSnap}
        onSelectedDriverIdChange={onSelectedDriverIdChange}
        autoFocusDriverOnMount={false}
        autoFocusMissionOnMount={false}
        initialSelectedDriverId={cockpit.initialSelectedDriverId}
        cockpitMapPolicy={orchestration}
        onMapSignalsChange={onMapSignalsChange}
        syncSelectedDriverId={selectedDriverId}
        nativeOverlaysEnabled={nativeOverlaysEnabled}
        upcomingTableExpanded={upcomingTableExpanded}
        onUpcomingTableExpandedChange={setUpcomingTableExpanded}
      />

      {uiState.overlays.topBar ? (
        <View
          style={[s.topBar, { top: layout.topBarTop, left: horizontalOverlayInset, right: horizontalOverlayInset }]}
          pointerEvents="box-none"
        >
          <View style={s.topBarShell}>
            <EnterpriseHeader
              variant="floating"
              metaDetail="networkOnly"
              date={date}
              mode={headerMode}
              realtimeStatus={realtimeStatus}
              liveStatusPill={
                uiState.overlays.livePill ? (
                  <LiveSystemStatusPill
                    variant="header"
                    status={uiState.liveStatus}
                    realtimeSocketExpected={uiState.realtimeSocketExpected}
                    activityHint={uiState.liveActivityHint}
                    dataFreshness={
                      uiState.liveStatus === "connected" &&
                      (realtimeDataFreshness === "fresh" ||
                        realtimeDataFreshness === "idle" ||
                        realtimeDataFreshness === "stale")
                        ? realtimeDataFreshness
                        : undefined
                    }
                    onPress={onLivePillPress}
                    animationIntensity={uiState.animationIntensity}
                  />
                ) : null
              }
              topSafeAreaPx={0}
              showModeChip={false}
              onOpenDatePicker={onOpenDatePicker}
            />
          </View>
        </View>
      ) : null}

      {uiState.trustMessage &&
      uiState.mode !== "focus_urgent" &&
      uiState.overlays.trustBanner !== false ? (
        <View style={[s.trustWrap, { top: layout.topBarTop + 94 }]} pointerEvents="none">
          <CockpitTrustBanner message={uiState.trustMessage} />
        </View>
      ) : null}

      {showOverlays && uiState.overlays.statusPills ? (
        <DashboardFloatingStatusBar
          stats={opsFeed.stats}
          top={topStatusPillsOffset}
          horizontalInset={horizontalOverlayInset}
          onPressStat={onPressStat}
        />
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

      {!uiState.overlays.topBar && uiState.overlays.livePill ? (
        <View style={[s.livePillWrap, { bottom: layout.livePillBottom }]} pointerEvents="box-none">
          <LiveSystemStatusPill
            status={uiState.liveStatus}
            realtimeSocketExpected={uiState.realtimeSocketExpected}
            activityHint={uiState.liveActivityHint}
            dataFreshness={
              uiState.liveStatus === "connected" &&
              (realtimeDataFreshness === "fresh" ||
                realtimeDataFreshness === "idle" ||
                realtimeDataFreshness === "stale")
                ? realtimeDataFreshness
                : undefined
            }
            onPress={onLivePillPress}
            animationIntensity={uiState.animationIntensity}
          />
        </View>
      ) : null}

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
        onMessage={onMessage}
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
  topBar: {
    position: "absolute",
    left: 0,
    right: 0,
    zIndex: 25,
    elevation: 25,
  },
  topBarShell: {
    borderRadius: 999,
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
  trustWrap: {
    position: "absolute",
    left: 16,
    right: 16,
    zIndex: 26,
    alignItems: "center",
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
  livePillWrap: {
    position: "absolute",
    left: 0,
    right: 0,
    alignItems: "center",
    zIndex: 34,
  },
});
