import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { realtimeManager } from "../../../src/core/realtime/realtimeManager";
import { Tabs } from "expo-router";
import { View } from "react-native";
import { DriverUnifiedGateGuard } from "../../../src/core/guards";
import { DriverFloatingTabBar } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import {
  useActiveDriverContextId,
  useDriverCompanyBookingsTodayQuery,
  useDriverMissionsQuery,
  useDriverRealtimeSync,
  useDriverTodayMissionsQuery,
  useDriverTracking,
} from "../../../src/features/driver/hooks";
import { useDriverSessionNetworkReady } from "../../../src/features/driver/sessionNetworkGate";
import { driverQueryKeys } from "../../../src/features/driver/queryKeys";
import {
  isMissionSourceSettledPostReady,
  resolveDriverMissionSnapshot,
} from "../../../src/features/driver/tracking/resolveMissionSnapshotReady";
import { startDriverLifecycleAttribution } from "../../../src/features/driver/driverLifecycleAttribution";
import { pickTrackingMission } from "../../../src/features/driver/domain/pickTrackingMission";
import type { DriverMission } from "../../../src/features/driver/types";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import { useTrackingWindowState } from "../../../src/features/driver/services/trackingWindow";
import {
  getDriverAvailabilityActive,
  subscribeDriverAvailability,
} from "../../../src/features/driver/services/driverAvailabilityBridge";
import {
  setDriverMissionSnapshot,
  setDriverPresenceContext,
} from "../../../src/features/driver/tracking";
import {
  startBackgroundTrackingHealthMonitor,
  stopBackgroundTrackingHealthMonitor,
} from "../../../src/features/driver/services/backgroundTrackingHealthMonitor";
import { installCanaryD5TransientLossInject } from "../../../src/features/driver/tracking/canaryD5TransientLoss";
import { installCanaryD5UnknownSelfHealInject } from "../../../src/features/driver/tracking/canaryD5UnknownSelfHeal";
import { installCanaryD5NativeBoundaryProbes } from "../../../src/features/driver/tracking/canaryD5NativeBoundaryProbe";
import {
  __getLifecycleGenerationForTests,
  getDriverTrackingBridgeSnapshot,
} from "../../../src/features/driver/services/driverTrackingBridge";
import {
  buildFloatingTabScreenOptions,
  FLOATING_TAB_IMPLEMENTATION,
  FLOATING_TAB_PAGE_BG,
} from "../../../src/navigation/floatingTabScreenOptions";
import { useAppViewport } from "../../../src/design/responsive";
import { AppFloatingBarMetricsProvider } from "../../../src/design/navigation/AppFloatingBarMetricsProvider";
import { useReduceMotion } from "../../../src/design/navigation/useReduceMotion";
import { usePerfRouteTracking } from "../../../src/core/observability/usePerfRouteTracking";
import { hydrateDriverMapViewport } from "../../../src/features/driver/services/driverMapViewportStore";
import { DriverPresenceDisclosureHost } from "../../../src/features/driver/components/DriverPresenceDisclosureHost";
import { DriverNotificationDisclosureHost } from "../../../src/features/driver/components/DriverNotificationDisclosureHost";
import { DriverTrackingBannerHost } from "../../../src/features/driver/components/DriverTrackingBannerHost";
import { DriverMissionLiveTrackingDisclosureHost } from "../../../src/features/driver/components/DriverMissionLiveTrackingDisclosureHost";
import {
  hasActiveDriverMissionStatus,
  setOtaAutoReloadMissionBlocking,
} from "../../../src/core/version/otaAutoReloadMissionGuard";

/**
 * Sélectionne la mission active pour le tracking GPS (priorité stricte PR2).
 */
function selectTrackingMission(missions: DriverMission[] | undefined): DriverMission | null {
  return pickTrackingMission(missions);
}

/**
 * Branche `useDriverTracking` au niveau du layout driver afin que le
 * tracking GPS reste actif quel que soit l'onglet courant (dashboard,
 * trips, chat…). Sans cet ancrage au layout, le tracking ne démarrait
 * qu'à l'ouverture de l'écran détail mission, ce qui empêchait toute
 * remontée GPS quand le chauffeur restait sur le dashboard.
 *
 * En complément, on pousse le contexte présence (disponibilité + fenêtre 07h–19h) :
 *   - FG + disponible + disclosure → tracking même hors fenêtre
 *   - BG + disponible → tracking seulement si fenêtre ouverte
 *   - mission → toujours tracking
 */
function DriverTrackingHost() {
  const queryClient = useQueryClient();
  const contextId = useActiveDriverContextId();
  const networkReady = useDriverSessionNetworkReady();
  const missionsQuery = useDriverMissionsQuery();
  const todayMissionsQuery = useDriverTodayMissionsQuery();
  const companyBookingsQuery = useDriverCompanyBookingsTodayQuery();
  const networkReadyAtRef = useRef(0);
  const networkReadyGenerationRef = useRef(0);
  const postReadyRefetchDoneRef = useRef(false);
  if (!networkReady) {
    networkReadyAtRef.current = 0;
    postReadyRefetchDoneRef.current = false;
  } else if (networkReadyAtRef.current === 0) {
    networkReadyAtRef.current = Date.now();
    networkReadyGenerationRef.current += 1;
  }
  const trackingMission = useMemo(
    () => selectTrackingMission(missionsQuery.data as DriverMission[] | undefined),
    [missionsQuery.data],
  );
  const todayTrackingMission = useMemo(
    () => selectTrackingMission(todayMissionsQuery.data as DriverMission[] | undefined),
    [todayMissionsQuery.data],
  );
  const missionSnapshot = useMemo(
    () =>
      resolveDriverMissionSnapshot({
        networkReady,
        networkReadyGeneration: networkReadyGenerationRef.current,
        sources: [
          {
            id: "bookings",
            settledPostReady: isMissionSourceSettledPostReady({
              networkReady,
              networkReadyAtMs: networkReadyAtRef.current,
              status: missionsQuery.status,
              fetchStatus: missionsQuery.fetchStatus,
              dataUpdatedAt: missionsQuery.dataUpdatedAt,
            }),
            missionId: trackingMission?.id ?? null,
          },
          {
            id: "today",
            settledPostReady: isMissionSourceSettledPostReady({
              networkReady,
              networkReadyAtMs: networkReadyAtRef.current,
              status: todayMissionsQuery.status,
              fetchStatus: todayMissionsQuery.fetchStatus,
              dataUpdatedAt: todayMissionsQuery.dataUpdatedAt,
            }),
            missionId: todayTrackingMission?.id ?? null,
          },
          {
            id: "company-bookings",
            settledPostReady: isMissionSourceSettledPostReady({
              networkReady,
              networkReadyAtMs: networkReadyAtRef.current,
              status: companyBookingsQuery.status,
              fetchStatus: companyBookingsQuery.fetchStatus,
              dataUpdatedAt: companyBookingsQuery.dataUpdatedAt,
            }),
            missionId: null,
          },
        ],
      }),
    [
      networkReady,
      trackingMission?.id,
      todayTrackingMission?.id,
      missionsQuery.status,
      missionsQuery.fetchStatus,
      missionsQuery.dataUpdatedAt,
      todayMissionsQuery.status,
      todayMissionsQuery.fetchStatus,
      todayMissionsQuery.dataUpdatedAt,
      companyBookingsQuery.status,
      companyBookingsQuery.fetchStatus,
      companyBookingsQuery.dataUpdatedAt,
    ],
  );
  const trackingMissionForBridge = trackingMission ?? todayTrackingMission;
  useEffect(() => {
    startDriverLifecycleAttribution();
  }, []);
  useEffect(() => {
    if (!networkReady || !contextId || postReadyRefetchDoneRef.current) return;
    postReadyRefetchDoneRef.current = true;
    void queryClient.refetchQueries({ queryKey: driverQueryKeys.missions(contextId) });
    void queryClient.refetchQueries({ queryKey: driverQueryKeys.companyBookingsToday(contextId) });
  }, [networkReady, contextId, queryClient]);
  useLayoutEffect(() => {
    setDriverMissionSnapshot(missionSnapshot);
  }, [missionSnapshot]);
  useLayoutEffect(() => {
    return () => setDriverMissionSnapshot({ status: "pending" });
  }, []);
  useDriverTracking(trackingMissionForBridge);

  // Canary D5-C3 / C4 : injects QA panel / production-apk uniquement.
  useEffect(() => {
    return installCanaryD5TransientLossInject({
      queryClient,
      getContextId: () => contextId,
    });
  }, [queryClient, contextId]);

  useEffect(() => {
    return installCanaryD5UnknownSelfHealInject();
  }, []);

  useEffect(() => {
    return installCanaryD5NativeBoundaryProbes({
      getMissionId: () => getDriverTrackingBridgeSnapshot().missionId,
      getGeneration: () => __getLifecycleGenerationForTests(),
    });
  }, []);

  const window = useTrackingWindowState();
  const workWindowEnabled = isFeatureEnabled("driver_tracking_work_window_enabled");
  const [driverAvailable, setDriverAvailable] = useState(() => getDriverAvailabilityActive());

  useEffect(() => subscribeDriverAvailability(() => {
    setDriverAvailable(getDriverAvailabilityActive());
  }), []);

  useEffect(() => {
    setDriverPresenceContext({
      available: driverAvailable,
      windowOpen: workWindowEnabled && window.isOpen,
    });
  }, [driverAvailable, workWindowEnabled, window.isOpen]);

  useEffect(() => {
    if (trackingMission?.id != null) {
      const stop = startBackgroundTrackingHealthMonitor();
      return () => stop();
    }
    stopBackgroundTrackingHealthMonitor();
    return undefined;
  }, [trackingMission?.id]);

  useEffect(() => {
    const missions = missionsQuery.data as DriverMission[] | undefined;
    const blocking =
      Array.isArray(missions) &&
      missions.some((mission) => hasActiveDriverMissionStatus(mission.status));
    setOtaAutoReloadMissionBlocking(blocking);
    return () => setOtaAutoReloadMissionBlocking(false);
  }, [missionsQuery.data]);

  return null;
}

/** Socket + polling missions : ancré au layout pour ne pas couper le socket en changeant d’onglet. */
function DriverRealtimeSyncHost() {
  useDriverRealtimeSync();
  useEffect(() => {
    return () => {
      realtimeManager.disconnect();
    };
  }, []);
  return null;
}

/** Fond identique à `app/(app)/(company)/_layout.tsx`. */
export default function DriverLayout() {
  usePerfRouteTracking("driver");
  useEffect(() => {
    void hydrateDriverMapViewport();
  }, []);
  const { width } = useAppViewport();
  const reduceMotion = useReduceMotion();
  const tabScreenOptions = useMemo(
    () => buildFloatingTabScreenOptions(FLOATING_TAB_PAGE_BG.driver, width, reduceMotion),
    [width, reduceMotion]
  );

  return (
    <DriverUnifiedGateGuard>
      <AppFloatingBarMetricsProvider preset="driver">
        <View style={{ flex: 1, backgroundColor: FLOATING_TAB_PAGE_BG.driver }}>
          <DriverTrackingHost />
          <DriverRealtimeSyncHost />
          <DriverPresenceDisclosureHost />
          <DriverNotificationDisclosureHost />
          <DriverTrackingBannerHost />
          <DriverMissionLiveTrackingDisclosureHost />
          <Tabs
            implementation={FLOATING_TAB_IMPLEMENTATION}
            screenOptions={{
              ...tabScreenOptions,
              tabBarActiveTintColor: "#0A8F7A",
              tabBarInactiveTintColor: "#7A808A",
            }}
            tabBar={(props) => <DriverFloatingTabBar {...props} />}
          >
            <Tabs.Screen name="index" options={{ title: "Accueil" }} />
            <Tabs.Screen name="trips" options={{ title: "Courses du jour" }} />
            <Tabs.Screen name="missions" options={{ title: "Missions", href: null }} />
            <Tabs.Screen name="messages" options={{ title: "Messages" }} />
            <Tabs.Screen name="chat" options={{ title: "Chat", href: null }} />
            <Tabs.Screen name="schedule" options={{ title: "Planning", href: null }} />
            <Tabs.Screen name="profile" options={{ title: "Profil", href: null }} />
            <Tabs.Screen name="missions/[missionId]" options={{ href: null }} />
            <Tabs.Screen name="trips/[tripId]" options={{ href: null }} />
          </Tabs>
        </View>
      </AppFloatingBarMetricsProvider>
    </DriverUnifiedGateGuard>
  );
}
