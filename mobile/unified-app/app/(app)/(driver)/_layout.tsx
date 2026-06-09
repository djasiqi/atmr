import { useEffect, useMemo, useState } from "react";
import { realtimeManager } from "../../../src/core/realtime/realtimeManager";
import { Tabs } from "expo-router";
import { View } from "react-native";
import { DriverUnifiedGateGuard } from "../../../src/core/guards";
import { DriverFloatingTabBar } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import {
  useDriverMissionsQuery,
  useDriverRealtimeSync,
  useDriverTracking,
} from "../../../src/features/driver/hooks";
import { filterNextMissionsOnly } from "../../../src/features/driver/domain/missionGrouping";
import { getDriverStatusUx } from "../../../src/features/driver/statusDictionary";
import type { DriverMission } from "../../../src/features/driver/types";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import { useTrackingWindowState } from "../../../src/features/driver/services/trackingWindow";
import {
  getDriverAvailabilityActive,
  subscribeDriverAvailability,
} from "../../../src/features/driver/services/driverAvailabilityBridge";
import { setDriverPresenceWindowActive } from "../../../src/features/driver/tracking";
import {
  startBackgroundTrackingHealthMonitor,
  stopBackgroundTrackingHealthMonitor,
} from "../../../src/features/driver/services/backgroundTrackingHealthMonitor";
import {
  buildFloatingTabScreenOptions,
  FLOATING_TAB_IMPLEMENTATION,
  FLOATING_TAB_PAGE_BG,
} from "../../../src/navigation/floatingTabScreenOptions";
import { useAppViewport } from "../../../src/design/responsive";
import { useReduceMotion } from "../../../src/design/navigation/useReduceMotion";
import { usePerfRouteTracking } from "../../../src/core/observability/usePerfRouteTracking";
import { DriverPresenceDisclosureHost } from "../../../src/features/driver/components/DriverPresenceDisclosureHost";
import { DriverTrackingBannerHost } from "../../../src/features/driver/components/DriverTrackingBannerHost";

/**
 * Sélectionne la mission active sur laquelle le tracking GPS doit être
 * démarré. Logique alignée sur `selectActiveMission` du dashboard, mais
 * sans dépendance écran-spécifique : on prend d'abord les missions
 * `EN_ROUTE`/`IN_PROGRESS`, sinon la première non terminale (ASSIGNED,
 * ARRIVED).
 */
function pickTrackingMission(missions: DriverMission[] | undefined): DriverMission | null {
  if (!Array.isArray(missions) || missions.length === 0) return null;
  const live = filterNextMissionsOnly(missions);
  if (live.length > 0) return live[0] ?? null;
  const firstNonTerminal = missions.find((mission) => {
    const ux = getDriverStatusUx(typeof mission.status === "string" ? mission.status : null);
    return !ux.terminal;
  });
  return firstNonTerminal ?? null;
}

/**
 * Branche `useDriverTracking` au niveau du layout driver afin que le
 * tracking GPS reste actif quel que soit l'onglet courant (dashboard,
 * trips, chat…). Sans cet ancrage au layout, le tracking ne démarrait
 * qu'à l'ouverture de l'écran détail mission, ce qui empêchait toute
 * remontée GPS quand le chauffeur restait sur le dashboard.
 *
 * En complément, on pilote ici la « fenêtre horaire 07h–19h » de présence :
 *   - dans la fenêtre  → tracking GPS actif même sans mission (présence pure)
 *   - hors fenêtre     → tracking GPS uniquement si mission éligible
 * Une mission qui démarre à 19h30 continue donc bien jusqu'à sa fin grâce
 * au pipeline mission existant.
 */
function DriverTrackingHost() {
  const missionsQuery = useDriverMissionsQuery();
  const trackingMission = useMemo(
    () => pickTrackingMission(missionsQuery.data as DriverMission[] | undefined),
    [missionsQuery.data],
  );
  useDriverTracking(
    trackingMission?.id ?? null,
    typeof trackingMission?.status === "string" ? trackingMission.status : null,
  );

  const window = useTrackingWindowState();
  const workWindowEnabled = isFeatureEnabled("driver_tracking_work_window_enabled");
  const [driverAvailable, setDriverAvailable] = useState(() => getDriverAvailabilityActive());

  useEffect(() => subscribeDriverAvailability(() => {
    setDriverAvailable(getDriverAvailabilityActive());
  }), []);

  useEffect(() => {
    setDriverPresenceWindowActive(
      driverAvailable && workWindowEnabled && window.isOpen
    );
  }, [driverAvailable, workWindowEnabled, window.isOpen]);

  useEffect(() => {
    if (trackingMission?.id != null) {
      const stop = startBackgroundTrackingHealthMonitor();
      return () => stop();
    }
    stopBackgroundTrackingHealthMonitor();
    return undefined;
  }, [trackingMission?.id]);

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
  const { width } = useAppViewport();
  const reduceMotion = useReduceMotion();
  const tabScreenOptions = useMemo(
    () => buildFloatingTabScreenOptions(FLOATING_TAB_PAGE_BG.driver, width, reduceMotion),
    [width, reduceMotion]
  );

  return (
    <DriverUnifiedGateGuard>
      <View style={{ flex: 1, backgroundColor: FLOATING_TAB_PAGE_BG.driver }}>
        <DriverTrackingHost />
        <DriverRealtimeSyncHost />
        <DriverPresenceDisclosureHost />
        <DriverTrackingBannerHost />
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
    </DriverUnifiedGateGuard>
  );
}
