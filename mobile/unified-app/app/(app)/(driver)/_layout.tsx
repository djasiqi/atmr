import { useEffect, useMemo } from "react";
import { Tabs } from "expo-router";
import { View } from "react-native";
import { DriverUnifiedGateGuard } from "../../../src/core/guards";
import { DriverFloatingTabBar } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import {
  useDriverMissionsQuery,
  useDriverTracking,
} from "../../../src/features/driver/hooks";
import { filterNextMissionsOnly } from "../../../src/features/driver/domain/missionGrouping";
import { getDriverStatusUx } from "../../../src/features/driver/statusDictionary";
import type { DriverMission } from "../../../src/features/driver/types";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import { useTrackingWindowState } from "../../../src/features/driver/services/trackingWindow";
import { setDriverPresenceWindowActive } from "../../../src/features/driver/tracking";

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
  useEffect(() => {
    setDriverPresenceWindowActive(workWindowEnabled && window.isOpen);
  }, [workWindowEnabled, window.isOpen]);

  return null;
}

/** Fond identique à `app/(app)/(company)/_layout.tsx`. */
export default function DriverLayout() {
  return (
    <DriverUnifiedGateGuard>
      <View style={{ flex: 1, backgroundColor: "#F5F7F6" }}>
        <DriverTrackingHost />
        <Tabs
          screenOptions={{
            headerShown: false,
            tabBarActiveTintColor: "#0A8F7A",
            tabBarInactiveTintColor: "#7A808A",
          }}
          tabBar={(props) => <DriverFloatingTabBar {...props} />}
        >
          <Tabs.Screen name="index" options={{ title: "Accueil" }} />
          <Tabs.Screen name="trips" options={{ title: "Courses" }} />
          <Tabs.Screen name="missions" options={{ title: "Missions", href: null }} />
          <Tabs.Screen name="chat" options={{ title: "Chat" }} />
          <Tabs.Screen name="schedule" options={{ title: "Planning", href: null }} />
          <Tabs.Screen name="profile" options={{ title: "Profil", href: null }} />
          <Tabs.Screen name="missions/[missionId]" options={{ href: null }} />
          <Tabs.Screen name="trips/[tripId]" options={{ href: null }} />
        </Tabs>
      </View>
    </DriverUnifiedGateGuard>
  );
}
