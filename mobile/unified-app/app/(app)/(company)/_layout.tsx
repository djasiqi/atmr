import { Redirect, Tabs } from "expo-router";
import { useEffect, useMemo } from "react";
import { View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import {
  isFeatureEnabled,
  isCompanyRealtimeSocketExpected,
} from "../../../src/core/featureFlags/registry";
import { useSession } from "../../../src/core/sessionProvider";
import { CompanyContextGuard } from "../../../src/core/guards";
import { companyRealtimeBridge } from "../../../src/features/company/realtime/companyRealtimeBridge";
import { CompanyFloatingTabBar } from "../../../src/features/company/navigation/CompanyFloatingTabBar";
import { E } from "../../../src/features/company/theme/enterpriseOpsTheme";
import { buildFloatingTabScreenOptions, FLOATING_TAB_IMPLEMENTATION } from "../../../src/navigation/floatingTabScreenOptions";
import { useCompanyUrgentAlertSound } from "../../../src/features/messaging/useCompanyUrgentAlertSound";
import { useCompanyRecoveryListener } from "../../../src/features/company/realtime/useCompanyRecoveryListener";
import { useCompanyRuntimeResume } from "../../../src/features/company/runtimeResume";
import { useAppViewport } from "../../../src/design/responsive";
import { useReduceMotion } from "../../../src/design/navigation/useReduceMotion";
import { usePerfRouteTracking } from "../../../src/core/observability/usePerfRouteTracking";

export default function CompanyLayout() {
  usePerfRouteTracking("company");
  const { width } = useAppViewport();
  const reduceMotion = useReduceMotion();
  const tabScreenOptions = useMemo(
    () => buildFloatingTabScreenOptions(E.BG, width, reduceMotion),
    [width, reduceMotion]
  );
  const { activeContext, status } = useSession();
  const dispatchEnabled = isFeatureEnabled("company_dispatch_enabled");
  const realtimeEnabled = isCompanyRealtimeSocketExpected();
  const companyRuntimeResumeEnabled = isFeatureEnabled("company_runtime_resume_enabled");
  const companyContextId =
    activeContext && activeContext.context_type === "company"
      ? activeContext.context_id
      : null;

  useCompanyUrgentAlertSound();
  // Phase 2 PR B/C — gate D3.2 : recovery cohérent dashboard/missions/inbox/chat
  // sur stale (5 min sans event) ou reconnect (background/foreground, transition réseau).
  useCompanyRecoveryListener(companyContextId);

  useCompanyRuntimeResume({
    contextId: companyContextId,
    enabled: companyRuntimeResumeEnabled && dispatchEnabled && status === "ready",
  });

  // Déconnexion à la sortie de la zone entreprise (évite socket orpheline).
  useEffect(() => {
    return () => {
      companyRealtimeBridge.disconnect();
    };
  }, []);

  // Ne lancer le socket qu’après bootstrap. Ne pas couper pendant `bootstrapping` :
  // sinon reconnexion alors que le JWT est réaligné par refresh → faux « jeton absent » / boucles.
  useEffect(() => {
    if (!dispatchEnabled || !realtimeEnabled || !companyContextId) {
      companyRealtimeBridge.disconnect();
      return;
    }
    if (status !== "ready") {
      return;
    }
    companyRealtimeBridge.connect(companyContextId);
  }, [companyContextId, dispatchEnabled, realtimeEnabled, status]);

  if (!activeContext) {
    return <Redirect href="/(app)/context-selector" />;
  }

  if (activeContext.context_type === "driver") {
    return <Redirect href="/(app)/(driver)" />;
  }
  if (activeContext.context_type !== "company") {
    return <Redirect href="/(app)/unauthorized" />;
  }

  /** Fond onglets aligné sur `operations-app` `(enterprise)/_layout` (#F5F7F6). */
  return (
    <CompanyContextGuard>
      <View style={{ flex: 1, backgroundColor: E.BG }}>
      <Tabs
        implementation={FLOATING_TAB_IMPLEMENTATION}
        screenOptions={{
          ...tabScreenOptions,
          tabBarActiveTintColor: E.BRAND,
          tabBarInactiveTintColor: "#64748B",
        }}
        tabBar={(props) => <CompanyFloatingTabBar {...props} />}
      >
        <Tabs.Screen
          name="index"
          options={{
            title: "Accueil",
            href: null,
          }}
        />
        <Tabs.Screen
          name="dashboard"
          options={{
            title: "Dashboard",
            tabBarIcon: ({ color, size }) => (
              <Ionicons name="speedometer-outline" size={size} color={color} />
            ),
          }}
        />
        <Tabs.Screen
          name="rides"
          options={{
            title: "Courses",
            tabBarIcon: ({ color, size }) => (
              <Ionicons name="car-outline" size={size} color={color} />
            ),
          }}
        />
        <Tabs.Screen
          name="chat"
          options={{
            title: "Chat",
            tabBarIcon: ({ color, size }) => (
              <Ionicons name="chatbubble-ellipses-outline" size={size} color={color} />
            ),
          }}
        />
        <Tabs.Screen name="messages" options={{ href: null }} />
        <Tabs.Screen
          name="clients-facturation"
          options={{
            title: "Clients & facturation",
            tabBarIcon: ({ color, size }) => (
              <Ionicons name="reader-outline" size={size} color={color} />
            ),
          }}
        />
        <Tabs.Screen
          name="invoices"
          options={{
            title: "Factures",
            tabBarIcon: ({ color, size }) => (
              <Ionicons name="receipt-outline" size={size} color={color} />
            ),
          }}
        />
        <Tabs.Screen
          name="settings"
          options={{
            title: "Parametres",
            tabBarIcon: ({ color, size }) => (
              <Ionicons name="settings-outline" size={size} color={color} />
            ),
          }}
        />
        <Tabs.Screen name="ride-details" options={{ href: null }} />
        <Tabs.Screen name="fleet-map" options={{ href: null }} />
        <Tabs.Screen name="dispatch" options={{ href: null }} />
      </Tabs>
      </View>
    </CompanyContextGuard>
  );
}
