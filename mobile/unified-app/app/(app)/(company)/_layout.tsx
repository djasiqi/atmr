import { Redirect, Tabs } from "expo-router";
import { useEffect } from "react";
import { AppState, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { isFeatureEnabled } from "../../../src/core/featureFlags/registry";
import { useSession } from "../../../src/core/sessionProvider";
import { CompanyContextGuard } from "../../../src/core/guards";
import { companyRealtimeBridge } from "../../../src/features/company/realtime/companyRealtimeBridge";
import { CompanyFloatingTabBar } from "../../../src/features/company/navigation/CompanyFloatingTabBar";

export default function CompanyLayout() {
  const { activeContext, status } = useSession();
  const dispatchEnabled = isFeatureEnabled("company_dispatch_enabled");
  const companyContextId =
    activeContext && activeContext.context_type === "company"
      ? activeContext.context_id
      : null;

  // Déconnexion à la sortie de la zone entreprise (évite socket orpheline).
  useEffect(() => {
    return () => {
      companyRealtimeBridge.disconnect();
    };
  }, []);

  // Ne lancer le socket qu’après bootstrap. Ne pas couper pendant `bootstrapping` :
  // sinon reconnexion alors que le JWT est réaligné par refresh → faux « jeton absent » / boucles.
  useEffect(() => {
    if (!dispatchEnabled || !companyContextId) {
      companyRealtimeBridge.disconnect();
      return;
    }
    if (status !== "ready") {
      return;
    }
    companyRealtimeBridge.connect(companyContextId);
  }, [companyContextId, dispatchEnabled, status]);

  // Après mise en arrière-plan / reconnexion réseau, relancer le bridge si le flux était en échec.
  useEffect(() => {
    if (!dispatchEnabled || !companyContextId || status !== "ready") {
      return;
    }
    const sub = AppState.addEventListener("change", (next) => {
      if (next !== "active") return;
      if (!isFeatureEnabled("company_realtime_enabled")) return;
      const snap = companyRealtimeBridge.getSnapshot();
      if (snap.status === "failed" || snap.status === "reconnecting" || snap.status === "degraded") {
        companyRealtimeBridge.reconnect();
      }
    });
    return () => sub.remove();
  }, [companyContextId, dispatchEnabled, status]);

  if (!activeContext) {
    return <Redirect href="/(app)/context-selector" />;
  }

  if (activeContext.context_type !== "company") {
    return <Redirect href="/(app)/unauthorized" />;
  }

  /** Fond onglets aligné sur `operations-app` `(enterprise)/_layout` (#F5F7F6). */
  return (
    <CompanyContextGuard>
      <View style={{ flex: 1, backgroundColor: "#F5F7F6" }}>
      <Tabs
        screenOptions={{
          headerShown: false,
          tabBarActiveTintColor: "#0A8F7A",
          tabBarInactiveTintColor: "#7A808A",
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
