import React from "react";
import { View, StyleSheet } from "react-native";
import { Tabs } from "expo-router";

import { HapticTab } from "@/components/HapticTab";
import { IconSymbol } from "@/components/ui/IconSymbol";
import { tabBarStyles } from "@/styles/tabBarStyles";
import { EnterpriseProvider } from "@/context/EnterpriseContext";
import { EnterpriseHeader } from "@/components/enterprise/EnterpriseHeader";
import { useAuth } from "@/hooks/useAuth";

export default function EnterpriseLayout() {
  const { enterpriseSession } = useAuth();
  const initialMode =
    (enterpriseSession?.company?.dispatchMode as
      | "manual"
      | "semi_auto"
      | "fully_auto"
      | undefined) ?? "semi_auto";

  return (
    <EnterpriseProvider initialMode={initialMode}>
      <View style={styles.container}>
        <Tabs
          screenOptions={{
            header: () => <EnterpriseHeader />,
            tabBarButton: HapticTab,
            tabBarStyle: tabBarStyles.tabBarStyle,
            tabBarItemStyle: tabBarStyles.tabBarItemStyle,
            tabBarLabelStyle: tabBarStyles.tabBarLabelStyle,
            tabBarActiveTintColor: tabBarStyles.palette.label,
            tabBarInactiveTintColor: tabBarStyles.palette.labelInactive,
            tabBarActiveBackgroundColor: "rgba(30,185,128,0.14)",
            tabBarInactiveBackgroundColor: "transparent",
            tabBarBadgeStyle: {
              backgroundColor: "rgba(30,185,128,0.9)",
              color: "#052015",
              fontWeight: "700",
            },
            sceneContainerStyle: { backgroundColor: styles.container.backgroundColor },
          }}
        >
          <Tabs.Screen
            name="dashboard"
            options={{
              title: "Tableau de bord",
              tabBarIcon: ({ color }) => (
                <IconSymbol name="chart.bar.fill" size={24} color={color} />
              ),
            }}
          />
          <Tabs.Screen
            name="rides"
            options={{
              title: "Courses",
              tabBarIcon: ({ color }) => (
                <IconSymbol name="list.bullet.rectangle" size={24} color={color} />
              ),
            }}
          />
          <Tabs.Screen
            name="settings"
            options={{
              title: "Paramètres",
              tabBarIcon: ({ color }) => (
                <IconSymbol name="gearshape.fill" size={24} color={color} />
              ),
            }}
          />
          <Tabs.Screen
            name="chat"
            options={{
              title: "Chat",
              tabBarIcon: ({ color }) => (
                <IconSymbol
                  name="bubble.left.and.bubble.right.fill"
                  size={24}
                  color={color}
                />
              ),
            }}
          />
          {/* Routes complémentaires (détails, écrans modaux) masquées de la barre */}
          <Tabs.Screen name="ride-details" options={{ href: null }} />
        </Tabs>
      </View>
    </EnterpriseProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#07130E",
  },
});
