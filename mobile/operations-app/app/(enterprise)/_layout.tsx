import React from "react";
import { View, StyleSheet, Text } from "react-native";
import { Tabs } from "expo-router";
import { StatusBar } from "expo-status-bar";

import { HapticTab } from "@/components/HapticTab";
import { IconSymbol } from "@/components/ui/IconSymbol";
import { tabBarStyles } from "@/styles/tabBarStyles";
import { EnterpriseProvider } from "@/context/EnterpriseContext";
import { EnterpriseHeader } from "@/components/enterprise/EnterpriseHeader";
import { EnterpriseReconnectBanner } from "@/components/common/EnterpriseReconnectBanner";
import { useAuth } from "@/hooks/useAuth";
import { useUnreadMessages } from "@/hooks/useUnreadMessages";

export default function EnterpriseLayout() {
  const { enterpriseSession } = useAuth();
  const { unreadCount } = useUnreadMessages();
  const initialMode =
    (enterpriseSession?.company?.dispatchMode as
      | "manual"
      | "semi_auto"
      | "fully_auto"
      | undefined) ?? "semi_auto";

  return (
    <EnterpriseProvider initialMode={initialMode}>
      <StatusBar style="dark" />
      <View style={styles.container}>
        <EnterpriseReconnectBanner />
        <Tabs
          screenOptions={{
            header: () => <EnterpriseHeader />,
            tabBarButton: HapticTab,
            tabBarStyle: tabBarStyles.tabBarStyle,
            tabBarItemStyle: tabBarStyles.tabBarItemStyle,
            tabBarLabelStyle: tabBarStyles.tabBarLabelStyle,
            tabBarActiveTintColor: tabBarStyles.palette.label, // #0A7F59 - Vert accent pour l'onglet actif
            tabBarInactiveTintColor: tabBarStyles.palette.labelInactive, // #5F7369 - Gris secondaire pour les onglets inactifs
          }}
        >
          <Tabs.Screen
            name="dashboard"
            options={{
              title: "Tableau de bord",
              tabBarIcon: ({ color }) => (
                <View style={tabBarStyles.tabBarIconContainer}>
                  <IconSymbol name="house.fill" size={24} color={color} />
                </View>
              ),
            }}
          />
          <Tabs.Screen
            name="rides"
            options={{
              title: "Courses",
              tabBarIcon: ({ color }) => (
                <View style={tabBarStyles.tabBarIconContainer}>
                  <IconSymbol name="list.bullet.rectangle" size={24} color={color} />
                </View>
              ),
            }}
          />
          <Tabs.Screen
            name="chat"
            options={{
              title: "Chat",
              tabBarIcon: ({ color }) => (
                <View style={tabBarStyles.tabBarIconContainer}>
                  <IconSymbol name="bubble.left.and.bubble.right.fill" size={24} color={color} />
                  {unreadCount > 0 && (
                    <View style={styles.badge}>
                      <Text style={styles.badgeText}>
                        {unreadCount > 99 ? "99+" : String(unreadCount)}
                      </Text>
                    </View>
                  )}
                </View>
              ),
            }}
          />
          <Tabs.Screen
            name="settings"
            options={{
              title: "Paramètres",
              tabBarIcon: ({ color }) => (
                <View style={tabBarStyles.tabBarIconContainer}>
                  <IconSymbol name="gearshape.fill" size={24} color={color} />
                </View>
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
    backgroundColor: "#F5F7F6",
  },
  badge: {
    position: "absolute",
    top: -6,
    right: -8,
    backgroundColor: "#EF4444",
    borderRadius: 10,
    minWidth: 20,
    height: 20,
    paddingHorizontal: 6,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 2,
    borderColor: "#FFFFFF",
    zIndex: 1,
  },
  badgeText: {
    color: "#FFFFFF",
    fontSize: 11,
    fontWeight: "700",
    textAlign: "center",
  },
});

