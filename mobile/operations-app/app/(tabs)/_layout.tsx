// C:\Users\jasiq\atmr\mobile\driver-app\app\(tabs)\_layout.tsx

import { Tabs } from "expo-router";
import React, { useMemo } from "react";
import { View, Platform } from "react-native";
import { StatusBar } from "expo-status-bar";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { HapticTab } from "@/components/HapticTab";
import { IconSymbol } from "@/components/ui/IconSymbol";
import { tabBarStyles } from "@/styles/tabBarStyles";
import { getLogger } from "@/utils/logger";

const log = getLogger("TabLayout");

// NOTE : On retire AuthProvider et useNotifications. Ils sont déjà gérés par le layout parent.

export default function TabLayout() {
  log.debug("tab layout rendered", { timestamp: new Date().toISOString() });
  
  const insets = useSafeAreaInsets();

  // ✅ Adapter le style de la tabBar pour prendre en compte les safe areas Android
  // Cela évite que la barre de navigation système recouvre la tabBar
  const dynamicTabBarStyle = useMemo(() => {
    const baseStyle = { ...tabBarStyles.tabBarStyle };

    // Sur Android, ajouter le paddingBottom pour éviter que la barre de navigation système recouvre
    // Le mode immersif devrait cacher cette barre, mais on s'assure avec le padding
    if (Platform.OS === "android") {
      // Utiliser le maximum entre le padding par défaut et les safe areas
      // La barre de navigation système fait généralement ~48px
      const navigationBarHeight = Math.max(insets.bottom, 0);
      const minPadding = 14; // Padding minimum défini dans tabBarStyles
      const baseHeight = 68; // Hauteur de base pour Android (iOS = 78)

      baseStyle.paddingBottom = Math.max(minPadding, navigationBarHeight);
      baseStyle.height = baseHeight + Math.max(0, navigationBarHeight - minPadding);
    }

    return baseStyle;
  }, [insets.bottom]);

  return (
    // Il n'y a plus besoin de AuthProvider ici.
    <>
      <StatusBar style="dark" />
      <Tabs
        screenOptions={{
          headerShown: false,
          tabBarButton: HapticTab,
          tabBarStyle: dynamicTabBarStyle,
          tabBarItemStyle: tabBarStyles.tabBarItemStyle,
          tabBarLabelStyle: tabBarStyles.tabBarLabelStyle,
          tabBarActiveTintColor: tabBarStyles.palette.label, // #0A7F59 - Vert accent pour l'onglet actif
          tabBarInactiveTintColor: tabBarStyles.palette.labelInactive, // #5F7369 - Gris secondaire pour les onglets inactifs
        }}
      >
        <Tabs.Screen
          name="mission"
          options={{
            title: "Mission",
            tabBarIcon: ({ color }) => (
              <View style={tabBarStyles.tabBarIconContainer}>
                <IconSymbol name="car.fill" size={24} color={color} />
              </View>
            ),
          }}
        />
        <Tabs.Screen
          name="trips"
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
            title: "Équipe",
            tabBarIcon: ({ color }) => (
              <View style={tabBarStyles.tabBarIconContainer}>
                <IconSymbol name="person.2.fill" size={24} color={color} />
              </View>
            ),
          }}
        />
        <Tabs.Screen
          name="profile"
          options={{
            title: "Profil",
            tabBarIcon: ({ color }) => (
              <View style={tabBarStyles.tabBarIconContainer}>
                <IconSymbol name="person.circle.fill" size={24} color={color} />
              </View>
            ),
          }}
        />
      </Tabs>
    </>
  );
}
