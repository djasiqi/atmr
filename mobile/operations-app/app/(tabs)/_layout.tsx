// C:\Users\jasiq\atmr\mobile\driver-app\app\(tabs)\_layout.tsx

import { Tabs } from "expo-router";
import React from "react";
import { View } from "react-native";

import { HapticTab } from "@/components/HapticTab";
import { IconSymbol } from "@/components/ui/IconSymbol";
import { tabBarStyles } from "@/styles/tabBarStyles";

// NOTE : On retire AuthProvider et useNotifications. Ils sont déjà gérés par le layout parent.

export default function TabLayout() {
  return (
    // Il n'y a plus besoin de AuthProvider ici.
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarButton: HapticTab,
        tabBarStyle: tabBarStyles.tabBarStyle,
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
  );
}
