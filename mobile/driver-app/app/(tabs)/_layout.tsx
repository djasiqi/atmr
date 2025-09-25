// C:\Users\jasiq\atmr\mobile\driver-app\app\(tabs)\_layout.tsx

import { Tabs } from 'expo-router';
import React from 'react';

import { HapticTab } from '@/components/HapticTab';
import { IconSymbol } from '@/components/ui/IconSymbol';
import { tabBarStyles } from '@/styles/tabBarStyles';

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
        tabBarActiveTintColor: '#ffffff',
        tabBarInactiveTintColor: '#cbd5e1',
      }}
    >
      <Tabs.Screen
        name="mission"
        options={{
          title: 'Mission',
          tabBarIcon: ({ color }) => (
            <IconSymbol name="car.fill" size={24} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="trips"
        options={{
          title: 'Courses',
          tabBarIcon: ({ color }) => (
            <IconSymbol name="list.bullet.rectangle" size={24} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="chat"
        options={{
          title: 'Équipe',
          tabBarIcon: ({ color }) => (
            <IconSymbol name="person.2.fill" size={24} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}