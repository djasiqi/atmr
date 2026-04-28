import { MaterialIcons } from '@expo/vector-icons';
import { Tabs, useRouter } from 'expo-router';
import React from 'react';
import { Platform, Pressable, StyleSheet, Text } from 'react-native';

import { HapticTab } from '@/components/HapticTab';
import TabBarBackground from '@/components/ui/TabBarBackground';
import { Colors } from '@/constants/Colors';
import { useColorScheme } from '@/hooks/useColorScheme';
import { featureFlags } from '@/services/featureFlags';
import { useInstitutionPermissions } from '@/services/useInstitutionPermissions';
import { useInstitutionRealtime } from '@/services/useInstitutionRealtime';

export default function InstitutionTabLayout() {
  const colorScheme = useColorScheme();
  const router = useRouter();
  const permissions = useInstitutionPermissions();
  useInstitutionRealtime(featureFlags.institutionMobileRealtimeEnabled);

  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: Colors[colorScheme ?? 'light'].tint,
        headerShown: true,
        tabBarButton: HapticTab,
        tabBarBackground: TabBarBackground,
        tabBarStyle: Platform.select({
          ios: { position: 'absolute' },
          default: {},
        }),
        headerRight: () => (permissions.canCreateRequest && featureFlags.institutionMobileRequestSendEnabled ? (
          <Pressable
            accessibilityRole="button"
            onPress={() => router.push('/(institution)/request-create')}
            style={styles.headerCta}
          >
            <MaterialIcons name="add" size={22} color={Colors[colorScheme ?? 'light'].tint} />
            <Text style={[styles.headerCtaText, { color: Colors[colorScheme ?? 'light'].tint }]}>
              Demande
            </Text>
          </Pressable>
        ) : null),
      }}
    >
      <Tabs.Screen
        name="dashboard"
        options={{
          title: 'Terrain',
          tabBarIcon: ({ color }) => <MaterialIcons name="dashboard" size={26} color={color} />,
        }}
      />
      <Tabs.Screen
        name="requests"
        options={{
          title: 'Demandes',
          tabBarIcon: ({ color }) => <MaterialIcons name="assignment" size={26} color={color} />,
        }}
      />
      <Tabs.Screen
        name="patients"
        options={{
          title: 'Patients',
          tabBarIcon: ({ color }) => <MaterialIcons name="people" size={26} color={color} />,
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Réglages',
          tabBarIcon: ({ color }) => <MaterialIcons name="settings" size={26} color={color} />,
        }}
      />
      <Tabs.Screen
        name="request-create"
        options={{
          href: null,
          title: 'Nouvelle demande',
        }}
      />
      <Tabs.Screen
        name="request/[requestId]"
        options={{
          href: null,
          title: 'Détail demande',
        }}
      />
      <Tabs.Screen
        name="patient/[patientId]"
        options={{
          href: null,
          title: 'Détail patient',
        }}
      />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  headerCta: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 12,
    gap: 4,
  },
  headerCtaText: {
    fontSize: 15,
    fontWeight: '600',
  },
});
