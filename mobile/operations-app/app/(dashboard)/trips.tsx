// app/(dashboard)/trips.tsx

import React, { useEffect, useState } from 'react';
import {
  FlatList,
  RefreshControl,
  Alert,
  TouchableOpacity,
  View,
  Text,
} from 'react-native';
import { useAuth } from '@/hooks/useAuth';
import api, { Booking } from '@/services/api';
import { Loader } from '@/components/ui/Loader';
import { router } from 'expo-router';
import { styles } from '@/styles/missionCardStyles';

export default function TripsHistoryScreen() {
  const { driver } = useAuth();
  const [trips, setTrips] = useState<Booking[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadTripsHistory = async () => {
    if (!driver?.id) return;
    setLoading(true);
    try {
      const response = await api.get<Booking[]>(`/drivers/${driver.id}/completed-trips`);
      setTrips(response.data || []);
    } catch (error) {
      Alert.alert('Erreur', 'Impossible de charger l\'historique des trajets.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadTripsHistory();
  }, [driver]);

  const onRefresh = async () => {
    setRefreshing(true);
    await loadTripsHistory();
  };

  const renderTrip = ({ item }: { item: Booking }) => {
    const pickup = item.pickup_location || 'Lieu inconnu';
    const dropoff = item.dropoff_location || 'Lieu inconnu';
    const client = item.client_name || 'Client inconnu';
    const status = item.status?.toUpperCase() === 'COMPLETED' ? 'Terminé' : item.status;

    return (
      <TouchableOpacity
        style={styles.containerEnhanced}
        onPress={() => router.push(`/(dashboard)/trip-details?id=${item.id}`)}
      >
        <View style={styles.headerRowEnhanced}>
          <Text style={styles.timeEnhanced}>
            {new Date(item.scheduled_time).toLocaleDateString()} à{' '}
            {new Date(item.scheduled_time).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit',
            })}
          </Text>
        </View>

        <View style={styles.statusBadgeContainer}>
          <Text style={styles.statusBadgeText}>{status}</Text>
        </View>

        <View style={styles.routeSection}>
          <Text style={styles.infoEnhanced}>
            {pickup} → {dropoff}
          </Text>
        </View>

        <View style={styles.metaInfoSection}>
          <Text style={styles.notesEnhanced}>Client : {client}</Text>
        </View>
      </TouchableOpacity>
    );
  };

  if (!driver || loading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#FFFFFF' }}>
        <Loader />
      </View>
    );
  }

  return (
    <FlatList
      style={{ flex: 1, backgroundColor: '#FFFFFF', paddingHorizontal: 16, paddingTop: 24 }}
      data={trips}
      keyExtractor={(item) => item.id.toString()}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
      ListEmptyComponent={() => (
        <View style={{ marginTop: 24, alignItems: 'center' }}>
          <Text style={{ color: '#616161', fontSize: 14 }}>Aucun trajet réalisé pour le moment.</Text>
        </View>
      )}
      renderItem={renderTrip}
      contentContainerStyle={{ paddingBottom: 20 }}
    />
  );
}
