import React, { useEffect, useState } from 'react';
import {
  View,
  ScrollView,
  RefreshControl,
  Alert,
  Linking,
} from 'react-native';
import MapView, { Marker } from 'react-native-maps';
import { useRouter } from 'expo-router';
import { useAuth } from '@/hooks/useAuth';
import { useSocket } from '@/hooks/useSocket';
import { useLocation } from '@/hooks/useLocation';
import { useNotifications } from '@/hooks/useNotifications';
import {
  getAssignedTrips,
  updateDriverAvailability,
  Booking,
} from '@/services/api';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Loader } from '@/components/ui/Loader';
import MissionCard from '@/components/dashboard/MissionCard';
import StatusSwitch from '@/components/dashboard/StatusSwitch';
import ConfirmCompletionModal from '@/components/dashboard/ConfirmCompletionModal';

export default function DashboardScreen() {
  const { driver, refreshProfile } = useAuth();
  const { location } = useLocation();
  const socket = useSocket();
  const router = useRouter();

  const [isLoading, setIsLoading] = useState(false);
  const [trips, setTrips] = useState<Booking[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [currentMission, setCurrentMission] = useState<Booking | null>(null);
  const [modalVisible, setModalVisible] = useState(false);

  useNotifications();

  const loadTrips = async () => {
    setIsLoading(true);
    try {
      const assignedTrips = await getAssignedTrips();
      setTrips(assignedTrips);
      setCurrentMission(assignedTrips.length > 0 ? assignedTrips[0] : null);
    } catch (error) {
      Alert.alert('Erreur', "Échec de chargement des missions.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleToggleAvailability = async () => {
    if (!driver) return;
    setIsLoading(true);
    try {
      await updateDriverAvailability(!driver.is_available);
      await refreshProfile();
    } catch (error) {
      Alert.alert('Erreur', "Impossible de changer le statut.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCompleteMission = () => {
    setModalVisible(true);
  };

  const confirmCompletion = () => {
    setModalVisible(false);
    Alert.alert('✅ Mission terminée');
    // TODO: appeler l'API pour marquer la mission comme terminée
  };

  const openNavigation = (destination: string) => {
    const url = `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(destination)}`;
    Linking.openURL(url);
  };

  useEffect(() => {
    if (driver) loadTrips();
  }, [driver]);

  useEffect(() => {
    if (!socket) return;

    const handleNewBooking = (data: Booking) => {
      setCurrentMission(data);
      loadTrips();
    };

    const handleBookingUpdated = (data: Booking) => {
      if (currentMission?.id === data.id) {
        setCurrentMission(data);
        loadTrips();
      }
    };

    const handleBookingCancelled = (data: { id: number }) => {
      if (currentMission?.id === data.id) {
        setCurrentMission(null);
        loadTrips();
        Alert.alert("❌ Mission annulée", "La mission a été annulée.");
      }
    };

    socket.on('new_booking', handleNewBooking);
    socket.on('booking_updated', handleBookingUpdated);
    socket.on('booking_cancelled', handleBookingCancelled);

    return () => {
      socket.off('new_booking', handleNewBooking);
      socket.off('booking_updated', handleBookingUpdated);
      socket.off('booking_cancelled', handleBookingCancelled);
    };
  }, [socket, currentMission]);

  if (!driver || isLoading) {
    return (
      <ThemedView className="flex-1 justify-center items-center">
        <Loader />
      </ThemedView>
    );
  }

  return (
    <ScrollView
      className="flex-1 bg-white"
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={loadTrips} />
      }
    >
      <View className="h-64">
        {location && (
          <MapView
            style={{ flex: 1 }}
            initialRegion={{
              latitude: location.coords.latitude,
              longitude: location.coords.longitude,
              latitudeDelta: 0.01,
              longitudeDelta: 0.01,
            }}
            showsUserLocation
          >
            <Marker
              coordinate={{
                latitude: location.coords.latitude,
                longitude: location.coords.longitude,
              }}
              title="Vous êtes ici"
            />
          </MapView>
        )}
      </View>

      <View className="px-4 py-3">
        <ThemedText className="text-xl font-semibold mb-2">
          Bonjour {driver.first_name}
        </ThemedText>

        <StatusSwitch
          isAvailable={driver.is_available}
          onStatusChange={handleToggleAvailability}
        />

      </View>

      {currentMission && (
        <View className="px-4 py-2">
          <ThemedText className="text-lg font-semibold mb-2">
            Mission actuelle
          </ThemedText>
          <MissionCard
            mission={currentMission}
            onCall={() =>
              Linking.openURL(`tel:${currentMission.client_phone}`)
            }
            onNavigate={() =>
              openNavigation(currentMission.pickup_location)
            }
            onComplete={handleCompleteMission}
          />
        </View>
      )}

      <View className="px-4 py-2">
        <ThemedText className="text-lg font-semibold mb-2">
          Prochaines missions
        </ThemedText>
        {trips.slice(1).map((trip) => (
          <MissionCard
            key={trip.id}
            mission={trip}
            onNavigate={() => openNavigation(trip.pickup_location)}
            onPressDetails={() =>
              router.push(`/(dashboard)/trip-details?id=${trip.id}`)
            }
          />
        ))}
      </View>

      <ConfirmCompletionModal
        visible={modalVisible}
        onClose={() => setModalVisible(false)}
        onConfirm={confirmCompletion}
      />
    </ScrollView>
  );
}