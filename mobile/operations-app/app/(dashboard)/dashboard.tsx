import React, { useEffect, useState } from 'react';
import {
  View,
  ScrollView,
  RefreshControl,
  Alert,
  Linking,
  Platform,
} from 'react-native';
import { useRouter } from 'expo-router';
import { openNavigation } from '@/services/deepLinks';
import { MissionStateManager } from '@/services/missionState';

// MapView n'est pas disponible sur web
let MapView: any = null;
let Marker: any = null;
if (Platform.OS !== 'web') {
  try {
    const maps = require('react-native-maps');
    MapView = maps.default;
    Marker = maps.Marker;
  } catch (e) {
    // react-native-maps non disponible
  }
}
import { useAuth } from '@/hooks/useAuth';
import { useSocket } from '@/hooks/useSocket';
import { useLocation } from '@/hooks/useLocation';
import { useNotifications } from '@/hooks/useNotifications';
import {
  updateDriverAvailability,
  Booking,
} from '@/services/api';
import { requestMissionSync } from '@/services/missionSyncOrchestrator';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Loader } from '@/components/ui/Loader';
import MissionCard from '@/components/dashboard/MissionCard';
import StatusSwitch from '@/components/dashboard/StatusSwitch';
import ConfirmCompletionModal from '@/components/dashboard/ConfirmCompletionModal';
import { getCallablePhone } from '@/utils/phone';

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
      const assignedTrips = await requestMissionSync('manual_screen');
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

  const handleOpenNavigation = (destination: string) => {
    if (MissionStateManager.isActive()) {
      MissionStateManager.setNavigating(true);
    }
    openNavigation(destination);
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
        {location && MapView ? (
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
        ) : location ? (
          <View className="flex-1 bg-gray-100 justify-center items-center">
            <ThemedText className="text-center">
              📍 Position: {location.coords.latitude.toFixed(4)}, {location.coords.longitude.toFixed(4)}
              {'\n'}
              🗺️ Carte non disponible
            </ThemedText>
          </View>
        ) : null}
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
            callablePhone={getCallablePhone(currentMission)}
            onCall={() => {
              const phone = getCallablePhone(currentMission);
              if (phone) {
                if (Platform.OS === "web") {
                  (window as any).open(`tel:${phone}`);
                  Alert.alert("Appel", "Ouverture de l'appel… Si rien ne se passe, aucun logiciel d'appel n'est peut-être configuré sur cet appareil.");
                } else {
                  Linking.openURL(`tel:${phone}`);
                }
              }
            }}
            onNavigate={() => {
              // ✅ Normaliser le statut en majuscules pour correspondre au backend
              const normalizedStatus = currentMission.status?.toUpperCase();

              // Déterminer la destination selon le statut :
              // - IN_PROGRESS : client à bord → dropoff (Point B)
              // - ASSIGNED/EN_ROUTE : aller chercher client → pickup (Point A)
              const dest =
                normalizedStatus === "IN_PROGRESS"
                  ? currentMission.dropoff_location
                  : currentMission.pickup_location;

              openNavigation(dest);
            }}
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
            callablePhone={getCallablePhone(trip)}
            onCall={() => {
              const phone = getCallablePhone(trip);
              if (phone) {
                if (Platform.OS === "web") {
                  (window as any).open(`tel:${phone}`);
                  Alert.alert("Appel", "Ouverture de l'appel… Si rien ne se passe, aucun logiciel d'appel n'est peut-être configuré sur cet appareil.");
                } else {
                  Linking.openURL(`tel:${phone}`);
                }
              }
            }}
            onNavigate={() => {
              // ✅ Même logique pour les prochaines missions
              const normalizedStatus = trip.status?.toUpperCase();
              const dest =
                normalizedStatus === "IN_PROGRESS"
                  ? trip.dropoff_location
                  : trip.pickup_location;
              openNavigation(dest);
            }}
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