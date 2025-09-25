import React, { useEffect, useState, useCallback } from 'react';
import { ScrollView, Alert, Linking, View } from 'react-native';
import { useAuth } from '@/hooks/useAuth';
import { useSocket } from '@/hooks/useSocket';
import { useLocation } from '@/hooks/useLocation';
import { useNotifications } from '@/hooks/useNotifications';
import MissionCard from '@/components/dashboard/MissionCard';
import MissionHeader from '@/components/dashboard/MissionHeader';
import MissionMap from '@/components/dashboard/MissionMap';
import ConfirmCompletionModal from '@/components/dashboard/ConfirmCompletionModal';
import { Loader } from '@/components/ui/Loader';
import { getAssignedTrips, updateTripStatus, Booking, BookingStatus } from '@/services/api';

/**
 * Détecte si la mission est un retour, quel que soit le type de donnée reçu (bool, int, string, etc.)
 */
function isMissionReturn(is_return: any): boolean {
  // Cas bool ou int
  if (is_return === true || is_return === 1) return true;
  if (is_return === false || is_return === 0) return false;
  // Cas string (même "False", "false", "0", etc.)
  if (typeof is_return === "string") {
    const v = is_return.trim().toLowerCase();
    if (["1", "true", "yes", "oui"].includes(v)) return true;
    if (["0", "false", "no", "non", ""].includes(v)) return false;
    // Patch anti-typo
    if (v === "return") return true;
  }
  // Cas null/undefined ou autre
  if (!is_return) return false;
  // Log tout le reste pour analyse
  console.log("[isMissionReturn] Valeur inattendue:", is_return, typeof is_return);
  return false;
}




export default function MissionScreen() {
  const { driver } = useAuth();
  const { location } = useLocation();
  const socket = useSocket();
  useNotifications();

  const [isLoading, setIsLoading] = useState(true);
  const [missions, setMissions] = useState<Booking[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [modalVisible, setModalVisible] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const currentMission = missions[currentIndex] || null;

const loadMissions = useCallback(async () => {
  setIsLoading(true);
  try {
    const assigned = await getAssignedTrips();
    console.log("🚦 Missions reçues (type is_return) :", assigned.map(m => [m.id, m.is_return, typeof m.is_return]));
    const sorted = assigned.sort(
      (a, b) => new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime()
    );
    setMissions(sorted);
    setCurrentIndex(0);
  } catch {
    Alert.alert('Erreur', 'Impossible de charger les missions.');
  } finally {
    setIsLoading(false);
  }
}, []);


  useEffect(() => {
    if (driver) {
      loadMissions();
    }
  }, [driver, loadMissions]);

  useEffect(() => {
    if (!socket) return;

    const onNew = (data: Booking) => {
      setMissions(prev => {
        const exists = prev.find(m => m.id === data.id);
        const updated = exists
          ? prev.map(m => (m.id === data.id ? data : m))
          : [...prev, data];
        return updated.sort(
          (a, b) => new Date(a.scheduled_time).getTime() - new Date(b.scheduled_time).getTime()
        );
      });
    };

    const onUpdate = (data: Booking) => {
      setMissions(prev => {
        const updated = prev.map(m => (m.id === data.id ? data : m))
          .filter(m => !["completed", "return_completed", "canceled"].includes(m.status));
        // recalcul de l'index pour éviter l'affichage d'une mission terminée
        if (updated.length === 0) setCurrentIndex(0);
        else setCurrentIndex(Math.min(currentIndex, updated.length - 1));
        return updated;
      });
    };

    const onCancel = ({ id }: { id: number }) => {
      setMissions(prev => prev.filter(m => m.id !== id));
      if (currentMission?.id === id) {
        Alert.alert('❌ Mission annulée', 'La mission en cours a été annulée.');
        setCurrentIndex(0); // Tu peux l'améliorer plus tard si besoin
      }
    };

    socket.on('new_booking', onNew);
    socket.on('booking_updated', onUpdate);
    socket.on('booking_cancelled', onCancel);
    return () => {
      socket.off('new_booking', onNew);
      socket.off('booking_updated', onUpdate);
      socket.off('booking_cancelled', onCancel);
    };
  }, [socket, currentMission?.id]);

  const openNavigation = (destination: string) => {
    const url = `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(destination)}`;
    Linking.openURL(url);
  };

  const handleOpenModal = () => {
    setModalVisible(true);
  };

  const confirmCompletion = useCallback(async () => {
    console.log("Confirmer la fin de mission");
  if (!currentMission || isSubmitting) return;

  // DEBUG : envoie à Pipedream ce que tu vas envoyer à l'API réelle
  const isReturn = isMissionReturn(currentMission.is_return);
  const statusToSend: BookingStatus = isReturn ? "return_completed" : "completed";
  const payload = {
    bookingId: currentMission.id,
    status: statusToSend,
    is_return: currentMission.is_return,
    mission: currentMission, // Si tu veux TOUT voir
  };

  // Envoi debug vers pipedream
  try {
    await fetch('https://eoxskun0j22ygoy.m.pipedream.net', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    console.log('DEBUG PIPEDREAM envoyé:', payload);
  } catch (err) {
    console.warn('Echec envoi pipedream', err);
  }

  setIsSubmitting(true);
  setModalVisible(false);

  try {
    console.log(
      "[PATCH DEBUG]",
      "currentMission.id:", currentMission?.id,
      "is_return (brut):", currentMission?.is_return,
      "typeof:", typeof currentMission?.is_return,
      "isMissionReturn:", isMissionReturn(currentMission?.is_return)
    );
    const isReturn = !!currentMission.is_return; // 100% safe, force en booléen
    const statusToSend: BookingStatus = isReturn ? "return_completed" : "completed";
    console.log("[PATCH DEBUG] statusToSend:", statusToSend);

    await updateTripStatus(currentMission.id, statusToSend);

    setMissions(prev =>
      prev
        .map(m =>
          m.id === currentMission.id ? { ...m, status: statusToSend } : m
        )
        .filter(m => !["completed", "return_completed", "canceled"].includes(m.status))
    );
    setCurrentIndex(0); // Repart à la prochaine mission disponible

  } catch (error: any) {
    const msg =
      error.response?.data?.error ||
      error.response?.data?.message ||
      'Impossible de terminer la mission.';
    Alert.alert('Erreur', msg);
  } finally {
    setIsSubmitting(false);
  }
}, [currentMission, isSubmitting, missions]);


  if (!driver || isLoading) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#FFF' }}>
        <Loader />
      </View>
    );
  }

  return (
    <ScrollView className="flex-1 bg-white">
      <MissionHeader
        driverName={driver.first_name || 'Chauffeur'}
        date={new Date().toLocaleDateString()}
      />

      {location && currentMission && (
        <MissionMap
          location={location}
          destination={
            currentMission.status === 'in_progress'
              ? currentMission.dropoff_location!
              : currentMission.pickup_location!
          }
        />
      )}

      {currentMission ? (
        <View className="px-4 pt-4">
          <MissionCard
            mission={currentMission}
            onComplete={handleOpenModal}
            onCall={() =>
              currentMission.client_phone &&
              Linking.openURL(`tel:${currentMission.client_phone}`)
            }
            onNavigate={() => {
              const dest =
                currentMission.status === 'in_progress'
                  ? currentMission.dropoff_location!
                  : currentMission.pickup_location!;
              openNavigation(dest);
            }}
          />
        </View>
      ) : (
        <View className="flex-1 items-center justify-center py-10 px-4">
          <MissionCard.EmptyState />
        </View>
      )}

      <ConfirmCompletionModal
        visible={modalVisible}
        onClose={() => setModalVisible(false)}
        onConfirm={confirmCompletion}
        isLoading={isSubmitting}
      />
    </ScrollView>
  );
}
