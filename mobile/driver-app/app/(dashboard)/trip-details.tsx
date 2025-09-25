// src/screens/trip-details.tsx
import React, { useEffect, useState } from 'react';
import { View, ScrollView, Text, Alert } from 'react-native';
import { useLocalSearchParams, router } from 'expo-router';
import { getTripDetails, updateTripStatus, Booking, BookingStatus } from '@/services/api';
import { Button } from '@/components/ui/Button';
import { Loader } from '@/components/ui/Loader';
import { styles } from '@/styles/tripDetailsStyles';

export default function TripDetailsScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const [trip, setTrip] = useState<Booking | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchTripDetails = async () => {
    setLoading(true);
    try {
      const details = await getTripDetails(Number(id));
      setTrip(details);
    } catch {
      Alert.alert('Erreur', 'Impossible de charger les détails du trajet.');
      router.back();
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (id) fetchTripDetails();
  }, [id]);

  const handleUpdateStatus = async (status: BookingStatus) => {
    if (!trip) return;
    setLoading(true);
    try {
      await updateTripStatus(trip.id, status);
      await fetchTripDetails();
      Alert.alert('Succès', `Le statut du trajet est passé à ${status}.`);
    } catch {
      Alert.alert('Erreur', 'Impossible de mettre à jour le statut.');
    } finally {
      setLoading(false);
    }
  };

  if (loading || !trip) {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <Loader />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.title}>Détails du trajet #{trip.id}</Text>

      {/* Client */}
      <View style={styles.section}>
        <View style={styles.rowBetween}>
          <Text style={styles.label}>Client :</Text>
          <Text style={styles.value}>{trip.client_name}</Text>
        </View>
      </View>

      {/* Lieux */}
      <View style={styles.section}>
        <View style={styles.rowBetween}>
          <Text style={styles.label}>De :</Text>
          <Text style={styles.value}>{trip.pickup_location}</Text>
        </View>
      </View>
      <View style={styles.section}>
        <View style={styles.rowBetween}>
          <Text style={styles.label}>Vers :</Text>
          <Text style={styles.value}>{trip.dropoff_location}</Text>
        </View>
      </View>

      {/* Horaires */}
      <View style={styles.section}>
        <View style={styles.rowBetween}>
          <Text style={styles.label}>Heure prévue :</Text>
          <Text style={styles.value}>
            {new Date(trip.scheduled_time).toLocaleString()}
          </Text>
        </View>
      </View>
      {trip.return_time && (
        <View style={styles.section}>
          <View style={styles.rowBetween}>
            <Text style={styles.label}>Heure de retour :</Text>
            <Text style={styles.value}>
              {new Date(trip.return_time).toLocaleString()}
            </Text>
          </View>
        </View>
      )}

      {/* Montant / Distance / Durée */}
      <View style={styles.section}>
        <View style={styles.rowBetween}>
          <Text style={styles.label}>Montant :</Text>
          <Text style={styles.value}>{trip.amount?.toFixed(2)} CHF</Text>
        </View>
      </View>

      <View style={styles.section}>
        <View style={styles.rowBetween}>
          <Text style={styles.label}>Distance :</Text>
          <Text style={styles.value}>
            {(trip.distance_meters / 1000).toFixed(1)} km
          </Text>
        </View>
      </View>

      <View style={styles.section}>
        <View style={styles.rowBetween}>
          <Text style={styles.label}>Durée :</Text>
          <Text style={styles.value}>
            {Math.ceil(trip.duration_seconds / 60)} min
          </Text>
        </View>
      </View>

      {/* Chauffeur */}
      {trip.driver_name && (
        <View style={styles.section}>
          <View style={styles.rowBetween}>
            <Text style={styles.label}>Chauffeur :</Text>
            <Text style={styles.value}>{trip.driver_name}</Text>
          </View>
        </View>
      )}

      {/* Infos médicales */}
      {trip.medical_facility && (
        <View style={styles.section}>
          <View style={styles.rowBetween}>
            <Text style={styles.label}>Établissement médical :</Text>
            <Text style={styles.value}>{trip.medical_facility}</Text>
          </View>
        </View>
      )}
      {trip.doctor_name && (
        <View style={styles.section}>
          <View style={styles.rowBetween}>
            <Text style={styles.label}>Médecin :</Text>
            <Text style={styles.value}>{trip.doctor_name}</Text>
          </View>
        </View>
      )}
      {trip.hospital_service && (
        <View style={styles.section}>
          <View style={styles.rowBetween}>
            <Text style={styles.label}>Service :</Text>
            <Text style={styles.value}>{trip.hospital_service}</Text>
          </View>
        </View>
      )}
      {trip.notes_medical && (
        <View style={styles.section}>
          <View style={styles.rowBetween}>
            <Text style={styles.label}>Notes :</Text>
            <Text style={styles.value}>{trip.notes_medical}</Text>
          </View>
        </View>
      )}

      {/* Statut */}
      <View style={styles.section}>
        <View style={styles.rowBetween}>
          <Text style={styles.label}>Statut :</Text>
          <Text style={[styles.value, { color: '#00796B' }]}>
            {trip.status}
          </Text>
        </View>
      </View>

      {/* Actions */}
      <View style={styles.actionsRow}>
        {trip.status === 'ASSIGNED' && (
          <View style={styles.actionButton}>
            <Button onPress={() => handleUpdateStatus('IN_PROGRESS' as BookingStatus)}>
              <Text style={styles.actionButtonText}>Commencer</Text>
            </Button>
          </View>
        )}
        {trip.status === 'IN_PROGRESS' && (
          <View style={styles.actionButton}>
            <Button
              onPress={() =>
                handleUpdateStatus(
                  trip.is_return ? 'return_completed' : 'completed'
                )
              }
            >
              <Text style={styles.actionButtonText}>
                {trip.is_return ? 'Terminer retour' : 'Terminer'}
              </Text>
            </Button>
          </View>
        )}
        <View style={styles.actionButton}>
          <Button variant="secondary" onPress={() => router.back()}>
            <Text style={styles.actionButtonText}>Retour</Text>
          </Button>
        </View>
      </View>
    </ScrollView>
  );
}
