// app/(tabs)/trips.tsx

import React, { useEffect, useState } from "react";
import {
  SectionList,
  RefreshControl,
  Alert,
  TouchableOpacity,
  View,
  Text,
} from "react-native";
import { useAuth } from "@/hooks/useAuth";
import { getCompletedTrips, getAssignedTrips, Booking } from "@/services/api";
import { Loader } from "@/components/ui/Loader";
import { tripCardStyles as cardStyles } from "@/styles/tripCardStyles";
import TripHeader from "@/components/dashboard/TripHeader";
import { useNotifications } from "@/hooks/useNotifications";
import TripDetailsModal from "@/components/dashboard/TripDetailsModal";

function categorizeTripByTime(trip: Booking) {
  const hour = new Date(trip.scheduled_time).getHours();
  if (hour < 12) return "Matin";
  if (hour < 18) return "Après-midi";
  return "Soirée";
}

export default function TripsScreen() {
  useNotifications();
  const { driver } = useAuth();
  const [completedTrips, setCompletedTrips] = useState<Booking[]>([]);
  const [assignedTrips, setAssignedTrips] = useState<Booking[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedTripId, setSelectedTripId] = useState<number | null>(null);
  const [modalVisible, setModalVisible] = useState(false);

  const loadTrips = async () => {
    if (!driver) return;
    try {
      setLoading(true);
      const [completed, assigned] = await Promise.all([
        getCompletedTrips(driver.id),
        getAssignedTrips(),
      ]);
      const today = new Date().toDateString();
      const todayTrips = completed.filter(
        (t) => new Date(t.scheduled_time).toDateString() === today
      );
      setCompletedTrips(todayTrips);
      setAssignedTrips(assigned);
    } catch (e) {
      Alert.alert("Erreur", "Impossible de charger les trajets.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadTrips();
  }, []);

  const onRefresh = () => {
    setRefreshing(true);
    loadTrips();
  };

  const groupTrips = (trips: Booking[]) => {
    const grouped = trips.reduce(
      (acc, trip) => {
        const key = categorizeTripByTime(trip);
        if (!acc[key]) acc[key] = [];
        acc[key].push(trip);
        return acc;
      },
      {} as Record<string, Booking[]>
    );

    return Object.entries(grouped).map(([title, data]) => ({ title, data }));
  };

  const renderTripCard = (trip: Booking) => {
    if ((trip as any).isPlaceholder) {
      return (
        <View style={cardStyles.cardContainer}>
          <Text style={{ fontSize: 14, color: "#616161", textAlign: "center" }}>
            En attente de course. Vous serez notifié dès qu’une mission vous
            sera assignée.
          </Text>
        </View>
      );
    }

    return (
      <TouchableOpacity
        key={trip.id}
        style={cardStyles.cardContainer}
        onPress={() => {
          setSelectedTripId(trip.id);
          setModalVisible(true);
        }}
      >
        <Text style={cardStyles.routeSection}>
          {trip.pickup_location} → {trip.dropoff_location}
        </Text>

        <Text style={cardStyles.timeEnhanced}>
          {new Date(trip.scheduled_time).toLocaleDateString()} à{" "}
          {new Date(trip.scheduled_time).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          })}
        </Text>

        <Text
          style={[
            cardStyles.statusBadge,
            {
              backgroundColor:
                trip.status === "completed" ? "#C8E6C9" : "#BBDEFB",
              color: trip.status === "completed" ? "#256029" : "#0D47A1",
            },
          ]}
        >
          Statut : {trip.status === "completed" ? "Terminé" : trip.status}
        </Text>
      </TouchableOpacity>
    );
  };

  if (loading) {
    return (
      <View
        style={{
          flex: 1,
          justifyContent: "center",
          alignItems: "center",
          backgroundColor: "#FFFFFF",
        }}
      >
        <Loader />
      </View>
    );
  }

  const sections = [
    {
      title: "🕒 Courses assignées",
      data:
        assignedTrips.length > 0
          ? assignedTrips
          : [
              {
                id: -1,
                pickup_location: "",
                dropoff_location: "",
                scheduled_time: new Date().toISOString(),
                status: "assigned",
                client_name: "",
                client_phone: "",
                company_id: 0,
                driver_id: 0,
                is_return: false, // Propriété manquante ajoutée
                isPlaceholder: true,
              } as Booking & { isPlaceholder: boolean },
            ],
    },
    ...groupTrips(completedTrips),
  ];

  return (
    <View style={{ flex: 1, backgroundColor: "#F4F6F8" }}>
      <TripHeader date={new Date().toLocaleDateString()} />

      <SectionList
        sections={sections}
        keyExtractor={(item) => item.id.toString()}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
        }
        ListEmptyComponent={() => (
          <View style={{ marginTop: 24, alignItems: "center" }}>
            <Text style={cardStyles.emptyText}>
              Aucun trajet prévu pour aujourd’hui.
            </Text>
          </View>
        )}
        renderSectionHeader={({ section: { title } }) => (
          <Text style={cardStyles.sectionHeader}>{title}</Text>
        )}
        renderItem={({ item }) => renderTripCard(item)}
        contentContainerStyle={{ paddingBottom: 80 }}
      />

      {/* Modal de détails */}
      <TripDetailsModal
        visible={modalVisible}
        tripId={selectedTripId}
        onClose={() => {
          setModalVisible(false);
          setSelectedTripId(null);
        }}
      />
    </View>
  );
}
