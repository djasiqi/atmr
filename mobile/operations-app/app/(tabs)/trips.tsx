// app/(tabs)/trips.tsx

import React, { useEffect, useState, useCallback, useMemo } from "react";
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
import { isCompletedStatus, isCanceledStatus } from "@/utils/bookingStatus";
import { Loader } from "@/components/ui/Loader";
import { tripCardStyles as cardStyles } from "@/styles/tripCardStyles";
import TripHeader from "@/components/dashboard/TripHeader";
import { useNotifications } from "@/hooks/useNotifications";
import TripDetailsModal from "@/components/dashboard/TripDetailsModal";
import {
  onBookingNew,
  onBookingUpdated,
  onBookingCancelled,
  onBookingReassigned,
} from "@/services/socket";
import { filterActiveMissions } from "@/utils/missionGrouping";

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

  const loadTrips = useCallback(async () => {
    if (!driver) {
      setCompletedTrips([]);
      setAssignedTrips([]);
      setLoading(false);
      setRefreshing(false);
      return;
    }

    try {
      setLoading(true);
      const [completed, assigned] = await Promise.all([
        getCompletedTrips(driver.id),
        getAssignedTrips(),
      ]);

      const today = new Date().toDateString();

      // Filtrer les courses complétées d'aujourd'hui uniquement
      const todayTrips = completed.filter(
        (t) => new Date(t.scheduled_time).toDateString() === today
      );

      // ✅ Filtrer les courses assignées selon la logique :
      // - Courses d'aujourd'hui
      // - Après 19h00 : afficher aussi les courses de demain
      const filteredAssigned = filterActiveMissions(assigned);

      setCompletedTrips(todayTrips);
      setAssignedTrips(filteredAssigned);
    } catch (e) {
      Alert.alert("Erreur", "Impossible de charger les trajets.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [driver]);

  useEffect(() => {
    loadTrips();
  }, [loadTrips]);

  // ✅ Listen to real-time booking events via EventEmitter
  useEffect(() => {
    if (!driver) return;

    // Listen to new booking
    const unsubscribeNew = onBookingNew((booking: Booking) => {
      console.log("📩 New booking received in trips:", booking.id);
      setAssignedTrips((prev) => {
        // Check if booking already exists
        const exists = prev.find((b) => b.id === booking.id);
        if (exists) {
          // Update existing booking
          return prev.map((b) => (b.id === booking.id ? booking : b));
        }
        // Add new booking and sort by scheduled_time
        const updated = [...prev, booking].sort(
          (a, b) =>
            new Date(a.scheduled_time).getTime() -
            new Date(b.scheduled_time).getTime()
        );
        return updated;
      });
    });

    // Listen to booking updated
    const unsubscribeUpdated = onBookingUpdated((booking: Booking) => {
      console.log("📩 Booking updated in trips:", booking.id, booking.status);
      const status = (booking.status || "").toLowerCase();

      // Update assigned trips
      setAssignedTrips((prev) => {
        const updated = prev
          .map((b) => (b.id === booking.id ? booking : b))
          .filter((b) => {
            // Remove if cancelled
            const s = (b.status || "").toLowerCase();
            // ✅ P0-1: Utiliser la fonction de normalisation
            return !isCanceledStatus(s);
          });
        return updated;
      });

      // ✅ P0-1: If booking is completed, move to completed trips
      if (isCompletedStatus(status)) {
        setCompletedTrips((prev) => {
          // Check if already in completed list
          const exists = prev.find((b) => b.id === booking.id);
          if (exists) {
            return prev.map((b) => (b.id === booking.id ? booking : b));
          }
          // Add to completed and sort
          const today = new Date().toDateString();
          const bookingDate = new Date(booking.scheduled_time).toDateString();
          if (bookingDate === today) {
            return [...prev, booking].sort(
              (a, b) =>
                new Date(a.scheduled_time).getTime() -
                new Date(b.scheduled_time).getTime()
            );
          }
          return prev;
        });
      }
    });

    // Listen to booking cancelled
    const unsubscribeCancelled = onBookingCancelled((data) => {
      const bookingId = typeof data === "object" && "id" in data ? data.id : null;
      if (!bookingId) return;

      console.log("📩 Booking cancelled in trips:", bookingId);

      // Remove from assigned trips
      setAssignedTrips((prev) => prev.filter((b) => b.id !== bookingId));

      // Remove from completed trips if present
      setCompletedTrips((prev) => prev.filter((b) => b.id !== bookingId));
    });

    // ✅ Mission retirée (réassignée à un autre chauffeur) → refresh
    const unsubscribeReassigned = onBookingReassigned(() => {
      console.log("📩 Booking reassigned in trips: refreshing trips list");
      loadTrips();
    });

    return () => {
      unsubscribeNew();
      unsubscribeUpdated();
      unsubscribeCancelled();
      unsubscribeReassigned();
    };
  }, [driver, loadTrips]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadTrips();
  }, [loadTrips]);

  const groupedCompletedSections = useMemo(() => {
    const grouped = completedTrips.reduce(
      (acc, trip) => {
        const key = categorizeTripByTime(trip);
        if (!acc[key]) acc[key] = [];
        acc[key].push(trip);
        return acc;
      },
      {} as Record<string, Booking[]>
    );

    return Object.entries(grouped).map(([title, data]) => ({
      title,
      data,
    }));
  }, [completedTrips]);

  const sections = useMemo(() => {
    const baseSections = [
      {
        title: "🕒 Courses assignées",
        data:
          assignedTrips.length > 0
            ? assignedTrips
            : ([
              {
                id: -1,
                pickup_location: "",
                dropoff_location: "",
                scheduled_time: new Date().toISOString(),
                status: "assigned",
                client_name: "",
                // ✅ P1-4 Phase 3.1: client_phone déprécié, mais gardé pour compatibilité avec le type
                client_phone: "",
                company_id: 0,
                driver_id: 0,
                is_return: false,
                isPlaceholder: true,
              } as Booking & { isPlaceholder: boolean },
            ] as Booking[]),
      },
      ...groupedCompletedSections,
    ];

    // ✅ DEBUG : Décommenter si besoin de diagnostiquer
    // console.log("[TripsScreen] sections =", JSON.stringify(baseSections));
    // console.log("[TripsScreen] completedTrips =", completedTrips);
    // console.log("[TripsScreen] assignedTrips =", assignedTrips);

    return baseSections;
  }, [assignedTrips, groupedCompletedSections, completedTrips]);

  const renderTripCard = (trip: Booking | (Booking & { isPlaceholder?: boolean })) => {
    const anyTrip = trip as any;

    if (anyTrip.isPlaceholder) {
      return (
        <View style={cardStyles.cardContainer}>
          <Text
            style={{
              fontSize: 16,
              color: "#15362B",
              textAlign: "center",
              fontWeight: "600",
              letterSpacing: 0.2,
            }}
          >
            🚗 En attente de course
          </Text>
          <Text
            style={{
              fontSize: 15,
              color: "#5F7369",
              textAlign: "center",
              marginTop: 10,
              lineHeight: 22,
            }}
          >
            Vous serez notifié dès qu'une mission vous sera assignée.
          </Text>
        </View>
      );
    }

    return (
      <TouchableOpacity
        style={cardStyles.cardContainer}
        onPress={() => {
          setSelectedTripId(trip.id);
          setModalVisible(true);
        }}
      >
        <Text style={cardStyles.routeSection}>
          {trip.pickup_location || "Point de départ"} →{" "}
          {trip.dropoff_location || "Destination"}
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
                trip.status === "completed"
                  ? "rgba(10,127,89,0.12)"
                  : "rgba(255,193,7,0.12)",
              color: trip.status === "completed" ? "#0A7F59" : "#8B6914",
              borderColor:
                trip.status === "completed"
                  ? "rgba(10,127,89,0.2)"
                  : "rgba(255,193,7,0.2)",
            },
          ]}
        >
          {trip.status === "completed"
            ? "✅ Terminé"
            : `🕓 ${trip.status || "En attente"}`}
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
          backgroundColor: "#F5F7F6",
        }}
      >
        <Loader />
      </View>
    );
  }

  return (
    <View style={{ flex: 1, backgroundColor: "#F5F7F6" }}>
      <TripHeader date={new Date().toLocaleDateString()} />

      <SectionList
        sections={sections}
        keyExtractor={(item) => {
          // ✅ Sécurité : garantir que keyExtractor retourne toujours une string valide
          if (!item || item.id == null) {
            return `item-${Math.random().toString(36).substr(2, 9)}`;
          }
          return String(item.id);
        }}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            colors={["#0A7F59"]}
            tintColor="#0A7F59"
          />
        }
        ListEmptyComponent={() => (
          <View style={{ marginTop: 24, alignItems: "center" }}>
            <Text style={cardStyles.emptyText}>
              Aucun trajet prévu pour aujourd'hui.
            </Text>
          </View>
        )}
        renderSectionHeader={({ section }) => {
          const rawTitle = (section as any).title;
          const safeTitle =
            typeof rawTitle === "string" ? rawTitle : String(rawTitle ?? "");

          return (
            <View style={{ paddingHorizontal: 16, paddingTop: 16 }}>
              <Text style={cardStyles.sectionHeader}>{safeTitle}</Text>
            </View>
          );
        }}
        renderItem={({ item }) => renderTripCard(item as any)}
        contentContainerStyle={{ paddingBottom: 80 }}
      />

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
