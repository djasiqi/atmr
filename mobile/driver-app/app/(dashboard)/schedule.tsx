// C:\Users\jasiq\atmr\mobile\driver-app\app\(dashboard)\schedule.tsx

import React, { useEffect, useState } from "react";
import { View, FlatList, RefreshControl, Alert } from "react-native";
import { useAuth } from "@/hooks/useAuth";
import { getAssignedTrips, Booking } from "@/services/api";
import { Loader } from "@/components/ui/Loader";
import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { Card } from "@/components/ui/Card";
import { router } from "expo-router";

export default function ScheduleScreen() {
  const { driver } = useAuth();
  const [schedule, setSchedule] = useState<Booking[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const loadSchedule = async () => {
    setLoading(true);
    try {
      const data = await getAssignedTrips();
      setSchedule(data);
    } catch {
      Alert.alert("Erreur", "Impossible de charger votre planning.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSchedule();
  }, [driver]);

  const onRefresh = async () => {
    setRefreshing(true);
    await loadSchedule();
    setRefreshing(false);
  };

  if (loading) {
    return (
      <ThemedView className="flex-1 justify-center items-center">
        <Loader />
      </ThemedView>
    );
  }

  return (
    <FlatList
      className="flex-1 bg-gray-50 dark:bg-black px-4 pt-6"
      data={schedule}
      keyExtractor={(item) => item.id.toString()}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
      ListHeaderComponent={() => (
        <ThemedText className="text-xl font-semibold mb-4">
          Mon Planning
        </ThemedText>
      )}
      ListEmptyComponent={() => (
        <ThemedView className="mt-6 items-center">
          <ThemedText className="text-gray-500">Aucun trajet prévu.</ThemedText>
        </ThemedView>
      )}
      renderItem={({ item }) => (
        <Card
          className="mb-4"
          onPress={() => router.push(`/(dashboard)/trip-details?id=${item.id}`)}
        >
          <ThemedText className="font-semibold">
            {item.pickup_location} → {item.dropoff_location}
          </ThemedText>
          <ThemedText className="text-gray-500">
            {new Date(item.scheduled_time).toLocaleString()}
          </ThemedText>
          <ThemedText className="text-sm mt-1">
            Client : {item.client_name}
          </ThemedText>
          <ThemedText className="text-sm text-blue-500">
            Statut : {item.status}
          </ThemedText>
        </Card>
      )}
    />
  );
}
