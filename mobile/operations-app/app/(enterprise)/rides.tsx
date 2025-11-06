import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Modal,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { router } from "expo-router";
import dayjs from "dayjs";
import "dayjs/locale/fr";

import { useAuth } from "@/hooks/useAuth";
import {
  getDispatchRides,
  markRideUrgent,
  scheduleRide,
} from "@/services/enterpriseDispatch";
import { RideSummary } from "@/types/enterpriseDispatch";

dayjs.locale("fr");

type TabValue = "unassigned" | "assigned" | "urgent";

const TABS: { label: string; value: TabValue }[] = [
  { label: "Non assignées", value: "unassigned" },
  { label: "Assignées", value: "assigned" },
  { label: "Urgences", value: "urgent" },
];

export default function EnterpriseRidesScreen() {
  const { enterpriseSession } = useAuth();

  const [selectedTab, setSelectedTab] = useState<TabValue>("unassigned");
  const [rides, setRides] = useState<RideSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [scheduleModal, setScheduleModal] = useState<{
    rideId: string | null;
    value: string;
  }>({ rideId: null, value: "" });
  const [actionLoading, setActionLoading] = useState(false);

  const currentDate = useMemo(() => {
    const now = dayjs().tz ? dayjs().tz("Europe/Zurich") : dayjs();
    return now.format("YYYY-MM-DD");
  }, []);

  const loadRides = useCallback(async () => {
    if (!enterpriseSession) return;
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await getDispatchRides({
        date: currentDate,
        status: selectedTab,
        query: search || undefined,
        page_size: 40,
      });
      setRides(response.items);
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de charger les courses.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  }, [currentDate, enterpriseSession, search, selectedTab]);

  useEffect(() => {
    loadRides();
  }, [loadRides]);

  const handleUrgent = useCallback(
    async (rideId: string) => {
      setActionLoading(true);
      try {
        await markRideUrgent(rideId, { extra_delay_minutes: 15 });
        await loadRides();
      } catch (error: any) {
        const message =
          error?.response?.data?.error ??
          error?.message ??
          "Impossible de marquer la course en urgence.";
        setErrorMessage(message);
      } finally {
        setActionLoading(false);
      }
    },
    [loadRides]
  );

  const confirmSchedule = useCallback(async () => {
    if (!scheduleModal.rideId) return;
    const raw = scheduleModal.value.trim();
    if (!raw) {
      setScheduleModal({ rideId: null, value: "" });
      return;
    }
    const [hour, minute] = raw.split(":");
    if (
      hour === undefined ||
      minute === undefined ||
      Number.isNaN(Number(hour)) ||
      Number.isNaN(Number(minute))
    ) {
      setErrorMessage("Format horaire invalide (HH:mm).");
      return;
    }
    const isoDate = dayjs(
      `${currentDate}T${hour.padStart(2, "0")}:${minute.padStart(2, "0")}:00`
    ).toISOString();
    setActionLoading(true);
    try {
      await scheduleRide(scheduleModal.rideId, { pickup_at: isoDate });
      await loadRides();
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de planifier l’horaire.";
      setErrorMessage(message);
    } finally {
      setActionLoading(false);
      setScheduleModal({ rideId: null, value: "" });
    }
  }, [currentDate, loadRides, scheduleModal]);

  const filteredRides = rides;

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl refreshing={loading} onRefresh={loadRides} />
        }
      >
        <Text style={styles.title}>
          Courses du {dayjs(currentDate).format("DD MMM YYYY")}
        </Text>
        <View style={styles.tabs}>
          {TABS.map((tab) => (
            <TouchableOpacity
              key={tab.value}
              style={[
                styles.tabButton,
                selectedTab === tab.value && styles.tabButtonActive,
              ]}
              onPress={() => setSelectedTab(tab.value)}
            >
              <Text
                style={[
                  styles.tabText,
                  selectedTab === tab.value && styles.tabTextActive,
                ]}
              >
                {tab.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <TextInput
          value={search}
          onChangeText={setSearch}
          placeholder="Rechercher (client, adresse, chauffeur...)"
          placeholderTextColor="#98A5C7"
          style={styles.searchInput}
          returnKeyType="search"
          onSubmitEditing={loadRides}
        />

        {loading ? (
          <View style={styles.loading}>
            <ActivityIndicator color="#4D6BFE" />
            <Text style={styles.loadingText}>Chargement des courses…</Text>
          </View>
        ) : filteredRides.length === 0 ? (
          <Text style={styles.muted}>Aucune course dans cette catégorie.</Text>
        ) : (
          filteredRides.map((ride) => (
            <View key={ride.id} style={styles.rideCard}>
              <View style={styles.rideHeader}>
                <Text style={styles.rideClient}>{ride.client.name}</Text>
                <Text style={styles.rideTime}>
                  {ride.time.pickup_at
                    ? dayjs(ride.time.pickup_at).format("HH:mm")
                    : "⏱️ À définir"}
                </Text>
              </View>
              <Text style={styles.rideRoute}>{ride.route.pickup_address}</Text>
              <Text style={styles.rideRoute}>
                → {ride.route.dropoff_address}
              </Text>
              <View style={styles.rideActions}>
                <TouchableOpacity
                  style={styles.primaryAction}
                  onPress={() =>
                    router.push({
                      pathname: "/(enterprise)/ride-details",
                      params: { rideId: ride.id },
                    } as any)
                  }
                >
                  <Text style={styles.primaryActionText}>Voir</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.secondaryAction}
                  onPress={() =>
                    setScheduleModal({ rideId: ride.id, value: "" })
                  }
                >
                  <Text style={styles.secondaryActionText}>Planifier</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={styles.secondaryAction}
                  onPress={() => handleUrgent(ride.id)}
                  disabled={actionLoading}
                >
                  <Text style={styles.secondaryActionText}>Urgent +15</Text>
                </TouchableOpacity>
              </View>
            </View>
          ))
        )}

        {errorMessage && <Text style={styles.error}>{errorMessage}</Text>}

        <TouchableOpacity
          style={styles.refreshButton}
          onPress={loadRides}
          disabled={loading}
        >
          <Text style={styles.refreshButtonText}>Rafraîchir</Text>
        </TouchableOpacity>
      </ScrollView>

      <Modal visible={!!scheduleModal.rideId} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Planifier l’horaire</Text>
            <TextInput
              style={styles.modalInput}
              value={scheduleModal.value}
              onChangeText={(text) =>
                setScheduleModal((prev) => ({ ...prev, value: text }))
              }
              placeholder="HH:mm"
              placeholderTextColor="#9AA5CC"
              keyboardType="numeric"
              autoFocus
            />
            <View style={styles.modalActions}>
              <Pressable
                style={styles.modalCancel}
                onPress={() => setScheduleModal({ rideId: null, value: "" })}
              >
                <Text style={styles.modalCancelText}>Annuler</Text>
              </Pressable>
              <Pressable
                style={styles.modalConfirm}
                onPress={confirmSchedule}
                disabled={actionLoading}
              >
                <Text style={styles.modalConfirmText}>Confirmer</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#0B1736",
  },
  scroll: {
    flex: 1,
  },
  content: {
    padding: 20,
  },
  title: {
    color: "#FFFFFF",
    fontSize: 22,
    fontWeight: "700",
    marginBottom: 16,
  },
  tabs: {
    flexDirection: "row",
    gap: 8,
    marginBottom: 16,
  },
  tabButton: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
    alignItems: "center",
  },
  tabButtonActive: {
    backgroundColor: "#4D6BFE",
    borderColor: "#4D6BFE",
  },
  tabText: {
    color: "#B7C5F5",
    fontWeight: "600",
  },
  tabTextActive: {
    color: "#FFFFFF",
  },
  searchInput: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 12,
    padding: 12,
    color: "#FFFFFF",
    marginBottom: 14,
  },
  loading: {
    alignItems: "center",
    paddingVertical: 40,
  },
  loadingText: {
    color: "#A9B6E5",
    marginTop: 10,
  },
  muted: {
    color: "#9AA5CC",
  },
  rideCard: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 16,
    padding: 16,
    marginBottom: 14,
  },
  rideHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  rideClient: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: 15,
  },
  rideTime: {
    color: "#E2E8FF",
    fontWeight: "600",
  },
  rideRoute: {
    color: "#C0CDF7",
    fontSize: 13,
  },
  rideActions: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 12,
  },
  primaryAction: {
    flex: 1,
    marginRight: 8,
    backgroundColor: "#4D6BFE",
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: "center",
  },
  primaryActionText: {
    color: "#FFFFFF",
    fontWeight: "600",
  },
  secondaryAction: {
    flex: 1,
    marginLeft: 8,
    backgroundColor: "rgba(255,255,255,0.12)",
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: "center",
  },
  secondaryActionText: {
    color: "#E2E8FF",
    fontWeight: "600",
  },
  error: {
    color: "#F87171",
    marginTop: 12,
  },
  refreshButton: {
    marginTop: 12,
    backgroundColor: "#4D6BFE",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  refreshButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    alignItems: "center",
    justifyContent: "center",
  },
  modalCard: {
    backgroundColor: "#102347",
    width: "80%",
    borderRadius: 16,
    padding: 18,
  },
  modalTitle: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 12,
  },
  modalInput: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 10,
    padding: 12,
    color: "#FFFFFF",
    marginBottom: 16,
  },
  modalActions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 12,
  },
  modalCancel: {
    paddingVertical: 10,
    paddingHorizontal: 14,
  },
  modalCancelText: {
    color: "#9AA5CC",
    fontWeight: "600",
  },
  modalConfirm: {
    backgroundColor: "#4D6BFE",
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 10,
  },
  modalConfirmText: {
    color: "#FFFFFF",
    fontWeight: "600",
  },
});
