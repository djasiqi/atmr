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
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import { router } from "expo-router";
import dayjs from "dayjs";
import "dayjs/locale/fr";

import { useAuth } from "@/hooks/useAuth";
import { useEnterpriseContext } from "@/context/EnterpriseContext";
import { RideSnippetCard } from "@/components/enterprise/cards/RideSnippetCard";
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
const TAB_ACCESSIBILITY: Record<TabValue, string> = {
  unassigned: "Courses non assignées",
  assigned: "Courses assignées",
  urgent: "Courses urgentes",
};

const palette = {
  background: "#07130E",
  heroGradient: ["#11412F", "#07130E"] as [string, string],
  heroBorder: "rgba(46,128,94,0.36)",
  heroText: "#E6F2EA",
  heroMeta: "rgba(184,214,198,0.72)",
  searchBackground: "rgba(10,34,26,0.82)",
  searchBorder: "rgba(59,143,105,0.28)",
  searchPlaceholder: "rgba(184,214,198,0.55)",
  tabBackground: "rgba(10,34,26,0.82)",
  tabBorder: "rgba(59,143,105,0.28)",
  tabActive: "#1EB980",
  tabActiveShadow: "rgba(30,185,128,0.25)",
  tabText: "rgba(184,214,198,0.65)",
  tabTextActive: "#052015",
  listGap: 18,
  emptyState: "rgba(184,214,198,0.7)",
  error: "#F87171",
  modalOverlay: "rgba(5,22,16,0.82)",
  modalBackground: "#08211A",
  modalBorder: "rgba(46,128,94,0.4)",
  modalTitle: "#E6F2EA",
  modalText: "rgba(184,214,198,0.8)",
  modalButton: "#1EB980",
  modalButtonText: "#052015",
  modalCancelText: "rgba(184,214,198,0.75)",
  divider: "rgba(46,128,94,0.2)",
  countPillBg: "rgba(30,185,128,0.12)",
  countPillText: "#1EB980",
  loadingText: "rgba(184,214,198,0.7)",
};

export default function EnterpriseRidesScreen() {
  const { enterpriseSession } = useAuth();
  const { selectedDate } = useEnterpriseContext();

  const [selectedTab, setSelectedTab] = useState<TabValue>("unassigned");
  const [rides, setRides] = useState<RideSummary[]>([]);
  const [tabCounts, setTabCounts] = useState<Record<TabValue, number>>({
    unassigned: 0,
    assigned: 0,
    urgent: 0,
  });
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [scheduleModal, setScheduleModal] = useState<{
    rideId: string | null;
    value: string;
    label?: string;
  }>({ rideId: null, value: "", label: undefined });
  const [actionLoading, setActionLoading] = useState(false);

  const currentDate = useMemo(() => {
    return selectedDate ?? dayjs().format("YYYY-MM-DD");
  }, [selectedDate]);

  const formattedDay = useMemo(() => {
    const base = dayjs(currentDate);
    return base.format("dddd D MMMM");
  }, [currentDate]);

  const loadRides = useCallback(async () => {
    if (!enterpriseSession) return;
    setLoading(true);
    setErrorMessage(null);
    try {
      const response = await getDispatchRides({
        date: currentDate,
        status: selectedTab,
        query: search || undefined,
        page_size: 120,
      });
      setRides(response.items);
      setTabCounts((prev) => ({
        ...prev,
        [selectedTab]: response.total ?? response.items.length,
      }));
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

  const refreshTabCounts = useCallback(async () => {
    if (!enterpriseSession) return;
    try {
      const responses = await Promise.all(
        TABS.map((tab) =>
          getDispatchRides({
            date: currentDate,
            status: tab.value,
            page_size: 1,
          })
        )
      );
      setTabCounts({
        unassigned: responses[0]?.total ?? 0,
        assigned: responses[1]?.total ?? 0,
        urgent: responses[2]?.total ?? 0,
      });
    } catch (error) {
      console.warn("Impossible de rafraîchir les compteurs de courses", error);
    }
  }, [currentDate, enterpriseSession]);

  useEffect(() => {
    loadRides();
  }, [loadRides]);

  useEffect(() => {
    refreshTabCounts();
  }, [refreshTabCounts]);

  const handleUrgent = useCallback(
    async (rideId: string) => {
      setActionLoading(true);
      try {
        await markRideUrgent(rideId, { extra_delay_minutes: 15 });
        await refreshTabCounts();
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
    [loadRides, refreshTabCounts]
  );

  const confirmSchedule = useCallback(async () => {
    if (!scheduleModal.rideId) return;
    const raw = scheduleModal.value.trim();
    if (!raw) {
      setScheduleModal({ rideId: null, value: "", label: undefined });
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
      await refreshTabCounts();
      await loadRides();
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de planifier l’horaire.";
      setErrorMessage(message);
    } finally {
      setActionLoading(false);
      setScheduleModal({ rideId: null, value: "", label: undefined });
    }
  }, [currentDate, loadRides, refreshTabCounts, scheduleModal]);

  const handleOpenDetails = useCallback((rideId: string) => {
    router.push({
      pathname: "/(enterprise)/ride-details",
      params: { rideId },
    } as any);
  }, []);

  const onSubmitSearch = useCallback(() => {
    loadRides();
  }, [loadRides]);

  const renderRideCard = (ride: RideSummary) => {
    const pickupTime = ride.time.pickup_at
      ? dayjs(ride.time.pickup_at).format("HH[h]mm")
      : "À définir";
    const badges =
      ride.client.priority === "HIGH"
        ? [{ label: "Priorité", tone: "danger" as const }]
        : ride.client.priority === "LOW"
          ? [{ label: "Confort", tone: "info" as const }]
          : undefined;

    return (
      <RideSnippetCard
        key={ride.id}
        ride={{
          id: ride.id,
          time: pickupTime,
          showUndefinedIcon: !ride.time.pickup_at,
          client: ride.client.name,
          pickup: ride.route.pickup_address,
          dropoff: ride.route.dropoff_address,
          assignedTo: ride.driver?.name ?? null,
          badges,
          footerActions: (
            <View style={styles.cardActions}>
              <TouchableOpacity
                style={styles.actionButtonPrimary}
                onPress={() =>
                  setScheduleModal({
                    rideId: ride.id,
                    value: ride.time.pickup_at
                      ? dayjs(ride.time.pickup_at).format("HH:mm")
                      : "",
                    label: ride.client.name,
                  })
                }
              >
                <Ionicons
                  name="time-outline"
                  size={16}
                  color={palette.modalButtonText}
                />
                <Text style={styles.actionButtonPrimaryText}>Planifier</Text>
              </TouchableOpacity>
              {selectedTab !== "urgent" && (
                <TouchableOpacity
                  style={styles.actionButtonGhost}
                  onPress={() => handleUrgent(ride.id)}
                  disabled={actionLoading}
                >
                  <Ionicons
                    name="flame-outline"
                    size={16}
                    color={palette.tabActive}
                  />
                  <Text style={styles.actionButtonGhostText}>Urgent +15</Text>
                </TouchableOpacity>
              )}
              <TouchableOpacity
                style={styles.actionButtonGhost}
                onPress={() => handleOpenDetails(ride.id)}
              >
                <Ionicons
                  name="open-outline"
                  size={16}
                  color={palette.heroText}
                />
                <Text style={styles.actionButtonGhostText}>Voir la fiche</Text>
              </TouchableOpacity>
            </View>
          ),
        }}
      />
    );
  };

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={styles.content}
        refreshControl={
          <RefreshControl refreshing={loading} onRefresh={loadRides} />
        }
      >
        <LinearGradient
          colors={palette.heroGradient}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={styles.hero}
        >
          <View style={{ flex: 1 }}>
            <Text style={styles.heroKicker}>Plan de transport</Text>
            <Text style={styles.heroTitle}>{formattedDay}</Text>
            <Text style={styles.heroSubtitle}>
              {tabCounts[selectedTab] ?? rides.length} course
              {((tabCounts[selectedTab] ?? rides.length) || 0) > 1
                ? "s"
                : ""}{" "}
              {selectedTab === "unassigned"
                ? "à traiter"
                : selectedTab === "assigned"
                  ? "en cours"
                  : "à prioriser"}
            </Text>
          </View>
          <View style={styles.heroCountPill}>
            <Ionicons
              name="clipboard-outline"
              size={18}
              color={palette.countPillText}
            />
            <Text style={styles.heroCount}>
              {tabCounts.unassigned + tabCounts.assigned + tabCounts.urgent}
            </Text>
          </View>
        </LinearGradient>

        <View style={styles.tabs}>
          {TABS.map((tab) => {
            const isActive = selectedTab === tab.value;
            return (
              <TouchableOpacity
                key={tab.value}
                style={[styles.tabButton, isActive && styles.tabButtonActive]}
                onPress={() => setSelectedTab(tab.value)}
                activeOpacity={0.85}
                accessibilityRole="button"
                accessibilityLabel={TAB_ACCESSIBILITY[tab.value]}
                accessibilityState={{ selected: isActive }}
              >
                <Text
                  style={[styles.tabCount, isActive && styles.tabCountActive]}
                >
                  {tabCounts[tab.value]}
                </Text>
                <Text
                  style={[styles.tabText, isActive && styles.tabTextActive]}
                  numberOfLines={1}
                  ellipsizeMode="tail"
                >
                  {tab.label}
                </Text>
              </TouchableOpacity>
            );
          })}
        </View>

        <View style={styles.searchBar}>
          <Ionicons
            name="search-outline"
            size={18}
            color={palette.searchPlaceholder}
          />
          <TextInput
            value={search}
            onChangeText={setSearch}
            placeholder="Rechercher client, adresse ou chauffeur"
            placeholderTextColor={palette.searchPlaceholder}
            style={styles.searchInput}
            returnKeyType="search"
            onSubmitEditing={onSubmitSearch}
          />
          <TouchableOpacity
            style={styles.searchTrigger}
            onPress={onSubmitSearch}
            activeOpacity={0.75}
          >
            <Ionicons name="arrow-forward" size={18} color={palette.heroText} />
          </TouchableOpacity>
        </View>

        <View style={styles.divider} />

        {loading ? (
          <View style={styles.loading}>
            <ActivityIndicator color={palette.tabActive} />
            <Text style={styles.loadingText}>Préparation des courses…</Text>
          </View>
        ) : rides.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons
              name="leaf-outline"
              size={32}
              color={palette.emptyState}
            />
            <Text style={styles.emptyStateTitle}>Pas de course ici</Text>
            <Text style={styles.emptyStateText}>
              Ajuste la date ou change d’onglet pour consulter d’autres trajets.
            </Text>
          </View>
        ) : (
          <View style={{ gap: palette.listGap }}>
            {rides.map((ride) => renderRideCard(ride))}
          </View>
        )}

        {errorMessage && (
          <View style={styles.errorBanner}>
            <Ionicons name="alert-circle" size={18} color={palette.error} />
            <Text style={styles.errorText}>{errorMessage}</Text>
          </View>
        )}
      </ScrollView>

      <Modal visible={!!scheduleModal.rideId} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Planifier la course</Text>
            {scheduleModal.label ? (
              <Text style={styles.modalSubtitle}>{scheduleModal.label}</Text>
            ) : null}
            <TextInput
              style={styles.modalInput}
              value={scheduleModal.value}
              onChangeText={(text) =>
                setScheduleModal((prev) => ({ ...prev, value: text }))
              }
              placeholder="HH:mm"
              placeholderTextColor={palette.modalText}
              keyboardType="numeric"
              autoFocus
            />
            <View style={styles.modalActions}>
              <Pressable
                style={styles.modalCancel}
                onPress={() =>
                  setScheduleModal({
                    rideId: null,
                    value: "",
                    label: undefined,
                  })
                }
              >
                <Text style={styles.modalCancelText}>Annuler</Text>
              </Pressable>
              <Pressable
                style={styles.modalConfirm}
                onPress={confirmSchedule}
                disabled={actionLoading}
              >
                <Text style={styles.modalConfirmText}>Enregistrer</Text>
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
    backgroundColor: palette.background,
  },
  scroll: {
    flex: 1,
  },
  content: {
    padding: 20,
    paddingBottom: 32,
    gap: 20,
  },
  hero: {
    borderRadius: 24,
    padding: 22,
    flexDirection: "row",
    alignItems: "center",
    gap: 18,
    borderWidth: 1,
    borderColor: palette.heroBorder,
  },
  heroKicker: {
    color: palette.heroMeta,
    textTransform: "uppercase",
    letterSpacing: 3,
    fontSize: 12,
    marginBottom: 6,
  },
  heroTitle: {
    color: palette.heroText,
    fontSize: 26,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "capitalize",
  },
  heroSubtitle: {
    color: palette.heroMeta,
    fontSize: 14,
    marginTop: 6,
  },
  heroCountPill: {
    backgroundColor: palette.countPillBg,
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: palette.tabBorder,
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 8,
  },
  heroCount: {
    color: palette.countPillText,
    fontWeight: "700",
    fontSize: 16,
  },
  tabs: {
    flexDirection: "row",
    gap: 10,
  },
  tabButton: {
    flex: 1,
    borderRadius: 16,
    paddingVertical: 12,
    paddingHorizontal: 10,
    minWidth: 0,
    backgroundColor: palette.tabBackground,
    borderWidth: 1,
    borderColor: palette.tabBorder,
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
  },
  tabButtonActive: {
    backgroundColor: palette.tabActive,
    borderColor: palette.tabActive,
    shadowColor: palette.tabActiveShadow,
    shadowOpacity: 0.45,
    shadowOffset: { width: 0, height: 8 },
    shadowRadius: 14,
    elevation: 8,
  },
  tabText: {
    color: palette.tabText,
    fontWeight: "600",
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 0.6,
    textAlign: "center",
    flexShrink: 1,
  },
  tabTextActive: {
    color: palette.tabTextActive,
  },
  tabCount: {
    color: palette.tabText,
    fontWeight: "600",
    fontSize: 12,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 999,
    backgroundColor: "rgba(10,34,26,0.55)",
  },
  tabCountActive: {
    color: palette.tabTextActive,
    backgroundColor: "rgba(5,32,22,0.22)",
  },
  searchBar: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: palette.searchBackground,
    borderRadius: 16,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: palette.searchBorder,
  },
  searchInput: {
    flex: 1,
    color: palette.heroText,
    paddingVertical: 12,
    paddingHorizontal: 10,
    fontSize: 15,
  },
  searchTrigger: {
    borderRadius: 999,
    padding: 8,
  },
  divider: {
    height: 1,
    backgroundColor: palette.divider,
  },
  loading: {
    alignItems: "center",
    paddingVertical: 40,
    gap: 12,
  },
  loadingText: {
    color: palette.loadingText,
    fontSize: 14,
  },
  emptyState: {
    alignItems: "center",
    paddingVertical: 48,
    gap: 12,
  },
  emptyStateTitle: {
    color: palette.heroText,
    fontWeight: "600",
    fontSize: 16,
  },
  emptyStateText: {
    color: palette.emptyState,
    fontSize: 14,
    textAlign: "center",
    paddingHorizontal: 20,
  },
  cardActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
  },
  actionButtonPrimary: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 14,
    backgroundColor: palette.modalButton,
  },
  actionButtonPrimaryText: {
    color: palette.modalButtonText,
    fontWeight: "600",
    fontSize: 13,
  },
  actionButtonGhost: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: palette.tabBorder,
    backgroundColor: "rgba(10,34,26,0.6)",
  },
  actionButtonGhostText: {
    color: palette.heroText,
    fontWeight: "600",
    fontSize: 13,
  },
  errorBanner: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    padding: 14,
    backgroundColor: "rgba(248,113,113,0.12)",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(248,113,113,0.24)",
  },
  errorText: {
    color: palette.error,
    flex: 1,
    fontSize: 13,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: palette.modalOverlay,
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  modalCard: {
    width: "100%",
    maxWidth: 420,
    backgroundColor: palette.modalBackground,
    borderRadius: 24,
    padding: 24,
    borderWidth: 1,
    borderColor: palette.modalBorder,
    gap: 16,
  },
  modalTitle: {
    color: palette.modalTitle,
    fontSize: 20,
    fontWeight: "700",
  },
  modalSubtitle: {
    color: palette.modalText,
    fontSize: 14,
  },
  modalInput: {
    backgroundColor: "rgba(10,34,26,0.82)",
    borderRadius: 14,
    paddingVertical: 12,
    paddingHorizontal: 16,
    color: palette.heroText,
    fontSize: 16,
    borderWidth: 1,
    borderColor: palette.searchBorder,
  },
  modalActions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 12,
  },
  modalCancel: {
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  modalCancelText: {
    color: palette.modalCancelText,
    fontWeight: "600",
  },
  modalConfirm: {
    backgroundColor: palette.modalButton,
    paddingHorizontal: 18,
    paddingVertical: 12,
    borderRadius: 14,
  },
  modalConfirmText: {
    color: palette.modalButtonText,
    fontWeight: "700",
  },
});
