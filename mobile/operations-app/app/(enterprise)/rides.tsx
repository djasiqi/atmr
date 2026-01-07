import React, { useCallback, useEffect, useMemo, useState, useRef } from "react";
import {
  ActivityIndicator,
  Alert,
  Modal,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
  AppState,
} from "react-native";
import { useBottomTabBarHeight } from "@react-navigation/bottom-tabs";
import { useFocusEffect } from "@react-navigation/native";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import "dayjs/locale/fr";

import { useAuth } from "@/hooks/useAuth";
import { useEnterpriseContext } from "@/context/EnterpriseContext";
import { RideSnippetCard } from "@/components/enterprise/cards/RideSnippetCard";
import { RideEditModal } from "@/components/enterprise/rides/RideEditModal";
import { RideCreateModal } from "@/components/enterprise/rides/RideCreateModal";
import { ClientCreateModal } from "@/components/enterprise/rides/ClientCreateModal";
import { TimeDatePicker } from "@/components/enterprise/rides/TimeDatePicker";
import {
  getDispatchRides,
  scheduleRide,
  cancelRide,
} from "@/services/enterpriseDispatch";
import { RideSummary } from "@/types/enterpriseDispatch";
import { useRideActions } from "@/hooks/useRideActions";
import { router } from "expo-router";

dayjs.locale("fr");


// ✅ Palette professionnelle cohérente avec le dashboard driver
const palette = {
  background: "#F5F7F6",
  heroGradient: ["#0A7F59", "#0D5F3F"] as [string, string],
  heroBorder: "rgba(15,54,43,0.08)",
  heroText: "#FFFFFF",
  heroMeta: "rgba(255,255,255,0.9)",
  searchBackground: "#FFFFFF",
  searchBorder: "rgba(15,54,43,0.08)",
  searchPlaceholder: "#91A59D",
  tabBackground: "#FFFFFF",
  tabBorder: "rgba(15,54,43,0.08)",
  tabActive: "#0A7F59",
  tabActiveShadow: "rgba(10,127,89,0.2)",
  tabText: "#5F7369",
  tabTextActive: "#FFFFFF",
  listGap: 18,
  emptyState: "#91A59D",
  error: "#EF4444",
  modalOverlay: "rgba(21,54,43,0.75)",
  modalBackground: "#FFFFFF",
  modalBorder: "rgba(15,54,43,0.12)",
  modalTitle: "#15362B",
  modalText: "#5F7369",
  modalButton: "#0A7F59",
  modalButtonText: "#FFFFFF",
  modalCancelText: "#5F7369",
  divider: "rgba(15,54,43,0.08)",
  countPillBg: "rgba(10,127,89,0.12)",
  countPillText: "#0A7F59",
  loadingText: "#91A59D",
};

export default function EnterpriseRidesScreen() {
  const { enterpriseSession } = useAuth();
  const { selectedDate } = useEnterpriseContext();
  const tabBarHeight = useBottomTabBarHeight();

  const [rides, setRides] = useState<RideSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [selectedRideForEdit, setSelectedRideForEdit] = useState<RideSummary | null>(null);
  const [expandedRideId, setExpandedRideId] = useState<string | null>(null);
  const [scheduleModal, setScheduleModal] = useState<{
    rideId: string | null;
    scheduledTime: Date | null;
    label?: string;
  }>({ rideId: null, scheduledTime: null, label: undefined });
  const [actionLoading, setActionLoading] = useState(false);
  const [cancelModal, setCancelModal] = useState<{
    ride: RideSummary | null;
    shouldBill: boolean;
  }>({ ride: null, shouldBill: false });
  const [clientCreateModalVisible, setClientCreateModalVisible] = useState(false);

  const currentDate = useMemo(() => {
    return selectedDate ?? dayjs().format("YYYY-MM-DD");
  }, [selectedDate]);

  const formattedDay = useMemo(() => {
    const base = dayjs(currentDate);
    return base.format("dddd D MMMM");
  }, [currentDate]);

  // ✅ Trier les courses : d'abord celles avec horaire (plus proche à plus éloignée), puis celles à définir
  const sortedRides = useMemo(() => {
    const withTime: RideSummary[] = [];
    const withoutTime: RideSummary[] = [];

    rides.forEach((ride) => {
      if (ride.time.pickup_at) {
        const moment = dayjs(ride.time.pickup_at);
        // Si l'heure est à minuit (00:00), c'est une heure non définie
        if (moment.hour() === 0 && moment.minute() === 0) {
          withoutTime.push(ride);
        } else {
          withTime.push(ride);
        }
      } else {
        withoutTime.push(ride);
      }
    });

    // Trier les courses avec heure de la plus proche à la plus éloignée
    withTime.sort(
      (a, b) =>
        dayjs(a.time.pickup_at!).valueOf() - dayjs(b.time.pickup_at!).valueOf()
    );

    return [...withTime, ...withoutTime];
  }, [rides]);

  const loadRides = useCallback(async () => {
    if (!enterpriseSession) return;
    setLoading(true);
    setErrorMessage(null);
    try {
      console.log("[rides.tsx] Chargement courses pour date:", currentDate);
      const response = await getDispatchRides({
        date: currentDate,
        query: search || undefined,
        page_size: 120,
      });
      console.log("[rides.tsx] Courses reçues:", response.items.length, "courses");
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
  }, [currentDate, enterpriseSession, search]);

  // Référence pour le polling automatique
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const appStateRef = useRef(AppState.currentState);

  // Charger les courses au montage et quand la date change
  useEffect(() => {
    loadRides();
  }, [loadRides]);

  // Polling automatique : récupérer les courses toutes les 30 secondes quand l'app est active
  useEffect(() => {
    if (!enterpriseSession) {
      // Nettoyer le polling si pas de session
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      return;
    }

    // Fonction pour démarrer le polling
    const startPolling = () => {
      // Nettoyer l'intervalle existant si présent
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }

      // Démarrer le polling toutes les 30 secondes
      pollingIntervalRef.current = setInterval(() => {
        const currentAppState = AppState.currentState;
        // Seulement charger si l'app est active
        if (currentAppState === "active") {
          console.log("[rides.tsx] Polling automatique : rechargement des courses");
          loadRides();
        }
      }, 30000); // 30 secondes
    };

    // Démarrer le polling si l'app est active
    if (appStateRef.current === "active") {
      startPolling();
    }

    // Écouter les changements d'état de l'application
    const subscription = AppState.addEventListener("change", (nextAppState) => {
      if (
        appStateRef.current.match(/inactive|background/) &&
        nextAppState === "active"
      ) {
        // L'app revient au premier plan : recharger immédiatement et redémarrer le polling
        console.log("[rides.tsx] Application revenue au premier plan : rechargement des courses");
        loadRides();
        startPolling();
      } else if (nextAppState.match(/inactive|background/)) {
        // L'app passe en arrière-plan : arrêter le polling
        console.log("[rides.tsx] Application en arrière-plan : arrêt du polling");
        if (pollingIntervalRef.current) {
          clearInterval(pollingIntervalRef.current);
          pollingIntervalRef.current = null;
        }
      }
      appStateRef.current = nextAppState;
    });

    // Cleanup
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      subscription.remove();
    };
  }, [enterpriseSession, loadRides]);

  // Recharger les courses quand l'écran revient au focus
  useFocusEffect(
    useCallback(() => {
      if (enterpriseSession) {
        console.log("[rides.tsx] Écran au focus : rechargement des courses");
        loadRides();
      }
    }, [enterpriseSession, loadRides])
  );


  // ✅ Utiliser le hook partagé pour les actions sur les courses
  const refreshData = useCallback(async () => {
    await loadRides();
  }, [loadRides]);
  const rideActions = useRideActions(refreshData);

  const handleUrgent = useCallback(
    async (ride: RideSummary) => {
      // Planifier la course pour l'heure actuelle + 15 minutes
      const now = dayjs();
      const urgentTime = now.add(15, "minute");
      const localISO = urgentTime.format("YYYY-MM-DDTHH:mm:ss");

      setActionLoading(true);
      try {
        await scheduleRide(ride.id, { pickup_at: localISO });
        await loadRides();
        Alert.alert("Urgent", `La course a été planifiée pour ${urgentTime.format("HH:mm")} (dans 15 minutes).`);
      } catch (error: any) {
        const message =
          error?.response?.data?.error ??
          error?.message ??
          "Impossible de planifier la course en urgence.";
        Alert.alert("Erreur", message);
      } finally {
        setActionLoading(false);
      }
    },
    [loadRides]
  );

  const confirmSchedule = useCallback(async () => {
    if (!scheduleModal.rideId || !scheduleModal.scheduledTime) return;

    // Combiner la date actuelle (currentDate) avec l'heure sélectionnée
    // Utiliser format() au lieu de toISOString() pour préserver l'heure locale
    // Le backend utilise parse_local_naive qui attend un format ISO sans timezone
    const selectedTime = dayjs(scheduleModal.scheduledTime);
    const dateWithTime = dayjs(currentDate)
      .hour(selectedTime.hour())
      .minute(selectedTime.minute())
      .second(0);
    const localISO = dateWithTime.format("YYYY-MM-DDTHH:mm:ss");
    setActionLoading(true);
    try {
      await scheduleRide(scheduleModal.rideId, { pickup_at: localISO });
      await loadRides();
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de planifier l'horaire.";
      setErrorMessage(message);
    } finally {
      setActionLoading(false);
      setScheduleModal({ rideId: null, scheduledTime: null, label: undefined });
    }
  }, [currentDate, loadRides, scheduleModal]);

  const handleOpenDetails = useCallback((rideId: string) => {
    router.push({
      pathname: "/(enterprise)/ride-details",
      params: { rideId },
    } as any);
  }, []);

  const onSubmitSearch = useCallback(() => {
    loadRides();
  }, [loadRides]);

  const handleEdit = useCallback((ride: RideSummary) => {
    setSelectedRideForEdit(ride);
    setEditModalVisible(true);
  }, []);

  const handleSchedule = useCallback((ride: RideSummary) => {
    // Si la course a déjà une heure, l'utiliser, sinon utiliser l'heure actuelle
    let initialTime: Date | null = null;
    if (ride.time.pickup_at) {
      initialTime = dayjs(ride.time.pickup_at).toDate();
    } else {
      // Utiliser la date actuelle avec l'heure actuelle comme valeur par défaut
      initialTime = dayjs().toDate();
    }
    setScheduleModal({
      rideId: ride.id,
      scheduledTime: initialTime,
      label: ride.client.name,
    });
  }, []);

  const handleCancel = useCallback(
    (ride: RideSummary) => {
      // Ouvrir le modal de confirmation avec option de facturation
      setCancelModal({ ride, shouldBill: false });
    },
    []
  );

  const confirmCancel = useCallback(async () => {
    if (!cancelModal.ride) return;

    setActionLoading(true);
    try {
      const reason = cancelModal.shouldBill
        ? "Annulée depuis l'application mobile - À facturer"
        : "Annulée depuis l'application mobile - Non facturée";
      await cancelRide(cancelModal.ride.id, "manual", reason);
      await refreshData();
      setCancelModal({ ride: null, shouldBill: false });
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible d'annuler la course.";
      setErrorMessage(message);
    } finally {
      setActionLoading(false);
    }
  }, [cancelModal, refreshData]);

  return (
    <View style={styles.container}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={[styles.content, { paddingBottom: Math.max(32, tabBarHeight + 40) }]}
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
              {sortedRides.length} course{sortedRides.length !== 1 ? "s" : ""} planifiée
              {sortedRides.length !== 1 ? "s" : ""}
            </Text>
          </View>
          <TouchableOpacity
            style={styles.heroFab}
            onPress={() => setCreateModalVisible(true)}
            activeOpacity={0.85}
          >
            <Ionicons name="add" size={24} color="#FFFFFF" />
          </TouchableOpacity>
        </LinearGradient>

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
        ) : sortedRides.length === 0 ? (
          <View style={styles.emptyState}>
            <Ionicons
              name="leaf-outline"
              size={32}
              color={palette.emptyState}
            />
            <Text style={styles.emptyStateTitle}>Pas de course ici</Text>
            <Text style={styles.emptyStateText}>
              Ajuste la date ou change d'onglet pour consulter d'autres trajets.
            </Text>
          </View>
        ) : (
          <View style={{ gap: palette.listGap }}>
            {sortedRides.map((ride, index) => {
              const isLast = index === sortedRides.length - 1;
              const isExpanded = expandedRideId === ride.id;
              const needsBottomMargin = isLast && isExpanded;

              // ✅ Calculer l'heure de pickup (identique au dashboard)
              let pickupTime: string | null = null;
              if (ride.time.pickup_at) {
                const pickupMoment = dayjs(ride.time.pickup_at);
                // Si l'heure est à minuit (00:00), c'est probablement une heure non définie
                pickupTime =
                  pickupMoment.hour() === 0 && pickupMoment.minute() === 0
                    ? null
                    : pickupMoment.format("HH[h]mm");
              }

              // ✅ Normaliser le statut en minuscules pour éviter les problèmes de casse
              const normalizedStatus = ride.status ? String(ride.status).toLowerCase().trim() : undefined;

              // ✅ Calcul du retard : uniquement si la course est assignée, l'heure prévue est passée, ET la course n'est pas terminée
              let delayMinutes: number | null = null;
              // ✅ P0-1: Utiliser la fonction de normalisation
              const { isCompletedStatus } = require("@/utils/bookingStatus");
              const isCompleted = isCompletedStatus(ride.status);
              if (!isCompleted && ride.driver?.name && ride.time.pickup_at) {
                const scheduledTime = dayjs(ride.time.pickup_at);
                const now = dayjs();
                if (scheduledTime.isValid() && scheduledTime.isBefore(now)) {
                  delayMinutes = Math.max(0, now.diff(scheduledTime, "minute"));
                }
              }

              const priorityBadge =
                ride.client.priority === "HIGH"
                  ? { label: "Priorité", tone: "danger" as const }
                  : ride.client.priority === "LOW"
                    ? { label: "Basse", tone: "info" as const }
                    : undefined;

              return (
                <RideSnippetCard
                  key={ride.id}
                  ride={{
                    id: ride.id,
                    time: pickupTime ?? "",
                    showUndefinedIcon: pickupTime === null,
                    client: ride.client.name,
                    pickup: ride.route.pickup_address,
                    dropoff: ride.route.dropoff_address,
                    assignedTo: ride.driver?.name ?? null,
                    status: normalizedStatus as "unassigned" | "assigned" | "completed" | "return_completed" | "in_progress" | "en_route" | undefined,
                    delayMinutes: delayMinutes,
                    badges: priorityBadge ? [priorityBadge] : undefined,
                    onPress: () => handleOpenDetails(ride.id),
                    onQuickAction: isCompleted ? undefined : () => handleUrgent(ride),
                    onPrimaryAction: isCompleted ? undefined : () => rideActions.handleOpenAssignModal(ride),
                    footerActions: (
                      <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
                        {isCompleted ? (
                          // ✅ Si la course est terminée, afficher uniquement "Détails"
                          <TouchableOpacity
                            style={[styles.actionButtonGhost, { flexBasis: "30%" }]}
                            onPress={() => handleOpenDetails(ride.id)}
                          >
                            <Ionicons name="open-outline" size={16} color={palette.modalText} />
                            <Text style={[styles.actionButtonGhostText, { color: palette.modalText }]}>Détails</Text>
                          </TouchableOpacity>
                        ) : (
                          // ✅ Sinon, afficher toutes les actions
                          <>
                            <TouchableOpacity
                              style={[styles.actionButtonGhost, { flexBasis: "30%" }]}
                              onPress={() => handleEdit(ride)}
                            >
                              <Ionicons name="create-outline" size={16} color={palette.modalButton} />
                              <Text style={[styles.actionButtonGhostText, { color: palette.modalButton }]}>Éditer</Text>
                            </TouchableOpacity>
                            <TouchableOpacity
                              style={[styles.actionButtonGhost, { flexBasis: "30%" }]}
                              onPress={() => handleSchedule(ride)}
                            >
                              <Ionicons name="time-outline" size={16} color={palette.modalButton} />
                              <Text style={[styles.actionButtonGhostText, { color: palette.modalButton }]}>Planifier</Text>
                            </TouchableOpacity>
                            <TouchableOpacity
                              style={[styles.actionButtonGhost, { flexBasis: "30%" }]}
                              onPress={() => handleOpenDetails(ride.id)}
                            >
                              <Ionicons name="open-outline" size={16} color={palette.modalText} />
                              <Text style={[styles.actionButtonGhostText, { color: palette.modalText }]}>Détails</Text>
                            </TouchableOpacity>
                            {ride.status !== "cancelled" && (
                              <TouchableOpacity
                                style={[styles.actionButtonGhost, { flexBasis: "30%" }]}
                                onPress={() => handleCancel(ride)}
                              >
                                <Ionicons name="close-circle-outline" size={16} color="#EF4444" />
                                <Text style={[styles.actionButtonGhostText, { color: "#EF4444" }]}>Annuler</Text>
                              </TouchableOpacity>
                            )}
                          </>
                        )}
                      </View>
                    ),
                  }}
                  expanded={isExpanded}
                  onToggle={() => {
                    setExpandedRideId(expandedRideId === ride.id ? null : ride.id);
                  }}
                />
              );
            })}
          </View>
        )}

        {errorMessage && (
          <View style={styles.errorBanner}>
            <Ionicons name="alert-circle" size={18} color={palette.error} />
            <Text style={styles.errorText}>{errorMessage}</Text>
          </View>
        )}
      </ScrollView>

      {/* Modal d'édition */}
      <RideEditModal
        visible={editModalVisible}
        ride={selectedRideForEdit}
        onClose={() => {
          setEditModalVisible(false);
          setSelectedRideForEdit(null);
        }}
        onSuccess={refreshData}
      />

      {/* Modal de création */}
      <RideCreateModal
        visible={createModalVisible}
        onClose={() => setCreateModalVisible(false)}
        onSuccess={refreshData}
        onOpenClientCreate={() => setClientCreateModalVisible(true)}
        onClientCreated={async (client) => {
          // Le client créé sera automatiquement sélectionné dans RideCreateModal
          // via le callback onClientCreated
        }}
      />

      {/* Modal de création de client */}
      <ClientCreateModal
        visible={clientCreateModalVisible}
        onClose={() => setClientCreateModalVisible(false)}
        onSuccess={async (newClient) => {
          // Recharger les informations complètes du client depuis l'API
          try {
            const { searchClients } = await import("@/services/enterpriseDispatch");
            const clients = await searchClients(newClient.name);
            const fullClient = clients.find((c) => c.id === newClient.id);

            if (fullClient && createModalVisible) {
              // Le client sera automatiquement sélectionné dans RideCreateModal
              // via le callback onClientCreated qui sera implémenté
            }
          } catch (error) {
            console.error("[rides.tsx] Erreur lors du rechargement du client:", error);
          }
          setClientCreateModalVisible(false);
        }}
      />

      {/* Modal de confirmation d'annulation avec option de facturation */}
      <Modal visible={!!cancelModal.ride} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <View style={styles.modalHeader}>
              <View>
                <Text style={styles.modalTitle}>Annuler la course</Text>
                {cancelModal.ride && (
                  <Text style={styles.modalSubtitle}>
                    Course #{cancelModal.ride.id.slice(-4)} • {cancelModal.ride.client.name}
                  </Text>
                )}
              </View>
              <TouchableOpacity
                onPress={() => setCancelModal({ ride: null, shouldBill: false })}
                style={styles.closeButton}
              >
                <Ionicons name="close" size={24} color={palette.modalText} />
              </TouchableOpacity>
            </View>
            <View style={styles.modalContent}>
              <Text style={styles.modalText}>
                Voulez-vous vraiment annuler cette course ?
              </Text>
              <View style={styles.billingSection}>
                <Text style={styles.billingLabel}>Facturation</Text>
                <View style={styles.billingOptions}>
                  <TouchableOpacity
                    style={[
                      styles.billingOption,
                      !cancelModal.shouldBill && styles.billingOptionActive,
                    ]}
                    onPress={() => setCancelModal((prev) => ({ ...prev, shouldBill: false }))}
                  >
                    <View style={styles.radioButton}>
                      {!cancelModal.shouldBill && (
                        <View style={styles.radioButtonInner} />
                      )}
                    </View>
                    <Text style={styles.billingOptionText}>Non facturée</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={[
                      styles.billingOption,
                      cancelModal.shouldBill && styles.billingOptionActive,
                    ]}
                    onPress={() => setCancelModal((prev) => ({ ...prev, shouldBill: true }))}
                  >
                    <View style={styles.radioButton}>
                      {cancelModal.shouldBill && (
                        <View style={styles.radioButtonInner} />
                      )}
                    </View>
                    <Text style={styles.billingOptionText}>À facturer</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </View>
            <View style={styles.modalActions}>
              <TouchableOpacity
                style={styles.modalCancel}
                onPress={() => setCancelModal({ ride: null, shouldBill: false })}
                disabled={actionLoading}
              >
                <Text style={styles.modalCancelText}>Retour</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalConfirm, actionLoading && styles.modalConfirmDisabled]}
                onPress={confirmCancel}
                disabled={actionLoading}
              >
                {actionLoading ? (
                  <ActivityIndicator color="#FFFFFF" size="small" />
                ) : (
                  <Text style={styles.modalConfirmText}>Confirmer l'annulation</Text>
                )}
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Sélecteur d'heure pour planification (TimeDatePicker a son propre modal) */}
      {scheduleModal.rideId && (
        <View style={{ position: "absolute", opacity: 0, pointerEvents: "none", width: 0, height: 0 }}>
          <TimeDatePicker
            label="Heure de prise en charge"
            value={scheduleModal.scheduledTime}
            onChange={(date) => {
              if (date) {
                // Quand l'heure est sélectionnée, enregistrer automatiquement
                const selectedTime = dayjs(date);
                const dateWithTime = dayjs(currentDate)
                  .hour(selectedTime.hour())
                  .minute(selectedTime.minute())
                  .second(0);
                const localISO = dateWithTime.format("YYYY-MM-DDTHH:mm:ss");

                setActionLoading(true);
                scheduleRide(scheduleModal.rideId!, { pickup_at: localISO })
                  .then(() => {
                    loadRides();
                    setScheduleModal({ rideId: null, scheduledTime: null, label: undefined });
                  })
                  .catch((error: any) => {
                    const message =
                      error?.response?.data?.error ??
                      error?.message ??
                      "Impossible de planifier l'horaire.";
                    setErrorMessage(message);
                  })
                  .finally(() => {
                    setActionLoading(false);
                  });
              } else {
                // Si l'utilisateur annule, fermer le modal
                setScheduleModal({ rideId: null, scheduledTime: null, label: undefined });
              }
            }}
            mode="time"
            autoOpen={true}
          />
        </View>
      )}
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
    padding: 20,
    height: 150,
    flexDirection: "row",
    alignItems: "center",
    gap: 18,
    borderWidth: 1,
    borderColor: palette.heroBorder,
    shadowColor: "rgba(10,127,89,0.15)",
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 1,
    shadowRadius: 24,
    elevation: 8,
  },
  heroKicker: {
    color: palette.heroMeta,
    textTransform: "uppercase",
    letterSpacing: 3,
    fontSize: 11,
    marginBottom: 4,
  },
  heroTitle: {
    color: palette.heroText,
    fontSize: 24,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "capitalize",
  },
  heroSubtitle: {
    color: palette.heroMeta,
    fontSize: 13,
    marginTop: 4,
  },
  heroFab: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: palette.modalButton,
    alignItems: "center",
    justifyContent: "center",
    shadowColor: "rgba(10,127,89,0.3)",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 1,
    shadowRadius: 12,
    elevation: 6,
  },
  searchBar: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: palette.searchBackground,
    borderRadius: 18,
    paddingHorizontal: 18,
    paddingVertical: 4,
    borderWidth: 1,
    borderColor: palette.searchBorder,
    shadowColor: "rgba(15,54,43,0.06)",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 2,
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
    paddingHorizontal: 18,
    paddingVertical: 12,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: palette.tabBorder,
    backgroundColor: "#FFFFFF",
    shadowColor: "rgba(15,54,43,0.06)",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 1,
    shadowRadius: 4,
    elevation: 1,
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
  modalHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    padding: 24,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: palette.divider,
  },
  closeButton: {
    padding: 4,
  },
  modalContent: {
    padding: 24,
    gap: 20,
  },
  modalConfirmDisabled: {
    opacity: 0.5,
  },
  modalText: {
    color: palette.modalText,
    fontSize: 15,
    lineHeight: 22,
    marginBottom: 20,
  },
  billingSection: {
    gap: 12,
  },
  billingLabel: {
    color: palette.modalTitle,
    fontSize: 14,
    fontWeight: "600",
  },
  billingOptions: {
    gap: 10,
  },
  billingOption: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    padding: 12,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: palette.divider,
    backgroundColor: palette.background,
  },
  billingOptionActive: {
    borderColor: palette.modalButton,
    backgroundColor: "rgba(10,127,89,0.06)",
  },
  radioButton: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: palette.modalButton,
    alignItems: "center",
    justifyContent: "center",
  },
  radioButtonInner: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: palette.modalButton,
  },
  billingOptionText: {
    color: palette.modalTitle,
    fontSize: 15,
    fontWeight: "500",
  },
});
