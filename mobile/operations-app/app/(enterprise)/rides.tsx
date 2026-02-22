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
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import "dayjs/locale/fr";

import { useAuth } from "@/hooks/useAuth";
import { getAuthNotReadyDisplayMessage, isAuthNotReadyError } from "@/services/authGuards";
import { useEnterpriseContext } from "@/context/EnterpriseContext";
import { useThrottledCallback } from "@/hooks/useDebouncedCallback";
import { createShadow } from "@/styles/shadowStyles";
import { RideSnippetCard } from "@/components/enterprise/cards/RideSnippetCard";
import { RideEditModal } from "@/components/enterprise/rides/RideEditModal";
import { RideCreateModal } from "@/components/enterprise/rides/RideCreateModal";
import { ClientCreateModal } from "@/components/enterprise/rides/ClientCreateModal";
import { TimeDatePicker } from "@/components/enterprise/rides/TimeDatePicker";
import { TransferRideModal } from "@/components/enterprise/transfers/TransferRideModal";
import {
  getDispatchRides,
  scheduleRide,
  markRideUrgent,
  cancelRide,
} from "@/services/enterpriseDispatch";
import { isPickupSentinel } from "@/utils/urgentTime";
import { RideSummary } from "@/types/enterpriseDispatch";
import { useRideActions } from "@/hooks/useRideActions";
import { router } from "expo-router";
import { secureStorage } from "@/services/storage";
import { getLogger } from "@/utils/logger";

const log = getLogger("Rides");

/**
 * Décoder un JWT et extraire le timestamp d'expiration (exp)
 * @returns timestamp d'expiration en millisecondes, ou null si décodage échoue
 */
const getTokenExpiration = (token: string): number | null => {
  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return payload.exp ? payload.exp * 1000 : null; // Convertir en ms
  } catch (error) {
    log.warn("jwt decode error", { error });
    return null;
  }
};

dayjs.locale("fr");


const BRAND = "#00796B";
const TEXT = "#1E293B";
const TEXT_SEC = "#64748B";
const TEXT_MUTED = "#94A3B8";
const BORDER = "rgba(0,121,107,0.08)";
const BG = "#f4f7fc";
const CARD = "#FFFFFF";
const DANGER = "#dc3545";

export default function EnterpriseRidesScreen() {
  const { enterpriseSession, refreshEnterprise } = useAuth();
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
  const [urgentLoadingRideId, setUrgentLoadingRideId] = useState<string | null>(null);
  const [cancelModal, setCancelModal] = useState<{
    ride: RideSummary | null;
    shouldBill: boolean;
  }>({ ride: null, shouldBill: false });
  const [clientCreateModalVisible, setClientCreateModalVisible] = useState(false);
  const [transferModal, setTransferModal] = useState<{
    ride: RideSummary | null;
  }>({ ride: null });

  useEffect(() => {
    if (__DEV__) {
      log.info("client create modal visibility changed", { clientCreateModalVisible });
    }
  }, [clientCreateModalVisible]);

  const currentDate = useMemo(() => {
    return selectedDate ?? dayjs().format("YYYY-MM-DD");
  }, [selectedDate]);

  const formattedDay = useMemo(() => {
    const base = dayjs(currentDate);
    return base.format("dddd D MMMM");
  }, [currentDate]);

  // ✅ Trier les courses : d'abord celles avec horaire (plus proche à plus éloignée), puis celles à définir (sentinelle 00:00)
  const sortedRides = useMemo(() => {
    const withTime: RideSummary[] = [];
    const withoutTime: RideSummary[] = [];

    rides.forEach((ride) => {
      if (ride.time.pickup_at && !isPickupSentinel(ride.time.pickup_at)) {
        withTime.push(ride);
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
      log.info("loading rides for date", { currentDate });
      const response = await getDispatchRides({
        date: currentDate,
        query: search || undefined,
        page_size: 120,
      });
      log.info("rides loaded", { count: response.items.length });
      setRides(response.items);
    } catch (error: any) {
      // ✅ Invariant C: refresh_token absent → forcer login (pas "connexion en cours" infini)
      if (isAuthNotReadyError(error) && ["missing_refresh_token", "auth_ready_timeout"].includes((error as any).reason)) {
        router.replace("/(enterprise-auth)/login" as any);
        return;
      }
      const message =
        getAuthNotReadyDisplayMessage(error) ??
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de charger les courses.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  }, [currentDate, enterpriseSession, search]);

  // ✅ Version throttled de loadRides pour éviter les requêtes en doublon
  // Maximum 1 appel par seconde même si déclenchée plusieurs fois
  const throttledLoadRides = useThrottledCallback(loadRides, 1000);

  // Référence pour le polling automatique
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const appStateRef = useRef(AppState.currentState);

  // Charger les courses au montage et quand la date change
  useEffect(() => {
    throttledLoadRides();
  }, [throttledLoadRides]);

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
          log.info("polling refresh rides");
          throttledLoadRides();
        }
      }, 30000); // 30 secondes
    };

    // Démarrer le polling si l'app est active
    if (appStateRef.current === "active") {
      startPolling();
    }

    // Écouter les changements d'état de l'application
    const subscription = AppState.addEventListener("change", async (nextAppState) => {
      if (
        appStateRef.current.match(/inactive|background/) &&
        nextAppState === "active"
      ) {
        // ✅ CORRECTION #3 : Vérifier la validité du token au retour foreground
        // Évite les erreurs 401 dues à un token expiré pendant que l'app était en background
        log.info("app foreground, checking token");
        try {
          // ✅ CORRECTION : Utiliser SecureStore au lieu d'AsyncStorage
          const token = await secureStorage.getEnterpriseToken();
          if (token) {
            const expiresAt = getTokenExpiration(token);
            const now = Date.now();
            // Si le token expire dans moins de 5 minutes, le rafraîchir
            if (expiresAt && expiresAt - now < 5 * 60 * 1000) {
              log.info("token near expiry, refreshing");
              await refreshEnterprise();
            }
          }
        } catch (error) {
          log.warn("token check failed", { error });
        }

        log.info("app foreground, reloading rides");
        throttledLoadRides();
        startPolling();
      } else if (nextAppState.match(/inactive|background/)) {
        log.info("app background, stopping polling");
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
  }, [enterpriseSession, throttledLoadRides]);

  // Recharger les courses quand l'écran revient au focus
  useFocusEffect(
    useCallback(() => {
      if (enterpriseSession) {
        log.info("screen focused, reloading rides");
        throttledLoadRides();
      }
    }, [enterpriseSession, throttledLoadRides])
  );


  // ✅ Utiliser le hook partagé pour les actions sur les courses
  const refreshData = useCallback(async () => {
    await loadRides();
  }, [loadRides]);
  const rideActions = useRideActions(refreshData);

  const handleUrgent = useCallback(
    async (ride: RideSummary) => {
      // Garde-fou local : urgent uniquement si pickup_at sentinelle (00:00 ou absent)
      if (!isPickupSentinel(ride.time?.pickup_at)) {
        Alert.alert(
          "Course déjà planifiée",
          "Urgent disponible uniquement pour une course sans heure (00:00)."
        );
        return;
      }
      setActionLoading(true);
      setUrgentLoadingRideId(ride.id);
      try {
        await markRideUrgent(ride.id, { extra_delay_minutes: 15 });
        await loadRides();
        const urgentTime = dayjs().add(15, "minute");
        Alert.alert(
          "Urgent",
          `La course a été planifiée pour ${urgentTime.format("HH:mm")} (dans 15 minutes).`
        );
      } catch (error: any) {
        if (isAuthNotReadyError(error) && ["missing_refresh_token", "auth_ready_timeout"].includes((error as any).reason)) {
          router.replace("/(enterprise-auth)/login" as any);
          return;
        }
        if (error?.response?.status === 409) {
          Alert.alert("Course déjà planifiée", "Course déjà planifiée (urgent indisponible).");
          return;
        }
        const message =
          getAuthNotReadyDisplayMessage(error) ??
          error?.response?.data?.error ??
          error?.message ??
          "Impossible de planifier la course en urgence.";
        Alert.alert("Erreur", message);
      } finally {
        setActionLoading(false);
        setUrgentLoadingRideId(null);
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
      if (isAuthNotReadyError(error) && ["missing_refresh_token", "auth_ready_timeout"].includes((error as any).reason)) {
        router.replace("/(enterprise-auth)/login" as any);
        return;
      }
      const message =
        getAuthNotReadyDisplayMessage(error) ??
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
    // Heure initiale : si course déjà planifiée (pickup_at != sentinelle 00:00), utiliser cette heure ; sinon maintenant
    let initialTime: Date | null = null;
    if (ride.time.pickup_at && !isPickupSentinel(ride.time.pickup_at)) {
      initialTime = dayjs(ride.time.pickup_at).toDate();
    } else {
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

  const handleTransfer = useCallback(
    (ride: RideSummary) => {
      // Ouvrir le modal de transfert
      setTransferModal({ ride });
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
      if (isAuthNotReadyError(error) && ["missing_refresh_token", "auth_ready_timeout"].includes((error as any).reason)) {
        router.replace("/(enterprise-auth)/login" as any);
        return;
      }
      const message =
        getAuthNotReadyDisplayMessage(error) ??
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
        {/* Header */}
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <View style={styles.headerIconWrap}>
              <Ionicons name="car-outline" size={20} color={BRAND} />
            </View>
            <View>
              <Text style={styles.headerTitle}>Courses</Text>
              <Text style={styles.headerDate}>{formattedDay}</Text>
            </View>
          </View>
          <View style={styles.headerRight}>
            <View style={styles.countPill}>
              <Text style={styles.countPillText}>
                {sortedRides.length}
              </Text>
            </View>
            <TouchableOpacity
              style={styles.addButton}
              onPress={() => setCreateModalVisible(true)}
              activeOpacity={0.85}
            >
              <Ionicons name="add" size={20} color="#FFFFFF" />
            </TouchableOpacity>
          </View>
        </View>

        {/* Search */}
        <View style={styles.searchBar}>
          <Ionicons name="search-outline" size={16} color={TEXT_MUTED} />
          <TextInput
            value={search}
            onChangeText={setSearch}
            placeholder="Client, adresse ou chauffeur…"
            placeholderTextColor={TEXT_MUTED}
            style={styles.searchInput}
            returnKeyType="search"
            onSubmitEditing={onSubmitSearch}
          />
          {search.length > 0 && (
            <TouchableOpacity onPress={() => { setSearch(""); loadRides(); }} style={styles.searchClear}>
              <Ionicons name="close-circle" size={16} color={TEXT_MUTED} />
            </TouchableOpacity>
          )}
        </View>

        {loading ? (
          <View style={styles.loading}>
            <ActivityIndicator color={BRAND} />
            <Text style={styles.loadingText}>Chargement…</Text>
          </View>
        ) : sortedRides.length === 0 ? (
          <View style={styles.emptyState}>
            <View style={styles.emptyIcon}>
              <Ionicons name="car-outline" size={28} color={BRAND} />
            </View>
            <Text style={styles.emptyTitle}>Aucune course</Text>
            <Text style={styles.emptySubtitle}>
              Aucune course pour cette date. Utilisez le bouton + pour en créer une.
            </Text>
          </View>
        ) : (
          <View style={styles.ridesListContainer}>
            {sortedRides.map((ride, index) => {
              const isLast = index === sortedRides.length - 1;
              const isExpanded = expandedRideId === ride.id;
              const needsBottomMargin = isLast && isExpanded;

              // ✅ Calculer l'heure de pickup : 00:00 = sentinelle "non définie", ne jamais afficher comme heure
              let pickupTime: string | null = null;
              if (ride.time.pickup_at && !isPickupSentinel(ride.time.pickup_at)) {
                pickupTime = dayjs(ride.time.pickup_at).format("HH[h]mm");
              }

              // ✅ Normaliser le statut en minuscules pour éviter les problèmes de casse
              const normalizedStatus = ride.status ? String(ride.status).toLowerCase().trim() : undefined;

              // ✅ Calcul du retard : uniquement si heure réelle (pas sentinelle 00:00), assignée, passée, non terminée
              let delayMinutes: number | null = null;
              // ✅ P0-1: Utiliser la fonction de normalisation
              const { isCompletedStatus } = require("@/utils/bookingStatus");
              const isCompleted = isCompletedStatus(ride.status);
              if (!isCompleted && ride.driver?.name && ride.time.pickup_at && !isPickupSentinel(ride.time.pickup_at)) {
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
                <View key={ride.id} style={styles.rideCardWrapper}>
                  <RideSnippetCard
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
                      onQuickAction: undefined,
                      onPrimaryAction: undefined,
                      footerActions: (
                        <View style={styles.footerActions}>
                          {isCompleted ? (
                            <TouchableOpacity style={styles.actionChip} onPress={() => handleOpenDetails(ride.id)}>
                              <Ionicons name="open-outline" size={14} color={TEXT_SEC} />
                              <Text style={[styles.actionChipText, { color: TEXT_SEC }]}>Détails</Text>
                            </TouchableOpacity>
                          ) : (
                            <>
                              <TouchableOpacity style={styles.actionChip} onPress={() => handleEdit(ride)}>
                                <Ionicons name="create-outline" size={14} color={BRAND} />
                                <Text style={[styles.actionChipText, { color: BRAND }]}>Éditer</Text>
                              </TouchableOpacity>
                              <TouchableOpacity style={styles.actionChip} onPress={() => handleSchedule(ride)}>
                                <Ionicons name="time-outline" size={14} color={BRAND} />
                                <Text style={[styles.actionChipText, { color: BRAND }]}>Planifier</Text>
                              </TouchableOpacity>
                              {isPickupSentinel(ride.time?.pickup_at) && (
                                <TouchableOpacity
                                  style={styles.actionChip}
                                  onPress={() => handleUrgent(ride)}
                                  disabled={!!actionLoading || urgentLoadingRideId === ride.id}
                                >
                                  {urgentLoadingRideId === ride.id ? (
                                    <ActivityIndicator size="small" color="#F59E0B" />
                                  ) : (
                                    <Ionicons name="flash" size={14} color="#F59E0B" />
                                  )}
                                  <Text style={[styles.actionChipText, { color: "#F59E0B" }]}>
                                    {urgentLoadingRideId === ride.id ? "…" : "Urgent"}
                                  </Text>
                                </TouchableOpacity>
                              )}
                              <TouchableOpacity style={styles.actionChip} onPress={() => handleOpenDetails(ride.id)}>
                                <Ionicons name="open-outline" size={14} color={TEXT_SEC} />
                                <Text style={[styles.actionChipText, { color: TEXT_SEC }]}>Détails</Text>
                              </TouchableOpacity>
                              <TouchableOpacity style={styles.actionChip} onPress={() => handleTransfer(ride)}>
                                <Ionicons name="swap-horizontal-outline" size={14} color="#0EA5E9" />
                                <Text style={[styles.actionChipText, { color: "#0EA5E9" }]}>Transférer</Text>
                              </TouchableOpacity>
                              {ride.status !== "cancelled" && (
                                <TouchableOpacity style={styles.actionChip} onPress={() => handleCancel(ride)}>
                                  <Ionicons name="close-circle-outline" size={14} color={DANGER} />
                                  <Text style={[styles.actionChipText, { color: DANGER }]}>Annuler</Text>
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
                </View>
              );
            })}
          </View>
        )}

        {errorMessage && (
          <View style={styles.errorBanner}>
            <Ionicons name="alert-circle" size={16} color={DANGER} />
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
        onOpenClientCreate={() => {
          if (__DEV__) {
            log.info("opening client create modal");
          }
          setClientCreateModalVisible(true);
        }}
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
            log.error("reload client after create failed", { error });
          }
          setClientCreateModalVisible(false);
        }}
      />

      {/* Modal d'annulation – bottom sheet */}
      <Modal visible={!!cancelModal.ride} transparent animationType="slide">
        <Pressable style={styles.sheetOverlay} onPress={() => setCancelModal({ ride: null, shouldBill: false })}>
          <View />
        </Pressable>
        <View style={styles.sheet}>
          <View style={styles.sheetHandle} />
          <View style={styles.sheetHeader}>
            <View style={styles.sheetIconWrap}>
              <Ionicons name="close-circle-outline" size={20} color={DANGER} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.sheetTitle}>Annuler la course</Text>
              {cancelModal.ride && (
                <Text style={styles.sheetSubtitle}>
                  #{cancelModal.ride.id.slice(-4)} · {cancelModal.ride.client.name}
                </Text>
              )}
            </View>
          </View>

          <Text style={styles.sheetBody}>
            Cette action est irréversible. Souhaitez-vous facturer cette annulation ?
          </Text>

          <View style={styles.billingOptions}>
            <TouchableOpacity
              style={[styles.billingOption, !cancelModal.shouldBill && styles.billingOptionActive]}
              onPress={() => setCancelModal((prev) => ({ ...prev, shouldBill: false }))}
            >
              <View style={styles.radioOuter}>
                {!cancelModal.shouldBill && <View style={styles.radioInner} />}
              </View>
              <Text style={styles.billingOptionText}>Non facturée</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.billingOption, cancelModal.shouldBill && styles.billingOptionActive]}
              onPress={() => setCancelModal((prev) => ({ ...prev, shouldBill: true }))}
            >
              <View style={styles.radioOuter}>
                {cancelModal.shouldBill && <View style={styles.radioInner} />}
              </View>
              <Text style={styles.billingOptionText}>À facturer</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.sheetActions}>
            <TouchableOpacity
              style={styles.sheetCancelBtn}
              onPress={() => setCancelModal({ ride: null, shouldBill: false })}
              disabled={actionLoading}
            >
              <Text style={styles.sheetCancelText}>Retour</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.sheetConfirmBtn, actionLoading && { opacity: 0.5 }]}
              onPress={confirmCancel}
              disabled={actionLoading}
            >
              {actionLoading ? (
                <ActivityIndicator color="#FFF" size="small" />
              ) : (
                <Text style={styles.sheetConfirmText}>Confirmer</Text>
              )}
            </TouchableOpacity>
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
                    if (isAuthNotReadyError(error) && ["missing_refresh_token", "auth_ready_timeout"].includes((error as any).reason)) {
                      router.replace("/(enterprise-auth)/login" as any);
                      return;
                    }
                    const message =
                      getAuthNotReadyDisplayMessage(error) ??
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

      {transferModal.ride && (
        <TransferRideModal
          visible={!!transferModal.ride}
          onClose={() => setTransferModal({ ride: null })}
          ride={transferModal.ride}
          onSuccess={() => {
            setTransferModal({ ride: null });
            loadRides();
          }}
        />
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: BG,
  },
  scroll: {
    flex: 1,
  },
  content: {
    padding: 16,
    paddingBottom: 32,
  },

  /* ── Header ── */
  header: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: CARD,
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: BORDER,
    ...createShadow({ shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.04, shadowRadius: 8, elevation: 2 }),
  },
  headerLeft: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  headerIconWrap: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  headerTitle: {
    fontSize: 17,
    fontWeight: "700",
    color: TEXT,
  },
  headerDate: {
    fontSize: 12,
    color: TEXT_SEC,
    textTransform: "capitalize",
    marginTop: 1,
  },
  headerRight: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  countPill: {
    backgroundColor: "rgba(0,121,107,0.1)",
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  countPillText: {
    fontSize: 13,
    fontWeight: "700",
    color: BRAND,
  },
  addButton: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: BRAND,
    alignItems: "center",
    justifyContent: "center",
    ...createShadow({ shadowColor: BRAND, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 6, elevation: 3 }),
  },

  /* ── Search ── */
  searchBar: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: CARD,
    borderRadius: 12,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: BORDER,
    marginTop: 12,
    ...createShadow({ shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.03, shadowRadius: 4, elevation: 1 }),
  },
  searchInput: {
    flex: 1,
    color: TEXT,
    paddingVertical: 10,
    paddingHorizontal: 8,
    fontSize: 14,
  },
  searchClear: {
    padding: 6,
  },

  /* ── Loading ── */
  loading: {
    alignItems: "center",
    paddingVertical: 48,
  },
  loadingText: {
    color: TEXT_MUTED,
    fontSize: 13,
    marginTop: 10,
  },

  /* ── Empty ── */
  emptyState: {
    alignItems: "center",
    paddingVertical: 56,
  },
  emptyIcon: {
    width: 52,
    height: 52,
    borderRadius: 14,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 14,
  },
  emptyTitle: {
    color: TEXT,
    fontWeight: "600",
    fontSize: 15,
  },
  emptySubtitle: {
    color: TEXT_MUTED,
    fontSize: 13,
    textAlign: "center",
    paddingHorizontal: 32,
    marginTop: 6,
    lineHeight: 19,
  },

  /* ── List ── */
  ridesListContainer: {
    marginTop: 14,
  },
  rideCardWrapper: {
    marginBottom: 10,
  },

  /* ── Footer actions (chips) ── */
  footerActions: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 6,
    marginTop: 10,
  },
  actionChip: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 8,
    backgroundColor: BG,
    borderWidth: 1,
    borderColor: BORDER,
  },
  actionChipText: {
    fontSize: 12,
    fontWeight: "600",
  },

  /* ── Error ── */
  errorBanner: {
    flexDirection: "row",
    alignItems: "center",
    padding: 12,
    backgroundColor: "rgba(220,53,69,0.06)",
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(220,53,69,0.15)",
    marginTop: 14,
  },
  errorText: {
    color: DANGER,
    flex: 1,
    fontSize: 13,
    marginLeft: 8,
  },

  /* ── Bottom sheet (cancel) ── */
  sheetOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.35)",
  },
  sheet: {
    backgroundColor: CARD,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingHorizontal: 20,
    paddingBottom: 32,
    ...createShadow({ shadowColor: "#000", shadowOffset: { width: 0, height: -4 }, shadowOpacity: 0.12, shadowRadius: 24, elevation: 12 }),
  },
  sheetHandle: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: "#D1D5DB",
    alignSelf: "center",
    marginTop: 10,
    marginBottom: 16,
  },
  sheetHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
    marginBottom: 16,
  },
  sheetIconWrap: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: "rgba(220,53,69,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  sheetTitle: {
    fontSize: 17,
    fontWeight: "700",
    color: TEXT,
  },
  sheetSubtitle: {
    fontSize: 13,
    color: TEXT_SEC,
    marginTop: 2,
  },
  sheetBody: {
    fontSize: 14,
    color: TEXT_SEC,
    lineHeight: 20,
    marginBottom: 16,
  },
  billingOptions: {
    gap: 8,
    marginBottom: 20,
  },
  billingOption: {
    flexDirection: "row",
    alignItems: "center",
    padding: 14,
    borderRadius: 12,
    borderWidth: 1.5,
    borderColor: BORDER,
    backgroundColor: BG,
  },
  billingOptionActive: {
    borderColor: BRAND,
    backgroundColor: "rgba(0,121,107,0.05)",
  },
  radioOuter: {
    width: 20,
    height: 20,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: BRAND,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  radioInner: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: BRAND,
  },
  billingOptionText: {
    color: TEXT,
    fontSize: 14,
    fontWeight: "500",
  },
  sheetActions: {
    flexDirection: "row",
    gap: 10,
  },
  sheetCancelBtn: {
    flex: 1,
    alignItems: "center",
    paddingVertical: 13,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: BORDER,
    backgroundColor: CARD,
  },
  sheetCancelText: {
    color: TEXT_SEC,
    fontWeight: "600",
    fontSize: 14,
  },
  sheetConfirmBtn: {
    flex: 1,
    alignItems: "center",
    paddingVertical: 13,
    borderRadius: 12,
    backgroundColor: DANGER,
    ...createShadow({ shadowColor: DANGER, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 6, elevation: 3 }),
  },
  sheetConfirmText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: 14,
  },
});
