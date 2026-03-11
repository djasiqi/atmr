import React, { useCallback, useEffect, useMemo, useState, useRef } from "react";
import {
  Alert,
  Modal,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
  ActivityIndicator,
  AppState,
  InteractionManager,
} from "react-native";
import { useBottomTabBarHeight } from "@react-navigation/bottom-tabs";
import { useFocusEffect, useIsFocused } from "@react-navigation/native";
import * as Crypto from "expo-crypto";

import { router } from "expo-router";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import timezone from "dayjs/plugin/timezone";
import relativeTime from "dayjs/plugin/relativeTime";
import "dayjs/locale/fr";
import { Ionicons } from "@expo/vector-icons";

import { useAuth } from "@/hooks/useAuth";
import { getAuthNotReadyDisplayMessage } from "@/services/authGuards";
import { useThrottledCallback } from "@/hooks/useDebouncedCallback";
import { isCompletedStatus } from "@/utils/bookingStatus";
import { isPickupSentinel } from "@/utils/urgentTime";
import { createShadow } from "@/styles/shadowStyles";
import { sendIngestEvent } from "@/src/config/telemetry";
import {
  getDispatchRides,
  getDispatchStatus,
  runDispatch,
  fetchRealtimeDashboard,
  runOptimizer,
  resetAssignments,
  applyOpportunity,
  type RealtimeDashboardData,
} from "@/services/enterpriseDispatch";
import {
  DispatchStatus,
  RideSummary,
} from "@/types/enterpriseDispatch";
import { RideSnippetCard } from "@/components/enterprise/cards/RideSnippetCard";
import { EnterpriseDriversMap } from "@/components/enterprise/EnterpriseDriversMap";
import { useEnterpriseDriverTracking } from "@/hooks/useEnterpriseDriverTracking";
import { useEnterpriseContext } from "@/context/EnterpriseContext";
import { useRideActions } from "@/hooks/useRideActions";
import { AssignDriverModal } from "@/components/enterprise/AssignDriverModal";
import {
  fetchIncomingTransfers,
  acceptTransfer,
  rejectTransfer,
  type Transfer,
} from "@/services/partnershipService";
import { TransferCard } from "@/components/enterprise/transfers/TransferCard";
import { TransferRideModal } from "@/components/enterprise/transfers/TransferRideModal";
import { getLogger } from "@/utils/logger";
import { getEnterpriseAuthRecoveryMessage } from "@/services/enterpriseAuth";

const log = getLogger("EntDash");

const BRAND = "#00796B";

const enterprisePalette = {
  background: "#f4f7fc",
  card: "#FFFFFF",


  surface: "#FFFFFF",
  surfaceBorder: "rgba(0,121,107,0.08)",
  surfaceMuted: "#64748B",
  sectionSurface: "#FFFFFF",
  sectionBorder: "rgba(0,121,107,0.08)",

  alertSurface: "rgba(239,68,68,0.06)",
  alertBorder: "rgba(239,68,68,0.18)",
  alertText: "#1E293B",

  textStrong: "#1E293B",
  textSecondary: "#64748B",
  hintText: "#94A3B8",

  dispatchButton: BRAND,
  dispatchButtonDisabled: "rgba(0,121,107,0.4)",
  dispatchText: "#FFFFFF",

  cardOverlay: "#FFFFFF",
  cardBorder: "rgba(0,121,107,0.08)",

  modalOverlay: "rgba(30,41,59,0.6)",
  modalBackground: "#FFFFFF",
  modalBorder: "rgba(0,121,107,0.12)",
  modalTitle: "#1E293B",
  modalText: "#64748B",
  modalButton: BRAND,
  modalButtonText: "#FFFFFF",
  modalCancelText: "#64748B",
  loadingText: "#94A3B8",
};

dayjs.extend(utc);
dayjs.extend(timezone);
dayjs.extend(relativeTime);
dayjs.locale("fr");

type RideFilter = "all" | "pending" | "unassigned" | "assigned" | "completed" | "delayed";

export default function EnterpriseDashboardScreen() {
  const { enterpriseSession, refreshEnterprise, enterpriseLoading } = useAuth();
  const { selectedDate } = useEnterpriseContext();
  const tabBarHeight = useBottomTabBarHeight();
  const isFocused = useIsFocused();

  const dispatchMode =
    (enterpriseSession?.company.dispatchMode as
      | "manual"
      | "semi_auto"
      | "fully_auto"
      | undefined) ?? "semi_auto";

  const companyName = enterpriseSession?.company.name ?? "Entreprise";

  const [status, setStatus] = useState<DispatchStatus | null>(null);
  const [urgentRides, setUrgentRides] = useState<RideSummary[]>([]);
  const [unassignedRides, setUnassignedRides] = useState<RideSummary[]>([]);
  const [allRides, setAllRides] = useState<RideSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [authRecoveryMessage, setAuthRecoveryMessage] = useState<string | null>(null);
  const [dispatching, setDispatching] = useState(false);

  // ✅ 3.4.2: État pour dashboard temps réel
  const [realtimeDashboard, setRealtimeDashboard] = useState<RealtimeDashboardData | null>(null);

  // ✅ États pour les nouvelles fonctionnalités
  const [applyingOpportunity, setApplyingOpportunity] = useState<number | null>(null);
  const [optimizerRunning, setOptimizerRunning] = useState(false);
  const [resetting, setResetting] = useState(false);

  // ✅ État pour l'accordéon des courses (une seule ouverte à la fois)
  const [expandedRideId, setExpandedRideId] = useState<string | null>(null);

  // ✅ État pour le filtre des courses
  const [rideFilter, setRideFilter] = useState<RideFilter>("all");

  // ✅ État pour les transferts entrants
  const [incomingTransfers, setIncomingTransfers] = useState<Transfer[]>([]);
  const [transferModalVisible, setTransferModalVisible] = useState(false);
  const [selectedRideForTransfer, setSelectedRideForTransfer] = useState<RideSummary | null>(null);

  const handleOpenTransferModal = useCallback((ride: RideSummary) => {
    setSelectedRideForTransfer(ride);
    setTransferModalVisible(true);
  }, []);

  const handleCloseTransferModal = useCallback(() => {
    setTransferModalVisible(false);
    setSelectedRideForTransfer(null);
  }, []);
  const [loadingTransfers, setLoadingTransfers] = useState(false);

  const { markers: driverMarkers, refreshLocations } =
    useEnterpriseDriverTracking();

  const currentDate = useMemo(() => selectedDate, [selectedDate]);

  // ✅ Utiliser le hook partagé pour les actions sur les courses
  // Note: loadData est défini plus bas, on passe une fonction qui l'appelle
  const rideActions = useRideActions(() => loadData());

  const formattedDay = useMemo(() => {
    const base = dayjs(selectedDate);
    const localized = dayjs.isDayjs(base)
      ? base
      : dayjs(selectedDate, "YYYY-MM-DD");
    const zoned = localized.tz ? localized.tz("Europe/Zurich") : localized;
    return zoned.format("dddd D MMMM");
  }, [selectedDate]);

  // ✅ Charger les transferts entrants (en attente)
  const loadIncomingTransfers = useCallback(async () => {
    try {
      setLoadingTransfers(true);
      const transfers = await fetchIncomingTransfers();
      setIncomingTransfers(transfers);
    } catch (error: any) {
      log.error("load transfers failed", { error });
    } finally {
      setLoadingTransfers(false);
    }
  }, []);

  const loadData = useCallback(async () => {
    if (!enterpriseSession) return;
    setLoading(true);
    setErrorMessage(null);
    try {
      const [
        statusResponse,
        urgentResponse,
        unassignedResponse,
        allResponse,
        realtimeResponse,
      ] = await Promise.all([
        getDispatchStatus(currentDate), // ✅ Passer la date pour obtenir dispatch_run
        getDispatchRides({
          date: currentDate,
          status: "urgent",
          page_size: 5,
        }),
        getDispatchRides({
          date: currentDate,
          status: "unassigned",
          page_size: 3,
        }),
        getDispatchRides({
          date: currentDate,
          page_size: 120,
        }),
        // ✅ 3.4.2: Charger dashboard temps réel
        fetchRealtimeDashboard(currentDate).catch((err) => {
          log.warn("realtime dashboard load failed", { error: err });
          return null;
        }),
      ]);
      setStatus(statusResponse);
      // ✅ Mettre à jour le statut de l'optimiseur
      setOptimizerRunning(statusResponse?.optimizer?.running ?? statusResponse?.optimizer?.active ?? false);
      setUrgentRides(urgentResponse.items);
      setUnassignedRides(unassignedResponse.items);
      setAllRides(allResponse.items);
      if (realtimeResponse) {
        setRealtimeDashboard(realtimeResponse);
      }
      refreshLocations();

      // ✅ Charger les transferts entrants en arrière-plan
      loadIncomingTransfers();
    } catch (error: any) {
      const authRecovery = getEnterpriseAuthRecoveryMessage(error);
      if (authRecovery) {
        setAuthRecoveryMessage(authRecovery);
        return;
      }
      const message =
        getAuthNotReadyDisplayMessage(error) ??
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de charger les dernières informations dispatch.";
      setErrorMessage(message);
    } finally {
      setLoading(false);
    }
  }, [currentDate, enterpriseSession, refreshLocations, loadIncomingTransfers]);

  // ✅ Version throttled de loadData pour éviter les requêtes en doublon
  // Maximum 1 appel par seconde même si déclenchée plusieurs fois
  const throttledLoadData = useThrottledCallback(loadData, 1000);

  // Référence pour le polling automatique
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const appStateRef = useRef(AppState.currentState);
  const lastFocusReloadAtRef = useRef(0);

  // Polling automatique : récupérer les données toutes les 30 secondes quand l'app est active
  useEffect(() => {
    if (!enterpriseSession || !isFocused) {
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
          log.info("polling refresh data");
          throttledLoadData();
        }
      }, 30000); // 30 secondes
    };

    // Démarrer le polling si l'app est active
    if (appStateRef.current === "active") {
      startPolling();
    }

    // Écouter les changements d'état de l'application
    const subscription = AppState.addEventListener("change", (nextAppState) => {
      const wasBackground = appStateRef.current.match(/inactive|background/);
      appStateRef.current = nextAppState;
      if (wasBackground && nextAppState === "active") {
        InteractionManager.runAfterInteractions(() => {
          setTimeout(() => {
            log.info("app foreground, reloading data");
            throttledLoadData();
            startPolling();
          }, 150);
        });
      } else if (nextAppState.match(/inactive|background/)) {
        InteractionManager.runAfterInteractions(() => {
          setTimeout(() => {
            log.info("app background, stopping polling");
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
              pollingIntervalRef.current = null;
            }
          }, 0);
        });
      }
    });

    // Cleanup
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
      subscription.remove();
    };
  }, [enterpriseSession, throttledLoadData, currentDate, isFocused]);

  // Recharger les données quand l'écran revient au focus
  useFocusEffect(
    useCallback(() => {
      if (enterpriseSession) {
        const now = Date.now();
        if (now - lastFocusReloadAtRef.current < 8000) {
          return;
        }
        lastFocusReloadAtRef.current = now;
        log.info("screen focused, reloading data");
        throttledLoadData();
      }
    }, [enterpriseSession, throttledLoadData])
  );

  // ✅ Actions sur les courses : utilisent le hook partagé
  const handleUrgentDelay = useCallback(
    (rideId: string) => {
      rideActions.handleMarkUrgent(rideId, 15);
    },
    [rideActions]
  );

  const manualStats = useMemo(() => {
    const total = allRides.length;
    const unassigned = allRides.filter((ride) => {
      const status = ride.status ? String(ride.status).toLowerCase().trim() : undefined;
      return status === "unassigned";
    });
    const assigned = allRides.filter((ride) => {
      const status = ride.status ? String(ride.status).toLowerCase().trim() : undefined;
      return status === "assigned";
    });
    // ✅ P0-1: Utiliser la fonction de normalisation
    const { isCompletedStatus } = require("@/utils/bookingStatus");
    const completed = allRides.filter((ride) => {
      return isCompletedStatus(ride.status);
    });
    return {
      total,
      unassignedCount: unassigned.length,
      assignedCount: assigned.length,
      completedCount: completed.length,
      assignmentRate:
        total > 0 ? Math.round((assigned.length / total) * 100) : 0,
    };
  }, [allRides]);


  const sortedManualRides = useMemo(() => {
    const withTime: RideSummary[] = [];
    const withoutTime: RideSummary[] = [];

    allRides.forEach((ride) => {
      if (ride.time.pickup_at) {
        const moment = dayjs(ride.time.pickup_at);
        if (moment.hour() === 0 && moment.minute() === 0) {
          withoutTime.push(ride);
        } else {
          withTime.push(ride);
        }
      } else {
        withoutTime.push(ride);
      }
    });

    withTime.sort(
      (a, b) =>
        dayjs(a.time.pickup_at!).valueOf() - dayjs(b.time.pickup_at!).valueOf()
    );

    return [...withTime, ...withoutTime];
  }, [allRides]);

  // ✅ Calcul des compteurs et filtrage des courses
  const { filteredRides, filterCounts } = useMemo(() => {
    const counts = {
      all: sortedManualRides.length,
      pending: 0,
      unassigned: 0,
      assigned: 0,
      completed: 0,
      delayed: 0,
    };

    // #region agent log
    try {
      sendIngestEvent({ location: 'dashboard.tsx:filteredRides', message: 'All rides before filtering', data: { total_rides: sortedManualRides.length, rides_sample: sortedManualRides.slice(0, 3).map(r => ({ id: r.id, status: r.status, status_type: typeof r.status, driver_name: r.driver?.name })) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run2', hypothesisId: 'H1' });
    } catch { }
    // #endregion

    sortedManualRides.forEach((ride) => {
      const status = ride.status ? String(ride.status).toLowerCase().trim() : undefined;
      // ✅ Compter les courses PENDING (transférées en attente)
      if (status === "pending") {
        counts.pending++;
        // #region agent log
        try {
          sendIngestEvent({ location: 'dashboard.tsx:pendingCount', message: 'Found PENDING ride', data: { ride_id: ride.id, status: ride.status, status_normalized: status }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run2', hypothesisId: 'H1' });
        } catch { }
        // #endregion
      }
      // ✅ Compter aussi les courses avec transfert PENDING (en attente d'acceptation)
      if (ride.transfer?.status === "PENDING") {
        counts.pending++;
      }
      if (status === "unassigned" || !ride.driver?.name) {
        counts.unassigned++;
      }
      if (status === "assigned" && !!ride.driver?.name) {
        counts.assigned++;
      }
      // ✅ P0-1: Utiliser la fonction de normalisation
      if (isCompletedStatus(status)) {
        counts.completed++;
      }
      // Vérifier si en retard (uniquement si la course n'est pas terminée)
      const isCompleted = isCompletedStatus(status);
      if (!isCompleted && ride.driver?.name && ride.time.pickup_at) {
        const scheduledTime = dayjs(ride.time.pickup_at);
        const now = dayjs();
        if (scheduledTime.isValid() && scheduledTime.isBefore(now)) {
          const delayMinutes = Math.max(0, now.diff(scheduledTime, "minute"));
          if (delayMinutes > 0) {
            counts.delayed++;
          }
        }
      }
    });

    // Filtrer selon le filtre sélectionné
    let filtered = [...sortedManualRides];
    if (rideFilter === "pending") {
      // #region agent log
      try {
        sendIngestEvent({ location: 'dashboard.tsx:pendingFilter', message: 'Applying PENDING filter', data: { total_before_filter: sortedManualRides.length, pending_count: counts.pending }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run2', hypothesisId: 'H3' });
      } catch { }
      // #endregion
      // ✅ Filtre pour afficher les courses PENDING ou avec transfert PENDING
      filtered = filtered.filter((r) => {
        const status = r.status ? String(r.status).toLowerCase().trim() : undefined;
        const hasPendingTransfer = r.transfer?.status === "PENDING";
        return status === "pending" || hasPendingTransfer;
      });
    } else if (rideFilter === "unassigned") {
      filtered = filtered.filter((r) => {
        const status = r.status ? String(r.status).toLowerCase().trim() : undefined;
        return status === "unassigned" || !r.driver?.name;
      });
    } else if (rideFilter === "assigned") {
      filtered = filtered.filter((r) => {
        const status = r.status ? String(r.status).toLowerCase().trim() : undefined;
        return status === "assigned" && !!r.driver?.name;
      });
    } else if (rideFilter === "completed") {
      // ✅ P0-1: Utiliser la fonction de normalisation
      filtered = filtered.filter((r) => {
        return isCompletedStatus(r.status);
      });
    } else if (rideFilter === "delayed") {
      filtered = filtered.filter((ride) => {
        if (!ride.driver?.name || !ride.time.pickup_at) return false;
        const scheduledTime = dayjs(ride.time.pickup_at);
        const now = dayjs();
        if (scheduledTime.isValid() && scheduledTime.isBefore(now)) {
          const delayMinutes = Math.max(0, now.diff(scheduledTime, "minute"));
          return delayMinutes > 0;
        }
        return false;
      });
    }
    // "all" : pas de filtre

    // #region agent log
    try {
      sendIngestEvent({ location: 'dashboard.tsx:filterResult', message: 'Filter result', data: { filter: rideFilter, filtered_count: filtered.length, counts: counts }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run2', hypothesisId: 'H3' });
    } catch { }
    // #endregion

    return { filteredRides: filtered, filterCounts: counts };
  }, [sortedManualRides, rideFilter]);

  const manualRidesList = (
    <View style={styles.manualListSection}>
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>Courses du jour</Text>
        {filteredRides.length > 0 && (
          <Text style={styles.sectionCount}>
            {filteredRides.length} course{filteredRides.length !== 1 ? "s" : ""}
          </Text>
        )}
      </View>

      <View style={styles.filtersRow}>
        {([
          { value: "all" as const, icon: "grid-outline" },
          { value: "pending" as const, icon: "hourglass-outline" },
          { value: "unassigned" as const, icon: "alert-circle-outline" },
          { value: "assigned" as const, icon: "checkmark-circle-outline" },
          { value: "completed" as const, icon: "checkmark-done-outline" },
          { value: "delayed" as const, icon: "alarm-outline" },
        ] as const).map((filter) => {
          const isActive = rideFilter === filter.value;
          const count = filterCounts[filter.value];
          return (
            <TouchableOpacity
              key={filter.value}
              style={[styles.filterIcon, isActive && styles.filterIconActive]}
              onPress={() => setRideFilter(filter.value)}
              activeOpacity={0.7}
            >
              <Ionicons
                name={filter.icon as any}
                size={16}
                color={isActive ? "#FFFFFF" : enterprisePalette.textSecondary}
              />
              {count > 0 && (
                <View style={[styles.filterBadge, isActive && styles.filterBadgeActive]}>
                  <Text style={[styles.filterBadgeText, isActive && styles.filterBadgeTextActive]}>
                    {count}
                  </Text>
                </View>
              )}
            </TouchableOpacity>
          );
        })}
      </View>

      {filteredRides.length === 0 ? (
        <View style={styles.emptyState}>
          <View style={styles.emptyStateIcon}>
            <Ionicons
              name={rideFilter === "delayed" ? "alarm-outline" : "car-outline"}
              size={28}
              color={enterprisePalette.hintText}
            />
          </View>
          <Text style={styles.emptyStateTitle}>
            {rideFilter === "all"
              ? "Aucune course planifiée"
              : rideFilter === "pending"
                ? "Aucune course en attente"
                : rideFilter === "unassigned"
                  ? "Aucune course non assignée"
                  : rideFilter === "assigned"
                    ? "Aucune course assignée"
                    : rideFilter === "completed"
                      ? "Aucune course terminée"
                      : "Aucune course en retard"}
          </Text>
          <Text style={styles.emptyStateSubtitle}>
            {rideFilter === "all"
              ? "Les courses apparaîtront ici dès qu'elles seront créées."
              : "Changez de filtre pour voir d'autres courses."}
          </Text>
        </View>
      ) : (
        filteredRides.map((ride) => {
          let pickupTime: string | null = null;
          if (ride.time.pickup_at) {
            const pickupMoment = dayjs(ride.time.pickup_at);
            // Si l'heure est à minuit (00:00), c'est probablement une heure non définie
            // Afficher null pour montrer l'icône d'horloge au lieu de "00h00"
            pickupTime =
              pickupMoment.hour() === 0 && pickupMoment.minute() === 0
                ? null
                : pickupMoment.format("HH[h]mm");
          }

          // ✅ P0-1: Normaliser le statut pour éviter les problèmes de casse
          const normalizedStatus = ride.status ? String(ride.status).toLowerCase().trim() : undefined;

          // ✅ Calcul du retard : uniquement si la course est assignée, l'heure prévue est passée, ET la course n'est pas terminée
          let delayMinutes: number | null = null;
          const isCompleted = isCompletedStatus(ride.status);

          // #region agent log
          try {
            sendIngestEvent({ location: 'dashboard.tsx:611', message: 'Ride processing', data: { rideId: ride.id, status: ride.status, normalizedStatus: normalizedStatus, isCompleted: isCompleted, hasTransfer: !!ride.transfer, transferStatus: ride.transfer?.status, isReceiver: ride.transfer?.is_receiver }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run3', hypothesisId: 'H1-H2-H3' });
          } catch { }
          // #endregion

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

          // ✅ Badge de transfert si la course est transférée
          const transferBadge = ride.transfer
            ? {
              label: ride.transfer.is_receiver
                ? `🔄 Reçu de ${ride.transfer.partner_company_name || "partenaire"}`
                : `🔄 Envoyé à ${ride.transfer.partner_company_name || "partenaire"}`,
              tone: "info" as const,
            }
            : undefined;

          // ✅ Déterminer si l'entreprise peut gérer cette course
          // L'entreprise peut gérer si :
          // - Pas de transfert, OU
          // - Transfert PENDING et on est sender/receiver, OU
          // - Transfert ACCEPTED et on est le receveur (on gère maintenant la course)
          const canManageRide = !ride.transfer
            || (ride.transfer.status === "PENDING" && (ride.transfer.is_receiver || ride.transfer.is_sender))
            || (ride.transfer.status === "ACCEPTED" && ride.transfer.is_receiver);
          const isPendingTransferReceiver = ride.transfer?.status === "PENDING" && ride.transfer.is_receiver;
          const isPendingTransferSender = ride.transfer?.status === "PENDING" && ride.transfer.is_sender;

          // #region agent log - Debug transfer roles
          if (ride.transfer?.status === "PENDING") {
            try {
              sendIngestEvent({ location: 'dashboard.tsx:663', message: 'Transfer roles check', data: { rideId: ride.id, transferId: ride.transfer?.id, transferStatus: ride.transfer?.status, is_sender: ride.transfer?.is_sender, is_receiver: ride.transfer?.is_receiver, isPendingTransferSender: isPendingTransferSender, isPendingTransferReceiver: isPendingTransferReceiver }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run13-roles', hypothesisId: 'ROLES' });
            } catch { }
          }
          // #endregion

          // ✅ Logique d'affichage des boutons :
          // - Receveur : Accepter (droite) + Refuser (gauche) + Transférer (footer)
          // - Émetteur : Annuler (footer)

          const badges = [priorityBadge, transferBadge].filter(Boolean) as Array<{ label: string; tone: "danger" | "info" | "warning" | "success" }>;

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
                status: normalizedStatus as "unassigned" | "assigned" | "completed" | "return_completed" | "in_progress" | "en_route" | "pending" | undefined,
                delayMinutes: delayMinutes,
                badges: badges.length > 0 ? badges : undefined,
                onPress: () =>
                  router.push({
                    pathname: "/(enterprise)/ride-details",
                    params: { rideId: ride.id },
                  } as any),
                // ✅ RECEVEUR : Accepter/Refuser à droite + Transférer dans footer
                // ✅ ÉMETTEUR : Annuler dans le footer
                // ✅ Autres courses : Urgence/Assigner à droite
                onQuickAction: (() => {
                  const value = isCompleted
                    ? undefined
                    : isPendingTransferReceiver && ride.transfer?.id
                      ? () => rideActions.handleRejectTransfer(ride.transfer!.id) // ❌ Refuser (receveur)
                      : !canManageRide || !isPickupSentinel(ride.time?.pickup_at)
                        ? undefined
                        : () => handleUrgentDelay(ride.id); // 🚨 Marquer urgent (sentinel 00:00 uniquement)
                  // #region agent log
                  try {
                    sendIngestEvent({ location: 'dashboard.tsx:697-onQuickAction', message: 'onQuickAction computed', data: { rideId: ride.id, transferId: ride.transfer?.id, hasValue: !!value, isCompleted: isCompleted, canManageRide: canManageRide, isPendingTransferReceiver: isPendingTransferReceiver }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run12-final', hypothesisId: 'H1-FIX' });
                  } catch { }
                  // #endregion
                  return value;
                })(),
                onPrimaryAction: (() => {
                  const value = isCompleted
                    ? undefined
                    : isPendingTransferReceiver && ride.transfer?.id
                      ? () => rideActions.handleAcceptTransfer(ride.transfer!.id) // ✅ Accepter (receveur)
                      : !canManageRide
                        ? undefined
                        : () => rideActions.handleOpenAssignModal(ride); // 👤 Assigner un chauffeur
                  // #region agent log
                  try {
                    sendIngestEvent({ location: 'dashboard.tsx:708-onPrimaryAction', message: 'onPrimaryAction computed', data: { rideId: ride.id, transferId: ride.transfer?.id, hasValue: !!value, isCompleted: isCompleted, canManageRide: canManageRide, isPendingTransferReceiver: isPendingTransferReceiver }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run12-final', hypothesisId: 'H1-FIX' });
                  } catch { }
                  // #endregion
                  return value;
                })(),
                // ✅ Icônes et couleurs pour le RECEVEUR uniquement
                quickIcon: isPendingTransferReceiver ? "close-circle-outline" : undefined,
                primaryIcon: isPendingTransferReceiver ? "checkmark-circle-outline" : undefined,
                quickIconColor: isPendingTransferReceiver ? "#EF4444" : undefined, // Rouge pour Refuser
                primaryIconColor: isPendingTransferReceiver ? "#0A7F59" : undefined, // Vert entreprise pour Accepter
                // ✅ Footer : Transférer (receveur) OU Annuler (émetteur)
                footerActions: isPendingTransferReceiver ? (
                  <TouchableOpacity
                    style={styles.transferButtonFooter}
                    onPress={() => handleOpenTransferModal(ride)}
                  >
                    <Ionicons name="git-network-outline" size={16} color="#0A7F59" />
                    <Text style={styles.transferButtonText}>Transférer</Text>
                  </TouchableOpacity>
                ) : isPendingTransferSender && ride.transfer?.id ? (
                  <TouchableOpacity
                    style={styles.cancelButtonFooter}
                    onPress={() => rideActions.handleRejectTransfer(ride.transfer!.id)}
                  >
                    <Ionicons name="close-circle-outline" size={16} color="#EF4444" />
                    <Text style={styles.cancelButtonText}>Annuler le transfert</Text>
                  </TouchableOpacity>
                ) : undefined,
              }}
              // #region agent log
              {...(() => { try { sendIngestEvent({ location: 'dashboard.tsx:679', message: 'RideSnippetCard props', data: { rideId: ride.id, hasOnQuickAction: !!(isCompleted || !canManageRide ? undefined : isPendingTransferReceiver ? () => { } : () => { }), hasOnPrimaryAction: !!(isCompleted || !canManageRide ? undefined : isPendingTransferReceiver ? () => { } : () => { }), quickIcon: isPendingTransferReceiver ? "close-circle-outline" : undefined, primaryIcon: isPendingTransferReceiver ? "checkmark-circle-outline" : undefined }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run3', hypothesisId: 'H4' }); } catch { } return {}; })()}
              // #endregion
              expanded={expandedRideId === ride.id}
              onToggle={() => {
                setExpandedRideId(expandedRideId === ride.id ? null : ride.id);
              }}
            />
          );
        })
      )}
    </View>
  );

  const manualMapSection = (
    <View style={styles.manualMapSection}>
      <EnterpriseDriversMap
        markers={driverMarkers}
        fallbackMessage="Activez le tracking pour visualiser les chauffeurs en temps réel."
      />
    </View>
  );

  const handleRunDispatch = useCallback(() => {
    if (dispatching) return;
    Alert.alert(
      "Lancer un dispatch ?",
      `Confirme le lancement d'une optimisation pour ${dayjs(
        currentDate
      ).format("dddd D MMMM")}.`,
      [
        { text: "Annuler", style: "cancel" },
        {
          text: "Lancer",
          style: "default",
          onPress: async () => {
            setDispatching(true);
            try {
              const response = await runDispatch(currentDate);
              const confirmation = (() => {
                if (!response?.message) {
                  return `Dispatch lancé pour le ${dayjs(currentDate).format(
                    "DD/MM/YYYY"
                  )}`;
                }
                return response.message
                  .replace("Dispatch lancé pour", "Dispatch lancé pour le")
                  .replace(/(\d{4})-(\d{2})-(\d{2})/, "$3/$2/$1");
              })();
              Alert.alert("Dispatch lancé", confirmation);
              await loadData();
            } catch (error: any) {
              const message =
                error?.response?.data?.error ??
                error?.message ??
                "Impossible de lancer le dispatch. Réessaie plus tard.";
              Alert.alert("Erreur dispatch", message);
            } finally {
              setDispatching(false);
            }
          },
        },
      ]
    );
  }, [currentDate, dispatching, formattedDay, loadData]);

  // ✅ Appliquer une opportunité d'optimisation
  const handleApplyOpportunity = useCallback(
    async (opportunity: RealtimeDashboardData["opportunities"][0]) => {
      if (applyingOpportunity === opportunity.assignment_id) return;
      setApplyingOpportunity(opportunity.assignment_id);
      try {
        await applyOpportunity(opportunity);
        Alert.alert(
          "Opportunité appliquée",
          "La réassignation a été effectuée avec succès."
        );
        await loadData();
      } catch (error: any) {
        const message =
          error?.response?.data?.error ??
          error?.response?.data?.message ??
          error?.message ??
          "Impossible d'appliquer l'opportunité.";
        Alert.alert("Erreur", message);
      } finally {
        setApplyingOpportunity(null);
      }
    },
    [applyingOpportunity, loadData]
  );

  // ✅ Contrôler l'optimiseur temps réel
  const handleToggleOptimizer = useCallback(async () => {
    if (optimizerRunning) {
      // Arrêter l'optimiseur (nécessite un endpoint stop si disponible)
      Alert.alert(
        "Arrêter l'optimiseur",
        "L'optimiseur temps réel sera arrêté. Les opportunités ne seront plus détectées automatiquement.",
        [
          { text: "Annuler", style: "cancel" },
          {
            text: "Arrêter",
            style: "destructive",
            onPress: async () => {
              // Note: Il faudrait un endpoint pour arrêter l'optimiseur
              // Pour l'instant, on ne peut que le démarrer
              setOptimizerRunning(false);
            },
          },
        ]
      );
    } else {
      // Démarrer l'optimiseur
      try {
        await runOptimizer(currentDate);
        setOptimizerRunning(true);
        Alert.alert("Optimiseur démarré", "L'optimiseur temps réel est maintenant actif.");
        await loadData();
      } catch (error: any) {
        const message =
          error?.response?.data?.error ??
          error?.message ??
          "Impossible de démarrer l'optimiseur.";
        Alert.alert("Erreur", message);
      }
    }
  }, [optimizerRunning, currentDate, loadData]);

  // ✅ Réinitialiser les assignations
  const handleResetAssignments = useCallback(() => {
    Alert.alert(
      "Réinitialiser les assignations ?",
      `Toutes les assignations pour le ${dayjs(currentDate).format(
        "dddd D MMMM"
      )} seront supprimées. Cette action est irréversible.`,
      [
        { text: "Annuler", style: "cancel" },
        {
          text: "Réinitialiser",
          style: "destructive",
          onPress: async () => {
            setResetting(true);
            try {
              await resetAssignments(currentDate);
              Alert.alert(
                "Assignations réinitialisées",
                "Toutes les assignations ont été supprimées."
              );
              await loadData();
            } catch (error: any) {
              const message =
                error?.response?.data?.error ??
                error?.message ??
                "Impossible de réinitialiser les assignations.";
              Alert.alert("Erreur", message);
            } finally {
              setResetting(false);
            }
          },
        },
      ]
    );
  }, [currentDate, loadData]);

  // ✅ Accepter un transfert
  const handleAcceptTransfer = useCallback(async (transfer: Transfer) => {
    Alert.alert(
      "Accepter le transfert",
      `Voulez-vous accepter ce transfert ? La course sera ajoutée à vos réservations.`,
      [
        { text: "Annuler", style: "cancel" },
        {
          text: "Accepter",
          onPress: async () => {
            try {
              await acceptTransfer(transfer.id);
              Alert.alert("Succès", "Transfert accepté avec succès");
              loadIncomingTransfers();
              loadData(); // Recharger les courses
            } catch (error: any) {
              Alert.alert(
                "Erreur",
                error?.response?.data?.error || "Impossible d'accepter le transfert"
              );
            }
          },
        },
      ]
    );
  }, [loadData, loadIncomingTransfers]);

  // ✅ Refuser un transfert
  const handleRejectTransfer = useCallback(async (transfer: Transfer) => {
    Alert.prompt(
      "Refuser le transfert",
      "Voulez-vous indiquer une raison (optionnel) ?",
      [
        { text: "Annuler", style: "cancel" },
        {
          text: "Refuser",
          style: "destructive",
          onPress: async (reason?: string) => {
            try {
              await rejectTransfer(transfer.id, reason);
              Alert.alert("Succès", "Transfert refusé");
              loadIncomingTransfers();
            } catch (error: any) {
              Alert.alert(
                "Erreur",
                error?.response?.data?.error || "Impossible de refuser le transfert"
              );
            }
          },
        },
      ],
      "plain-text"
    );
  }, [loadIncomingTransfers]);

  const semiAutoControls = (
    <View style={styles.semiAutoControls}>
      <View style={styles.semiAutoHeader}>
        <View style={styles.semiAutoIconWrap}>
          <Ionicons name="flash-outline" size={16} color={BRAND} />
        </View>
        <Text style={styles.semiAutoTitle}>Mode semi-automatique</Text>
      </View>
      <Text style={styles.dispatchHint}>
        Laisse l’optimisation préparer les assignations et finalise-les en un
        clic. Relance le dispatch à chaque nouvelle vague de courses.
      </Text>
      <TouchableOpacity
        style={[
          styles.dispatchButton,
          dispatching && styles.dispatchButtonDisabled,
        ]}
        onPress={handleRunDispatch}
        disabled={dispatching}
        activeOpacity={0.85}
      >
        {dispatching ? (
          <ActivityIndicator size="small" color="#FFFFFF" />
        ) : (
          <Ionicons name="flash" size={16} color="#FFFFFF" />
        )}
        <Text style={styles.dispatchButtonText}>
          {dispatching ? "Dispatch en cours…" : "Lancer un dispatch"}
        </Text>
      </TouchableOpacity>
    </View>
  );

  const isRefreshing = loading || enterpriseLoading;
  const kpis = status?.kpis;
  const osrm = status?.osrm;
  const agent = status?.agent;
  const optimizer = status?.optimizer;
  const isManual = dispatchMode === "manual";
  const isSemiAuto = dispatchMode === "semi_auto";

  const urgentSection = (
    <Section title="Alertes urgentes" icon="warning-outline">
      {urgentRides.length === 0 ? (
        <View style={styles.emptyStateSmall}>
          <Ionicons name="checkmark-circle-outline" size={20} color={enterprisePalette.hintText} />
          <Text style={styles.emptyStateSmallText}>Aucune urgence en cours</Text>
        </View>
      ) : (
        urgentRides.map((ride) => (
          <RideAlert key={ride.id} ride={ride} badge="Urgent" />
        ))
      )}
    </Section>
  );

  // ✅ Section transferts entrants
  const incomingTransfersSection = incomingTransfers.length > 0 ? (
    <Section title={`Transferts entrants (${incomingTransfers.length})`} icon="swap-horizontal-outline">
      {incomingTransfers.slice(0, 3).map((transfer) => (
        <View key={transfer.id} style={{ marginBottom: 12 }}>
          <TransferCard
            transfer={transfer}
            type="incoming"
            onAccept={handleAcceptTransfer}
            onReject={handleRejectTransfer}
            onViewDetails={(t) => {
              if (t.booking_id) {
                router.push({
                  pathname: "/(enterprise)/ride-details",
                  params: { rideId: t.booking_id },
                } as any);
              }
            }}
          />
        </View>
      ))}
      {incomingTransfers.length > 3 && (
        <Text style={styles.muted}>
          + {incomingTransfers.length - 3} autres transferts
        </Text>
      )}
    </Section>
  ) : null;

  // ✅ 3.4.2: Section opportunités critiques avec bouton "Appliquer"
  const criticalOpportunitiesSection =
    realtimeDashboard?.opportunities &&
      realtimeDashboard.opportunities.filter(
        (opp) => opp.severity === "critical" || opp.severity === "high"
      ).length > 0 ? (
      <Section title="Opportunités critiques" icon="flash-outline">
        {realtimeDashboard.opportunities
          .filter((opp) => opp.severity === "critical" || opp.severity === "high")
          .slice(0, 3)
          .map((opp) => (
            <View
              key={opp.assignment_id}
              style={[
                styles.alertCard,
                opp.severity === "critical" && {
                  backgroundColor: "rgba(239,68,68,0.08)",
                  borderColor: "rgba(239,68,68,0.2)",
                },
                opp.severity === "high" && {
                  backgroundColor: "rgba(251,191,36,0.08)",
                  borderColor: "rgba(251,191,36,0.2)",
                },
              ]}
            >
              <TouchableOpacity
                onPress={() => {
                  router.push({
                    pathname: "/(enterprise)/ride-details",
                    params: { rideId: String(opp.booking_id) },
                  } as any);
                }}
                style={{ flex: 1 }}
              >
                <View style={styles.alertHeader}>
                  <Text
                    style={[
                      styles.alertBadge,
                      {
                        color:
                          opp.severity === "critical" ? "#EF4444" : "#F59E0B",
                      },
                    ]}
                  >
                    {opp.severity === "critical" ? "Critique" : "Élevée"}
                  </Text>
                  {opp.current_delay_minutes !== undefined && (
                    <Text style={styles.alertTime}>
                      {opp.current_delay_minutes > 0 ? "+" : ""}
                      {opp.current_delay_minutes} min
                    </Text>
                  )}
                </View>
                {opp.suggestions && opp.suggestions.length > 0 && (
                  <Text style={styles.alertRoute} numberOfLines={2}>
                    {opp.suggestions[0].message || opp.suggestions[0].action}
                  </Text>
                )}
              </TouchableOpacity>
              {/* ✅ Bouton "Appliquer" */}
              <TouchableOpacity
                style={[
                  styles.applyOpportunityButton,
                  applyingOpportunity === opp.assignment_id &&
                  styles.applyOpportunityButtonDisabled,
                ]}
                onPress={() => handleApplyOpportunity(opp)}
                disabled={applyingOpportunity === opp.assignment_id}
              >
                {applyingOpportunity === opp.assignment_id ? (
                  <ActivityIndicator
                    size="small"
                    color={enterprisePalette.modalButtonText}
                  />
                ) : (
                  <>
                    <Ionicons
                      name="checkmark-circle-outline"
                      size={16}
                      color={enterprisePalette.modalButtonText}
                    />
                    <Text style={styles.applyOpportunityText}>Appliquer</Text>
                  </>
                )}
              </TouchableOpacity>
            </View>
          ))}
      </Section>
    ) : null;

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={[styles.content, { paddingBottom: Math.max(80, tabBarHeight + 40) }]}
      refreshControl={
        <RefreshControl refreshing={isRefreshing} onRefresh={loadData} />
      }
    >
      {manualMapSection}

      {isManual && manualRidesList}

      {isSemiAuto && (
        <>
          {semiAutoControls}
          {manualRidesList}
          {urgentSection}
          {incomingTransfersSection}
        </>
      )}

      {dispatchMode === "fully_auto" && (
        <>
          {urgentSection}
          {incomingTransfersSection}
        </>
      )}

      {/* ✅ Statut dispatch en cours */}
      {status?.dispatch_run && status.dispatch_run.status !== "completed" && (
        <View style={styles.dispatchRunBanner}>
          <View style={styles.dispatchRunHeader}>
            <Ionicons
              name="sync-outline"
              size={18}
              color={enterprisePalette.dispatchButton}
            />
            <Text style={styles.dispatchRunTitle}>Dispatch en cours</Text>
            <View style={styles.dispatchRunStatus}>
              <Text style={styles.dispatchRunStatusText}>
                {status.dispatch_run.status}
              </Text>
            </View>
          </View>
          <Text style={styles.dispatchRunDetails}>
            {status.dispatch_run.assignments_count} assignation(s) créée(s)
            {status.dispatch_run.started_at &&
              ` • Débuté ${dayjs(status.dispatch_run.started_at).fromNow()}`}
          </Text>
          {status.is_running && (
            <View style={styles.dispatchRunProgress}>
              <ActivityIndicator
                size="small"
                color={enterprisePalette.dispatchButton}
              />
              <Text style={styles.dispatchRunProgressText}>
                Traitement en cours...
              </Text>
            </View>
          )}
        </View>
      )}

      {/* ✅ Statut optimiseur temps réel (mode fully-auto) */}
      {dispatchMode === "fully_auto" && (
        <View style={styles.optimizerSection}>
          <View style={styles.optimizerHeader}>
            <View style={styles.optimizerInfo}>
              <Ionicons
                name={optimizerRunning ? "flash" : "flash-outline"}
                size={20}
                color={
                  optimizerRunning
                    ? enterprisePalette.dispatchButton
                    : enterprisePalette.surfaceMuted
                }
              />
              <Text style={styles.optimizerTitle}>Optimiseur temps réel</Text>
              <View
                style={[
                  styles.optimizerStatusBadge,
                  optimizerRunning && styles.optimizerStatusBadgeActive,
                ]}
              >
                <Text
                  style={[
                    styles.optimizerStatusText,
                    optimizerRunning && styles.optimizerStatusTextActive,
                  ]}
                >
                  {optimizerRunning ? "🟢 Actif" : "🔴 Inactif"}
                </Text>
              </View>
            </View>
            <TouchableOpacity
              style={[
                styles.optimizerToggleButton,
                optimizerRunning && styles.optimizerToggleButtonActive,
              ]}
              onPress={handleToggleOptimizer}
              disabled={resetting}
            >
              <Text
                style={[
                  styles.optimizerToggleText,
                  optimizerRunning && styles.optimizerToggleTextActive,
                ]}
              >
                {optimizerRunning ? "Arrêter" : "Démarrer"}
              </Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.optimizerDescription}>
            {optimizerRunning
              ? "L'optimiseur détecte automatiquement les opportunités d'optimisation et les applique."
              : "Démarrez l'optimiseur pour activer la détection automatique des opportunités."}
          </Text>
        </View>
      )}

      {/* ✅ Actions rapides (mode semi-auto et fully-auto) */}
      {(isSemiAuto || dispatchMode === "fully_auto") && (
        <View style={styles.quickActionsSection}>
          <Text style={styles.sectionTitle}>Actions rapides</Text>
          <View style={styles.quickActionsRow}>
            {isSemiAuto && (
              <TouchableOpacity
                style={styles.quickActionButton}
                onPress={handleRunDispatch}
                disabled={dispatching}
              >
                <Ionicons
                  name="flash-outline"
                  size={18}
                  color={enterprisePalette.dispatchButton}
                />
                <Text style={styles.quickActionText}>Relancer dispatch</Text>
              </TouchableOpacity>
            )}
            <TouchableOpacity
              style={[styles.quickActionButton, styles.quickActionButtonDanger]}
              onPress={handleResetAssignments}
              disabled={resetting}
            >
              <Ionicons
                name="refresh-outline"
                size={18}
                color="#F87171"
              />
              <Text
                style={[
                  styles.quickActionText,
                  styles.quickActionTextDanger,
                ]}
              >
                {resetting ? "Réinitialisation..." : "Réinitialiser"}
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* Sections dashboard temps réel (semi-auto / fully-auto uniquement) */}
      {!isManual && criticalOpportunitiesSection}

      {errorMessage && <Text style={styles.error}>{errorMessage}</Text>}
      {authRecoveryMessage && (
        <View style={styles.authRecoveryBanner}>
          <Text style={styles.error}>{authRecoveryMessage}</Text>
          <TouchableOpacity onPress={() => router.replace("/(enterprise-auth)/login" as any)}>
            <Text style={styles.authRecoveryLink}>Se reconnecter</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* ✅ Modal d'assignation partagé */}
      <AssignDriverModal
        visible={rideActions.assignModalVisible}
        ride={rideActions.selectedRide}
        suggestions={rideActions.rideSuggestions}
        loading={rideActions.loadingSuggestions}
        assigning={rideActions.assigning}
        allDrivers={rideActions.allDrivers}
        loadingAllDrivers={rideActions.loadingAllDrivers}
        isManualMode={dispatchMode === "manual"}
        onClose={rideActions.handleCloseAssignModal}
        onAssign={rideActions.handleAssignDriver}
      />

      {/* ✅ Modal de transfert pour les courses PENDING */}
      <TransferRideModal
        visible={transferModalVisible}
        ride={selectedRideForTransfer}
        onClose={handleCloseTransferModal}
        onSuccess={() => {
          handleCloseTransferModal();
          loadData();
        }}
      />
    </ScrollView>
  );
}

const Section = ({
  title,
  icon,
  children,
}: {
  title: string;
  icon?: string;
  children: React.ReactNode;
}) => (
  <View style={styles.section}>
    <View style={styles.sectionTitleRow}>
      {icon && (
        <View style={styles.sectionIconWrap}>
          <Ionicons name={icon as any} size={16} color={BRAND} />
        </View>
      )}
      <Text style={styles.sectionTitle}>{title}</Text>
    </View>
    {children}
  </View>
);

const StatusCard = ({
  label,
  status,
  detail,
}: {
  label: string;
  status: "OK" | "WARNING" | "DOWN";
  detail: string;
}) => {
  const color =
    status === "OK" ? "#4ADE80" : status === "WARNING" ? "#FACC15" : "#F87171";
  return (
    <View style={styles.statusCard}>
      <Text style={styles.statusLabel}>{label}</Text>
      <Text style={[styles.statusValue, { color }]}>{status}</Text>
      <Text style={styles.statusDetail}>{detail}</Text>
    </View>
  );
};

const RideAlert = ({ ride, badge }: { ride: RideSummary; badge: string }) => {
  const pickupDisplay = ride.time.pickup_at
    ? (() => {
      const time = dayjs(ride.time.pickup_at);
      if (time.hour() === 0 && time.minute() === 0) return null;
      return time.format("HH:mm");
    })()
    : null;

  return (
    <TouchableOpacity
      style={styles.alertCard}
      activeOpacity={0.7}
      onPress={() =>
        router.push({
          pathname: "/(enterprise)/ride-details",
          params: { rideId: ride.id },
        } as any)
      }
    >
      <View style={styles.alertHeader}>
        <View style={styles.alertBadgeWrap}>
          <Ionicons name="warning-outline" size={14} color="#EF4444" />
          <Text style={styles.alertBadge}>{badge}</Text>
        </View>
        {pickupDisplay ? (
          <Text style={styles.alertTime}>{pickupDisplay}</Text>
        ) : (
          <Ionicons name="time-outline" size={16} color={enterprisePalette.hintText} />
        )}
      </View>
      <Text style={styles.alertClient} numberOfLines={1}>{ride.client.name}</Text>
      <View style={styles.alertRouteRow}>
        <Ionicons name="location-outline" size={13} color={BRAND} />
        <Text style={styles.alertRoute} numberOfLines={1}>{ride.route.pickup_address}</Text>
      </View>
      <View style={styles.alertRouteRow}>
        <Ionicons name="flag-outline" size={13} color={BRAND} />
        <Text style={styles.alertRoute} numberOfLines={1}>{ride.route.dropoff_address}</Text>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: enterprisePalette.background,
  },
  content: {
    padding: 16,
    paddingBottom: 80,
  },
  statusRow: {
    flexDirection: "row",
    gap: 12,
    marginBottom: 18,
  },
  statusCard: {
    flex: 1,
    backgroundColor: enterprisePalette.cardOverlay,
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: enterprisePalette.cardBorder,
  },
  statusLabel: {
    color: enterprisePalette.textSecondary,
    fontSize: 12,
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  statusValue: {
    fontSize: 18,
    fontWeight: "700",
    marginTop: 8,
    letterSpacing: 0.2,
  },
  statusDetail: {
    color: enterprisePalette.textSecondary,
    marginTop: 6,
    fontSize: 13,
    lineHeight: 18,
  },
  section: {
    backgroundColor: enterprisePalette.sectionSurface,
    borderRadius: 16,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: enterprisePalette.sectionBorder,
    ...createShadow({
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.04,
      shadowRadius: 8,
      elevation: 2,
    }),
  },
  sectionTitleRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 12,
  },
  sectionIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  sectionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },
  sectionTitle: {
    color: enterprisePalette.textStrong,
    fontSize: 16,
    fontWeight: "700",
  },
  sectionCount: {
    color: enterprisePalette.textSecondary,
    fontSize: 13,
    fontWeight: "600",
  },
  filtersRow: {
    flexDirection: "row",
    gap: 6,
    marginBottom: 14,
  },
  filterIcon: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 9,
    borderRadius: 12,
    backgroundColor: enterprisePalette.card,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    gap: 4,
  },
  filterIconActive: {
    backgroundColor: BRAND,
    borderColor: BRAND,
    ...createShadow({
      shadowColor: BRAND,
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.2,
      shadowRadius: 4,
      elevation: 2,
    }),
  },
  filterBadge: {
    backgroundColor: "rgba(0,121,107,0.12)",
    borderRadius: 7,
    paddingHorizontal: 5,
    paddingVertical: 1,
    minWidth: 18,
    alignItems: "center",
    justifyContent: "center",
  },
  filterBadgeActive: {
    backgroundColor: "rgba(255,255,255,0.28)",
  },
  filterBadgeText: {
    color: BRAND,
    fontSize: 10,
    fontWeight: "700",
  },
  filterBadgeTextActive: {
    color: "#FFFFFF",
  },
  emptyState: {
    alignItems: "center",
    paddingVertical: 28,
    gap: 6,
  },
  emptyStateIcon: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: "rgba(0,121,107,0.06)",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 4,
  },
  emptyStateTitle: {
    color: enterprisePalette.textStrong,
    fontSize: 14,
    fontWeight: "600",
  },
  emptyStateSubtitle: {
    color: enterprisePalette.hintText,
    fontSize: 12,
    textAlign: "center",
    lineHeight: 17,
  },
  emptyStateSmall: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingVertical: 6,
  },
  emptyStateSmallText: {
    color: enterprisePalette.hintText,
    fontSize: 13,
    fontWeight: "500",
  },
  muted: {
    color: enterprisePalette.hintText,
    fontSize: 13,
  },
  alertCard: {
    backgroundColor: enterprisePalette.card,
    borderRadius: 12,
    paddingVertical: 10,
    paddingHorizontal: 12,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    ...createShadow({
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.04,
      shadowRadius: 4,
      elevation: 1,
    }),
  },
  alertHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
  },
  alertBadgeWrap: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
  },
  alertBadge: {
    fontWeight: "700",
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 0.5,
    color: "#EF4444",
  },
  alertTime: {
    color: enterprisePalette.textStrong,
    fontSize: 14,
    fontWeight: "700",
  },
  alertClient: {
    color: enterprisePalette.textStrong,
    fontWeight: "600",
    fontSize: 14,
    marginBottom: 4,
  },
  alertRouteRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    marginBottom: 2,
  },
  alertRoute: {
    color: enterprisePalette.textSecondary,
    fontSize: 12,
    fontWeight: "500",
    flex: 1,
  },
  semiAutoControls: {
    marginTop: 14,
    padding: 16,
    borderRadius: 16,
    backgroundColor: enterprisePalette.surface,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    gap: 10,
    ...createShadow({
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.04,
      shadowRadius: 8,
      elevation: 2,
    }),
  },
  semiAutoHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
  },
  semiAutoIconWrap: {
    width: 28,
    height: 28,
    borderRadius: 8,
    backgroundColor: "rgba(0,121,107,0.08)",
    alignItems: "center",
    justifyContent: "center",
  },
  semiAutoTitle: {
    color: enterprisePalette.textStrong,
    fontSize: 16,
    fontWeight: "700",
  },
  dispatchHint: {
    color: enterprisePalette.hintText,
    fontSize: 13,
    lineHeight: 18,
  },
  dispatchButton: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    alignSelf: "flex-start",
    backgroundColor: BRAND,
    paddingHorizontal: 18,
    paddingVertical: 10,
    borderRadius: 12,
    ...createShadow({
      shadowColor: BRAND,
      shadowOffset: { width: 0, height: 3 },
      shadowOpacity: 0.2,
      shadowRadius: 8,
      elevation: 4,
    }),
  },
  dispatchButtonDisabled: {
    backgroundColor: enterprisePalette.dispatchButtonDisabled,
  },
  dispatchButtonText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: 13,
    letterSpacing: 0.3,
  },
  rideCard: {
    backgroundColor: enterprisePalette.cardOverlay,
    borderRadius: 14,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: enterprisePalette.cardBorder,
    ...createShadow({
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.04,
      shadowRadius: 6,
      elevation: 1,
    }),
  },
  rideTitle: {
    color: enterprisePalette.textStrong,
    fontWeight: "600",
    marginBottom: 6,
  },
  rideText: {
    color: enterprisePalette.textSecondary,
    fontSize: 13,
  },
  rideTime: {
    color: enterprisePalette.textStrong,
    marginTop: 8,
    fontWeight: "700",
  },
  error: {
    color: "#EF4444",
    marginTop: 10,
    fontSize: 13,
  },
  authRecoveryBanner: {
    marginTop: 8,
    padding: 10,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "rgba(239,68,68,0.2)",
    backgroundColor: "rgba(239,68,68,0.06)",
  },
  authRecoveryLink: {
    color: "#EF4444",
    fontWeight: "700",
    marginTop: 4,
    textDecorationLine: "underline",
  },
  manualListSection: {
    marginTop: 4,
    marginBottom: 14,
    gap: 10,
  },
  manualActionsRow: {
    flexDirection: "row",
    gap: 8,
  },
  manualSecondaryAction: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  manualSecondaryText: {
    color: enterprisePalette.textSecondary,
    fontWeight: "600",
    fontSize: 12,
  },
  manualPrimaryAction: {
    borderRadius: 999,
    backgroundColor: BRAND,
    paddingHorizontal: 16,
    paddingVertical: 6,
  },
  manualPrimaryText: {
    color: "#FFFFFF",
    fontWeight: "700",
    fontSize: 12,
  },
  manualMapSection: {
    marginTop: 2,
    gap: 6,
    marginBottom: 8,
  },
  dispatchRunBanner: {
    backgroundColor: enterprisePalette.sectionSurface,
    borderRadius: 16,
    padding: 14,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: enterprisePalette.sectionBorder,
    ...createShadow({
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.04,
      shadowRadius: 8,
      elevation: 2,
    }),
  },
  dispatchRunHeader: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: 8,
  },
  dispatchRunTitle: {
    color: enterprisePalette.textStrong,
    fontSize: 16,
    fontWeight: "600",
    flex: 1,
  },
  dispatchRunStatus: {
    backgroundColor: "rgba(0,121,107,0.1)",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(0,121,107,0.2)",
  },
  dispatchRunStatusText: {
    color: enterprisePalette.dispatchButton,
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  dispatchRunDetails: {
    color: enterprisePalette.textSecondary,
    fontSize: 13,
    marginBottom: 8,
  },
  dispatchRunProgress: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginTop: 8,
  },
  dispatchRunProgressText: {
    color: enterprisePalette.dispatchButton,
    fontSize: 13,
    fontWeight: "500",
  },
  optimizerSection: {
    backgroundColor: enterprisePalette.sectionSurface,
    borderRadius: 16,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: enterprisePalette.sectionBorder,
    ...createShadow({
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 0.04,
      shadowRadius: 8,
      elevation: 2,
    }),
  },
  optimizerHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: 12,
  },
  optimizerInfo: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    flex: 1,
  },
  optimizerTitle: {
    color: enterprisePalette.textStrong,
    fontSize: 16,
    fontWeight: "600",
    flex: 1,
  },
  optimizerStatusBadge: {
    backgroundColor: "rgba(239,68,68,0.08)",
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: "rgba(239,68,68,0.18)",
  },
  optimizerStatusBadgeActive: {
    backgroundColor: "rgba(0,121,107,0.1)",
    borderColor: "rgba(0,121,107,0.25)",
  },
  optimizerStatusText: {
    color: "#EF4444",
    fontSize: 11,
    fontWeight: "700",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  optimizerStatusTextActive: {
    color: enterprisePalette.dispatchButton,
  },
  optimizerToggleButton: {
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 14,
    paddingVertical: 9,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    ...createShadow({
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.04,
      shadowRadius: 4,
      elevation: 1,
    }),
  },
  optimizerToggleButtonActive: {
    backgroundColor: enterprisePalette.alertSurface,
    borderColor: enterprisePalette.alertBorder,
  },
  optimizerToggleText: {
    color: enterprisePalette.textStrong,
    fontSize: 13,
    fontWeight: "600",
  },
  optimizerToggleTextActive: {
    color: "#EF4444",
  },
  optimizerDescription: {
    color: enterprisePalette.textSecondary,
    fontSize: 13,
    lineHeight: 18,
  },
  quickActionsSection: {
    backgroundColor: enterprisePalette.sectionSurface,
    borderRadius: 16,
    padding: 16,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: enterprisePalette.sectionBorder,
  },
  quickActionsRow: {
    flexDirection: "row",
    gap: 10,
    marginTop: 10,
  },
  quickActionButton: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: "#FFFFFF",
    paddingVertical: 11,
    paddingHorizontal: 14,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    ...createShadow({
      shadowColor: "#000",
      shadowOffset: { width: 0, height: 1 },
      shadowOpacity: 0.04,
      shadowRadius: 4,
      elevation: 1,
    }),
  },
  quickActionButtonDanger: {
    backgroundColor: enterprisePalette.alertSurface,
    borderColor: enterprisePalette.alertBorder,
  },
  quickActionText: {
    color: enterprisePalette.textStrong,
    fontSize: 13,
    fontWeight: "600",
  },
  quickActionTextDanger: {
    color: "#EF4444",
  },
  applyOpportunityButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: BRAND,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 10,
    marginTop: 10,
    alignSelf: "flex-start",
  },
  applyOpportunityButtonDisabled: {
    backgroundColor: enterprisePalette.dispatchButtonDisabled,
  },
  applyOpportunityText: {
    color: "#FFFFFF",
    fontSize: 13,
    fontWeight: "600",
  },
  transferButtonFooter: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 5,
    backgroundColor: "rgba(0,121,107,0.07)",
    paddingVertical: 7,
    paddingHorizontal: 12,
    borderRadius: 10,
    marginTop: 8,
    alignSelf: "flex-start",
  },
  transferButtonText: {
    color: BRAND,
    fontSize: 13,
    fontWeight: "600",
  },
  cancelButtonFooter: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 5,
    backgroundColor: "rgba(239,68,68,0.06)",
    paddingVertical: 7,
    paddingHorizontal: 12,
    borderRadius: 10,
    marginTop: 8,
    alignSelf: "flex-start",
  },
  cancelButtonText: {
    color: "#EF4444",
    fontSize: 13,
    fontWeight: "600",
  },
});
