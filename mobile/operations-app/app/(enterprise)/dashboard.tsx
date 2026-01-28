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
} from "react-native";
import { useBottomTabBarHeight } from "@react-navigation/bottom-tabs";
import { useFocusEffect } from "@react-navigation/native";
import * as Crypto from "expo-crypto";
import { LinearGradient } from "expo-linear-gradient";
import { router } from "expo-router";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import timezone from "dayjs/plugin/timezone";
import relativeTime from "dayjs/plugin/relativeTime";
import "dayjs/locale/fr";
import { Ionicons } from "@expo/vector-icons";

import { useAuth } from "@/hooks/useAuth";
import { getAuthNotReadyDisplayMessage, isAuthNotReadyError } from "@/services/authGuards";
import { useEnterpriseNotifications } from "@/hooks/useEnterpriseNotifications";
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

// ✅ Palette professionnelle cohérente avec le dashboard driver
const enterprisePalette = {
  // Backgrounds - Clair et professionnel
  background: "#F5F7F6",
  card: "#FFFFFF",

  // Hero section - Gradient élégant
  heroGradient: ["#0A7F59", "#0D5F3F"] as [string, string],
  heroKpiSurface: "rgba(255,255,255,0.25)",
  heroKpiBorder: "rgba(255,255,255,0.35)",
  heroKicker: "rgba(255,255,255,0.85)",
  heroTitle: "#FFFFFF",
  heroMeta: "rgba(255,255,255,0.9)",
  heroTick: "#A8E6CF",

  // Surfaces - Cartes et sections
  surface: "#FFFFFF",
  surfaceBorder: "rgba(15,54,43,0.08)",
  surfaceMuted: "#5F7369",
  sectionSurface: "#FFFFFF",
  sectionBorder: "rgba(15,54,43,0.08)",

  // Alertes et états
  alertSurface: "rgba(239,68,68,0.1)",
  alertBorder: "rgba(239,68,68,0.25)",
  alertText: "#15362B",

  // Texte
  textStrong: "#15362B",
  textSecondary: "#5F7369",
  hintText: "#91A59D",

  // Boutons et actions
  dispatchButton: "#0A7F59",
  dispatchButtonDisabled: "rgba(10,127,89,0.4)",
  dispatchText: "#FFFFFF",

  // Cards
  cardOverlay: "#FFFFFF",
  cardBorder: "rgba(15,54,43,0.08)",

  // Modales
  modalOverlay: "rgba(21,54,43,0.75)",
  modalBackground: "#FFFFFF",
  modalBorder: "rgba(15,54,43,0.12)",
  modalTitle: "#15362B",
  modalText: "#5F7369",
  modalButton: "#0A7F59",
  modalButtonText: "#FFFFFF",
  modalCancelText: "#5F7369",
  loadingText: "#91A59D",
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

  // Activer les notifications push pour l'entreprise
  useEnterpriseNotifications();

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
      console.error("[Dashboard] Erreur chargement transferts:", error);
      // Ne pas afficher d'erreur, juste logger
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
          console.warn("[Dashboard] Failed to load realtime dashboard:", err);
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
      // ✅ Invariant C: refresh_token absent → forcer login (pas "connexion en cours" infini)
      if (isAuthNotReadyError(error) && ["missing_refresh_token", "auth_ready_timeout"].includes((error as any).reason)) {
        router.replace("/(enterprise-auth)/login" as any);
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

  useEffect(() => {
    if (!enterpriseSession) return;
    throttledLoadData();
  }, [enterpriseSession, throttledLoadData, currentDate]);

  // Polling automatique : récupérer les données toutes les 30 secondes quand l'app est active
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
          console.log("[dashboard.tsx] Polling automatique : rechargement des données");
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
      if (
        appStateRef.current.match(/inactive|background/) &&
        nextAppState === "active"
      ) {
        // L'app revient au premier plan : recharger immédiatement et redémarrer le polling
        console.log("[dashboard.tsx] Application revenue au premier plan : rechargement des données");
        throttledLoadData();
        startPolling();
      } else if (nextAppState.match(/inactive|background/)) {
        // L'app passe en arrière-plan : arrêter le polling
        console.log("[dashboard.tsx] Application en arrière-plan : arrêt du polling");
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
  }, [enterpriseSession, throttledLoadData, currentDate]);

  // Recharger les données quand l'écran revient au focus
  useFocusEffect(
    useCallback(() => {
      if (enterpriseSession) {
        console.log("[dashboard.tsx] Écran au focus : rechargement des données");
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

  const heroKpis = useMemo(() => {
    if (dispatchMode === "manual") {
      return [
        {
          id: "manual-total",
          label: "Total",
          value: String(manualStats.total),
        },
        {
          id: "manual-assigned",
          label: "En cours",
          value: String(manualStats.assignedCount),
        },
      ];
    }

    const kpis = status?.kpis;
    return [
      {
        id: "auto-total",
        label: "Total",
        value: kpis ? String(kpis.total_bookings) : "—",
      },
      {
        id: "auto-assigned",
        label: "En cours",
        value: kpis ? String(kpis.assigned_bookings) : "—",
      },
    ];
  }, [dispatchMode, manualStats, status?.kpis]);

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

      {/* ✅ Filtres horizontaux compacts (tous visibles sur une ligne) */}
      <View style={styles.filtersContainer}>
        {[
          { label: "Toutes", value: "all" as const, icon: "grid-outline" },
          { label: "En attente", value: "pending" as const, icon: "hourglass-outline" },
          { label: "Non assignées", value: "unassigned" as const, icon: "alert-circle-outline" },
          { label: "Assignées", value: "assigned" as const, icon: "checkmark-circle-outline" },
          { label: "Terminées", value: "completed" as const, icon: "checkmark-done-circle-outline" },
          { label: "En retard", value: "delayed" as const, icon: "alarm-outline" },
        ].map((filter) => {
          const isActive = rideFilter === filter.value;
          const count = filterCounts[filter.value];
          return (
            <TouchableOpacity
              key={filter.value}
              style={[styles.filterChip, isActive && styles.filterChipActive]}
              onPress={() => setRideFilter(filter.value)}
              activeOpacity={0.7}
            >
              <Ionicons
                name={filter.icon as any}
                size={16}
                color={isActive ? enterprisePalette.dispatchText : enterprisePalette.textSecondary}
              />
              {count > 0 && (
                <View style={[styles.filterCount, isActive && styles.filterCountActive]}>
                  <Text style={[styles.filterCountText, isActive && styles.filterCountTextActive]}>
                    {count}
                  </Text>
                </View>
              )}
            </TouchableOpacity>
          );
        })}
      </View>

      {filteredRides.length === 0 ? (
        <Text style={styles.muted}>
          {rideFilter === "all"
            ? "Aucune course planifiée pour cette date."
            : rideFilter === "pending"
              ? "Aucune course en attente."
              : rideFilter === "unassigned"
                ? "Aucune course non assignée."
                : rideFilter === "assigned"
                  ? "Aucune course assignée."
                  : rideFilter === "completed"
                    ? "Aucune course terminée."
                    : "Aucune course en retard."}
        </Text>
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
      <Text style={styles.sectionTitle}>Mode semi-automatique</Text>
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
        <Ionicons
          name={dispatching ? "time-outline" : "flash-outline"}
          size={18}
          color="#0B1736"
        />
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
    <Section title="Alertes urgentes">
      {urgentRides.length === 0 ? (
        <Text style={styles.muted}>Aucune urgence en cours.</Text>
      ) : (
        urgentRides.map((ride) => (
          <RideAlert key={ride.id} ride={ride} badge="Urgent" />
        ))
      )}
    </Section>
  );

  // ✅ Section transferts entrants
  const incomingTransfersSection = incomingTransfers.length > 0 ? (
    <Section title={`Transferts entrants (${incomingTransfers.length})`}>
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
      <Section title="Opportunités critiques">
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
      <LinearGradient
        colors={enterprisePalette.heroGradient}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.hero}
      >
        <View style={styles.heroHeader}>
          <View style={{ flex: 1 }}>
            <Text style={styles.heroKicker}>Tableau de bord</Text>
            <Text style={styles.heroCompany}>{companyName}</Text>
          </View>
          <View style={styles.heroMeta}>
            <Text style={styles.heroDate}>{formattedDay}</Text>
            <Text style={styles.heroTick}>
              Agent {agent?.last_tick ? dayjs(agent.last_tick).fromNow() : "—"}
            </Text>
          </View>
        </View>

        {heroKpis.length > 0 && (
          <View style={styles.heroKpiRow}>
            {heroKpis.map((kpi) => (
              <View key={kpi.id} style={styles.heroKpiCard}>
                <Text style={styles.heroKpiValue}>{kpi.value}</Text>
                <Text style={styles.heroKpiLabel}>{kpi.label}</Text>
              </View>
            ))}
          </View>
        )}
      </LinearGradient>

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

      {/* ✅ 3.4.2: Sections dashboard temps réel */}
      {criticalOpportunitiesSection}

      {errorMessage && <Text style={styles.error}>{errorMessage}</Text>}

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
  children,
}: {
  title: string;
  children: React.ReactNode;
}) => (
  <View style={styles.section}>
    <Text style={styles.sectionTitle}>{title}</Text>
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

const RideAlert = ({ ride, badge }: { ride: RideSummary; badge: string }) => (
  <TouchableOpacity
    style={styles.alertCard}
    onPress={() =>
      router.push({
        pathname: "/(enterprise)/ride-details",
        params: { rideId: ride.id },
      } as any)
    }
  >
    <View style={styles.alertHeader}>
      <Text style={styles.alertBadge}>{badge}</Text>
      <Text style={styles.alertTime}>
        {ride.time.pickup_at
          ? (() => {
            const time = dayjs(ride.time.pickup_at);
            // Si l'heure est à minuit (00:00), c'est probablement une heure non définie
            // Afficher une icône d'horloge au lieu de "00:00"
            if (time.hour() === 0 && time.minute() === 0) {
              return "⏱️";
            }
            return time.format("HH:mm");
          })()
          : "⏱️"}
      </Text>
    </View>
    <Text style={styles.alertClient}>{ride.client.name}</Text>
    <Text style={styles.alertRoute}>{ride.route.pickup_address}</Text>
    <Text style={styles.alertRoute}>→ {ride.route.dropoff_address}</Text>
  </TouchableOpacity>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: enterprisePalette.background,
  },
  content: {
    padding: 20,
    paddingBottom: 80,
  },
  hero: {
    borderRadius: 24,
    paddingHorizontal: 20,
    paddingVertical: 12,
    marginBottom: 6,
    height: 150,
    overflow: "hidden",
    ...createShadow({
      shadowColor: "rgba(10,127,89,0.15)",
      shadowOffset: { width: 0, height: 8 },
      shadowOpacity: 1,
      shadowRadius: 24,
      elevation: 8,
    }), // ✅ Compatible web/native
    justifyContent: "space-between",
  },
  heroHeader: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 10,
    marginBottom: 4,
  },
  heroKpiRow: {
    flexDirection: "row",
    gap: 8,
    marginTop: 0,
    justifyContent: "flex-end",
  },
  heroKpiCard: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 16,
    backgroundColor: enterprisePalette.heroKpiSurface,
    borderWidth: 1,
    borderColor: enterprisePalette.heroKpiBorder,
    backdropFilter: "blur(10px)",
  },
  heroKpiValue: {
    color: enterprisePalette.heroTitle,
    fontWeight: "700",
    fontSize: 13,
  },
  heroKpiLabel: {
    color: enterprisePalette.heroMeta,
    fontSize: 11,
  },
  heroKicker: {
    color: enterprisePalette.heroKicker,
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 0.8,
    marginBottom: 4,
  },
  heroCompany: {
    color: enterprisePalette.heroTitle,
    fontSize: 24,
    fontWeight: "700",
    lineHeight: 28,
  },
  heroMeta: {
    alignItems: "flex-end",
    gap: 2,
  },
  heroDate: {
    color: enterprisePalette.heroMeta,
    fontSize: 12,
    textTransform: "capitalize",
  },
  heroTick: {
    color: enterprisePalette.heroTick,
    fontSize: 11,
  },
  modeSwitch: {
    flexDirection: "row",
    backgroundColor: "rgba(10,17,38,0.55)",
    borderRadius: 16,
    padding: 6,
    gap: 6,
  },
  modePill: {
    flex: 1,
    borderRadius: 12,
    paddingVertical: 10,
    alignItems: "center",
  },
  modePillActive: {
    backgroundColor: "rgba(255,255,255,0.16)",
  },
  modePillText: {
    color: "rgba(255,255,255,0.65)",
    fontWeight: "600",
  },
  modePillTextActive: {
    color: "#FFFFFF",
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
    borderRadius: 20,
    padding: 20,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: enterprisePalette.sectionBorder,
    ...createShadow({
      shadowColor: "rgba(15,54,43,0.08)",
      shadowOffset: { width: 0, height: 4 },
      shadowOpacity: 1,
      shadowRadius: 12,
      elevation: 2,
    }), // ✅ Compatible web/native
  },
  sectionHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  sectionTitle: {
    color: enterprisePalette.textStrong,
    fontSize: 17,
    fontWeight: "600",
  },
  sectionCount: {
    color: enterprisePalette.textSecondary,
    fontSize: 14,
    fontWeight: "600",
  },
  filtersContainer: {
    flexDirection: "row",
    flexWrap: "nowrap",
    gap: 6,
    marginBottom: 16,
  },
  filterChip: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    flex: 1,
    paddingHorizontal: 8,
    paddingVertical: 8,
    borderRadius: 16,
    backgroundColor: enterprisePalette.card,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    gap: 6,
    minWidth: 0, // Permet au flex de réduire la taille
  },
  filterChipActive: {
    backgroundColor: enterprisePalette.dispatchButton,
    borderColor: enterprisePalette.dispatchButton,
  },
  filterCount: {
    backgroundColor: enterprisePalette.surfaceBorder,
    borderRadius: 8,
    paddingHorizontal: 5,
    paddingVertical: 2,
    minWidth: 20,
    alignItems: "center",
    justifyContent: "center",
  },
  filterCountActive: {
    backgroundColor: "rgba(255,255,255,0.25)",
  },
  filterCountText: {
    color: enterprisePalette.textSecondary,
    fontSize: 10,
    fontWeight: "700",
  },
  filterCountTextActive: {
    color: enterprisePalette.dispatchText,
  },
  muted: {
    color: enterprisePalette.surfaceMuted,
  },
  alertCard: {
    backgroundColor: enterprisePalette.card,
    borderRadius: 14,
    paddingVertical: 10,
    paddingHorizontal: 14,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    ...createShadow({
      shadowColor: "rgba(15,54,43,0.06)",
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 1,
      shadowRadius: 6,
      elevation: 2,
    }), // ✅ Compatible web/native
  },
  alertHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 6,
  },
  alertBadgeContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  alertBadge: {
    fontWeight: "700",
    fontSize: 11,
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  alertTime: {
    color: enterprisePalette.textStrong,
    fontSize: 14,
    fontWeight: "700",
    letterSpacing: 0.2,
  },
  alertContent: {
    gap: 3,
  },
  alertClient: {
    color: enterprisePalette.textStrong,
    fontWeight: "700",
    fontSize: 14,
    letterSpacing: 0.1,
  },
  alertTimeRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  alertRoute: {
    color: enterprisePalette.textSecondary,
    fontSize: 12,
    fontWeight: "500",
  },
  semiAutoControls: {
    marginTop: 24,
    padding: 20,
    borderRadius: 18,
    backgroundColor: enterprisePalette.surface,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    gap: 12,
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
    backgroundColor: enterprisePalette.dispatchButton,
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 16,
    ...createShadow({
      shadowColor: enterprisePalette.dispatchButton,
      shadowOffset: { width: 0, height: 6 },
      shadowOpacity: 0.3,
      shadowRadius: 16,
      elevation: 6,
    }), // ✅ Compatible web/native
  },
  dispatchButtonDisabled: {
    backgroundColor: enterprisePalette.dispatchButtonDisabled,
    shadowOpacity: 0.15,
  },
  dispatchButtonText: {
    color: enterprisePalette.dispatchText,
    fontWeight: "700",
    fontSize: 14,
    letterSpacing: 0.2,
    textTransform: "uppercase",
  },
  rideCard: {
    backgroundColor: enterprisePalette.cardOverlay,
    borderRadius: 18,
    padding: 18,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: enterprisePalette.cardBorder,
    ...createShadow({
      shadowColor: "rgba(15,54,43,0.06)",
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 1,
      shadowRadius: 8,
      elevation: 2,
    }), // ✅ Compatible web/native
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
    color: "#F87171",
    marginTop: 12,
  },
  manualListSection: {
    marginTop: 8,
    marginBottom: 24,
    gap: 12,
  },
  manualActionsRow: {
    flexDirection: "row",
    gap: 8,
  },
  manualSecondaryAction: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(148,163,255,0.45)",
    paddingHorizontal: 14,
    paddingVertical: 6,
  },
  manualSecondaryText: {
    color: "rgba(214,224,255,0.85)",
    fontWeight: "600",
    fontSize: 12,
  },
  manualPrimaryAction: {
    borderRadius: 999,
    backgroundColor: "#5EEAD4",
    paddingHorizontal: 16,
    paddingVertical: 6,
  },
  manualPrimaryText: {
    color: "#0B1736",
    fontWeight: "700",
    fontSize: 12,
  },
  manualMapSection: {
    marginTop: 2,
    gap: 12,
    marginBottom: 8,
  },
  // ✅ Styles pour le statut dispatch en cours
  dispatchRunBanner: {
    backgroundColor: enterprisePalette.sectionSurface,
    borderRadius: 18,
    padding: 16,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: enterprisePalette.sectionBorder,
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
    backgroundColor: "rgba(10,127,89,0.12)",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(10,127,89,0.2)",
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
  // ✅ Styles pour l'optimiseur temps réel
  optimizerSection: {
    backgroundColor: enterprisePalette.sectionSurface,
    borderRadius: 18,
    padding: 18,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: enterprisePalette.sectionBorder,
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
    backgroundColor: "rgba(239,68,68,0.1)",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: "rgba(239,68,68,0.2)",
  },
  optimizerStatusBadgeActive: {
    backgroundColor: "rgba(10,127,89,0.12)",
    borderColor: enterprisePalette.dispatchButton,
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
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    ...createShadow({
      shadowColor: "rgba(15,54,43,0.08)",
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 1,
      shadowRadius: 4,
      elevation: 2,
    }), // ✅ Compatible web/native
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
  // ✅ Styles pour les actions rapides
  quickActionsSection: {
    backgroundColor: enterprisePalette.sectionSurface,
    borderRadius: 18,
    padding: 18,
    marginBottom: 20,
    borderWidth: 1,
    borderColor: enterprisePalette.sectionBorder,
  },
  quickActionsRow: {
    flexDirection: "row",
    gap: 12,
    marginTop: 12,
  },
  quickActionButton: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    backgroundColor: "#FFFFFF",
    paddingVertical: 14,
    paddingHorizontal: 18,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: enterprisePalette.surfaceBorder,
    ...createShadow({
      shadowColor: "rgba(15,54,43,0.08)",
      shadowOffset: { width: 0, height: 2 },
      shadowOpacity: 1,
      shadowRadius: 8,
      elevation: 2,
    }), // ✅ Compatible web/native
  },
  quickActionButtonDanger: {
    backgroundColor: enterprisePalette.alertSurface,
    borderColor: enterprisePalette.alertBorder,
  },
  quickActionText: {
    color: enterprisePalette.textStrong,
    fontSize: 14,
    fontWeight: "600",
    letterSpacing: 0.2,
  },
  quickActionTextDanger: {
    color: "#EF4444",
  },
  // ✅ Styles pour le bouton "Appliquer" sur les opportunités
  applyOpportunityButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: enterprisePalette.dispatchButton,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 12,
    marginTop: 12,
    alignSelf: "flex-start",
  },
  applyOpportunityButtonDisabled: {
    backgroundColor: enterprisePalette.dispatchButtonDisabled,
  },
  applyOpportunityText: {
    color: enterprisePalette.dispatchText,
    fontSize: 13,
    fontWeight: "600",
  },
  // ✅ Styles pour le bouton "Transférer" dans le footer (receveur)
  transferButtonFooter: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: "rgba(10,127,89,0.08)",
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    marginTop: 8,
    alignSelf: "flex-start",
  },
  transferButtonText: {
    color: "#0A7F59",
    fontSize: 14,
    fontWeight: "600",
  },
  // ✅ Styles pour le bouton "Annuler" dans le footer (émetteur)
  cancelButtonFooter: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
    backgroundColor: "rgba(239,68,68,0.08)",
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    marginTop: 8,
    alignSelf: "flex-start",
  },
  cancelButtonText: {
    color: "#EF4444",
    fontSize: 14,
    fontWeight: "600",
  },
});
