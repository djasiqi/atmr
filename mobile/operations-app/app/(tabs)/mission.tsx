import React, { useEffect, useState, useCallback, useMemo, useRef } from "react";
import { ScrollView, Alert, Linking, View, Text, RefreshControl, Platform, AppState, InteractionManager, ActivityIndicator, StyleSheet } from "react-native";
import { useAuth } from "@/hooks/useAuth";
import { useSocket } from "@/hooks/useSocket";
import { useLocation } from "@/hooks/useLocation";
import { useTrackingState } from "@/hooks/useTrackingState";
import { useNotifications } from "@/hooks/useNotifications";
import { useDynamicETA } from "@/hooks/useDynamicETA";
import { useMissionLayout } from "@/hooks/useMissionLayout";
import MissionCard from "@/components/dashboard/MissionCard";
import MissionGroupHeader from "@/components/dashboard/MissionGroupHeader";
import MissionHeader from "@/components/dashboard/MissionHeader";
import MissionMap from "@/components/dashboard/MissionMap";
import { MissionListSkeleton } from "@/components/dashboard/MissionListSkeleton";
import ConfirmCompletionModal from "@/components/dashboard/ConfirmCompletionModal";
// SocketStatusIndicator is now integrated into MissionHeader
import { Loader } from "@/components/ui/Loader";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  Booking,
  BookingStatus,
} from "@/services/api";
import { requestMissionSync } from "@/services/missionSyncOrchestrator";
import { MissionStateManager, type MissionBarStatus } from "@/services/missionState";
import { showMissionNotification, dismissMissionNotification } from "@/services/missionBarAndroid";
import { openNavigation as openNavigationApp } from "@/services/deepLinks";
import { registerNotifeeForegroundHandler } from "@/services/missionBarBackground";
import { onBookingsResync } from "@/services/socket";
import {
  organizeMissionsForDisplay,
  getNextDestination,
  getNextDestinationCoords,
  filterActiveMissions,
  filterNextMissionsOnly,
  type DisplayMission,
} from "@/utils/missionGrouping";
import { getCallablePhone } from "@/utils/phone";
import {
  scheduleMissionReminder,
  cancelMissionReminder,
  scheduleRemindersForActiveMissions,
  cleanupExpiredReminders,
} from "@/services/localNotifications";
import { getLogger } from "@/utils/logger";
import { isAuthNotReadyError } from "@/services/authGuards";
import { useAppAlert } from "@/contexts/AppAlertContext";
import { TrackingStateBanner } from "@/components/common/TrackingStateBanner";
import {
  requestBackgroundPermissionIfNeeded,
  buildBgTrackingInputs,
  refreshBackgroundTrackingNotification,
} from "@/services/locationTracker";

const log = getLogger("Mission");

/**
 * Détecte si la mission est un retour, quel que soit le type de donnée reçu (bool, int, string, etc.)
 */
function isMissionReturn(is_return: any): boolean {
  // Cas bool ou int
  if (is_return === true || is_return === 1) return true;
  if (is_return === false || is_return === 0) return false;
  // Cas string (même "False", "false", "0", etc.)
  if (typeof is_return === "string") {
    const v = is_return.trim().toLowerCase();
    if (["1", "true", "yes", "oui"].includes(v)) return true;
    if (["0", "false", "no", "non", ""].includes(v)) return false;
    // Patch anti-typo
    if (v === "return") return true;
  }
  // Cas null/undefined ou autre
  if (!is_return) return false;
  log.info("unexpected is_return value", { is_return, type: typeof is_return });
  return false;
}

export default function MissionScreen() {
  const { driver, mode, isDriverAuthenticated } = useAuth();
  const appAlert = useAppAlert();
  const { location } = useLocation();
  const trackingState = useTrackingState({
    isDriverAuthenticated: !!isDriverAuthenticated,
    role: mode === "driver" ? "driver" : "enterprise",
  });
  const socket = useSocket();
  useNotifications();
  const { contentWidth, mapHeight, horizontalPadding } = useMissionLayout();

  // Hook pour les ETAs dynamiques basés sur la position GPS (GET /driver/me/bookings/eta uniquement)
  const {
    etas,
    hasGPS,
    getDuration,
    getETAToPickup,
    getETAToDropoff,
    getEstimatedArrival,
    getEstimatedArrivalDropoff,
    getDelayMinutes,
    isLoading: etaLoading,
  } = useDynamicETA(!!driver);

  const [isLoading, setIsLoading] = useState(true);
  const [missions, setMissions] = useState<Booking[]>([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [completingMissionId, setCompletingMissionId] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  /** True dès qu’on a restauré des missions actives depuis AsyncStorage (STW-01 / STW-05). */
  const cacheHadActiveMissionsRef = useRef(false);
  const missionsNonEmptyRef = useRef(false);

  useEffect(() => {
    missionsNonEmptyRef.current = missions.length > 0;
  }, [missions.length]);

  useEffect(() => {
    if (!driver?.id) {
      cacheHadActiveMissionsRef.current = false;
    }
  }, [driver?.id]);

  // v2: backend envoie désormais client.phone et client.gp_phone (bouton Appeler) — bump pour invalider ancien cache
  const MISSIONS_CACHE_KEY = "missions_cache_v2";

  // Filtrer les missions actives (aujourd'hui ou demain si après 19h)
  const activeMissions = useMemo(() => {
    const todayMissions = filterActiveMissions(missions);
    // Ne garder que le prochain groupe de missions
    return filterNextMissionsOnly(todayMissions);
  }, [missions]);

  // Organiser les missions pour l'affichage avec groupement (avec intervalle de 5min)
  const displayMissions = useMemo(() => {
    return organizeMissionsForDisplay(activeMissions);
  }, [activeMissions]);

  // Trouver la prochaine destination pour la carte
  const nextDestination = useMemo(() => {
    return getNextDestination(activeMissions);
  }, [activeMissions]);

  // Coords backend (priorité) — évite le géocodage mobile
  const nextDestinationCoords = useMemo(() => {
    return getNextDestinationCoords(activeMissions);
  }, [activeMissions]);

  // ✅ Phase 1 - Quick Wins: Planifier rappels pour les missions actives
  useEffect(() => {
    if (__DEV__) {
      log.info("mission screen mounted");
    }
  }, []);

  // Log map readiness (pour debug temps de chargement carte)
  useEffect(() => {
    if (__DEV__ && location && nextDestination) {
      log.info("Mission map ready to render", {
        hasNextDestCoords: nextDestinationCoords != null,
      });
    }
  }, [location, nextDestination, nextDestinationCoords]);

  // ✅ Phase 1 - Quick Wins: Planifier rappels pour les missions actives
  useEffect(() => {
    if (activeMissions.length > 0) {
      scheduleRemindersForActiveMissions(activeMissions);
    }
  }, [activeMissions]);

  // ✅ Phase 1 - Quick Wins: Nettoyer rappels expirés au démarrage
  useEffect(() => {
    cleanupExpiredReminders();
  }, []);

  // Charger missions actives depuis le cache au démarrage
  useEffect(() => {
    (async () => {
      try {
        const raw = await AsyncStorage.getItem(MISSIONS_CACHE_KEY);
        if (raw) {
          const cached: Booking[] = JSON.parse(raw);
          const active = cached.filter(
            (m) =>
              ![
                "completed",
                "return_completed",
                "canceled",
                "cancelled",
              ].includes((m.status || "").toLowerCase()) ||
              // ✅ P0-1: Vérifier aussi les statuts en uppercase
              ["COMPLETED", "RETURN_COMPLETED", "CANCELED"].includes(m.status || "")
          );
          if (active.length) {
            const sorted = active.sort(
              (a, b) =>
                new Date(a.scheduled_time).getTime() -
                new Date(b.scheduled_time).getTime()
            );
            setMissions(sorted);
          }
        }
      } catch { }
    })();
  }, []);

  const loadMissions = useCallback(async (isRefreshAction = false) => {
    if (!isRefreshAction) {
      setIsLoading(true);
    }
    try {
      const assigned = await requestMissionSync("manual_screen");

      // 🔒 SÉCURITÉ : Utiliser UNIQUEMENT les données du backend
      // Ne pas merger avec le cache pour éviter de voir les missions d'autres chauffeurs
      const sorted = assigned.sort(
        (a, b) =>
          new Date(a.scheduled_time).getTime() -
          new Date(b.scheduled_time).getTime()
      );

      // Mettre à jour le cache avec les nouvelles données uniquement
      AsyncStorage.setItem(MISSIONS_CACHE_KEY, JSON.stringify(sorted)).catch(
        () => { }
      );

      setMissions(sorted);
    } catch (err) {
      if (!isAuthNotReadyError(err)) {
        appAlert.showAlert("Erreur", "Impossible de charger les missions.");
      }
    } finally {
      if (!isRefreshAction) {
        setIsLoading(false);
      }
      setIsRefreshing(false);
    }
  }, [driver, appAlert]);

  // Fonction de rafraîchissement pour pull-to-refresh
  const onRefresh = useCallback(async () => {
    setIsRefreshing(true);
    try {
      await loadMissions(true);
    } finally {
      setIsRefreshing(false);
    }
  }, [loadMissions]);

  useEffect(() => {
    if (driver) {
      loadMissions();
    }
  }, [driver, loadMissions]);

  useEffect(() => {
    if (!socket) return;

    const onNew = (data: Booking) => {
      setMissions((prev) => {
        const exists = prev.find((m) => m.id === data.id);
        const updated = exists
          ? prev.map((m) => (m.id === data.id ? data : m))
          : [...prev, data];
        const sorted = updated.sort(
          (a, b) =>
            new Date(a.scheduled_time).getTime() -
            new Date(b.scheduled_time).getTime()
        );
        AsyncStorage.setItem(MISSIONS_CACHE_KEY, JSON.stringify(sorted)).catch(
          () => { }
        );
        return sorted;
      });
      // Pas besoin de réinitialiser l'index, toutes les missions sont affichées
    };

    const onUpdate = (data: Booking) => {
      setMissions((prev) => {
        const updated = prev
          .map((m) => (m.id === data.id ? data : m))
          .filter((m) => {
            const s = (m.status || "").toLowerCase();
            // ✅ P0-1: Filtrer les missions terminées ou annulées (mais pas les libérées qui reviennent à ASSIGNED)
            // Vérifier aussi les statuts en uppercase
            return !["completed", "return_completed", "canceled", "cancelled"].includes(s) &&
              !["COMPLETED", "RETURN_COMPLETED", "CANCELED"].includes(m.status || "");
          })
          .sort(
            (a, b) =>
              new Date(a.scheduled_time).getTime() -
              new Date(b.scheduled_time).getTime()
          );
        // Pas besoin de gérer l'index, toutes les missions sont affichées
        AsyncStorage.setItem(MISSIONS_CACHE_KEY, JSON.stringify(updated)).catch(
          () => { }
        );
        return updated;
      });
      // ✅ Mettre à jour MissionStateManager pour réconciliation du tracking background (EN_ROUTE)
      if (MissionStateManager.isActive() && MissionStateManager.getState().activeMission?.id === data.id) {
        MissionStateManager.applyBookingUpdate(data).catch(() => {});
      }
      // ✅ Si la mission a été annulée, afficher un message
      // Note: Si la mission est libérée (retour à ASSIGNED), elle reste dans la liste
      // et sera réassignée automatiquement, pas besoin d'alerte
      const statusLower = (data.status || "").toLowerCase();
      if (statusLower === "canceled" || statusLower === "cancelled") {
        // ✅ Phase 1 - Quick Wins: Annuler le rappel local
        cancelMissionReminder(data.id);

        Alert.alert(
          "Course annulée",
          "Elle sera facturée comme annulation."
        );
      }
    };

    const onCancel = ({ id }: { id: number }) => {
      // ✅ Phase 1 - Quick Wins: Annuler le rappel local
      cancelMissionReminder(id);

      setMissions((prev) => {
        const next = prev.filter((m) => m.id !== id);
        AsyncStorage.setItem(MISSIONS_CACHE_KEY, JSON.stringify(next)).catch(
          () => { }
        );
        return next;
      });
      // Vérifier si une mission visible a été annulée
      const cancelledMission = missions.find((m) => m.id === id);
      if (cancelledMission) {
        appAlert.showAlert("Mission annulée", "Une mission a été annulée.");
      }
    };

    // ✅ Mission réassignée à un autre chauffeur → rafraîchir la liste + notifier
    const onReassigned = async (payload: any) => {
      const bookingId = payload?.booking_id ?? payload?.id ?? null;
      log.info("booking reassigned", { bookingId });
      appAlert.showAlert(
        "Mission réassignée",
        "Cette course a été assignée à un autre chauffeur. Vos courses sont mises à jour."
      );
      loadMissions(true);
    };

    // Gestion de la déconnexion WebSocket
    const onDisconnect = () => {
      log.warn("socket disconnected, reconnecting", {});
      // Le client Socket.IO reconnecte automatiquement, mais on peut forcer un refresh
    };

    // Gestion de la reconnexion WebSocket
    const onReconnect = () => {
      log.info("socket reconnected, reloading missions", {});
      loadMissions(true);
      void (async () => {
        try {
          const fresh = await requestMissionSync("socket_connect");
          await MissionStateManager.updateFromServer(fresh);
        } catch {
          /* best-effort */
        }
      })();
    };

    socket.on("new_booking", onNew);
    socket.on("booking_updated", onUpdate);
    socket.on("booking_cancelled", onCancel);
    socket.on("booking_reassigned", onReassigned);
    socket.on("disconnect", onDisconnect);
    socket.on("reconnect", onReconnect);

    // ✅ Écouter l'événement de resync pour mettre à jour les missions
    const unsubscribeResync = onBookingsResync((bookings) => {
      const sorted = bookings.sort(
        (a, b) =>
          new Date(a.scheduled_time).getTime() -
          new Date(b.scheduled_time).getTime()
      );
      setMissions(sorted);
      // Mettre à jour le cache
      AsyncStorage.setItem(MISSIONS_CACHE_KEY, JSON.stringify(sorted)).catch(
        () => { }
      );
    });

    return () => {
      socket.off("new_booking", onNew);
      socket.off("booking_updated", onUpdate);
      socket.off("booking_cancelled", onCancel);
      socket.off("booking_reassigned", onReassigned);
      socket.off("disconnect", onDisconnect);
      socket.off("reconnect", onReconnect);
      unsubscribeResync();
    };
  }, [socket, loadMissions, appAlert]);

  // ✅ Mission Bar: foreground Notifee handler + AppState reconciliation
  useEffect(() => {
    if (Platform.OS === "web") return;

    const unsubNotifee = registerNotifeeForegroundHandler();

    const appStateSub = AppState.addEventListener("change", (state) => {
      if (state === "active" && MissionStateManager.isActive()) {
        InteractionManager.runAfterInteractions(async () => {
          await MissionStateManager.syncPendingActions();
          try {
            const fresh = await requestMissionSync("foreground");
            await MissionStateManager.updateFromServer(fresh);
            if (MissionStateManager.isActive()) {
              const inputs = await buildBgTrackingInputs({
                isAuthenticated: !!isDriverAuthenticated,
                role: mode === "driver" ? "driver" : "enterprise",
                hasActiveMission: MissionStateManager.isActive(),
              });
              const refreshed = await refreshBackgroundTrackingNotification(inputs);
              if (!refreshed) {
                await showMissionNotification(MissionStateManager.getState());
              }
            } else {
              await dismissMissionNotification();
            }
          } catch { /* best-effort */ }
        });
      }
    });

    const unsubState = MissionStateManager.subscribe(async (event) => {
      if (event === "mission_stopped") {
        await dismissMissionNotification();
      } else if (event === "state_changed" || event === "transition_confirmed" || event === "reconciliation") {
        if (MissionStateManager.isActive()) {
          const inputs = await buildBgTrackingInputs({
            isAuthenticated: !!isDriverAuthenticated,
            role: mode === "driver" ? "driver" : "enterprise",
            hasActiveMission: MissionStateManager.isActive(),
          });
          const refreshed = await refreshBackgroundTrackingNotification(inputs);
          if (!refreshed) {
            await showMissionNotification(MissionStateManager.getState());
          }
        }
      }
    });

    return () => {
      unsubNotifee();
      appStateSub.remove();
      unsubState();
    };
  }, [isDriverAuthenticated, mode]);

  // Plan 2G/3G Phase 6 : fallback 60s et heartbeat 60s migrés vers syncEngine

  const openNavigation = useCallback(async (destination: string, mission?: Booking) => {
    if (mission && Platform.OS !== "web") {
      await MissionStateManager.startMission(mission, destination);
      await showMissionNotification(MissionStateManager.getState());
    }
    await openNavigationApp(destination);
  }, []);

  const handleOpenModal = useCallback((missionId: number) => {
    setCompletingMissionId(missionId);
    setModalVisible(true);
  }, []);

  const confirmCompletion = useCallback(async () => {
    if (!completingMissionId || isSubmitting) return;

    const mission = missions.find((m) => m.id === completingMissionId);
    if (!mission) {
      setModalVisible(false);
      return;
    }

    // Bloquer les doubles clics
    setIsSubmitting(true);

    try {
      log.info("updating status to completed", { bookingId: mission.id });

      if (Platform.OS !== "web" && MissionStateManager.isActive()) {
        const res = await MissionStateManager.requestTransition("COMPLETED");
        if (!res.ok) {
          if (res.reason === "network_unavailable") {
            appAlert.showAlert(
              "Connexion",
              "Impossible de confirmer que cette course vous est toujours assignée. Veuillez actualiser."
            );
          } else if (
            res.reason === "invalidated_reassigned" ||
            res.reason === "not_assigned_to_driver"
          ) {
            appAlert.showAlert(
              "Mission réassignée",
              "Cette course n'est plus assignée à vous."
            );
          } else {
            appAlert.showAlert(
              "Action impossible",
              "La mise à jour du statut a été refusée. Actualisez la liste des missions."
            );
          }
          setIsSubmitting(false);
          return;
        }
        await MissionStateManager.stopMission();
        await dismissMissionNotification();
      } else {
        const { updateTripStatus } = await import("@/services/api");
        const statusToSend: BookingStatus = mission.is_return ? "RETURN_COMPLETED" : "COMPLETED";
        await updateTripStatus(mission.id, statusToSend);
      }

      // ✅ Phase 1 - Quick Wins: Annuler le rappel local pour cette mission
      await cancelMissionReminder(mission.id);

      setMissions((prev) =>
        prev.filter((m) => {
          if (m.id === mission.id) return false;
          const s = (m.status || "").toLowerCase();
          return !["completed", "return_completed", "canceled", "cancelled"].includes(s);
        })
      );

      // Fermer le modal après succès
      setModalVisible(false);
      setCompletingMissionId(null);

      log.success("mission completed", { bookingId: mission.id });
    } catch (error: any) {
      const msg =
        error.response?.data?.error ||
        error.response?.data?.message ||
        "Impossible de terminer la mission.";
      appAlert.showAlert("Erreur", msg);
      log.error("confirm completion failed", { error });
    } finally {
      // Toujours débloquer le bouton
      setIsSubmitting(false);
    }
  }, [completingMissionId, missions, isSubmitting, appAlert]);

  if (!driver) {
    return (
      <View
        style={{
          flex: 1,
          justifyContent: "center",
          alignItems: "center",
          backgroundColor: "#f4f7fc",
        }}
      >
        <Loader />
      </View>
    );
  }

  const showMissionSkeleton = isLoading && displayMissions.length === 0;
  const showSyncBanner = isRefreshing || (isLoading && missions.length > 0);

  return (
    <View style={{ flex: 1, backgroundColor: "#f4f7fc" }}>
      <ScrollView
        style={{ flex: 1 }}
        refreshControl={
          <RefreshControl
            refreshing={isRefreshing}
            onRefresh={onRefresh}
            colors={["#00796b"]}
            tintColor="#00796b"
          />
        }
      >
        <MissionHeader
          driverName={driver.first_name || "Chauffeur"}
          missionCount={activeMissions.length}
        />

        {showSyncBanner ? (
          <View
            style={[
              styles.syncBanner,
              { paddingHorizontal: horizontalPadding },
            ]}
          >
            <ActivityIndicator size="small" color="#00796b" />
            <Text style={styles.syncBannerText}>
              Synchronisation des missions…
            </Text>
          </View>
        ) : null}

        <TrackingStateBanner
          displayState={trackingState.displayState}
          onRequestPermission={async () => {
            await requestBackgroundPermissionIfNeeded();
          }}
        />

        {location ? (
          <MissionMap
            location={location}
            destination={nextDestination ?? ""}
            destinationCoords={
              nextDestination ? nextDestinationCoords : null
            }
            allowGeocodeFallback={false}
            contentWidth={contentWidth}
            mapHeight={mapHeight}
          />
        ) : null}

        {showMissionSkeleton ? (
          <MissionListSkeleton
            count={3}
            horizontalPadding={horizontalPadding}
          />
        ) : displayMissions.length > 0 ? (
          <View style={{ paddingHorizontal: horizontalPadding, paddingTop: 4 }}>
            {displayMissions.map((displayMission, index) => {
              const { mission, missionNumber, groupInfo } = displayMission;
              const previousMission = index > 0 ? displayMissions[index - 1] : null;
              const showGroupHeader =
                groupInfo.isGrouped &&
                groupInfo.isFirstInGroup &&
                (!previousMission ||
                  previousMission.groupInfo.groupId !== groupInfo.groupId);

              return (
                <React.Fragment key={mission.id}>
                  {showGroupHeader && (
                    <MissionGroupHeader
                      location={groupInfo.groupLocationDisplay}
                      count={groupInfo.groupSize}
                      type={groupInfo.groupType}
                    />
                  )}
                  <MissionCard
                    contentWidth={contentWidth}
                    mission={{
                      ...mission,
                      // Utiliser la durée dynamique si disponible, sinon la durée statique
                      duration_seconds:
                        getDuration(mission.id) || mission.duration_seconds,
                    }}
                    missionNumber={missionNumber}
                    isGrouped={groupInfo.isGrouped}
                    getETAToPickup={getETAToPickup}
                    getETAToDropoff={getETAToDropoff}
                    getEstimatedArrival={getEstimatedArrival}
                    getEstimatedArrivalDropoff={getEstimatedArrivalDropoff}
                    getDelayMinutes={getDelayMinutes}
                    hasGPS={hasGPS}
                    etaLoading={etaLoading}
                    onComplete={() => handleOpenModal(mission.id)}
                    callablePhone={getCallablePhone(mission)}
                    onCall={() => {
                      const phone = getCallablePhone(mission);
                      if (phone) {
                        if (Platform.OS === "web") {
                          (window as any).open(`tel:${phone}`);
                          Alert.alert("Appel", "Ouverture de l'appel… Si rien ne se passe, aucun logiciel d'appel n'est peut-être configuré sur cet appareil.");
                        } else {
                          Linking.openURL(`tel:${phone}`);
                        }
                      }
                    }}
                    onNavigate={() => {
                      const normalizedStatus = mission.status?.toUpperCase();
                      const dest =
                        normalizedStatus === "IN_PROGRESS"
                          ? mission.dropoff_location!
                          : mission.pickup_location!;
                      openNavigation(dest, mission);
                    }}
                    onStatusChange={(missionId, newStatus) => {
                      setMissions((prev) => {
                        const updated = prev.map((m) =>
                          m.id === missionId ? { ...m, status: newStatus } : m
                        );
                        // Mettre à jour le cache pour que le statut soit persistant
                        AsyncStorage.setItem(
                          MISSIONS_CACHE_KEY,
                          JSON.stringify(updated)
                        ).catch(() => { });
                        return updated;
                      });
                    }}
                  />
                </React.Fragment>
              );
            })}
          </View>
        ) : (
          <View style={{ flex: 1, alignItems: "center", justifyContent: "center", paddingVertical: 40, paddingHorizontal: horizontalPadding }}>
            <MissionCard.EmptyState contentWidth={contentWidth} />
          </View>
        )}

        <ConfirmCompletionModal
          visible={modalVisible}
          onClose={() => setModalVisible(false)}
          onConfirm={confirmCompletion}
          isLoading={isSubmitting}
        />
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  syncBanner: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 8,
    backgroundColor: "rgba(0, 121, 107, 0.08)",
  },
  syncBannerText: {
    marginLeft: 8,
    fontSize: 13,
    color: "#004d40",
  },
});
