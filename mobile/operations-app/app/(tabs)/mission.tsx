import React, { useEffect, useState, useCallback, useMemo } from "react";
import { ScrollView, Alert, Linking, View, RefreshControl, Platform, AppState } from "react-native";
import { useAuth } from "@/hooks/useAuth";
import { useSocket } from "@/hooks/useSocket";
import { useLocation } from "@/hooks/useLocation";
import { useNotifications } from "@/hooks/useNotifications";
import { useDynamicETA } from "@/hooks/useDynamicETA";
import { useMissionLayout } from "@/hooks/useMissionLayout";
import MissionCard from "@/components/dashboard/MissionCard";
import MissionGroupHeader from "@/components/dashboard/MissionGroupHeader";
import MissionHeader from "@/components/dashboard/MissionHeader";
import MissionMap from "@/components/dashboard/MissionMap";
import ConfirmCompletionModal from "@/components/dashboard/ConfirmCompletionModal";
// SocketStatusIndicator is now integrated into MissionHeader
import { Loader } from "@/components/ui/Loader";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  getAssignedTrips,
  Booking,
  BookingStatus,
} from "@/services/api";
import { MissionStateManager, type MissionBarStatus } from "@/services/missionState";
import { showMissionNotification, dismissMissionNotification } from "@/services/missionBarAndroid";
import { openNavigation as openNavigationApp } from "@/services/deepLinks";
import { registerNotifeeForegroundHandler } from "@/services/missionBarBackground";
import { sendDriverHeartbeat, onBookingsResync } from "@/services/socket";
import {
  organizeMissionsForDisplay,
  getNextDestination,
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
  log.info("mission screen rendered", { timestamp: new Date().toISOString() });

  const { driver } = useAuth();
  const { location } = useLocation();
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
  const [lastUpdate, setLastUpdate] = useState<number>(Date.now()); // Track last update time

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
      const assigned = await getAssignedTrips();

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
      setLastUpdate(Date.now()); // Mettre à jour le timestamp de dernière mise à jour
    } catch {
      Alert.alert("Erreur", "Impossible de charger les missions.");
    } finally {
      if (!isRefreshAction) {
        setIsLoading(false);
      }
      setIsRefreshing(false);
    }
  }, [driver]);

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
      // ✅ Si la mission a été annulée, afficher un message
      // Note: Si la mission est libérée (retour à ASSIGNED), elle reste dans la liste
      // et sera réassignée automatiquement, pas besoin d'alerte
      const statusLower = (data.status || "").toLowerCase();
      if (statusLower === "canceled" || statusLower === "cancelled") {
        // ✅ Phase 1 - Quick Wins: Annuler le rappel local
        cancelMissionReminder(data.id);

        Alert.alert(
          "Course annulée",
          "La course a été annulée et sera facturée comme booking annulé."
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
        Alert.alert("❌ Mission annulée", "Une mission a été annulée.");
      }
      setLastUpdate(Date.now()); // Mettre à jour le timestamp
    };

    // ✅ Mission réassignée à un autre chauffeur → rafraîchir la liste + notifier
    const onReassigned = async (payload: any) => {
      try {
        const bookingId = payload?.booking_id ?? payload?.id ?? null;
        log.info("booking reassigned", { bookingId });
        Alert.alert(
          "🔄 Mission réassignée",
          "Une mission a été réassignée. Vos courses vont être mises à jour."
        );
      } catch { }
      // Refresh silencieux pour être sûr de ne plus voir la mission
      loadMissions(true);
      setLastUpdate(Date.now());
    };

    // Gestion de la déconnexion WebSocket
    const onDisconnect = () => {
      log.warn("socket disconnected, reconnecting", {});
      // Le client Socket.IO reconnecte automatiquement, mais on peut forcer un refresh
    };

    // Gestion de la reconnexion WebSocket
    const onReconnect = () => {
      log.info("socket reconnected, reloading missions", {});
      // Recharger les missions après reconnexion
      loadMissions(true); // Refresh silencieux
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
      setLastUpdate(Date.now());
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
  }, [socket, loadMissions]);

  // ✅ Mission Bar: foreground Notifee handler + AppState reconciliation
  useEffect(() => {
    if (Platform.OS === "web") return;

    const unsubNotifee = registerNotifeeForegroundHandler();

    const appStateSub = AppState.addEventListener("change", async (state) => {
      if (state === "active" && MissionStateManager.isActive()) {
        await MissionStateManager.syncPendingActions();
        try {
          const fresh = await getAssignedTrips();
          await MissionStateManager.updateFromServer(fresh);
          if (MissionStateManager.isActive()) {
            await showMissionNotification(MissionStateManager.getState());
          } else {
            await dismissMissionNotification();
          }
        } catch { /* best-effort */ }
      }
    });

    const unsubState = MissionStateManager.subscribe(async (event) => {
      if (event === "mission_stopped") {
        await dismissMissionNotification();
      } else if (event === "state_changed" || event === "transition_confirmed" || event === "reconciliation") {
        if (MissionStateManager.isActive()) {
          await showMissionNotification(MissionStateManager.getState());
        }
      }
    });

    return () => {
      unsubNotifee();
      appStateSub.remove();
      unsubState();
    };
  }, []);

  // ✅ Polling de secours : actualiser toutes les 60s si socket déconnecté ou si pas de missions depuis >60s
  useEffect(() => {
    const pollingInterval = setInterval(() => {
      const now = Date.now();
      const timeSinceLastUpdate = now - lastUpdate;
      const shouldPoll =
        !socket?.connected || // Socket déconnecté
        (missions.length === 0 && timeSinceLastUpdate > 60000); // Pas de missions depuis >60s

      if (shouldPoll) {
        log.info("fallback polling triggered", {
          socketConnected: socket?.connected,
          timeSinceLastUpdateSeconds: Math.round(timeSinceLastUpdate / 1000),
        });
        loadMissions(true); // Refresh silencieux
      }
    }, 60000); // Toutes les 60s

    return () => clearInterval(pollingInterval);
  }, [socket, missions.length, lastUpdate, loadMissions]);

  // ✅ Heartbeat métier : envoyer métadonnées toutes les 60s si socket connecté et mission active
  useEffect(() => {
    if (!socket?.connected || missions.length === 0) return;

    const heartbeatInterval = setInterval(() => {
      if (socket?.connected && missions.length > 0) {
        // Utiliser la première mission active pour le heartbeat
        const firstMission = missions[0];
        sendDriverHeartbeat({
          last_mission_id: firstMission.id,
          location: location?.coords ? {
            lat: location.coords.latitude,
            lon: location.coords.longitude
          } : undefined
        }).catch((err) => {
          log.warn("driver heartbeat error", {
            error: err?.message || String(err),
            timestamp: new Date().toISOString(),
          });
        });
      }
    }, 60000); // Toutes les 60s

    return () => clearInterval(heartbeatInterval);
  }, [socket, missions, location]);

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
        const ok = await MissionStateManager.requestTransition("COMPLETED");
        if (!ok) {
          throw new Error("Transition COMPLETED refusée par le state manager");
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
      Alert.alert("Erreur", msg);
      log.error("confirm completion failed", { error });
    } finally {
      // Toujours débloquer le bouton
      setIsSubmitting(false);
    }
  }, [completingMissionId, missions, isSubmitting]);

  if (!driver || isLoading) {
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

        {location && nextDestination && (
          <MissionMap
            location={location}
            destination={nextDestination}
            contentWidth={contentWidth}
            mapHeight={mapHeight}
          />
        )}

        {displayMissions.length > 0 ? (
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
