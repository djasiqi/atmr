import React, { useEffect, useState, useCallback, useMemo } from "react";
import { ScrollView, Alert, Linking, View, RefreshControl } from "react-native";
import { useAuth } from "@/hooks/useAuth";
import { useSocket } from "@/hooks/useSocket";
import { useLocation } from "@/hooks/useLocation";
import { useNotifications } from "@/hooks/useNotifications";
import { useDynamicETA } from "@/hooks/useDynamicETA";
import MissionCard from "@/components/dashboard/MissionCard";
import MissionGroupHeader from "@/components/dashboard/MissionGroupHeader";
import MissionHeader from "@/components/dashboard/MissionHeader";
import MissionMap from "@/components/dashboard/MissionMap";
import ConfirmCompletionModal from "@/components/dashboard/ConfirmCompletionModal";
import SocketStatusIndicator from "@/components/common/SocketStatusIndicator";
import { Loader } from "@/components/ui/Loader";
import AsyncStorage from "@react-native-async-storage/async-storage";
import {
  getAssignedTrips,
  updateTripStatus,
  Booking,
  BookingStatus,
} from "@/services/api";
import { sendDriverHeartbeat, onBookingsResync } from "@/services/socket";
import {
  organizeMissionsForDisplay,
  getNextDestination,
  type DisplayMission,
} from "@/utils/missionGrouping";

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
  // Log tout le reste pour analyse
  console.log(
    "[isMissionReturn] Valeur inattendue:",
    is_return,
    typeof is_return
  );
  return false;
}

export default function MissionScreen() {
  const { driver } = useAuth();
  const { location } = useLocation();
  const socket = useSocket();
  useNotifications();

  // Hook pour les ETAs dynamiques basés sur la position GPS
  const { etas, hasGPS, getDuration } = useDynamicETA(!!driver);

  const [isLoading, setIsLoading] = useState(true);
  const [missions, setMissions] = useState<Booking[]>([]);
  const [modalVisible, setModalVisible] = useState(false);
  const [completingMissionId, setCompletingMissionId] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<number>(Date.now()); // Track last update time

  const MISSIONS_CACHE_KEY = "missions_cache_v1";

  // Organiser les missions pour l'affichage avec groupement
  const displayMissions = useMemo(() => {
    return organizeMissionsForDisplay(missions);
  }, [missions]);

  // Trouver la prochaine destination pour la carte
  const nextDestination = useMemo(() => {
    return getNextDestination(missions);
  }, [missions]);

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
        Alert.alert(
          "Course annulée",
          "La course a été annulée et sera facturée comme booking annulé."
        );
      }
    };

    const onCancel = ({ id }: { id: number }) => {
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

    // Gestion de la déconnexion WebSocket
    const onDisconnect = () => {
      console.log("⚠️ [Mission] Socket déconnecté, tentative de reconnexion...");
      // Le client Socket.IO reconnecte automatiquement, mais on peut forcer un refresh
    };

    // Gestion de la reconnexion WebSocket
    const onReconnect = () => {
      console.log("✅ [Mission] Socket reconnecté, rechargement des missions...");
      // Recharger les missions après reconnexion
      loadMissions(true); // Refresh silencieux
    };

    socket.on("new_booking", onNew);
    socket.on("booking_updated", onUpdate);
    socket.on("booking_cancelled", onCancel);
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
      socket.off("disconnect", onDisconnect);
      socket.off("reconnect", onReconnect);
      unsubscribeResync();
    };
  }, [socket, loadMissions]);

  // ✅ Polling de secours : actualiser toutes les 60s si socket déconnecté ou si pas de missions depuis >60s
  useEffect(() => {
    const pollingInterval = setInterval(() => {
      const now = Date.now();
      const timeSinceLastUpdate = now - lastUpdate;
      const shouldPoll =
        !socket?.connected || // Socket déconnecté
        (missions.length === 0 && timeSinceLastUpdate > 60000); // Pas de missions depuis >60s

      if (shouldPoll) {
        console.log(`🔄 [Mission] Polling de secours déclenché (socket: ${socket?.connected ? 'connecté' : 'déconnecté'}, timeSinceLastUpdate: ${Math.round(timeSinceLastUpdate / 1000)}s)`);
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
          console.warn(JSON.stringify({
            event: "driver_heartbeat_error",
            error: err?.message || String(err),
            timestamp: new Date().toISOString()
          }));
        });
      }
    }, 60000); // Toutes les 60s

    return () => clearInterval(heartbeatInterval);
  }, [socket, missions, location]);

  const openNavigation = (destination: string) => {
    const url = `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(destination)}`;
    Linking.openURL(url);
  };

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
      const isReturn = !!mission.is_return;
      // ✅ P0-1: Utiliser les statuts en UPPERCASE pour correspondre au backend
      const statusToSend: BookingStatus = isReturn
        ? "RETURN_COMPLETED"
        : "COMPLETED";

      console.log("[Mission] Mise à jour du statut:", statusToSend, "pour booking", mission.id);

      await updateTripStatus(mission.id, statusToSend);

      // Mettre à jour la liste des missions (retirer la mission terminée)
      setMissions((prev) =>
        prev
          .map((m) =>
            m.id === mission.id ? { ...m, status: statusToSend } : m
          )
          .filter(
            (m) => {
              const s = (m.status || "").toLowerCase();
              return !["completed", "return_completed", "canceled", "cancelled"].includes(s);
            }
          )
      );

      // Fermer le modal après succès
      setModalVisible(false);
      setCompletingMissionId(null);

      console.log("✅ Mission terminée avec succès");
    } catch (error: any) {
      const msg =
        error.response?.data?.error ||
        error.response?.data?.message ||
        "Impossible de terminer la mission.";
      Alert.alert("Erreur", msg);
      console.error("[Mission] Erreur lors de la confirmation:", error);
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
          backgroundColor: "#F5F7F6", // ✅ Fond épuré cohérent avec le login
        }}
      >
        <Loader />
      </View>
    );
  }

  return (
    <View style={{ flex: 1, backgroundColor: "#F5F7F6" }}>
      {/* 🆕 Indicateur de statut connexion Socket.IO */}
      <SocketStatusIndicator />

      <ScrollView
        style={{ flex: 1 }}
        refreshControl={
          <RefreshControl
            refreshing={isRefreshing}
            onRefresh={onRefresh}
            colors={["#0A7F59"]} // Android - accent color
            tintColor="#0A7F59" // iOS - accent color
          />
        }
      >
        <MissionHeader
          driverName={driver.first_name || "Chauffeur"}
          date={new Date().toLocaleDateString()}
        />

        {location && nextDestination && (
          <MissionMap
            location={location}
            destination={nextDestination}
          />
        )}

        {displayMissions.length > 0 ? (
          <View className="px-4 pt-4">
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
                    mission={{
                      ...mission,
                      // Utiliser la durée dynamique si disponible, sinon la durée statique
                      duration_seconds:
                        getDuration(mission.id) || mission.duration_seconds,
                    }}
                    missionNumber={missionNumber}
                    isGrouped={groupInfo.isGrouped}
                    onComplete={() => handleOpenModal(mission.id)}
                    onCall={() =>
                      // ✅ P1-4 Phase 3.1: Utiliser client.contact_phone au lieu de client_phone
                      (mission.client?.contact_phone || mission.client_phone) &&
                      Linking.openURL(`tel:${mission.client?.contact_phone || mission.client_phone}`)
                    }
                    onNavigate={() => {
                      const dest =
                        mission.status === "in_progress"
                          ? mission.dropoff_location!
                          : mission.pickup_location!;
                      openNavigation(dest);
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
          <View className="flex-1 items-center justify-center py-10 px-4">
            <MissionCard.EmptyState />
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
