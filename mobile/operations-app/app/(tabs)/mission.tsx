import React, { useEffect, useState, useCallback } from "react";
import { ScrollView, Alert, Linking, View, RefreshControl } from "react-native";
import { useAuth } from "@/hooks/useAuth";
import { useSocket } from "@/hooks/useSocket";
import { useLocation } from "@/hooks/useLocation";
import { useNotifications } from "@/hooks/useNotifications";
import { useDynamicETA } from "@/hooks/useDynamicETA";
import MissionCard from "@/components/dashboard/MissionCard";
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
import { sendDriverHeartbeat } from "@/services/socket";

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
  const [currentIndex, setCurrentIndex] = useState(0);
  const [modalVisible, setModalVisible] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<number>(Date.now()); // Track last update time

  const currentMission = missions[currentIndex] || null;
  const MISSIONS_CACHE_KEY = "missions_cache_v1";

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
              ].includes((m.status || "").toLowerCase())
          );
          if (active.length) {
            const sorted = active.sort(
              (a, b) =>
                new Date(a.scheduled_time).getTime() -
                new Date(b.scheduled_time).getTime()
            );
            setMissions(sorted);
            setCurrentIndex(0);
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
      setCurrentIndex(0);
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
      // Réinitialiser l'index pour afficher la mission la plus proche (première dans la liste triée)
      setCurrentIndex(0);
    };

    const onUpdate = (data: Booking) => {
      setMissions((prev) => {
        const updated = prev
          .map((m) => (m.id === data.id ? data : m))
          .filter((m) => {
            const s = (m.status || "").toLowerCase();
            // ✅ Filtrer les missions terminées ou annulées (mais pas les libérées qui reviennent à ASSIGNED)
            return !["completed", "return_completed", "canceled", "cancelled"].includes(s);
          })
          .sort(
            (a, b) =>
              new Date(a.scheduled_time).getTime() -
              new Date(b.scheduled_time).getTime()
          );
        // recalcul de l'index pour éviter l'affichage d'une mission terminée
        if (updated.length === 0) {
          setCurrentIndex(0);
        } else {
          // Réinitialiser à 0 pour toujours afficher la mission la plus proche
          setCurrentIndex(0);
        }
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
      if (currentMission?.id === id) {
        Alert.alert("❌ Mission annulée", "La mission en cours a été annulée.");
        setCurrentIndex(0); // Tu peux l'améliorer plus tard si besoin
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
    return () => {
      socket.off("new_booking", onNew);
      socket.off("booking_updated", onUpdate);
      socket.off("booking_cancelled", onCancel);
      socket.off("disconnect", onDisconnect);
      socket.off("reconnect", onReconnect);
    };
  }, [socket, currentMission?.id, loadMissions]);

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
    if (!socket?.connected || !currentMission) return;

    const heartbeatInterval = setInterval(() => {
      if (socket?.connected && currentMission) {
        sendDriverHeartbeat({
          last_mission_id: currentMission.id,
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
  }, [socket, currentMission, location]);

  const openNavigation = (destination: string) => {
    const url = `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(destination)}`;
    Linking.openURL(url);
  };

  const handleOpenModal = () => {
    setModalVisible(true);
  };

  const confirmCompletion = useCallback(async () => {
    console.log("Confirmer la fin de mission");
    if (!currentMission || isSubmitting) return;

    // Bloquer les doubles clics
    setIsSubmitting(true);

    try {
      const isReturn = !!currentMission.is_return;
      const statusToSend: BookingStatus = isReturn
        ? "return_completed"
        : "completed";

      console.log("[Mission] Mise à jour du statut:", statusToSend, "pour booking", currentMission.id);

      await updateTripStatus(currentMission.id, statusToSend);

      // Mettre à jour la liste des missions (retirer la mission terminée)
      setMissions((prev) =>
        prev
          .map((m) =>
            m.id === currentMission.id ? { ...m, status: statusToSend } : m
          )
          .filter(
            (m) => {
              const s = (m.status || "").toLowerCase();
              return !["completed", "return_completed", "canceled", "cancelled"].includes(s);
            }
          )
      );

      // Passer à la prochaine mission
      setCurrentIndex(0);

      // Fermer le modal après succès
      setModalVisible(false);

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
  }, [currentMission, isSubmitting]);

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

        {location && currentMission && (
          <MissionMap
            location={location}
            destination={
              currentMission.status === "in_progress"
                ? currentMission.dropoff_location!
                : currentMission.pickup_location!
            }
          />
        )}

        {currentMission ? (
          <View className="px-4 pt-4">
            <MissionCard
              mission={{
                ...currentMission,
                // Utiliser la durée dynamique si disponible, sinon la durée statique
                duration_seconds: getDuration(currentMission.id) || currentMission.duration_seconds
              }}
              onComplete={handleOpenModal}
              onCall={() =>
                currentMission.client_phone &&
                Linking.openURL(`tel:${currentMission.client_phone}`)
              }
              onNavigate={() => {
                const dest =
                  currentMission.status === "in_progress"
                    ? currentMission.dropoff_location!
                    : currentMission.pickup_location!;
                openNavigation(dest);
              }}
            />
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
