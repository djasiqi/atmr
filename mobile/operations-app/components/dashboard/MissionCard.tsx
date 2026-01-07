import React, { useEffect, useState } from "react";
import { View, Text, TouchableOpacity, Alert } from "react-native";
import { Ionicons, MaterialIcons } from "@expo/vector-icons";
import type { Booking as Mission, BookingStatus } from "@/services/api";
import { styles } from "@/styles/missionCardStyles";
import { styles as groupStyles } from "@/styles/missionGroupStyles";
import { updateTripStatus } from "@/services/api";
import CancelJustificationModal from "./CancelJustificationModal";
import { isCompletedStatus, isCanceledStatus, normalizeBookingStatus } from "@/utils/bookingStatus";

type Props = {
  mission: Mission | null;
  missionNumber?: number; // Numéro de la mission dans le groupe
  isGrouped?: boolean; // true si la mission fait partie d'un groupe
  onCall?: () => void;
  onNavigate?: (destination: string) => void;
  onComplete?: (missionId: number) => void; // Prend maintenant l'ID de la mission
  onPressDetails?: () => void;
  onStatusChange?: (missionId: number, status: BookingStatus) => void;
};

interface MissionCardType extends React.FC<Props> {
  EmptyState: React.FC;
}

// ✅ Composant visuel réutilisable lorsqu'il n'y a pas de mission (style épuré)
const EmptyStateComponent: React.FC = () => (
  <View style={styles.containerEnhanced}>
    <Text style={{ fontSize: 18, textAlign: "center", color: "#15362B", fontWeight: "600", letterSpacing: 0.2 }}>
      🚗 En attente de mission
    </Text>
    <Text
      style={{ fontSize: 15, textAlign: "center", color: "#5F7369", marginTop: 10, lineHeight: 22 }}
    >
      Vous serez notifié dès qu'une mission vous sera assignée.
    </Text>
  </View>
);

const MissionCard: MissionCardType = ({
  mission,
  missionNumber,
  isGrouped = false,
  onCall,
  onNavigate,
  onComplete,
  onPressDetails,
  onStatusChange,
}) => {
  const [status, setStatus] = useState<Mission["status"] | undefined>(
    mission?.status
  );
  const [cancelModalVisible, setCancelModalVisible] = useState(false);
  const [releaseModalVisible, setReleaseModalVisible] = useState(false);
  const [isUpdatingStatus, setIsUpdatingStatus] = useState(false);

  useEffect(() => {
    setStatus(mission?.status);
  }, [mission?.status]);

  // ✅ P0-1: Normaliser les statuts pour correspondre au backend (uppercase)
  const formatStatus = (s?: string): string => {
    const normalized = s?.toUpperCase();
    switch (normalized) {
      case "ASSIGNED":
        return "📦 Assignée";
      case "EN_ROUTE":
        return "🚗 En route";
      case "IN_PROGRESS":
        return "🟡 En cours";
      case "COMPLETED":
        return "✅ Terminée";
      case "CANCELED":
        return "❌ Annulée";
      default:
        return "🕓 À venir";
    }
  };

  // ✅ P0-1: Utiliser les statuts en uppercase
  const handleStatusUpdate = async (
    newStatus: "EN_ROUTE" | "IN_PROGRESS" | "COMPLETED" | "CANCELED",
    cancelReason?: "CANCEL" | "RELEASE" | string
  ) => {
    if (!mission || isUpdatingStatus) return;
    try {
      setIsUpdatingStatus(true);
      await updateTripStatus(mission.id, newStatus, cancelReason as any);
      setStatus(newStatus);
      Object.assign(mission, { status: newStatus });
      onStatusChange?.(mission.id, newStatus);
      if (newStatus === "COMPLETED") onComplete?.(mission.id);
      if (newStatus === "CANCELED") {
        // ✅ Mission annulée : notifier selon le type
        if (cancelReason === "RELEASE") {
          Alert.alert(
            "Course libérée",
            "La course a été libérée. Un autre chauffeur pourra être assigné."
          );
        } else {
          Alert.alert(
            "Course annulée",
            "La course a été annulée avec justification."
          );
        }
      }
    } catch (error: any) {
      const errorMsg =
        error?.response?.data?.error || error?.message || "Erreur inconnue";
      Alert.alert(
        "Erreur",
        `Impossible de mettre à jour le statut : ${errorMsg}`
      );
    } finally {
      setIsUpdatingStatus(false);
    }
  };

  const handleCancelJustification = (reason: string, isClientFault: boolean) => {
    // La raison sera envoyée au backend qui décidera de la facturation
    // ✅ P0-1: Utiliser uppercase
    handleStatusUpdate("CANCELED", reason);
    setCancelModalVisible(false);
  };

  const handleReleaseConfirm = () => {
    handleStatusUpdate("CANCELED", "RELEASE");
    setReleaseModalVisible(false);
  };

  const handleReleasePress = () => {
    console.log("[MissionCard] Bouton Libérer pressé");
    Alert.alert(
      "Libérer la course",
      "Êtes-vous sûr de vouloir libérer cette course pour réassignation ?",
      [
        {
          text: "Non",
          style: "cancel",
          onPress: () => {
            console.log("[MissionCard] Libération annulée");
            setReleaseModalVisible(false);
          },
        },
        {
          text: "Oui, libérer",
          style: "default",
          onPress: () => {
            console.log("[MissionCard] Confirmation de libération");
            handleReleaseConfirm();
          },
        },
      ],
      { cancelable: true, onDismiss: () => setReleaseModalVisible(false) }
    );
  };

  // Gérer le modal de libération avec un hook (fallback si appelé depuis ailleurs)
  React.useEffect(() => {
    if (releaseModalVisible) {
      // Délai pour éviter les doubles appels
      const timer = setTimeout(() => {
        handleReleasePress();
      }, 100);
      return () => clearTimeout(timer);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [releaseModalVisible]);

  // ✅ P0-1: Normaliser le statut pour les comparaisons
  const normalizedStatus = normalizeBookingStatus(status) as BookingStatus;

  const getCurrentDestination = (): string => {
    if (!mission) return "";
    if (normalizedStatus === "IN_PROGRESS") return mission.dropoff_location || "";
    if (normalizedStatus === "EN_ROUTE") return mission.pickup_location || "";
    return "";
  };

  const shouldShowNavigation =
    !isCompletedStatus(status) && !isCanceledStatus(status);

  if (!mission) {
    return <EmptyStateComponent />;
  }

  // DEBUG : Afficher les champs de durée
  console.log("[MissionCard] mission.id:", mission.id);
  console.log(
    "[MissionCard] mission.duration_seconds:",
    mission.duration_seconds
  );
  console.log(
    "[MissionCard] mission.estimated_duration:",
    mission.estimated_duration
  );
  console.log(
    "[MissionCard] mission.distance_meters:",
    mission.distance_meters
  );

  return (
    <View
      style={[
        styles.containerEnhanced,
        isGrouped && groupStyles.groupedCardBorder,
      ]}
    >
      {/* Badge numéroté pour toutes les missions (utile pour référence) */}
      {missionNumber && (
        <View style={groupStyles.missionNumberBadge}>
          <Text style={groupStyles.missionNumberText}>{missionNumber}</Text>
        </View>
      )}

      {/* Ligne 1 : Nom et Statut */}
      <View style={styles.headerRowEnhanced}>
        <View style={{ flex: 1 }}>
          <Text style={styles.clientName}>
            {mission.client_name ||
              // ✅ P1-4 Phase 3.3: Utiliser client_name au lieu de customer_name
              mission.client_name ||
              mission.client?.full_name ||
              "Non spécifié"}
          </Text>
          {mission.client?.birth_date && (
            <Text style={[styles.detailText, { fontSize: 12, marginTop: 4, color: "#5F7369" }]}>
              📅 {new Date(mission.client.birth_date).toLocaleDateString('fr-FR', {
                day: '2-digit',
                month: '2-digit',
                year: 'numeric'
              })}
            </Text>
          )}
        </View>
        <View style={styles.statusBadgeContainer}>
          <Text style={styles.statusBadgeText}>
            {formatStatus(status ?? "")}
          </Text>
        </View>
      </View>

      {/* Ligne 2 : Départ + Heure */}
      <View style={styles.rowBetween}>
        <Text style={styles.infoEnhanced}>📍 Départ :</Text>
        <View style={styles.timeRow}>
          <Ionicons
            name="time-outline"
            size={15}
            color="#5F7369"
            style={{ marginRight: 4 }}
          />
          <Text style={styles.timeEnhanced}>
            {new Date(mission.scheduled_time).toLocaleTimeString([], {
              hour: "2-digit",
              minute: "2-digit",
            })}
          </Text>
        </View>
      </View>
      {/* Ligne 3 : Adresse Départ */}
      <Text style={styles.detailText}>{mission.pickup_location}</Text>

      {/* Ligne 4 : Arrivée + Durée estimée */}
      {mission.dropoff_location && (
        <>
          <View style={styles.rowBetween}>
            <Text style={styles.infoEnhanced}>🏁 Arrivée :</Text>
            {/* Durée estimée formatée depuis duration_seconds */}
            <Text style={styles.timeEnhanced}>
              {mission.duration_seconds
                ? `${Math.round(mission.duration_seconds / 60)} min`
                : mission.estimated_duration || "Durée inconnue"}
            </Text>
          </View>
          {/* Ligne 5 : Adresse Arrivée */}
          <Text style={styles.detailText}>{mission.dropoff_location}</Text>
        </>
      )}

      {/* Infos supplémentaires */}
      <View style={styles.metaInfoSection}>
        {/* AVANT le client à bord (assigned, en_route) : Afficher les infos chaise roulante */}
        {normalizedStatus !== "IN_PROGRESS" && (mission.wheelchair_client_has || mission.wheelchair_need) && (
          <View style={styles.wheelchairSection}>
            {mission.wheelchair_client_has && (
              <Text style={styles.wheelchairAlert}>
                ♿ Client en chaise roulante
              </Text>
            )}
            {mission.wheelchair_need && (
              <Text style={styles.wheelchairAlert}>
                🏥 Prendre une chaise roulante
              </Text>
            )}
          </View>
        )}

        {/* Ancien champ wheelchair (gardé pour compatibilité) - seulement avant client à bord */}
        {normalizedStatus !== "IN_PROGRESS" &&
          mission.wheelchair &&
          !mission.wheelchair_client_has &&
          !mission.wheelchair_need && (
            <Text style={styles.infoEnhanced}>
              ♿ Transport fauteuil roulant
            </Text>
          )}

        {/* APRÈS le client à bord (in_progress) : Afficher les infos médicales */}
        {normalizedStatus === "IN_PROGRESS" && (mission.medical_facility ||
          mission.doctor_name ||
          mission.hospital_service) && (
            <View style={styles.medicalInfoSection}>
              <Text style={styles.medicalTitle}>🏥 Destination médicale</Text>
              {mission.medical_facility && (
                <Text style={styles.medicalDetail}>
                  📍 {mission.medical_facility}
                </Text>
              )}
              {mission.doctor_name && (
                <Text style={styles.medicalDetail}>
                  👨‍⚕️ Dr {mission.doctor_name}
                </Text>
              )}
              {mission.hospital_service && (
                <Text style={styles.medicalDetail}>
                  🚪 {mission.hospital_service}
                </Text>
              )}
            </View>
          )}

        {/* Notes médicales - toujours visibles */}
        {mission.notes_medical && (
          <Text style={styles.notesEnhanced}>
            📝 Notes : {mission.notes_medical}
          </Text>
        )}
        {mission.notes && (
          <Text style={styles.notesEnhanced}>📝 {mission.notes}</Text>
        )}
      </View>

      {/* Actions principales : Appeler, GPS, En route/À bord/Terminer */}
      <View style={styles.actionsRowEnhanced}>
        {onCall && (
          <TouchableOpacity onPress={onCall} style={styles.actionItemEnhanced}>
            <Ionicons name="call" size={18} color="white" />
            <Text style={styles.actionLabel}>Appeler</Text>
          </TouchableOpacity>
        )}

        {shouldShowNavigation && onNavigate && (
          <TouchableOpacity
            onPress={() => onNavigate(getCurrentDestination())}
            style={styles.actionItemEnhanced}
            disabled={isUpdatingStatus}
          >
            <MaterialIcons name="navigation" size={18} color="white" />
            <Text style={styles.actionLabel}>GPS</Text>
          </TouchableOpacity>
        )}

        {normalizedStatus === "ASSIGNED" && (
          <TouchableOpacity
            onPress={() => handleStatusUpdate("EN_ROUTE")}
            style={styles.actionItemEnhanced}
            disabled={isUpdatingStatus}
          >
            <Ionicons name="walk" size={18} color="white" />
            <Text style={styles.actionLabel}>En route</Text>
          </TouchableOpacity>
        )}

        {normalizedStatus === "EN_ROUTE" && (
          <TouchableOpacity
            onPress={() => handleStatusUpdate("IN_PROGRESS")}
            style={styles.actionItemEnhanced}
            disabled={isUpdatingStatus}
          >
            <Ionicons name="person" size={18} color="white" />
            <Text style={styles.actionLabel}>À bord</Text>
          </TouchableOpacity>
        )}

        {normalizedStatus === "IN_PROGRESS" && (
          <TouchableOpacity
            onPress={() => {
              // Ouvrir le modal de confirmation au lieu de terminer directement
              if (mission && onComplete) {
                onComplete(mission.id);
              } else {
                handleStatusUpdate("COMPLETED");
              }
            }}
            style={styles.actionItemEnhanced}
          >
            <Ionicons name="checkmark-done" size={18} color="white" />
            <Text style={styles.actionLabel}>Terminer</Text>
          </TouchableOpacity>
        )}

        {onPressDetails && (
          <TouchableOpacity
            onPress={onPressDetails}
            style={styles.actionItemEnhanced}
          >
            <Ionicons
              name="information-circle-outline"
              size={18}
              color="white"
            />
            <Text style={styles.actionLabel}>Détails</Text>
          </TouchableOpacity>
        )}
      </View>

      {/* Actions secondaires : Annuler (en dessous) */}
      {(normalizedStatus === "ASSIGNED" || normalizedStatus === "EN_ROUTE") && (
        <View style={styles.actionsRowSecondary}>
          <TouchableOpacity
            onPress={handleReleasePress}
            activeOpacity={0.7}
            style={[
              styles.actionItemEnhanced,
              { backgroundColor: "#6c757d", flex: 1, maxWidth: "48%" },
            ]}
          >
            <Ionicons name="refresh" size={18} color="white" />
            <Text style={styles.actionLabel}>Libérer</Text>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => setCancelModalVisible(true)}
            style={[
              styles.actionItemEnhanced,
              { backgroundColor: "#dc3545", flex: 1, maxWidth: "48%" },
            ]}
          >
            <Ionicons name="close-circle" size={18} color="white" />
            <Text style={styles.actionLabel}>Annuler</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Modal de justification d'annulation */}
      <CancelJustificationModal
        visible={cancelModalVisible}
        onClose={() => setCancelModalVisible(false)}
        onConfirm={handleCancelJustification}
      />

    </View>
  );
};

// ✅ Attacher EmptyStateComponent comme propriété statique pour compatibilité
MissionCard.EmptyState = EmptyStateComponent;

export default MissionCard;
