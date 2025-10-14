import React, { useEffect, useState } from "react";
import { View, Text, TouchableOpacity, Alert } from "react-native";
import { Ionicons, MaterialIcons } from "@expo/vector-icons";
import type { Booking as Mission } from "@/services/api";
import { styles } from "@/styles/missionCardStyles";
import { updateTripStatus } from "@/services/api";

type Props = {
  mission: Mission | null;
  onCall?: () => void;
  onNavigate?: (destination: string) => void;
  onComplete?: () => void;
  onPressDetails?: () => void;
};

interface MissionCardType extends React.FC<Props> {
  EmptyState: React.FC;
}

const MissionCard: MissionCardType = ({
  mission,
  onCall,
  onNavigate,
  onComplete,
  onPressDetails,
}) => {
  const [status, setStatus] = useState<Mission["status"] | undefined>(
    mission?.status
  );

  useEffect(() => {
    setStatus(mission?.status);
  }, [mission?.status]);

  const formatStatus = (s?: string): string => {
    switch (s) {
      case "assigned":
        return "📦 Assignée";
      case "en_route":
        return "🚗 En route";
      case "in_progress":
        return "🟡 En cours";
      case "completed":
        return "✅ Terminée";
      default:
        return "🕓 À venir";
    }
  };

  const handleStatusUpdate = async (
    newStatus: "en_route" | "in_progress" | "completed"
  ) => {
    if (!mission) return;
    try {
      await updateTripStatus(mission.id, newStatus);
      setStatus(newStatus);
      Object.assign(mission, { status: newStatus });
      if (newStatus === "completed") onComplete?.();
    } catch (error) {
      Alert.alert(
        "Erreur",
        `Impossible de mettre à jour le statut : ${formatStatus(newStatus)}`
      );
    }
  };

  const getCurrentDestination = (): string => {
    if (!mission) return "";
    if (status === "in_progress") return mission.dropoff_location || "";
    if (status === "en_route") return mission.pickup_location || "";
    return "";
  };

  const shouldShowNavigation = status !== "completed" && status !== "canceled";

  if (!mission) {
    return <MissionCard.EmptyState />;
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
    <View style={styles.containerEnhanced}>
      {/* Ligne 1 : Nom et Statut */}
      <View style={styles.headerRowEnhanced}>
        <Text style={styles.clientName}>
          {mission.client_name ||
            mission.customer_name ||
            mission.client?.full_name ||
            "Non spécifié"}
        </Text>
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
            color="#666"
            style={{ marginRight: 2 }}
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
        {status !== "in_progress" && (mission.wheelchair_client_has || mission.wheelchair_need) && (
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
        {status !== "in_progress" &&
          mission.wheelchair &&
          !mission.wheelchair_client_has &&
          !mission.wheelchair_need && (
            <Text style={styles.infoEnhanced}>
              ♿ Transport fauteuil roulant
            </Text>
          )}

        {/* APRÈS le client à bord (in_progress) : Afficher les infos médicales */}
        {status === "in_progress" && (mission.medical_facility ||
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

      {/* Actions */}
      <View style={styles.actionsRowEnhanced}>
        {onCall && (
          <TouchableOpacity onPress={onCall} style={styles.actionItemEnhanced}>
            <Ionicons name="call" size={22} color="white" />
            <Text style={styles.actionLabel}>Appeler</Text>
          </TouchableOpacity>
        )}

        {shouldShowNavigation && onNavigate && (
          <TouchableOpacity
            onPress={() => onNavigate(getCurrentDestination())}
            style={styles.actionItemEnhanced}
          >
            <MaterialIcons name="navigation" size={22} color="white" />
            <Text style={styles.actionLabel}>GPS</Text>
          </TouchableOpacity>
        )}

        {status === "assigned" && (
          <TouchableOpacity
            onPress={() => handleStatusUpdate("en_route")}
            style={styles.actionItemEnhanced}
          >
            <Ionicons name="walk" size={22} color="white" />
            <Text style={styles.actionLabel}>En route</Text>
          </TouchableOpacity>
        )}

        {status === "en_route" && (
          <TouchableOpacity
            onPress={() => handleStatusUpdate("in_progress")}
            style={styles.actionItemEnhanced}
          >
            <Ionicons name="person" size={22} color="white" />
            <Text style={styles.actionLabel}>À bord</Text>
          </TouchableOpacity>
        )}

        {status === "in_progress" && (
          <TouchableOpacity
            onPress={() => handleStatusUpdate("completed")}
            style={styles.actionItemEnhanced}
          >
            <Ionicons name="checkmark-done" size={22} color="white" />
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
              size={22}
              color="white"
            />
            <Text style={styles.actionLabel}>Détails</Text>
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
};

// ✅ Composant visuel réutilisable lorsqu'il n'y a pas de mission
MissionCard.EmptyState = () => (
  <View style={styles.containerEnhanced}>
    <Text style={{ fontSize: 16, textAlign: "center", color: "#666" }}>
      🚗 En attente de course
    </Text>
    <Text
      style={{ fontSize: 14, textAlign: "center", color: "#999", marginTop: 8 }}
    >
      Vous serez notifié dès qu'une mission vous sera assignée.
    </Text>
  </View>
);

export default MissionCard;
