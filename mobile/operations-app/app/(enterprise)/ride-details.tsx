import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Modal,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { useAppAlert } from "@/contexts/AppAlertContext";
import { router, useLocalSearchParams } from "expo-router";
import * as Crypto from "expo-crypto";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import "dayjs/locale/fr";
import { Ionicons } from "@expo/vector-icons";
import { isCompletedStatus, normalizeBookingStatus } from "@/utils/bookingStatus";
import { createShadow } from "@/styles/shadowStyles";
import { isPickupSentinel } from "@/utils/urgentTime";
import { TransferRideModal } from "@/components/enterprise/transfers/TransferRideModal";
import { getLogger } from "@/utils/logger";

const log = getLogger("RideDetail");

const BRAND = "#00796B";
const TEXT = "#1E293B";
const TEXT_SEC = "#64748B";
const TEXT_MUTED = "#94A3B8";
const BORDER = "rgba(0,121,107,0.08)";
const BG = "#f4f7fc";
const CARD = "#FFFFFF";
const DANGER = "#dc3545";

// ✅ Couleurs de statut alignées avec le frontend web et RideSnippetCard
const statusColors = {
  pending: {
    bg: "#fef3c7", // --warning-bg
    text: "#f59e0b", // --warning-primary
  },
  accepted: {
    bg: "#dbeafe", // --info-bg
    text: "#3b82f6", // --info-primary
  },
  assigned: {
    bg: "#dbeafe", // --info-bg
    text: "#3b82f6", // --info-primary
  },
  en_route: {
    bg: "#fef3c7", // --warning-bg (orange clair)
    text: "#f59e0b", // --warning-primary (orange)
  },
  in_progress: {
    bg: "#fef3c7", // --warning-bg (orange clair)
    text: "#f59e0b", // --warning-primary (orange)
  },
  completed: {
    bg: "#dcfce7", // --success-bg
    text: "#16a34a", // --success-primary
  },
  return_completed: {
    bg: "#dcfce7", // --success-bg
    text: "#16a34a", // --success-primary
  },
  cancelled: {
    bg: "#f3f4f6", // --bg-hover
    text: "#6b7280", // --text-tertiary
  },
  canceled: {
    bg: "#f3f4f6", // --bg-hover
    text: "#6b7280", // --text-tertiary
  },
};

// ✅ Fonction pour obtenir les couleurs selon le statut
const getStatusColors = (status?: string) => {
  if (!status) return statusColors.pending;
  const normalizedStatus = status.toLowerCase().trim();
  // ✅ Vérifier si le statut existe dans statusColors
  if (normalizedStatus in statusColors) {
    return statusColors[normalizedStatus as keyof typeof statusColors];
  }
  // ✅ Fallback vers pending si le statut n'est pas reconnu
  return statusColors.pending;
};

import { useAuth } from "@/hooks/useAuth";
import {
  assignRide,
  cancelRide,
  getDispatchRideDetails,
  reassignRide,
  markRideUrgent,
  scheduleRide,
} from "@/services/enterpriseDispatch";
import {
  DriverSuggestion,
  RideConflict,
  RideDetail,
  RideEvent,
} from "@/types/enterpriseDispatch";

dayjs.extend(relativeTime);
dayjs.locale("fr");

const CANCEL_REASONS = [
  { code: "CLIENT_CANCELLED", label: "Annulation côté client" },
  { code: "MEDICAL_CANCELLED", label: "Annulation clinique/médical" },
  { code: "DISPATCH_ERROR", label: "Erreur dispatch" },
  { code: "OTHER", label: "Autre raison" },
];

export default function RideDetailsScreen() {
  const { rideId } = useLocalSearchParams<{ rideId?: string }>();
  const { enterpriseSession } = useAuth();
  const appAlert = useAppAlert();
  const dispatchMode = (enterpriseSession?.company?.dispatchMode as "manual" | "semi_auto" | "fully_auto" | undefined) || "manual";
  const isManualMode = dispatchMode === "manual";

  const [detail, setDetail] = useState<RideDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [manualDriverId, setManualDriverId] = useState("");
  const [manualReason, setManualReason] = useState("");
  const [allowEmergency, setAllowEmergency] = useState(false);
  const [scheduleValue, setScheduleValue] = useState("");
  const [scheduleVisible, setScheduleVisible] = useState(false);
  const [driverPickerVisible, setDriverPickerVisible] = useState(false);
  const [transferModalVisible, setTransferModalVisible] = useState(false);

  const summary = detail?.summary;
  const suggestions = detail?.suggestions ?? [];
  const history = detail?.history ?? [];
  const conflicts = detail?.conflicts ?? [];

  const isAssigned = summary?.status === "assigned" || !!summary?.driver?.id;

  // ✅ Vérifier si l'assignation/annulation est désactivée (completed ou in_board uniquement)
  // Note: in_progress et en_route permettent toujours l'assignation/annulation
  const statusLower = summary?.status?.toLowerCase() || "";
  const isCompleted = statusLower === "completed";
  const isInBoard = statusLower === "in_board";
  const isActionDisabled = isCompleted || isInBoard;

  const loadDetail = useCallback(async () => {
    if (!rideId) return;
    setLoading(true);
    setErrorMessage(null);
    try {
      const data = await getDispatchRideDetails(rideId);
      setDetail(data);
      // ✅ Réinitialiser le message d'erreur en cas de succès
      setErrorMessage(null);
    } catch (error: any) {
      // ✅ Gestion améliorée des erreurs avec messages plus clairs et actions possibles
      const status = error?.response?.status;
      const errorData = error?.response?.data;
      let message = "Impossible de charger la fiche course.";

      if (status === 404) {
        message = "Course introuvable. Elle a peut-être été supprimée.";
      } else if (status === 403) {
        message = "Vous n'avez pas l'autorisation d'accéder à cette course.";
      } else if (status === 500) {
        // ✅ Message plus informatif pour les erreurs 500 avec suggestion de réessayer
        const backendMessage = errorData?.error || errorData?.message;
        if (backendMessage && typeof backendMessage === "string" && backendMessage.length > 0) {
          message = `Erreur serveur: ${backendMessage}\n\nVeuillez réessayer dans quelques instants.`;
        } else {
          message = "Erreur serveur lors du chargement des détails.\n\nVeuillez réessayer dans quelques instants. Si le problème persiste, contactez le support.";
        }
      } else if (errorData?.error) {
        // Utiliser le message d'erreur du backend s'il est disponible
        message = errorData.error;
      } else if (errorData?.message) {
        message = errorData.message;
      } else if (error?.message && error.message !== "Request failed with status code") {
        // Éviter les messages génériques comme "Request failed with status code 500"
        message = error.message;
      }

      log.error("load ride details failed", {
        status,
        message,
        error: errorData || error,
        rideId,
      });

      setErrorMessage(message);
      // ✅ Ne pas définir detail à null pour permettre un retry sans perdre l'état précédent
      // setDetail(null); // Commenté pour permettre un retry
    } finally {
      setLoading(false);
    }
  }, [rideId]);

  useEffect(() => {
    loadDetail();
  }, [loadDetail]);

  const handleAssign = useCallback(
    async (
      driverId: string,
      reason?: string,
      allowEmergencyOverride?: boolean
    ) => {
      if (!rideId) return;
      setActionLoading(true);
      setErrorMessage(null);
      try {
        if (isAssigned) {
          await reassignRide(rideId, {
            driver_id: driverId,
            reason: reason ?? manualReason ?? undefined,
            allow_emergency: allowEmergencyOverride ?? allowEmergency,
            respect_preferences: true,
            idempotency_key: Crypto.randomUUID(),
          });
        } else {
          await assignRide(rideId, {
            driver_id: driverId,
            reason: reason ?? manualReason ?? undefined,
            allow_emergency: allowEmergencyOverride ?? allowEmergency,
            respect_preferences: true,
            idempotency_key: Crypto.randomUUID(),
          });
        }
        await loadDetail();
        Alert.alert("Assignation effectuée", "La course a été mise à jour.");
      } catch (error: any) {
        // ✅ Gestion spécifique des erreurs 409 (conflit d'assignation)
        const status = error?.response?.status;
        const responseMessage =
          error?.response?.data?.error ??
          error?.response?.data?.message ??
          error?.message;

        if (status === 409) {
          // Erreur de conflit : afficher un message clair avec Alert
          const conflictMessage = responseMessage ||
            "Le chauffeur est déjà assigné à une autre course à ce moment. Veuillez choisir un autre chauffeur ou modifier l'horaire.";
          appAlert.showAlert(
            "⚠️ Conflit d'assignation",
            conflictMessage + "\n\nVeuillez choisir un autre chauffeur ou modifier l'horaire de la course."
          );
          setErrorMessage(conflictMessage);
        } else {
          // Autres erreurs
          setErrorMessage(
            responseMessage ||
            "Impossible de finaliser l'assignation. Vérifiez les validations (fairness, préférences, conflits)."
          );
        }
      } finally {
        setActionLoading(false);
      }
    },
    [allowEmergency, isAssigned, loadDetail, manualReason, rideId, appAlert]
  );

  const handleMarkUrgent = useCallback(async () => {
    if (!rideId) return;
    if (!isPickupSentinel(detail?.summary?.time?.pickup_at)) {
      setErrorMessage("Course déjà planifiée (urgent indisponible).");
      return;
    }
    setActionLoading(true);
    setErrorMessage(null);
    try {
      await markRideUrgent(rideId, { extra_delay_minutes: 15 });
      await loadDetail();
      appAlert.showAlert(
        "Urgence enregistrée",
        "La course est marquée urgente (+15 min)."
      );
    } catch (error: any) {
      if (error?.response?.status === 409) {
        setErrorMessage("Course déjà planifiée (urgent indisponible).");
        return;
      }
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de marquer la course en urgence.";
      setErrorMessage(message);
    } finally {
      setActionLoading(false);
    }
  }, [detail?.summary?.time?.pickup_at, loadDetail, rideId]);

  const handleScheduleConfirm = useCallback(async () => {
    if (!rideId) return;
    const raw = scheduleValue.trim();
    if (!raw) {
      setScheduleVisible(false);
      setScheduleValue("");
      return;
    }
    const [hour, minute] = raw.split(":");
    if (
      hour === undefined ||
      minute === undefined ||
      Number.isNaN(Number(hour)) ||
      Number.isNaN(Number(minute))
    ) {
      setErrorMessage("Format horaire invalide (HH:mm).");
      return;
    }
    const baseTime =
      detail?.summary?.time?.pickup_at ??
      detail?.summary?.time?.window_start ??
      dayjs().toISOString();
    const isoDate = dayjs(baseTime)
      .set("hour", Number(hour))
      .set("minute", Number(minute))
      .set("second", 0)
      .toISOString();
    setActionLoading(true);
    try {
      await scheduleRide(rideId, { pickup_at: isoDate });
      await loadDetail();
      const isSentinel = Number(hour) === 0 && Number(minute) === 0;
      appAlert.showAlert(
        "Horaire planifié",
        isSentinel
          ? "Heure replacée à « À définir »."
          : `Pickup replanifié à ${hour.padStart(2, "0")}:${minute.padStart(2, "0")}.`
      );
      setScheduleVisible(false);
      setScheduleValue("");
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de planifier l’horaire.";
      setErrorMessage(message);
    } finally {
      setActionLoading(false);
    }
  }, [
    detail?.summary?.time?.pickup_at,
    detail?.summary?.time?.window_start,
    loadDetail,
    rideId,
    scheduleValue,
    appAlert,
  ]);

  const handleCancel = useCallback(async () => {
    if (!rideId) return;
    Alert.alert(
      "Annuler la course",
      "Choisissez une raison d’annulation :",
      CANCEL_REASONS.map((reason) => ({
        text: reason.label,
        onPress: async () => {
          setActionLoading(true);
          setErrorMessage(null);
          try {
            await cancelRide(rideId, reason.code);
            await loadDetail();
            appAlert.showAlert("Course annulée", "");
          } catch (error: any) {
            const message =
              error?.response?.data?.error ??
              error?.message ??
              "Impossible d’annuler la course.";
            setErrorMessage(message);
          } finally {
            setActionLoading(false);
          }
        },
      })),
      { cancelable: true }
    );
  }, [loadDetail, rideId, appAlert]);

  const manualAssignDisabled =
    manualDriverId.trim().length === 0 || actionLoading;

  const headerTitle = summary
    ? `${summary.client.name} • ${!isPickupSentinel(summary?.time?.pickup_at) ? dayjs(summary.time.pickup_at).format("DD MMM HH:mm") : "⏱️ À définir"}`
    : "Course";

  // ✅ Urgent autorisé uniquement si sentinelle 00:00 (heure non définie)
  const showUrgentActions = isPickupSentinel(summary?.time?.pickup_at);

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator color={BRAND} />
        <Text style={styles.loadingText}>Chargement…</Text>
      </View>
    );
  }

  if (!detail || !summary) {
    return (
      <View style={styles.loadingContainer}>
        <Ionicons name="alert-circle-outline" size={48} color={DANGER} style={{ marginBottom: 16 }} />
        <Text style={styles.errorText}>
          {errorMessage ?? "Course introuvable."}
        </Text>
        <Text style={styles.errorHint}>
          {errorMessage?.includes("500") || errorMessage?.includes("serveur")
            ? "L'erreur peut être temporaire. Réessayez dans quelques instants."
            : "Vérifiez votre connexion et réessayez."}
        </Text>
        <View style={styles.errorActions}>
          <TouchableOpacity style={styles.primaryButton} onPress={loadDetail} disabled={loading}>
            {loading ? (
              <ActivityIndicator color="#FFFFFF" />
            ) : (
              <Text style={styles.primaryButtonText}>Réessayer</Text>
            )}
          </TouchableOpacity>
          <TouchableOpacity style={styles.secondaryButton} onPress={() => router.back()}>
            <Text style={styles.secondaryButtonText}>Retour</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.headerBack} onPress={() => router.back()}>
          <Ionicons name="chevron-back" size={20} color={TEXT_SEC} />
        </TouchableOpacity>
        <View style={styles.headerCenter}>
          <Text style={styles.headerTitle} numberOfLines={1}>{headerTitle}</Text>
          <Text style={styles.headerSub}>#{summary.id.slice(-6)}</Text>
        </View>
        <View style={styles.headerStatusBadge}>
          <View style={[styles.statusDot, { backgroundColor: getStatusColors(summary.status).text }]} />
          <Text style={[styles.statusLabel, { color: getStatusColors(summary.status).text }]}>
            {normalizeBookingStatus(summary.status) === "ASSIGNED" ? "Assignée"
              : isCompletedStatus(summary.status) ? "Terminée"
              : summary.status === "cancelled" ? "Annulée"
              : summary.status === "en_route" ? "En route"
              : summary.status === "in_progress" ? "En cours"
              : "En attente"}
          </Text>
        </View>
      </View>

      {/* ✅ Section Informations client (enrichie) */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Informations client</Text>
        {(() => {
          // ✅ Formatage de la civilité (backend utilise "HOMME", "FEMME", "AUTRE")
          const formatGender = (gender?: string) => {
            if (!gender) return null;
            const normalized = String(gender).toUpperCase();
            if (normalized === "HOMME" || normalized === "MALE") return "Monsieur";
            if (normalized === "FEMME" || normalized === "FEMALE") return "Madame";
            if (normalized === "AUTRE" || normalized === "OTHER") return "Autre";
            return null;
          };

          const clientFields = [
            { label: "Nom complet", value: summary.client.name, show: true },
            { label: "Civilité", value: formatGender(summary.client.gender), show: !!summary.client.gender },
            { label: "Prénom", value: summary.client.first_name, show: !!summary.client.first_name },
            { label: "Nom", value: summary.client.last_name, show: !!summary.client.last_name },
            { label: "Date de naissance", value: summary.client.birth_date ? dayjs(summary.client.birth_date).format("DD MMMM YYYY") : null, show: !!summary.client.birth_date },
            { label: "Numéro AVS", value: summary.client.avs_number, show: !!summary.client.avs_number },
            { label: "Téléphone", value: summary.client.phone, show: !!summary.client.phone },
            { label: "Téléphone de contact", value: summary.client.contact_phone, show: !!summary.client.contact_phone && summary.client.contact_phone !== summary.client.phone },
            { label: "Email de contact", value: summary.client.contact_email, show: !!summary.client.contact_email },
            { label: "Adresse de domicile", value: summary.client.home_address, show: !!summary.client.home_address },
            { label: "Établissement de résidence", value: summary.client.residence_facility, show: !!summary.client.residence_facility },
            { label: "Adresse de facturation", value: summary.client.billing_address, show: !!summary.client.billing_address && summary.client.billing_address !== summary.client.home_address },
            { label: "Institution", value: summary.client.institution_name, show: !!summary.client.is_institution && !!summary.client.institution_name },
            { label: "Tarif préférentiel", value: summary.client.preferential_rate ? `${summary.client.preferential_rate.toFixed(2)} CHF` : null, show: !!summary.client.preferential_rate },
          ].filter(f => f.show);

          return clientFields.map((field, index) => (
            <InfoRow
              key={field.label}
              label={field.label}
              value={field.value || ""}
              isLast={index === clientFields.length - 1}
            />
          ));
        })()}
      </View>

      {/* ✅ Section Actions rapides : urgent (si heure à définir) + modifier horaire (toujours) */}
      {(!isCompletedStatus(summary?.status ?? "") && summary?.status !== "cancelled" && summary?.status !== "canceled") && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Actions rapides</Text>
          <View style={styles.quickActions}>
            {showUrgentActions && (
              <TouchableOpacity
                style={styles.quickActionButton}
                onPress={handleMarkUrgent}
                disabled={actionLoading}
              >
                {actionLoading ? (
                  <ActivityIndicator size="small" color={BRAND} />
                ) : (
                  <Text style={styles.quickActionText}>Marquer urgent +15 min</Text>
                )}
              </TouchableOpacity>
            )}
            <TouchableOpacity
              style={styles.quickActionButton}
              onPress={() => {
                const pickup = summary?.time?.pickup_at;
                setScheduleValue(
                  pickup && !isPickupSentinel(pickup)
                    ? dayjs(pickup).format("HH:mm")
                    : ""
                );
                setScheduleVisible(true);
              }}
              disabled={actionLoading}
            >
              <Text style={styles.quickActionText}>
                {showUrgentActions ? "Planifier l'horaire" : "Modifier l'horaire"}
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      )}

      {/* ✅ Section Informations générales (simplifiée) */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Informations générales</Text>
        {(() => {
          const generalFields = [
            { label: "Priorité", value: summary.client.priority },
            { label: "Départ", value: summary.route.pickup_address },
            { label: "Arrivée", value: summary.route.dropoff_address },
            ...(summary.route.distance_km ? [{ label: "Distance", value: `${summary.route.distance_km} km` }] : []),
            {
              label: "Statut",
              value:
                summary.status === "assigned" || normalizeBookingStatus(summary.status) === "ASSIGNED"
                  ? "Assignée"
                  : isCompletedStatus(summary.status)
                    ? "Terminée"
                    : summary.status === "cancelled" || normalizeBookingStatus(summary.status) === "CANCELED"
                      ? "Annulée"
                      : "Non assignée",
            },
            {
              label: "Chauffeur",
              value: summary.driver?.name
                ? `${summary.driver.name}${summary.driver.is_emergency ? " (urgence)" : ""}`
                : "Non assigné",
            },
          ];

          // ✅ Déterminer le statut à utiliser pour les couleurs du texte du chauffeur
          let displayStatus: string | undefined = summary?.status;
          const statusStr = String(displayStatus || "").toLowerCase().trim();

          // ✅ Normaliser le statut en minuscules pour la correspondance avec statusColors
          if (summary?.driver?.name) {
            // ✅ Si un chauffeur est assigné, normaliser le statut pour l'affichage
            // ✅ P0-1: Utiliser la fonction de normalisation pour vérifier les statuts complétés
            if (isCompletedStatus(displayStatus) || statusStr === "cancelled" || statusStr === "canceled") {
              // ✅ Utiliser le statut en minuscules pour la correspondance avec statusColors
              displayStatus = statusStr;
            } else if (!displayStatus || statusStr === "unassigned" || statusStr === "accepted" || statusStr === "") {
              // ✅ Si pas de statut ou "unassigned"/"accepted" avec chauffeur, utiliser "assigned"
              displayStatus = "assigned";
            } else {
              // ✅ Sinon, utiliser le statut en minuscules (assigned, en_route, in_progress, etc.)
              displayStatus = statusStr;
            }
          } else {
            // ✅ Si pas de chauffeur, normaliser quand même le statut en minuscules
            // ✅ Mais ne pas appliquer de couleur si pas de chauffeur assigné
            displayStatus = statusStr || undefined;
          }

          return generalFields.map((field, index) => (
            <InfoRow
              key={field.label}
              label={field.label}
              value={field.value}
              isLast={index === generalFields.length - 1}
              status={displayStatus} // ✅ Passer le statut normalisé pour appliquer les couleurs
            />
          ));
        })()}
      </View>

      {/* ✅ Section Historique (améliorée) */}
      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Historique</Text>
        {history.length === 0 ? (
          <Text style={styles.muted}>Aucun événement pour le moment.</Text>
        ) : (
          history.map((event: RideEvent, index: number) => (
            <View key={`${event.ts}-${index}`} style={styles.historyItem}>
              <View style={styles.historyHeader}>
                <Text style={styles.historyTitle}>
                  {formatEventType(event.event)} • {formatActor(event.actor || "système")}
                </Text>
                <Text style={styles.historyDate}>
                  {dayjs(event.ts).format("DD MMM YYYY HH:mm")} (
                  {dayjs(event.ts).fromNow()})
                </Text>
              </View>
              {/* ✅ Afficher les détails formatés si disponibles, sinon formater le JSON */}
              {event.details_formatted ? (
                <Text style={styles.historyDetailsFormatted}>
                  {event.details_formatted}
                </Text>
              ) : event.details ? (
                <Text style={styles.historyDetailsFormatted}>
                  {formatEventDetails(event.details, summary)}
                </Text>
              ) : null}
            </View>
          ))
        )}
      </View>

      {detail.notes && detail.notes.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Notes</Text>
          {detail.notes.map((note, idx) => (
            <Text key={`${note}-${idx}`} style={styles.noteItem}>
              • {note}
            </Text>
          ))}
        </View>
      )}

      {errorMessage && <Text style={styles.errorText}>{errorMessage}</Text>}

      {/* Bouton retour */}
      <TouchableOpacity
        style={styles.backButton}
        onPress={() => router.back()}
        activeOpacity={0.85}
      >
        <Ionicons name="arrow-back" size={18} color="#FFFFFF" style={{ marginRight: 8 }} />
        <Text style={styles.backButtonText}>Retour aux courses</Text>
      </TouchableOpacity>

      {actionLoading && (
        <View style={styles.overlay}>
          <ActivityIndicator color="#FFFFFF" />
          <Text style={styles.overlayText}>Traitement en cours…</Text>
        </View>
      )}

      <Modal visible={scheduleVisible} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>
              {showUrgentActions ? "Planifier l'horaire" : "Modifier l'horaire"}
            </Text>
            <View style={{ flexDirection: "row", alignItems: "center", gap: 8, marginBottom: 12 }}>
              <TextInput
                style={[styles.modalInput, { flex: 1 }]}
                value={scheduleValue}
                onChangeText={setScheduleValue}
                placeholder="HH:mm"
                placeholderTextColor={TEXT_MUTED}
                keyboardType="numeric"
                autoFocus
              />
              <Pressable
                style={({ pressed }) => [
                  {
                    paddingVertical: 10,
                    paddingHorizontal: 14,
                    backgroundColor: pressed ? "rgba(0,121,107,0.15)" : "rgba(0,121,107,0.1)",
                    borderRadius: 8,
                    borderWidth: 1,
                    borderColor: BRAND,
                  },
                ]}
                onPress={() => setScheduleValue("00:00")}
              >
                <Text style={{ color: BRAND, fontWeight: "600", fontSize: 14 }}>À définir</Text>
              </Pressable>
            </View>
            <View style={styles.modalActions}>
              <Pressable
                style={styles.modalCancel}
                onPress={() => {
                  setScheduleVisible(false);
                  setScheduleValue("");
                }}
              >
                <Text style={styles.modalCancelText}>Annuler</Text>
              </Pressable>
              <Pressable
                style={styles.modalConfirm}
                onPress={handleScheduleConfirm}
                disabled={actionLoading}
              >
                <Text style={styles.modalConfirmText}>Confirmer</Text>
              </Pressable>
            </View>
          </View>
        </View>
      </Modal>

      {/* ✅ Modal pour sélectionner un chauffeur */}
      <Modal
        visible={driverPickerVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setDriverPickerVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Choisir un chauffeur</Text>
            <ScrollView style={styles.driverListModal} nestedScrollEnabled>
              {/* ✅ En mode manuel, ne pas afficher les suggestions */}
              {isManualMode ? (
                <Text style={styles.muted}>
                  Mode manuel : Sélectionnez un chauffeur depuis la liste complète des chauffeurs disponibles.
                </Text>
              ) : suggestions.length === 0 ? (
                <Text style={styles.muted}>Aucun chauffeur disponible</Text>
              ) : (
                suggestions.map((suggestion: DriverSuggestion) => (
                  <TouchableOpacity
                    key={suggestion.driver_id}
                    style={[
                      styles.driverOption,
                      manualDriverId === suggestion.driver_id &&
                      styles.driverOptionSelected,
                    ]}
                    onPress={() => {
                      setManualDriverId(suggestion.driver_id);
                      setDriverPickerVisible(false);
                    }}
                  >
                    <View style={styles.driverOptionContent}>
                      <Text style={styles.driverOptionName}>
                        {suggestion.driver_name}
                      </Text>
                      <Text style={styles.driverOptionMeta}>
                        Score: {suggestion.score.toFixed(2)}
                        {suggestion.preferred_match && " • Préféré"}
                        {suggestion.is_emergency && " • Urgence"}
                      </Text>
                    </View>
                    {manualDriverId === suggestion.driver_id && (
                      <Ionicons name="checkmark" size={20} color="#1EB980" />
                    )}
                  </TouchableOpacity>
                ))
              )}
              {/* Option pour réinitialiser */}
              <TouchableOpacity
                style={styles.driverOption}
                onPress={() => {
                  setManualDriverId("");
                  setDriverPickerVisible(false);
                }}
              >
                <Text style={styles.driverOptionReset}>
                  Effacer la sélection
                </Text>
              </TouchableOpacity>
            </ScrollView>
            <Pressable
              style={styles.modalCancel}
              onPress={() => setDriverPickerVisible(false)}
            >
              <Text style={styles.modalCancelText}>Fermer</Text>
            </Pressable>
          </View>
        </View>
      </Modal>

      {/* ✅ Modal de transfert de course */}
      <TransferRideModal
        visible={transferModalVisible}
        onClose={() => setTransferModalVisible(false)}
        ride={summary || null}
        onSuccess={() => {
          setTransferModalVisible(false);
          loadDetail();
        }}
      />
    </ScrollView>
  );
}

const InfoRow = ({
  label,
  value,
  isLast = false,
  status,
}: {
  label: string;
  value: string;
  isLast?: boolean;
  status?: string; // ✅ Statut pour appliquer les couleurs
}) => {
  // ✅ Si c'est le champ "Chauffeur" et qu'il y a un statut, appliquer les couleurs
  const isDriverField = label === "Chauffeur";
  const isAssigned = isDriverField && value !== "Non assigné" && value !== "";

  // ✅ Nettoyer la valeur (enlever "(urgence)" si présent)
  const cleanDriverName = isDriverField && isAssigned
    ? value.replace(/\s*\(urgence\)\s*/gi, "").trim()
    : value;

  // ✅ Normaliser le statut : toujours en minuscules pour la correspondance avec statusColors
  let normalizedStatus = status ? String(status).toLowerCase().trim() : undefined;

  if (isDriverField && isAssigned) {
    // ✅ Si un chauffeur est assigné, normaliser le statut pour l'affichage
    const statusStr = normalizedStatus || "";
    // ✅ P0-1: Utiliser la fonction de normalisation pour vérifier les statuts complétés
    if (isCompletedStatus(status) || statusStr === "cancelled" || statusStr === "canceled") {
      normalizedStatus = statusStr;
    } else if (!status || statusStr === "accepted" || statusStr === "unassigned" || statusStr === "") {
      normalizedStatus = "assigned"; // Traiter "accepted" ou undefined avec driver comme "assigned"
    } else {
      // ✅ Garder le statut original en minuscules (assigned, en_route, in_progress, etc.)
      normalizedStatus = statusStr;
    }
  }

  // ✅ Toujours obtenir les couleurs si c'est le champ chauffeur et qu'un chauffeur est assigné
  // ✅ Utiliser "assigned" par défaut si pas de statut normalisé
  const statusColors = isDriverField && isAssigned && normalizedStatus
    ? getStatusColors(normalizedStatus)
    : null;

  // ✅ Appliquer la couleur du statut au texte du nom du chauffeur
  return (
    <View style={[styles.infoRow, isLast && styles.infoRowLast]}>
      <Text style={styles.infoLabel}>{label}</Text>
      <Text
        style={[
          styles.infoValue,
          isDriverField && isAssigned && statusColors
            ? {
              color: statusColors.text,
              fontWeight: "600",
            }
            : {},
        ]}
      >
        {isDriverField && isAssigned ? cleanDriverName : value}
      </Text>
    </View>
  );
};

// ✅ Fonctions utilitaires pour formater l'historique
const formatEventType = (event: string): string => {
  const eventMap: Record<string, string> = {
    created: "Création",
    assigned: "Assignation",
    reassigned: "Réassignation",
    cancelled: "Annulation",
    completed: "Terminaison",
    status_changed: "Changement de statut",
  };
  return eventMap[event] || event;
};

const formatActor = (actor: string): string => {
  const actorMap: Record<string, string> = {
    system: "Système",
    dispatcher: "Répartiteur",
    driver: "Chauffeur",
  };
  return actorMap[actor] || actor;
};

const formatEventDetails = (details: any, summary?: RideDetail["summary"]): string => {
  if (typeof details === "string") {
    try {
      details = JSON.parse(details);
    } catch {
      return details;
    }
  }

  if (typeof details !== "object" || details === null) {
    return String(details);
  }

  const parts: string[] = [];

  if (details.status) {
    const statusMap: Record<string, string> = {
      ACCEPTED: "Acceptée",
      SCHEDULED: "Planifiée",
      ASSIGNED: "Assignée",
      IN_PROGRESS: "En cours",
      COMPLETED: "Terminée",
      CANCELLED: "Annulée",
      PENDING: "En attente",
    };
    parts.push(`Statut: ${statusMap[details.status] || details.status}`);
  }

  // ✅ Afficher le nom du chauffeur au lieu de l'ID
  if (details.driver_id) {
    // Essayer de récupérer le nom depuis summary.driver si disponible
    const driverName = summary?.driver?.name || `#${details.driver_id}`;
    parts.push(`Chauffeur: ${driverName}`);
  }

  // Ajouter les autres champs
  Object.entries(details).forEach(([key, value]) => {
    if (key !== "status" && key !== "driver_id") {
      parts.push(`${key}: ${value}`);
    }
  });

  return parts.length > 0 ? parts.join("\n") : JSON.stringify(details, null, 2);
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: BG,
  },
  content: {
    paddingBottom: 60,
  },

  /* ── Header ── */
  header: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: CARD,
    paddingHorizontal: 16,
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
    ...createShadow({ shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.04, shadowRadius: 6, elevation: 2 }),
  },
  headerBack: {
    width: 36,
    height: 36,
    borderRadius: 10,
    backgroundColor: BG,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 10,
  },
  headerCenter: {
    flex: 1,
  },
  headerTitle: {
    color: TEXT,
    fontSize: 16,
    fontWeight: "700",
  },
  headerSub: {
    color: TEXT_MUTED,
    fontSize: 12,
    marginTop: 2,
  },
  headerStatusBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 5,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
    backgroundColor: BG,
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: 4,
  },
  statusLabel: {
    fontSize: 11,
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: 0.3,
  },

  /* ── Section ── */
  section: {
    backgroundColor: CARD,
    borderRadius: 14,
    padding: 16,
    margin: 16,
    marginBottom: 0,
    borderWidth: 1,
    borderColor: BORDER,
    ...createShadow({ shadowColor: "#000", shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.03, shadowRadius: 4, elevation: 1 }),
  },
  sectionTitle: {
    color: TEXT,
    fontSize: 15,
    fontWeight: "700",
    marginBottom: 12,
  },
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 10,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
  },
  infoRowLast: {
    marginBottom: 0,
    paddingBottom: 0,
    borderBottomWidth: 0,
  },
  infoLabel: {
    color: TEXT_SEC,
    fontSize: 13,
    flex: 1,
  },
  infoValue: {
    color: TEXT,
    fontSize: 13,
    flex: 1,
    textAlign: "right",
    fontWeight: "500",
  },
  driverBadge: {
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderWidth: 1,
    maxWidth: 140,
    minWidth: 80,
  },
  driverBadgeText: {
    fontSize: 11,
    fontWeight: "600",
    letterSpacing: 0.3,
    textTransform: "uppercase",
  },
  conflictCard: {
    borderRadius: 12,
    padding: 12,
    marginBottom: 8,
    backgroundColor: "rgba(251,191,36,0.08)",
    borderWidth: 1,
    borderColor: BORDER,
  },
  conflictBlocking: {
    backgroundColor: "rgba(220,53,69,0.06)",
  },
  conflictTitle: {
    color: TEXT,
    fontSize: 14,
    fontWeight: "600",
  },
  conflictMessage: {
    color: TEXT_SEC,
    marginTop: 4,
    fontSize: 13,
  },
  conflictBadge: {
    color: DANGER,
    fontSize: 11,
    marginTop: 4,
    fontWeight: "600",
  },
  suggestionCard: {
    backgroundColor: BG,
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
    flexDirection: "row",
    justifyContent: "space-between",
    borderWidth: 1,
    borderColor: BORDER,
  },
  suggestionMain: {
    flex: 1,
    paddingRight: 12,
  },
  suggestionName: {
    color: TEXT,
    fontSize: 15,
    fontWeight: "600",
  },
  suggestionReason: {
    color: TEXT_SEC,
    marginTop: 3,
    fontSize: 13,
  },
  suggestionMeta: {
    color: TEXT_MUTED,
    marginTop: 3,
    fontSize: 12,
  },
  badgePreferred: {
    color: BRAND,
    marginTop: 4,
    fontWeight: "600",
    fontSize: 11,
  },
  badgeEmergency: {
    color: DANGER,
    marginTop: 3,
    fontWeight: "600",
    fontSize: 11,
  },
  assignButton: {
    alignSelf: "center",
    backgroundColor: BRAND,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: 10,
  },
  assignButtonText: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: 13,
  },
  input: {
    backgroundColor: BG,
    borderRadius: 12,
    padding: 12,
    color: TEXT,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: BORDER,
    fontSize: 14,
  },
  inputMultiline: {
    minHeight: 80,
    textAlignVertical: "top",
  },
  checkboxRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 10,
  },
  checkbox: {
    width: 20,
    height: 20,
    borderRadius: 6,
    borderWidth: 1.5,
    borderColor: BORDER,
    marginRight: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  checkboxChecked: {
    backgroundColor: BRAND,
    borderColor: BRAND,
  },
  checkboxInner: {
    width: 10,
    height: 10,
    borderRadius: 2,
    backgroundColor: "#FFFFFF",
  },
  checkboxLabel: {
    color: TEXT_SEC,
    fontSize: 13,
  },
  primaryButton: {
    backgroundColor: BRAND,
    borderRadius: 12,
    paddingVertical: 13,
    alignItems: "center",
    marginTop: 6,
    ...createShadow({ shadowColor: BRAND, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 6, elevation: 3 }),
  },
  primaryButtonDisabled: {
    opacity: 0.5,
    backgroundColor: TEXT_MUTED,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "600",
  },
  actionsRow: {
    flexDirection: "row",
    gap: 10,
    marginTop: 10,
  },
  secondaryButton: {
    borderWidth: 1,
    borderColor: BORDER,
    borderRadius: 12,
    paddingVertical: 13,
    alignItems: "center",
    backgroundColor: CARD,
  },
  secondaryButtonText: {
    color: TEXT,
    fontWeight: "600",
    fontSize: 14,
  },
  flexButton: {
    flex: 1,
  },
  quickActions: {
    flexDirection: "row",
    gap: 10,
  },
  quickActionButton: {
    flex: 1,
    backgroundColor: BG,
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: "center",
    borderWidth: 1,
    borderColor: BORDER,
  },
  quickActionText: {
    color: TEXT,
    fontWeight: "600",
    fontSize: 13,
  },
  backButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: BRAND,
    borderRadius: 12,
    paddingVertical: 13,
    paddingHorizontal: 20,
    marginTop: 20,
    marginBottom: 20,
    marginHorizontal: 16,
    ...createShadow({ shadowColor: BRAND, shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.2, shadowRadius: 6, elevation: 3 }),
  },
  backButtonText: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "600",
  },
  errorText: {
    color: DANGER,
    marginTop: 8,
    fontSize: 15,
    marginHorizontal: 16,
    textAlign: "center",
    fontWeight: "600",
    lineHeight: 22,
  },
  errorHint: {
    color: TEXT_SEC,
    marginTop: 6,
    fontSize: 13,
    marginHorizontal: 16,
    textAlign: "center",
    lineHeight: 18,
  },
  errorActions: {
    marginTop: 20,
    width: "100%",
    gap: 10,
    paddingHorizontal: 16,
  },
  muted: {
    color: TEXT_MUTED,
    fontSize: 13,
  },
  historyItem: {
    marginBottom: 10,
    padding: 12,
    backgroundColor: BG,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: BORDER,
  },
  historyHeader: {
    marginBottom: 6,
  },
  historyTitle: {
    color: TEXT,
    fontWeight: "600",
    fontSize: 13,
    marginBottom: 2,
  },
  historyDate: {
    color: TEXT_MUTED,
    fontSize: 11,
    marginTop: 2,
  },
  historyDetails: {
    color: TEXT_SEC,
    fontSize: 12,
    marginTop: 6,
  },
  historyDetailsFormatted: {
    color: TEXT_SEC,
    fontSize: 12,
    marginTop: 6,
    lineHeight: 18,
  },
  noteItem: {
    color: TEXT_SEC,
    marginBottom: 4,
    fontSize: 13,
    lineHeight: 19,
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: BG,
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  loadingText: {
    color: TEXT_MUTED,
    marginTop: 10,
    fontSize: 13,
  },
  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0,0,0,0.4)",
    alignItems: "center",
    justifyContent: "center",
  },
  overlayText: {
    color: "#FFFFFF",
    marginTop: 8,
    fontSize: 14,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.35)",
    alignItems: "center",
    justifyContent: "center",
    padding: 24,
  },
  modalCard: {
    backgroundColor: CARD,
    width: "100%",
    maxWidth: 420,
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: BORDER,
    gap: 14,
    ...createShadow({ shadowColor: "#000", shadowOffset: { width: 0, height: 8 }, shadowOpacity: 0.12, shadowRadius: 24, elevation: 10 }),
  },
  modalTitle: {
    color: TEXT,
    fontSize: 17,
    fontWeight: "700",
  },
  modalInput: {
    backgroundColor: BG,
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 14,
    color: TEXT,
    fontSize: 15,
    borderWidth: 1,
    borderColor: BORDER,
  },
  modalActions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 10,
  },
  modalCancel: {
    paddingVertical: 10,
    paddingHorizontal: 14,
  },
  modalCancelText: {
    color: TEXT_SEC,
    fontWeight: "600",
    fontSize: 14,
  },
  modalConfirm: {
    backgroundColor: BRAND,
    paddingVertical: 11,
    paddingHorizontal: 18,
    borderRadius: 12,
  },
  modalConfirmText: {
    color: "#FFFFFF",
    fontWeight: "600",
    fontSize: 14,
  },
  pickerInput: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  pickerText: {
    color: TEXT,
    fontSize: 14,
    flex: 1,
  },
  pickerPlaceholder: {
    color: TEXT_MUTED,
  },
  inputDisabled: {
    opacity: 0.5,
  },
  disabledHint: {
    color: DANGER,
    fontSize: 11,
    marginTop: 6,
    fontStyle: "italic",
  },
  driverListModal: {
    maxHeight: 400,
    marginBottom: 14,
  },
  driverOption: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    padding: 12,
    borderRadius: 12,
    backgroundColor: BG,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: BORDER,
  },
  driverOptionSelected: {
    backgroundColor: "rgba(0,121,107,0.06)",
    borderColor: BRAND,
  },
  driverOptionContent: {
    flex: 1,
  },
  driverOptionName: {
    color: TEXT,
    fontSize: 15,
    fontWeight: "600",
    marginBottom: 2,
  },
  driverOptionMeta: {
    color: TEXT_SEC,
    fontSize: 12,
  },
  driverOptionReset: {
    color: TEXT_SEC,
    fontSize: 14,
    fontStyle: "italic",
  },
  assignButtonDisabled: {
    opacity: 0.5,
    backgroundColor: TEXT_MUTED,
  },
  assignButtonTextDisabled: {
    color: TEXT_SEC,
  },
});
