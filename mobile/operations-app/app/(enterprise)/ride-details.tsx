import React, { useCallback, useEffect, useState } from "react";
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
import { router, useLocalSearchParams } from "expo-router";
import * as Crypto from "expo-crypto";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";
import "dayjs/locale/fr";

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

  const [detail, setDetail] = useState<RideDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [manualDriverId, setManualDriverId] = useState("");
  const [manualReason, setManualReason] = useState("");
  const [allowEmergency, setAllowEmergency] = useState(false);
  const [scheduleValue, setScheduleValue] = useState("");
  const [scheduleVisible, setScheduleVisible] = useState(false);

  const summary = detail?.summary;
  const suggestions = detail?.suggestions ?? [];
  const history = detail?.history ?? [];
  const conflicts = detail?.conflicts ?? [];

  const isAssigned = summary?.status === "assigned" || !!summary?.driver?.id;

  const loadDetail = useCallback(async () => {
    if (!rideId) return;
    setLoading(true);
    setErrorMessage(null);
    try {
      const data = await getDispatchRideDetails(rideId);
      setDetail(data);
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de charger la fiche course.";
      setErrorMessage(message);
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
        const responseMessage =
          error?.response?.data?.error ??
          error?.response?.data?.message ??
          error?.message;
        setErrorMessage(
          responseMessage ||
            "Impossible de finaliser l’assignation. Vérifiez les validations (fairness, préférences, conflits)."
        );
      } finally {
        setActionLoading(false);
      }
    },
    [allowEmergency, isAssigned, loadDetail, manualReason, rideId]
  );

  const handleMarkUrgent = useCallback(async () => {
    if (!rideId) return;
    setActionLoading(true);
    setErrorMessage(null);
    try {
      await markRideUrgent(rideId, { extra_delay_minutes: 15 });
      await loadDetail();
      Alert.alert(
        "Urgence enregistrée",
        "La course est marquée urgente (+15 min)."
      );
    } catch (error: any) {
      const message =
        error?.response?.data?.error ??
        error?.message ??
        "Impossible de marquer la course en urgence.";
      setErrorMessage(message);
    } finally {
      setActionLoading(false);
    }
  }, [loadDetail, rideId]);

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
      Alert.alert(
        "Horaire planifié",
        `Pickup replanifié à ${hour.padStart(2, "0")}:${minute.padStart(
          2,
          "0"
        )}.`
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
            Alert.alert("Course annulée");
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
  }, [loadDetail, rideId]);

  const manualAssignDisabled =
    manualDriverId.trim().length === 0 || actionLoading;

  const headerTitle = summary
    ? `${summary.client.name} • ${summary.time.pickup_at ? dayjs(summary.time.pickup_at).format("DD MMM HH:mm") : "⏱️ À définir"}`
    : "Course";

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator color="#4D6BFE" />
        <Text style={styles.loadingText}>Chargement de la course…</Text>
      </View>
    );
  }

  if (!detail || !summary) {
    return (
      <View style={styles.loadingContainer}>
        <Text style={styles.errorText}>
          {errorMessage ?? "Course introuvable."}
        </Text>
        <TouchableOpacity style={styles.primaryButton} onPress={loadDetail}>
          <Text style={styles.primaryButtonText}>Réessayer</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>{headerTitle}</Text>
      <Text style={styles.subtitle}>{summary.id}</Text>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Informations générales</Text>
        <InfoRow label="Client" value={summary.client.name} />
        <InfoRow label="Priorité" value={summary.client.priority} />
        <InfoRow label="Départ" value={summary.route.pickup_address} />
        <InfoRow label="Arrivée" value={summary.route.dropoff_address} />
        <InfoRow
          label="Statut"
          value={summary.status === "assigned" ? "Assignée" : "Non assignée"}
        />
        <InfoRow
          label="Chauffeur"
          value={
            summary.driver?.name
              ? `${summary.driver.name}${summary.driver.is_emergency ? " (urgence)" : ""}`
              : "Non assigné"
          }
        />
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Actions rapides</Text>
        <View style={styles.quickActions}>
          <TouchableOpacity
            style={styles.quickActionButton}
            onPress={handleMarkUrgent}
            disabled={actionLoading}
          >
            <Text style={styles.quickActionText}>Marquer urgent +15 min</Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={styles.quickActionButton}
            onPress={() => {
              const existing = summary.time.pickup_at
                ? dayjs(summary.time.pickup_at).format("HH:mm")
                : "";
              setScheduleValue(existing);
              setScheduleVisible(true);
            }}
            disabled={actionLoading}
          >
            <Text style={styles.quickActionText}>Planifier l’horaire</Text>
          </TouchableOpacity>
        </View>
      </View>

      {conflicts.length > 0 && (
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Conflits détectés</Text>
          {conflicts.map((conflict: RideConflict, index: number) => (
            <View
              key={`${conflict.type}-${index}`}
              style={[
                styles.conflictCard,
                conflict.blocking && styles.conflictBlocking,
              ]}
            >
              <Text style={styles.conflictTitle}>{conflict.type}</Text>
              <Text style={styles.conflictMessage}>{conflict.message}</Text>
              <Text style={styles.conflictBadge}>
                {conflict.blocking ? "Bloquant" : "Avertissement"}
              </Text>
            </View>
          ))}
        </View>
      )}

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Suggestions chauffeurs</Text>
        {suggestions.length === 0 ? (
          <Text style={styles.muted}>Aucune suggestion disponible.</Text>
        ) : (
          suggestions.map((suggestion: DriverSuggestion) => (
            <View key={suggestion.driver_id} style={styles.suggestionCard}>
              <View style={styles.suggestionMain}>
                <Text style={styles.suggestionName}>
                  {suggestion.driver_name}
                </Text>
                <Text style={styles.suggestionReason}>{suggestion.reason}</Text>
                <Text style={styles.suggestionMeta}>
                  Score: {suggestion.score.toFixed(2)} • Fairness:{" "}
                  {suggestion.fairness_delta != null
                    ? suggestion.fairness_delta.toFixed(2)
                    : "n/a"}
                </Text>
                {suggestion.preferred_match && (
                  <Text style={styles.badgePreferred}>Chauffeur préféré</Text>
                )}
                {suggestion.is_emergency && (
                  <Text style={styles.badgeEmergency}>Chauffeur d’urgence</Text>
                )}
              </View>
              <TouchableOpacity
                style={styles.assignButton}
                disabled={actionLoading}
                onPress={() =>
                  handleAssign(
                    suggestion.driver_id,
                    suggestion.reason,
                    suggestion.is_emergency
                  )
                }
              >
                <Text style={styles.assignButtonText}>
                  {isAssigned ? "Réassigner" : "Assigner"}
                </Text>
              </TouchableOpacity>
            </View>
          ))
        )}
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Assignation manuelle</Text>
        <TextInput
          style={styles.input}
          placeholder="ID chauffeur"
          placeholderTextColor="#8DA0C1"
          value={manualDriverId}
          onChangeText={setManualDriverId}
          autoCapitalize="none"
        />
        <TextInput
          style={[styles.input, styles.inputMultiline]}
          placeholder="Commentaire (optionnel)"
          placeholderTextColor="#8DA0C1"
          value={manualReason}
          onChangeText={setManualReason}
          multiline
        />

        <View style={styles.checkboxRow}>
          <TouchableOpacity
            style={[styles.checkbox, allowEmergency && styles.checkboxChecked]}
            onPress={() => setAllowEmergency((prev) => !prev)}
          >
            {allowEmergency && <View style={styles.checkboxInner} />}
          </TouchableOpacity>
          <Text style={styles.checkboxLabel}>
            Autoriser l’usage d’un chauffeur d’urgence
          </Text>
        </View>

        <TouchableOpacity
          style={[
            styles.primaryButton,
            manualAssignDisabled && styles.primaryButtonDisabled,
          ]}
          disabled={manualAssignDisabled}
          onPress={() =>
            handleAssign(manualDriverId.trim(), manualReason || undefined)
          }
        >
          <Text style={styles.primaryButtonText}>
            {isAssigned ? "Réassigner" : "Assigner"} le chauffeur
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.section}>
        <Text style={styles.sectionTitle}>Historique</Text>
        {history.length === 0 ? (
          <Text style={styles.muted}>Aucun événement pour le moment.</Text>
        ) : (
          history.map((event: RideEvent, index: number) => (
            <View key={`${event.ts}-${index}`} style={styles.historyItem}>
              <Text style={styles.historyTitle}>
                {event.event} • {event.actor || "système"}
              </Text>
              <Text style={styles.historyDate}>
                {dayjs(event.ts).format("DD MMM YYYY HH:mm")} (
                {dayjs(event.ts).fromNow()})
              </Text>
              {event.details && (
                <Text style={styles.historyDetails}>
                  {JSON.stringify(event.details, null, 2)}
                </Text>
              )}
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

      <View style={styles.actionsRow}>
        <TouchableOpacity
          style={[styles.secondaryButton, styles.flexButton]}
          onPress={loadDetail}
          disabled={actionLoading}
        >
          <Text style={styles.secondaryButtonText}>Rafraîchir</Text>
        </TouchableOpacity>
        {isAssigned && (
          <TouchableOpacity
            style={[styles.secondaryButton, styles.flexButton]}
            onPress={handleCancel}
            disabled={actionLoading}
          >
            <Text style={styles.secondaryButtonText}>Annuler la course</Text>
          </TouchableOpacity>
        )}
      </View>

      <TouchableOpacity style={styles.linkButton} onPress={() => router.back()}>
        <Text style={styles.linkButtonText}>Retour aux courses</Text>
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
            <Text style={styles.modalTitle}>Planifier l’horaire</Text>
            <TextInput
              style={styles.modalInput}
              value={scheduleValue}
              onChangeText={setScheduleValue}
              placeholder="HH:mm"
              placeholderTextColor="#9AA5CC"
              keyboardType="numeric"
              autoFocus
            />
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
    </ScrollView>
  );
}

const InfoRow = ({ label, value }: { label: string; value: string }) => (
  <View style={styles.infoRow}>
    <Text style={styles.infoLabel}>{label}</Text>
    <Text style={styles.infoValue}>{value}</Text>
  </View>
);

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#09162E",
  },
  content: {
    padding: 20,
    paddingBottom: 60,
  },
  title: {
    color: "#FFFFFF",
    fontSize: 24,
    fontWeight: "700",
  },
  subtitle: {
    color: "#7D8CB2",
    marginBottom: 16,
  },
  section: {
    backgroundColor: "rgba(255,255,255,0.06)",
    borderRadius: 18,
    padding: 16,
    marginBottom: 18,
  },
  sectionTitle: {
    color: "#FFFFFF",
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 12,
  },
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
  },
  infoLabel: {
    color: "#94A3C1",
    fontSize: 14,
  },
  infoValue: {
    color: "#FFFFFF",
    fontSize: 14,
    flexShrink: 1,
    textAlign: "right",
  },
  conflictCard: {
    borderRadius: 14,
    padding: 12,
    marginBottom: 10,
    backgroundColor: "rgba(251,191,36,0.12)",
  },
  conflictBlocking: {
    backgroundColor: "rgba(248,113,113,0.15)",
  },
  conflictTitle: {
    color: "#FFFFFF",
    fontSize: 15,
    fontWeight: "600",
  },
  conflictMessage: {
    color: "#E3E9FF",
    marginTop: 4,
  },
  conflictBadge: {
    color: "#FBBF24",
    fontSize: 12,
    marginTop: 6,
  },
  suggestionCard: {
    backgroundColor: "rgba(77,107,254,0.12)",
    borderRadius: 12,
    padding: 12,
    marginBottom: 12,
    flexDirection: "row",
    justifyContent: "space-between",
  },
  suggestionMain: {
    flex: 1,
    paddingRight: 12,
  },
  suggestionName: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
  suggestionReason: {
    color: "#CBD6FF",
    marginTop: 4,
  },
  suggestionMeta: {
    color: "#A9B6E5",
    marginTop: 4,
    fontSize: 13,
  },
  badgePreferred: {
    color: "#4ADE80",
    marginTop: 6,
    fontWeight: "600",
  },
  badgeEmergency: {
    color: "#F87171",
    marginTop: 4,
    fontWeight: "600",
  },
  assignButton: {
    alignSelf: "center",
    backgroundColor: "#4D6BFE",
    paddingVertical: 8,
    paddingHorizontal: 16,
    borderRadius: 10,
  },
  assignButtonText: {
    color: "#FFFFFF",
    fontWeight: "600",
  },
  input: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 12,
    padding: 12,
    color: "#FFFFFF",
    marginBottom: 12,
  },
  inputMultiline: {
    minHeight: 80,
    textAlignVertical: "top",
  },
  checkboxRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
  },
  checkbox: {
    width: 20,
    height: 20,
    borderRadius: 4,
    borderWidth: 1,
    borderColor: "#6C7AA5",
    marginRight: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  checkboxChecked: {
    backgroundColor: "#4D6BFE",
    borderColor: "#4D6BFE",
  },
  checkboxInner: {
    width: 10,
    height: 10,
    borderRadius: 2,
    backgroundColor: "#FFFFFF",
  },
  checkboxLabel: {
    color: "#CDD7F6",
  },
  primaryButton: {
    backgroundColor: "#4D6BFE",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
    marginTop: 6,
  },
  primaryButtonDisabled: {
    opacity: 0.5,
  },
  primaryButtonText: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
  },
  actionsRow: {
    flexDirection: "row",
    gap: 12,
    marginTop: 10,
  },
  secondaryButton: {
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.3)",
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: "center",
  },
  secondaryButtonText: {
    color: "#FFFFFF",
    fontWeight: "600",
  },
  flexButton: {
    flex: 1,
  },
  quickActions: {
    flexDirection: "row",
    gap: 12,
  },
  quickActionButton: {
    flex: 1,
    backgroundColor: "rgba(255,255,255,0.12)",
    borderRadius: 12,
    paddingVertical: 12,
    alignItems: "center",
  },
  quickActionText: {
    color: "#FFFFFF",
    fontWeight: "600",
  },
  linkButton: {
    alignItems: "center",
    marginTop: 20,
  },
  linkButtonText: {
    color: "#AAB6FF",
    fontSize: 15,
    textDecorationLine: "underline",
  },
  errorText: {
    color: "#F87171",
    marginTop: 8,
    fontSize: 14,
  },
  muted: {
    color: "#8FA0C6",
  },
  historyItem: {
    marginBottom: 12,
    padding: 12,
    backgroundColor: "rgba(255,255,255,0.05)",
    borderRadius: 10,
  },
  historyTitle: {
    color: "#FFFFFF",
    fontWeight: "600",
  },
  historyDate: {
    color: "#9CAEEC",
    fontSize: 12,
    marginTop: 4,
  },
  historyDetails: {
    color: "#E5EDFF",
    fontSize: 12,
    marginTop: 8,
  },
  noteItem: {
    color: "#E5EDFF",
    marginBottom: 6,
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: "#09162E",
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  loadingText: {
    color: "#E5EDFF",
    marginTop: 12,
  },
  overlay: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(9,22,46,0.75)",
    alignItems: "center",
    justifyContent: "center",
  },
  overlayText: {
    color: "#FFFFFF",
    marginTop: 8,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    alignItems: "center",
    justifyContent: "center",
  },
  modalCard: {
    backgroundColor: "#102347",
    width: "80%",
    borderRadius: 16,
    padding: 20,
  },
  modalTitle: {
    color: "#FFFFFF",
    fontSize: 16,
    fontWeight: "600",
    marginBottom: 10,
  },
  modalInput: {
    backgroundColor: "rgba(255,255,255,0.08)",
    borderRadius: 12,
    padding: 12,
    color: "#FFFFFF",
    marginBottom: 16,
  },
  modalActions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: 12,
  },
  modalCancel: {
    paddingVertical: 10,
    paddingHorizontal: 14,
  },
  modalCancelText: {
    color: "#9AA5CC",
    fontWeight: "600",
  },
  modalConfirm: {
    backgroundColor: "#4D6BFE",
    paddingVertical: 10,
    paddingHorizontal: 18,
    borderRadius: 10,
  },
  modalConfirmText: {
    color: "#FFFFFF",
    fontWeight: "600",
  },
});
