import React, { useEffect, useState } from "react";
import { View, Text, TouchableOpacity, Alert, Modal, Pressable, ScrollView, TouchableWithoutFeedback, Platform } from "react-native";
import { Ionicons, MaterialIcons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import type { Booking as Mission, BookingStatus } from "@/services/api";
import { styles, palette } from "@/styles/missionCardStyles";
import { styles as groupStyles } from "@/styles/missionGroupStyles";
import { updateTripStatus } from "@/services/api";
import CancelJustificationModal from "./CancelJustificationModal";
import { isCompletedStatus, isCanceledStatus, normalizeBookingStatus } from "@/utils/bookingStatus";
import {
  getAuthNotReadyDisplayMessage,
  isAuthNotReadyError,
  shouldShowAuthNotReadyAlert,
} from "@/services/authGuards";
import { getPickupHints, getDropoffHints } from "@/src/domain/missionHints";

// ——— Helpers (pas de logique métier dans le JSX) ———
const formatTime = (d: Date) =>
  d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const formatDurationMinutes = (seconds: number) => Math.max(0, Math.round(seconds / 60));
/** Afficher la ligne "Départ" uniquement avant le début de la course (assigned/en_route). */
const shouldShowDeparture = (normalizedStatus: string) =>
  normalizedStatus !== "IN_PROGRESS";
const formatAddressFallback = (value: string | undefined | null): string =>
  value?.trim() || "Adresse non renseignée";

/** Civilité pour l’en-tête : Madame / Monsieur selon le genre client (HOMME/FEMME). */
const getCivilityLabel = (gender: string | undefined | null): string | null => {
  if (!gender) return null;
  const g = String(gender).toUpperCase();
  if (g === "FEMME" || g === "FEMALE") return "Madame";
  if (g === "HOMME" || g === "MALE") return "Monsieur";
  return null; // AUTRE ou inconnu : pas de civilité affichée
};

const normNotes = (s: string) => s.trim().replace(/\s+/g, " ").toLowerCase();

/** Retourne le texte des notes (dédupliqué / superset / concat) ou null si vide. Robuste : pas de "—". */
function getNotesDisplay(
  notes: string | undefined | null,
  notesMedical: string | undefined | null
): string | null {
  const n = (notes ?? "").trim();
  const nm = (notesMedical ?? "").trim();
  if (!n && !nm) return null;
  if (!n) return nm;
  if (!nm) return n;
  const nNorm = normNotes(n);
  const nmNorm = normNotes(nm);
  if (nNorm === nmNorm) return n;
  if (nmNorm.includes(nNorm)) return nm;
  if (nNorm.includes(nmNorm)) return n;
  return `${n} — ${nm}`;
}

const NOTES_SEE_MORE_THRESHOLD = 100;

/** Retour haptique léger (natif uniquement) pour les actions En route / À bord / Terminer */
const triggerSelectionHaptic = () => {
  if (Platform.OS === "web") return;
  Haptics.selectionAsync().catch(() => { });
};

type Props = {
  mission: Mission | null;
  missionNumber?: number; // Numéro de la mission dans le groupe
  isGrouped?: boolean; // true si la mission fait partie d'un groupe
  /** Largeur cible (responsive) — aligne card + map sur tous les formats */
  contentWidth?: number;
  /** Premier numéro appelable valide (contact_phone > phone > gp_phone) ou null si aucun */
  callablePhone?: string | null;
  onCall?: () => void;
  onNavigate?: (destination: string) => void;
  onComplete?: (missionId: number) => void; // Prend maintenant l'ID de la mission
  onPressDetails?: () => void;
  onStatusChange?: (missionId: number, status: BookingStatus) => void;
  /** Temps restant / arrivée (GET /driver/me/bookings/eta) : avant pickup = client, après pickup = destination */
  getETAToPickup?: (bookingId: number) => number | null;
  getETAToDropoff?: (bookingId: number) => number | null;
  getEstimatedArrival?: (bookingId: number) => Date | null;
  getEstimatedArrivalDropoff?: (bookingId: number) => Date | null;
  getDelayMinutes?: (bookingId: number, scheduledTime: string) => number | null;
  hasGPS?: boolean;
  etaLoading?: boolean;
};

interface MissionCardType extends React.FC<Props> {
  EmptyState: React.FC<{ contentWidth?: number }>;
}

// ✅ Composant visuel réutilisable lorsqu'il n'y a pas de mission
const EmptyStateComponent: React.FC<{ contentWidth?: number }> = ({ contentWidth: contentWidthProp }) => (
  <View
    style={[
      styles.emptyStateContainer,
      contentWidthProp != null
        ? { width: contentWidthProp, alignSelf: "center" as const, marginHorizontal: 0 }
        : Platform.OS === "web" && styles.emptyStateWebFixed,
    ]}
  >
    <Text style={styles.emptyStateTitle}>🚗 En attente de mission</Text>
    <Text style={styles.emptyStateSubtitle}>
      Vous serez notifié dès qu'une mission vous sera assignée.
    </Text>
  </View>
);

const MissionCard: MissionCardType = ({
  mission,
  missionNumber,
  isGrouped = false,
  contentWidth: contentWidthProp,
  callablePhone = null,
  onCall,
  onNavigate,
  onComplete,
  onPressDetails,
  onStatusChange,
  getETAToPickup,
  getETAToDropoff,
  getEstimatedArrival,
  getEstimatedArrivalDropoff,
  getDelayMinutes,
  hasGPS,
  etaLoading = false,
}) => {
  const [status, setStatus] = useState<Mission["status"] | undefined>(
    mission?.status
  );
  const [cancelModalVisible, setCancelModalVisible] = useState(false);
  const [releaseModalVisible, setReleaseModalVisible] = useState(false);
  const [isUpdatingStatus, setIsUpdatingStatus] = useState(false);
  const [notesModalVisible, setNotesModalVisible] = useState(false);
  const [notesFullText, setNotesFullText] = useState<string | null>(null);
  const [detailsSheetVisible, setDetailsSheetVisible] = useState(false);

  /** Mode compact par défaut : réduit la hauteur de la card (~25–35 %) pour libérer la map. */
  const isCompact = true;

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
      // Dedupe : ne pas afficher un second popup si même erreur AUTH_NOT_READY dans les 2–3 s
      if (isAuthNotReadyError(error) && !shouldShowAuthNotReadyAlert(error)) {
        return;
      }
      const friendlyMsg = isAuthNotReadyError(error)
        ? getAuthNotReadyDisplayMessage(error)
        : null;
      const errorMsg =
        friendlyMsg ??
        error?.response?.data?.error ??
        error?.message ??
        "Erreur inconnue";
      Alert.alert("Erreur", `Impossible de mettre à jour le statut : ${errorMsg}`);
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

  return (
    <View
      style={[
        styles.containerEnhanced,
        contentWidthProp != null
          ? { width: contentWidthProp, alignSelf: "center" as const, marginHorizontal: 0 }
          : Platform.OS === "web" && styles.containerWebFixed,
        isCompact && styles.containerCompact,
        isGrouped && groupStyles.groupedCardBorder,
      ]}
    >
      {/* Badge numéroté pour toutes les missions (utile pour référence) */}
      {missionNumber && (
        <View style={groupStyles.missionNumberBadge}>
          <Text style={groupStyles.missionNumberText}>{missionNumber}</Text>
        </View>
      )}

      {/* 1. MissionCardHeader : civilité (Madame/Monsieur) + identité client + statut */}
      <View style={[styles.headerRowEnhanced, isCompact && styles.headerRowCompact]}>
        <View style={styles.headerClientWrap}>
          {(() => {
            const civility = getCivilityLabel(mission.client?.gender);
            return civility != null ? (
              <Text style={styles.clientCivility}>{civility}</Text>
            ) : null;
          })()}
          <Text style={styles.clientName}>
            {mission.client_name ||
              mission.client?.full_name ||
              "Non spécifié"}
          </Text>
          {mission.client?.birth_date != null && mission.client.birth_date !== "" && (
            <Text style={styles.clientBirthDate}>
              📅 {new Date(mission.client.birth_date).toLocaleDateString("fr-FR", {
                day: "2-digit",
                month: "2-digit",
                year: "numeric",
              })}
            </Text>
          )}
        </View>
        <View style={styles.statusBadgeContainer}>
          <Text style={styles.statusBadgeText}>{formatStatus(status ?? "")}</Text>
        </View>
      </View>

      {/* 2. MissionTimingBlock : départ prévu / arrivée estimée (si données) */}
      {(getEstimatedArrival != null || getEstimatedArrivalDropoff != null) && (() => {
        const isAfterPickup = normalizedStatus === "IN_PROGRESS";
        const secondsToTarget = isAfterPickup
          ? (getETAToDropoff?.(mission.id) ?? null)
          : (getETAToPickup?.(mission.id) ?? null);
        const arrivalTime = isAfterPickup
          ? (getEstimatedArrivalDropoff?.(mission.id) ?? null)
          : (getEstimatedArrival?.(mission.id) ?? null);
        const hasCoords = isAfterPickup
          ? !!(mission.dropoff_lat != null && mission.dropoff_lon != null)
          : !!(mission.pickup_lat != null && mission.pickup_lon != null);
        const arrivalAvailable =
          !etaLoading &&
          hasGPS !== false &&
          hasCoords &&
          secondsToTarget != null &&
          arrivalTime != null;
        const minutesRounded =
          secondsToTarget != null ? formatDurationMinutes(secondsToTarget) : null;
        const arrivalLabel = isAfterPickup ? "Arrivée à destination" : "Arrivée estimée";
        let arrivalValueStr: string;
        if (!arrivalAvailable) {
          arrivalValueStr = "indisponible";
        } else if (minutesRounded != null && minutesRounded <= 0) {
          arrivalValueStr = "maintenant";
        } else {
          const scheduledDate = new Date(mission.scheduled_time).getTime();
          const arrivalDate = arrivalTime.getTime();
          const courseNotStarted = normalizedStatus !== "IN_PROGRESS";
          if (courseNotStarted && arrivalDate < scheduledDate && minutesRounded != null && minutesRounded > 0) {
            arrivalValueStr = `dans ${minutesRounded} min`;
          } else if (courseNotStarted && arrivalDate < scheduledDate) {
            arrivalValueStr = "estimée";
          } else {
            arrivalValueStr =
              minutesRounded != null && minutesRounded > 0
                ? `${formatTime(arrivalTime)} (dans ${minutesRounded} min)`
                : formatTime(arrivalTime);
          }
        }
        if (isCompact) {
          const showDeparture = shouldShowDeparture(normalizedStatus);
          const departureStr = formatTime(new Date(mission.scheduled_time));
          return (
            <View style={[styles.timingSection, styles.timingSectionCompact]}>
              <View style={[styles.timingRow, styles.timingRowCompact]}>
                {showDeparture && (
                  <>
                    <Ionicons name="time-outline" size={14} color={palette.timingDeparture} />
                    <Text style={styles.timingTextCompact}>{departureStr}</Text>
                    <Text style={styles.timingTextSecondaryCompact}>→</Text>
                  </>
                )}
                <Ionicons name="timer-outline" size={14} color={palette.timingArrival} />
                <Text style={[styles.timingTextCompact, !arrivalAvailable && styles.timingUnavailable]}>
                  {arrivalValueStr}
                </Text>
              </View>
            </View>
          );
        }
        return (
          <View style={styles.timingSection}>
            {shouldShowDeparture(normalizedStatus) && (
              <View style={styles.timingRow}>
                <Ionicons name="time-outline" size={16} color={palette.timingDeparture} />
                <Text style={styles.timingDeparture}>
                  Départ prévu : {formatTime(new Date(mission.scheduled_time))}
                </Text>
              </View>
            )}
            <View style={styles.timingRow}>
              <Ionicons name="timer-outline" size={16} color={palette.timingArrival} />
              <Text style={[styles.timingArrival, !arrivalAvailable && styles.timingUnavailable]}>
                {arrivalLabel} : {arrivalValueStr}
              </Text>
            </View>
          </View>
        );
      })()}

      {/* 3. MissionRouteBlock : départ → destination (icônes) */}
      <View style={[styles.routeSection, isCompact && styles.routeSectionCompact]}>
        {isCompact ? (
          <>
            <View style={[styles.routeRowCompact, styles.routeRowCompactLast]}>
              <Ionicons name="location-outline" size={16} color={palette.accent} />
              <Text style={styles.routeAddressCompact} numberOfLines={1} ellipsizeMode="tail">
                Départ : {formatAddressFallback(mission.pickup_location)}
              </Text>
            </View>
            <View style={styles.routeRowCompact}>
              <Ionicons name="flag-outline" size={16} color={palette.accent} />
              <Text style={styles.routeAddressCompact} numberOfLines={1} ellipsizeMode="tail">
                Destination : {formatAddressFallback(mission.dropoff_location)}
              </Text>
            </View>
          </>
        ) : (
          <>
            <View style={styles.routeRow}>
              <View style={styles.routeIconWrap}>
                <Ionicons name="location-outline" size={18} color={palette.accent} />
              </View>
              <View style={styles.routeContentWrap}>
                <Text style={styles.routeLabel}>Départ</Text>
                <Text style={styles.routeAddress} numberOfLines={2} ellipsizeMode="tail">
                  {formatAddressFallback(mission.pickup_location)}
                </Text>
              </View>
            </View>
            <View style={[styles.routeRow, styles.routeRowLast]}>
              <View style={styles.routeIconWrap}>
                <Ionicons name="flag-outline" size={18} color={palette.accent} />
              </View>
              <View style={styles.routeContentWrap}>
                <Text style={styles.routeLabel}>Destination</Text>
                <Text style={styles.routeAddress} numberOfLines={2} ellipsizeMode="tail">
                  {formatAddressFallback(mission.dropoff_location)}
                </Text>
              </View>
            </View>
          </>
        )}
      </View>

      {/* 4. MissionHintsSection : accès contextuel (uniquement si données) */}
      <View style={[styles.metaInfoSection, isCompact && styles.metaInfoSectionCompact]}>
        {normalizedStatus !== "IN_PROGRESS" && (() => {
          const hints = getPickupHints(mission);
          if (hints.length === 0) return null;
          const title = "À l'arrivée au point de départ";
          const displayHints = isCompact ? hints.slice(0, 3) : hints;
          const hasMoreHints = isCompact && hints.length > 3;
          return (
            <View style={[styles.hintsSection, isCompact && styles.hintsSectionCompact]}>
              <Text style={[styles.hintsSectionTitle, isCompact && styles.hintsSectionTitleCompact]}>
                {title}
              </Text>
              {displayHints.map((hint, i) => (
                <View
                  key={i}
                  style={[
                    isCompact ? styles.hintRowCompact : styles.hintRow,
                    (isCompact ? i === displayHints.length - 1 && !hasMoreHints : i === hints.length - 1)
                      ? (isCompact ? styles.hintRowCompactLast : styles.hintRowLast)
                      : undefined,
                  ]}
                >
                  {!isCompact && (
                    <View style={styles.hintIconWrap}>
                      <Ionicons name={hint.icon as any} size={16} color={palette.secondary} />
                    </View>
                  )}
                  <View style={styles.routeContentWrap}>
                    {isCompact ? (
                      <Text style={styles.hintLineCompact} numberOfLines={1} ellipsizeMode="tail">
                        {hint.label} : {hint.value}
                      </Text>
                    ) : (
                      <>
                        <Text style={styles.hintLabel}>{hint.label}</Text>
                        <Text style={styles.hintValue} numberOfLines={2} ellipsizeMode="tail">
                          {hint.value}
                        </Text>
                      </>
                    )}
                  </View>
                </View>
              ))}
              {hasMoreHints && (
                <TouchableOpacity
                  onPress={() => setDetailsSheetVisible(true)}
                  style={styles.notesSeeMoreButton}
                  hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                  accessible
                  accessibilityRole="button"
                  accessibilityLabel="Voir plus de détails"
                >
                  <Text style={styles.notesSeeMoreText}>Voir plus</Text>
                </TouchableOpacity>
              )}
            </View>
          );
        })()}
        {normalizedStatus === "IN_PROGRESS" && (() => {
          const hints = getDropoffHints(mission);
          if (hints.length === 0) return null;
          const title = "À l'arrivée à destination";
          const displayHints = isCompact ? hints.slice(0, 3) : hints;
          const hasMoreHints = isCompact && hints.length > 3;
          return (
            <View style={[styles.hintsSection, isCompact && styles.hintsSectionCompact]}>
              <Text style={[styles.hintsSectionTitle, isCompact && styles.hintsSectionTitleCompact]}>
                {title}
              </Text>
              {displayHints.map((hint, i) => (
                <View
                  key={i}
                  style={[
                    isCompact ? styles.hintRowCompact : styles.hintRow,
                    (isCompact ? i === displayHints.length - 1 && !hasMoreHints : i === hints.length - 1)
                      ? (isCompact ? styles.hintRowCompactLast : styles.hintRowLast)
                      : undefined,
                  ]}
                >
                  {!isCompact && (
                    <View style={styles.hintIconWrap}>
                      <Ionicons name={hint.icon as any} size={16} color={palette.secondary} />
                    </View>
                  )}
                  <View style={styles.routeContentWrap}>
                    {isCompact ? (
                      <Text style={styles.hintLineCompact} numberOfLines={1} ellipsizeMode="tail">
                        {hint.label} : {hint.value}
                      </Text>
                    ) : (
                      <>
                        <Text style={styles.hintLabel}>{hint.label}</Text>
                        <Text style={styles.hintValue} numberOfLines={2} ellipsizeMode="tail">
                          {hint.value}
                        </Text>
                      </>
                    )}
                  </View>
                </View>
              ))}
              {hasMoreHints && (
                <TouchableOpacity
                  onPress={() => setDetailsSheetVisible(true)}
                  style={styles.notesSeeMoreButton}
                  hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                  accessible
                  accessibilityRole="button"
                  accessibilityLabel="Voir plus de détails"
                >
                  <Text style={styles.notesSeeMoreText}>Voir plus</Text>
                </TouchableOpacity>
              )}
            </View>
          );
        })()}

        {/* 5. MissionNotes : uniquement quand client à bord (IN_PROGRESS). Notes générales uniquement :
            les notes médicales sont déjà affichées dans les hints "À l'arrivée à destination". */}
        {normalizedStatus === "IN_PROGRESS" &&
          (() => {
            const notesText = (mission.notes ?? "").trim();
            if (!notesText) return null;
            const showSeeMore = notesText.length > NOTES_SEE_MORE_THRESHOLD;
            return (
              <View style={[styles.notesBlock, isCompact && styles.notesBlockCompact]}>
                <Text
                  style={[styles.notesEnhanced, isCompact && styles.notesEnhancedCompact]}
                  numberOfLines={isCompact ? 1 : 2}
                  ellipsizeMode="tail"
                >
                  Notes : {notesText}
                </Text>
                {showSeeMore && (
                  <TouchableOpacity
                    onPress={() => {
                      setNotesFullText(notesText);
                      setNotesModalVisible(true);
                    }}
                    style={isCompact ? styles.notesSeeMoreButtonCompact : styles.notesSeeMoreButton}
                    hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                    accessible
                    accessibilityRole="button"
                    accessibilityLabel="Voir plus"
                  >
                    <Text style={styles.notesSeeMoreText}>Voir plus</Text>
                  </TouchableOpacity>
                )}
              </View>
            );
          })()}
      </View>

      {/* Modal Notes : backdrop = Pressable (ferme), inner = View (100% safe cross-platform) */}
      <Modal
        visible={notesModalVisible}
        transparent
        animationType="fade"
        onRequestClose={() => setNotesModalVisible(false)}
      >
        <Pressable style={styles.notesModalBackdrop} onPress={() => setNotesModalVisible(false)}>
          <TouchableWithoutFeedback onPress={() => { }}>
            <View style={styles.notesModalCard}>
              <Text style={styles.notesModalTitle}>Notes</Text>
              <ScrollView style={styles.notesModalScroll} showsVerticalScrollIndicator>
                <Text style={[styles.hintValue, styles.notesModalBody]} selectable>
                  {notesFullText}
                </Text>
              </ScrollView>
              <TouchableOpacity
                onPress={() => setNotesModalVisible(false)}
                style={styles.notesModalCloseButton}
                accessible
                accessibilityRole="button"
                accessibilityLabel="Fermer"
              >
                <Text style={styles.notesModalCloseText}>Fermer</Text>
              </TouchableOpacity>
            </View>
          </TouchableWithoutFeedback>
        </Pressable>
      </Modal>

      {/* 6. MissionActionsPrimary : max 3 (Appeler, GPS, En route|À bord|Terminer) + Plus → Détails */}
      <View style={[styles.actionsRowEnhanced, isCompact && styles.actionsRowCompact]}>
        {callablePhone != null && (
          <TouchableOpacity
            onPress={onCall}
            style={styles.actionItemEnhanced}
            accessible
            accessibilityRole="button"
            accessibilityLabel="Appeler le client"
          >
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
              if (mission && onComplete) {
                onComplete(mission.id);
              } else {
                handleStatusUpdate("COMPLETED");
              }
            }}
            style={styles.actionItemEnhanced}
            disabled={isUpdatingStatus}
          >
            <Ionicons name="checkmark-done" size={18} color="white" />
            <Text style={styles.actionLabel}>Terminer</Text>
          </TouchableOpacity>
        )}

        {onPressDetails && (
          <TouchableOpacity
            onPress={() => setDetailsSheetVisible(true)}
            style={styles.actionItemMore}
            accessible
            accessibilityRole="button"
            accessibilityLabel="Plus d’actions"
          >
            <Ionicons name="ellipsis-horizontal" size={20} color={palette.text} />
          </TouchableOpacity>
        )}
      </View>

      {/* Sheet Plus : slide depuis le bas, backdrop plus sombre */}
      <Modal
        visible={detailsSheetVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setDetailsSheetVisible(false)}
      >
        <Pressable style={styles.detailsSheetBackdrop} onPress={() => setDetailsSheetVisible(false)}>
          <TouchableWithoutFeedback onPress={() => { }}>
            <View style={styles.detailsSheetCard}>
              <Text style={styles.detailsSheetTitle}>Actions</Text>

              <ScrollView style={styles.detailsSheetScroll} showsVerticalScrollIndicator>
                {/* Client */}
                <View style={styles.detailsSheetSection}>
                  <Text style={styles.detailsSheetSectionTitle}>Client</Text>
                  {getCivilityLabel(mission.client?.gender) != null && (
                    <Text style={styles.detailsSheetLine}>
                      <Text style={styles.detailsSheetLineLabel}>Civilité : </Text>
                      {getCivilityLabel(mission.client?.gender)}
                    </Text>
                  )}
                  <Text style={styles.detailsSheetLine}>
                    <Text style={styles.detailsSheetLineLabel}>Nom : </Text>
                    {mission.client_name || mission.client?.full_name || "Non spécifié"}
                  </Text>
                  {mission.client?.birth_date != null && mission.client.birth_date !== "" && (
                    <Text style={styles.detailsSheetLine}>
                      <Text style={styles.detailsSheetLineLabel}>Date de naissance : </Text>
                      {new Date(mission.client.birth_date).toLocaleDateString("fr-FR", {
                        day: "2-digit",
                        month: "2-digit",
                        year: "numeric",
                      })}
                    </Text>
                  )}
                  <Text style={styles.detailsSheetLine}>
                    <Text style={styles.detailsSheetLineLabel}>Statut : </Text>
                    {formatStatus(status ?? "")}
                  </Text>
                </View>

                {/* Horaires */}
                <View style={styles.detailsSheetSection}>
                  <Text style={styles.detailsSheetSectionTitle}>Horaires</Text>
                  <Text style={styles.detailsSheetLine}>
                    <Text style={styles.detailsSheetLineLabel}>Heure prévue : </Text>
                    {formatTime(new Date(mission.scheduled_time))}
                  </Text>
                  {(getEstimatedArrival != null || getEstimatedArrivalDropoff != null) && (() => {
                    const isAfterPickup = normalizedStatus === "IN_PROGRESS";
                    const arrivalTime = isAfterPickup
                      ? (getEstimatedArrivalDropoff?.(mission.id) ?? null)
                      : (getEstimatedArrival?.(mission.id) ?? null);
                    const secondsToTarget = isAfterPickup
                      ? (getETAToDropoff?.(mission.id) ?? null)
                      : (getETAToPickup?.(mission.id) ?? null);
                    const minutesRounded = secondsToTarget != null ? formatDurationMinutes(secondsToTarget) : null;
                    if (arrivalTime && minutesRounded != null) {
                      return (
                        <Text style={styles.detailsSheetLine}>
                          <Text style={styles.detailsSheetLineLabel}>Arrivée estimée : </Text>
                          {minutesRounded > 0 ? `${formatTime(arrivalTime)} (dans ${minutesRounded} min)` : formatTime(arrivalTime)}
                        </Text>
                      );
                    }
                    return null;
                  })()}
                </View>

                {/* Adresses */}
                <View style={styles.detailsSheetSection}>
                  <Text style={styles.detailsSheetSectionTitle}>Adresses</Text>
                  <Text style={styles.detailsSheetLine}>
                    <Text style={styles.detailsSheetLineLabel}>Prise en charge : </Text>
                    {formatAddressFallback(mission.pickup_location)}
                  </Text>
                  <Text style={styles.detailsSheetLine}>
                    <Text style={styles.detailsSheetLineLabel}>Destination : </Text>
                    {formatAddressFallback(mission.dropoff_location)}
                  </Text>
                </View>

                {/* Hints prise en charge */}
                {(() => {
                  const pickupHints = getPickupHints(mission);
                  if (pickupHints.length === 0) return null;
                  return (
                    <View style={styles.detailsSheetSection}>
                      <Text style={styles.detailsSheetSectionTitle}>Infos prise en charge</Text>
                      {pickupHints.map((h, i) => (
                        <Text key={i} style={styles.detailsSheetLine}>
                          <Text style={styles.detailsSheetLineLabel}>{h.label} : </Text>
                          {h.value}
                        </Text>
                      ))}
                    </View>
                  );
                })()}

                {/* Hints destination */}
                {(() => {
                  const dropoffHints = getDropoffHints(mission);
                  if (dropoffHints.length === 0) return null;
                  return (
                    <View style={styles.detailsSheetSection}>
                      <Text style={styles.detailsSheetSectionTitle}>Infos destination</Text>
                      {dropoffHints.map((h, i) => (
                        <Text key={i} style={styles.detailsSheetLine}>
                          <Text style={styles.detailsSheetLineLabel}>{h.label} : </Text>
                          {h.value}
                        </Text>
                      ))}
                    </View>
                  );
                })()}

                {/* Notes générales (pas notes_medical, affichées dans Médical) */}
                {mission.notes?.trim() && (
                  <View style={styles.detailsSheetSection}>
                    <Text style={styles.detailsSheetSectionTitle}>Notes</Text>
                    <Text style={styles.detailsSheetLine}>{mission.notes.trim()}</Text>
                  </View>
                )}

                {/* Médical */}
                {(mission.medical_facility ?? mission.hospital_service ?? mission.doctor_name ?? mission.notes_medical) && (
                  <View style={styles.detailsSheetSection}>
                    <Text style={styles.detailsSheetSectionTitle}>Médical</Text>
                    {mission.medical_facility?.trim() && (
                      <Text style={styles.detailsSheetLine}>
                        <Text style={styles.detailsSheetLineLabel}>Établissement : </Text>
                        {mission.medical_facility.trim()}
                      </Text>
                    )}
                    {mission.hospital_service?.trim() && (
                      <Text style={styles.detailsSheetLine}>
                        <Text style={styles.detailsSheetLineLabel}>Service : </Text>
                        {mission.hospital_service.trim()}
                      </Text>
                    )}
                    {mission.doctor_name?.trim() && (
                      <Text style={styles.detailsSheetLine}>
                        <Text style={styles.detailsSheetLineLabel}>Médecin : </Text>
                        {mission.doctor_name.trim()}
                      </Text>
                    )}
                    {mission.notes_medical?.trim() && (
                      <Text style={styles.detailsSheetLine}>
                        <Text style={styles.detailsSheetLineLabel}>Notes médicales : </Text>
                        {mission.notes_medical.trim()}
                      </Text>
                    )}
                  </View>
                )}

                {/* Instructions accès */}
                {(mission.pickup_access_notes?.trim() || mission.dropoff_access_notes?.trim()) && (
                  <View style={styles.detailsSheetSection}>
                    <Text style={styles.detailsSheetSectionTitle}>Instructions d'accès</Text>
                    {mission.pickup_access_notes?.trim() && (
                      <Text style={styles.detailsSheetLine}>
                        <Text style={styles.detailsSheetLineLabel}>Prise en charge : </Text>
                        {mission.pickup_access_notes.trim()}
                      </Text>
                    )}
                    {mission.dropoff_access_notes?.trim() && (
                      <Text style={styles.detailsSheetLine}>
                        <Text style={styles.detailsSheetLineLabel}>Destination : </Text>
                        {mission.dropoff_access_notes.trim()}
                      </Text>
                    )}
                  </View>
                )}

                {/* Chaise roulante */}
                {(mission.wheelchair === true || mission.wheelchair_client_has === true || mission.wheelchair_need === true) && (
                  <View style={styles.detailsSheetSection}>
                    <Text style={styles.detailsSheetSectionTitle}>Chaise roulante</Text>
                    <Text style={styles.detailsSheetLine}>
                      {mission.wheelchair_need === true ? "Chaise roulante requise à la destination" : "Client avec chaise roulante"}
                    </Text>
                  </View>
                )}
              </ScrollView>

              <TouchableOpacity
                onPress={() => {
                  setDetailsSheetVisible(false);
                  onPressDetails?.();
                }}
                style={styles.detailsSheetItem}
                accessible
                accessibilityRole="button"
                accessibilityLabel="Voir les détails complets"
              >
                <Ionicons name="information-circle-outline" size={22} color={palette.accent} />
                <Text style={styles.detailsSheetItemText}>Voir les détails complets</Text>
              </TouchableOpacity>
              <TouchableOpacity
                onPress={() => setDetailsSheetVisible(false)}
                style={styles.notesModalCloseButton}
                accessible
                accessibilityRole="button"
                accessibilityLabel="Fermer"
              >
                <Text style={styles.notesModalCloseText}>Fermer</Text>
              </TouchableOpacity>
            </View>
          </TouchableWithoutFeedback>
        </Pressable>
      </Modal>

      {/* 7. MissionActionsDanger : Libérer / Annuler (uniquement ASSIGNED ou EN_ROUTE) */}
      {(normalizedStatus === "ASSIGNED" || normalizedStatus === "EN_ROUTE") && (
        <View style={[styles.actionsRowSecondary, isCompact && styles.actionsRowSecondaryCompact]}>
          <TouchableOpacity
            onPress={handleReleasePress}
            activeOpacity={0.7}
            style={styles.actionItemSecondary}
          >
            <Ionicons name="refresh" size={18} color="white" />
            <Text style={styles.actionLabel}>Libérer</Text>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => setCancelModalVisible(true)}
            style={styles.actionItemDanger}
          >
            <Ionicons name="close-circle" size={18} color="white" />
            <Text style={styles.actionLabel}>Annuler course</Text>
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
