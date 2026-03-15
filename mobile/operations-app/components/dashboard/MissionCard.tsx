import React, { useEffect, useState } from "react";
import { View, Text, TouchableOpacity, Alert, Modal, Pressable, ScrollView, TouchableWithoutFeedback, Platform } from "react-native";
import { useAppAlert } from "@/contexts/AppAlertContext";
import { Ionicons, MaterialIcons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import type { Booking as Mission, BookingStatus } from "@/services/api";
import { styles, palette, dsStyles } from "@/styles/missionCardStyles";
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
import { getLogger } from "@/utils/logger";
import { formatTimeLocal } from "@/utils/formatTimeLocal";

const log = getLogger("MissionCard");
// ——— Helpers : heure locale sans fuseau horaire ———
const formatTime = (d: Date) => formatTimeLocal(d);
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

/** Livraison matériel : mission_type === material_delivery (case-insensitive, tirets/espaces normalisés) */
const isMaterialDelivery = (m: Mission | null): boolean => {
  const missionType = String(m?.mission_type ?? "patient_transport")
    .toLowerCase()
    .replace(/[\s-]+/g, "_");
  return missionType === "material_delivery";
};

/** Description livraison (fallback si absente : null, "", espaces) */
const getDeliveryDescriptionDisplay = (m: Mission | null): string | null => {
  if (!isMaterialDelivery(m)) return null;
  const desc = (m?.delivery_description ?? "").trim();
  return desc || "Livraison (description manquante)";
};

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
    <View style={styles.emptyStateIconWrap}>
      <Ionicons name="car-outline" size={32} color="#00796b" />
    </View>
    <Text style={styles.emptyStateTitle}>Aucune course pour le moment</Text>
    <Text style={styles.emptyStateSubtitle}>
      Vous serez notifie des qu'une course vous sera assignee.
    </Text>
    <View style={styles.emptyStateBadge}>
      <Ionicons name="notifications-outline" size={14} color="#16a34a" />
      <Text style={styles.emptyStateBadgeText}>Notifications actives</Text>
    </View>
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
  const appAlert = useAppAlert();
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
        return "Assignée";
      case "EN_ROUTE":
        return "En route";
      case "IN_PROGRESS":
        return "En cours";
      case "COMPLETED":
        return "Terminée";
      case "CANCELED":
        return "Annulée";
      default:
        return "À venir";
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
        // ✅ Mission annulée : notifier selon le type (modal plateforme)
        if (cancelReason === "RELEASE") {
          appAlert.showAlert(
            "Course libérée",
            "Elle sera réassignée à un autre chauffeur."
          );
        } else {
          appAlert.showAlert("Course annulée", "Course annulée avec justification.");
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
      appAlert.showAlert("Erreur", `Impossible de mettre à jour le statut : ${errorMsg}`);
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
    log.info("release confirmed");
    handleStatusUpdate("CANCELED", "RELEASE");
    setReleaseModalVisible(false);
  };

  const handleReleasePress = () => {
    log.info("release button pressed");
    setReleaseModalVisible(true);
  };

  // ✅ P0-1: Normaliser le statut pour les comparaisons
  const normalizedStatus = normalizeBookingStatus(status) as BookingStatus;
  const isDelivery = isMaterialDelivery(mission);

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

      {/* 1. MissionCardHeader : civilité + identité + badges (type + statut) */}
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
            <View style={{ flexDirection: "row", alignItems: "center", marginTop: 6 }}>
              <Ionicons name="calendar-outline" size={13} color={palette.secondary} style={{ marginRight: 4 }} />
              <Text style={styles.clientBirthDate}>
                {new Date(mission.client.birth_date).toLocaleDateString("fr-FR", {
                  day: "2-digit",
                  month: "2-digit",
                  year: "numeric",
                })}
              </Text>
            </View>
          )}
        </View>
        <View style={styles.headerBadgesWrap}>
          {isDelivery && (
            <View style={[styles.statusBadgeContainer, styles.deliveryTypeBadge]}>
              <View style={{ flexDirection: "row", alignItems: "center", gap: 4 }}>
                <Ionicons name="cube-outline" size={13} color="#B45309" />
                <Text style={styles.deliveryTypeBadgeText}>Livraison</Text>
              </View>
            </View>
          )}
          <View style={styles.statusBadgeContainer}>
            <Text style={styles.statusBadgeText}>{formatStatus(status ?? "")}</Text>
          </View>
        </View>
      </View>

      {/* 1b. Ligne description livraison (si material_delivery) */}
      {isDelivery && getDeliveryDescriptionDisplay(mission) && (
        <View style={[styles.deliveryDescRow, isCompact && styles.deliveryDescRowCompact]}>
          <Ionicons name="cube" size={14} color="#B45309" style={{ marginRight: 4 }} />
          <Text style={styles.deliveryDescLabel}>Livraison — </Text>
          <Text style={styles.deliveryDescText} numberOfLines={1} ellipsizeMode="tail">
            {getDeliveryDescriptionDisplay(mission)}
          </Text>
        </View>
      )}

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
        const arrivalLabel = isDelivery
          ? (isAfterPickup ? "Arrivée au point de dépôt" : "Arrivée au point de retrait")
          : (isAfterPickup ? "Arrivée à destination" : "Arrivée estimée");
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
                {isDelivery ? "Point de retrait : " : "Départ : "}
                {formatAddressFallback(mission.pickup_location)}
              </Text>
            </View>
            <View style={styles.routeRowCompact}>
              <Ionicons name="flag-outline" size={16} color={palette.accent} />
              <Text style={styles.routeAddressCompact} numberOfLines={1} ellipsizeMode="tail">
                {isDelivery ? "Point de dépôt : " : "Destination : "}
                {formatAddressFallback(mission.dropoff_location)}
              </Text>
            </View>
          </>
        ) : (
          <>
            <View style={styles.routeRow}>
              <View style={styles.routeTimelineWrap}>
                <View style={styles.routeDot} />
                <View style={styles.routeConnector} />
              </View>
              <View style={styles.routeContentWrap}>
                <Text style={styles.routeLabel}>{isDelivery ? "Point de retrait" : "Départ"}</Text>
                <Text style={styles.routeAddress} numberOfLines={2} ellipsizeMode="tail">
                  {formatAddressFallback(mission.pickup_location)}
                </Text>
              </View>
            </View>
            <View style={[styles.routeRow, styles.routeRowLast]}>
              <View style={styles.routeTimelineWrap}>
                <View style={[styles.routeDot, styles.routeDotDropoff]} />
              </View>
              <View style={styles.routeContentWrap}>
                <Text style={styles.routeLabel}>{isDelivery ? "Point de dépôt" : "Destination"}</Text>
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
          const title = isDelivery ? "À l'arrivée au point de retrait" : "À l'arrivée au point de départ";
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
                  accessibilityLabel={`Voir plus de détails (${hints.length - 3} info${hints.length - 3 > 1 ? "s" : ""})`}
                >
                  <Text style={styles.notesSeeMoreText}>
                    Voir plus (+{hints.length - 3} info{hints.length - 3 > 1 ? "s" : ""})
                  </Text>
                </TouchableOpacity>
              )}
            </View>
          );
        })()}
        {normalizedStatus === "IN_PROGRESS" && (() => {
          const hints = getDropoffHints(mission);
          if (hints.length === 0) return null;
          const title = isDelivery ? "À l'arrivée au point de dépôt" : "À l'arrivée à destination";
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
                  accessibilityLabel={`Voir plus de détails (${hints.length - 3} info${hints.length - 3 > 1 ? "s" : ""})`}
                >
                  <Text style={styles.notesSeeMoreText}>
                    Voir plus (+{hints.length - 3} info{hints.length - 3 > 1 ? "s" : ""})
                  </Text>
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
            <Ionicons name="call" size={15} color="white" />
            <Text style={styles.actionLabel}>Appeler</Text>
          </TouchableOpacity>
        )}

        {shouldShowNavigation && onNavigate && (
          <TouchableOpacity
            onPress={() => onNavigate(getCurrentDestination())}
            style={styles.actionItemEnhanced}
            disabled={isUpdatingStatus}
          >
            <MaterialIcons name="navigation" size={15} color="white" />
            <Text style={styles.actionLabel}>GPS</Text>
          </TouchableOpacity>
        )}

        {normalizedStatus === "ASSIGNED" && (
          <TouchableOpacity
            onPress={() => handleStatusUpdate("EN_ROUTE")}
            style={styles.actionItemEnhanced}
            disabled={isUpdatingStatus}
          >
            <Ionicons name="walk" size={15} color="white" />
            <Text style={styles.actionLabel}>En route</Text>
          </TouchableOpacity>
        )}

        {normalizedStatus === "EN_ROUTE" && (
          <TouchableOpacity
            onPress={() => handleStatusUpdate("IN_PROGRESS")}
            style={styles.actionItemEnhanced}
            disabled={isUpdatingStatus}
          >
            <Ionicons name={isDelivery ? "cube-outline" : "person"} size={15} color="white" />
            <Text style={styles.actionLabel}>
              {isDelivery ? "Colis récupéré" : "À bord"}
            </Text>
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
            <Ionicons name="checkmark-done" size={15} color="white" />
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
            <Ionicons name="ellipsis-horizontal" size={16} color={palette.text} />
          </TouchableOpacity>
        )}
      </View>

      {/* Bottom sheet — même structure que RideEditModal / RideCreateModal */}
      <Modal
        visible={detailsSheetVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setDetailsSheetVisible(false)}
      >
        <View style={dsStyles.dsRoot}>
          <Pressable style={dsStyles.dsOverlay} onPress={() => setDetailsSheetVisible(false)} />
          <View style={dsStyles.dsSheet}>
            <View style={dsStyles.dsHandle} />

            {/* Header */}
            <View style={dsStyles.dsHeaderBar}>
              <View style={dsStyles.dsHeaderIcon}>
                <Ionicons name="document-text-outline" size={17} color="#00796B" />
              </View>
              <View style={{ flex: 1 }}>
                <Text style={dsStyles.dsHeaderTitle}>Informations course</Text>
                <Text style={dsStyles.dsHeaderSub}>{mission.client_name || mission.client?.full_name || "Client"}</Text>
              </View>
              <View style={dsStyles.dsStatusBadge}>
                <View style={{ width: 6, height: 6, borderRadius: 3, backgroundColor: palette.accent }} />
                <Text style={dsStyles.dsStatusText}>{formatStatus(status ?? "")}</Text>
              </View>
              <TouchableOpacity onPress={() => setDetailsSheetVisible(false)} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
                <Ionicons name="close" size={22} color="#94A3B8" />
              </TouchableOpacity>
            </View>

            {/* Scrollable content */}
            <ScrollView style={dsStyles.dsScroll} contentContainerStyle={dsStyles.dsScrollContent} showsVerticalScrollIndicator={false}>
              {/* Client */}
              <View style={dsStyles.dsCard}>
                <View style={dsStyles.dsCardHeader}>
                  <Ionicons name="person-outline" size={14} color="#00796B" />
                  <Text style={dsStyles.dsCardTitle}>Client</Text>
                </View>
                <View style={dsStyles.dsCardBody}>
                  {getCivilityLabel(mission.client?.gender) != null && (
                    <Text style={dsStyles.dsSecText}>{getCivilityLabel(mission.client?.gender)}</Text>
                  )}
                  <Text style={dsStyles.dsMainText}>{mission.client_name || mission.client?.full_name || "Non spécifié"}</Text>
                  {mission.client?.birth_date != null && mission.client.birth_date !== "" && (
                    <View style={dsStyles.dsChipRow}>
                      <Ionicons name="calendar-outline" size={12} color="#64748B" />
                      <Text style={dsStyles.dsSecText}>
                        {new Date(mission.client.birth_date).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit", year: "numeric" })}
                      </Text>
                    </View>
                  )}
                  {mission.client?.contact_phone && (
                    <View style={dsStyles.dsChipRow}>
                      <Ionicons name="call-outline" size={12} color="#00796B" />
                      <Text style={dsStyles.dsPhoneText}>{mission.client.contact_phone}</Text>
                    </View>
                  )}
                </View>
              </View>

              {/* Horaire + Infos trajet — combinés en une seule carte */}
              <View style={dsStyles.dsCard}>
                <View style={dsStyles.dsCardHeader}>
                  <Ionicons name="time-outline" size={14} color="#00796B" />
                  <Text style={dsStyles.dsCardTitle}>Horaire & trajet</Text>
                </View>
                <View style={dsStyles.dsCardBody}>
                  <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
                    <Text style={{ fontSize: 15, fontWeight: "700", color: "#1E293B" }}>{formatTime(new Date(mission.scheduled_time))}</Text>
                    {mission.is_return && (
                      <View style={dsStyles.dsReturnChip}>
                        <Ionicons name="repeat-outline" size={11} color="#B45309" />
                        <Text style={{ fontSize: 11, fontWeight: "600", color: "#B45309" }}>
                          Retour{mission.return_time ? ` ${formatTime(new Date(mission.return_time))}` : ""}
                        </Text>
                      </View>
                    )}
                  </View>
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
                        <View style={dsStyles.dsChipRow}>
                          <Ionicons name="timer-outline" size={12} color="#64748B" />
                          <Text style={dsStyles.dsSecText}>
                            Arrivée {minutesRounded > 0 ? `${formatTime(arrivalTime)} (dans ${minutesRounded} min)` : formatTime(arrivalTime)}
                          </Text>
                        </View>
                      );
                    }
                    return null;
                  })()}
                  {(mission.distance_meters || mission.duration_seconds) && (
                    <View style={{ flexDirection: "row", flexWrap: "wrap", gap: 8, marginTop: 6 }}>
                      {mission.distance_meters != null && mission.distance_meters > 0 && (
                        <View style={dsStyles.dsMetricChip}>
                          <Ionicons name="car-outline" size={12} color="#00796B" />
                          <Text style={dsStyles.dsMetricText}>{(mission.distance_meters / 1000).toFixed(1)} km</Text>
                        </View>
                      )}
                      {mission.duration_seconds != null && mission.duration_seconds > 0 && (
                        <View style={dsStyles.dsMetricChip}>
                          <Ionicons name="hourglass-outline" size={12} color="#00796B" />
                          <Text style={dsStyles.dsMetricText}>{Math.round(mission.duration_seconds / 60)} min</Text>
                        </View>
                      )}
                    </View>
                  )}
                </View>
              </View>

              {/* Livraison */}
              {isMaterialDelivery(mission) && (
                <View style={dsStyles.dsCard}>
                  <View style={[dsStyles.dsCardHeader, { backgroundColor: "rgba(245,158,11,0.08)", borderBottomColor: "rgba(245,158,11,0.12)" }]}>
                    <Ionicons name="cube-outline" size={14} color="#B45309" />
                    <Text style={[dsStyles.dsCardTitle, { color: "#B45309" }]}>Livraison</Text>
                  </View>
                  <View style={dsStyles.dsCardBody}>
                    <Text style={dsStyles.dsMainText}>{getDeliveryDescriptionDisplay(mission) ?? "—"}</Text>
                  </View>
                </View>
              )}

              {/* Trajet */}
              <View style={dsStyles.dsCard}>
                <View style={dsStyles.dsCardHeader}>
                  <Ionicons name="navigate-outline" size={14} color="#00796B" />
                  <Text style={dsStyles.dsCardTitle}>Trajet</Text>
                </View>
                <View style={dsStyles.dsCardBody}>
                  <View style={{ flexDirection: "row", gap: 10, marginBottom: 10 }}>
                    <View style={{ alignItems: "center", width: 14, paddingTop: 5 }}>
                      <View style={{ width: 10, height: 10, borderRadius: 5, backgroundColor: "#00796B" }} />
                      <View style={{ width: 2, flex: 1, backgroundColor: "#E2E8F0", marginVertical: 3, minHeight: 14 }} />
                    </View>
                    <View style={{ flex: 1 }}>
                      <Text style={dsStyles.dsRouteLabel}>{isMaterialDelivery(mission) ? "Point de retrait" : "Prise en charge"}</Text>
                      <Text style={dsStyles.dsMainText}>{formatAddressFallback(mission.pickup_location)}</Text>
                    </View>
                  </View>
                  <View style={{ flexDirection: "row", gap: 10 }}>
                    <View style={{ alignItems: "center", width: 14, paddingTop: 5 }}>
                      <View style={{ width: 10, height: 10, borderRadius: 3, backgroundColor: "#1E293B" }} />
                    </View>
                    <View style={{ flex: 1 }}>
                      <Text style={dsStyles.dsRouteLabel}>{isMaterialDelivery(mission) ? "Point de dépôt" : "Destination"}</Text>
                      <Text style={dsStyles.dsMainText}>{formatAddressFallback(mission.dropoff_location)}</Text>
                    </View>
                  </View>
                </View>
              </View>

              {/* Infos prise en charge */}
              {(() => {
                const pickupHints = getPickupHints(mission);
                if (pickupHints.length === 0) return null;
                return (
                  <View style={dsStyles.dsCard}>
                    <View style={dsStyles.dsCardHeader}>
                      <Ionicons name="log-in-outline" size={14} color="#00796B" />
                      <Text style={dsStyles.dsCardTitle}>Infos prise en charge</Text>
                    </View>
                    <View style={dsStyles.dsCardBody}>
                      {pickupHints.map((h, i) => (
                        <View key={i} style={[dsStyles.dsInfoRow, i === pickupHints.length - 1 && { marginBottom: 0 }]}>
                          <Ionicons name={h.icon as any} size={14} color="#64748B" />
                          <View style={{ flex: 1 }}>
                            <Text style={dsStyles.dsInfoLabel}>{h.label}</Text>
                            <Text style={dsStyles.dsMainText}>{h.value}</Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  </View>
                );
              })()}

              {/* Infos destination */}
              {(() => {
                const dropoffHints = getDropoffHints(mission);
                if (dropoffHints.length === 0) return null;
                return (
                  <View style={dsStyles.dsCard}>
                    <View style={dsStyles.dsCardHeader}>
                      <Ionicons name="log-out-outline" size={14} color="#00796B" />
                      <Text style={dsStyles.dsCardTitle}>Infos destination</Text>
                    </View>
                    <View style={dsStyles.dsCardBody}>
                      {dropoffHints.map((h, i) => (
                        <View key={i} style={[dsStyles.dsInfoRow, i === dropoffHints.length - 1 && { marginBottom: 0 }]}>
                          <Ionicons name={h.icon as any} size={14} color="#64748B" />
                          <View style={{ flex: 1 }}>
                            <Text style={dsStyles.dsInfoLabel}>{h.label}</Text>
                            <Text style={dsStyles.dsMainText}>{h.value}</Text>
                          </View>
                        </View>
                      ))}
                    </View>
                  </View>
                );
              })()}

              {/* Médical */}
              {(mission.medical_facility ?? mission.hospital_service ?? mission.doctor_name ?? mission.notes_medical) && (
                <View style={dsStyles.dsCard}>
                  <View style={dsStyles.dsCardHeader}>
                    <Ionicons name="medkit-outline" size={14} color="#00796B" />
                    <Text style={dsStyles.dsCardTitle}>Médical</Text>
                  </View>
                  <View style={dsStyles.dsCardBody}>
                    {mission.medical_facility?.trim() && (
                      <View style={dsStyles.dsInfoRow}>
                        <Ionicons name="business-outline" size={14} color="#64748B" />
                        <View style={{ flex: 1 }}><Text style={dsStyles.dsInfoLabel}>Établissement</Text><Text style={dsStyles.dsMainText}>{mission.medical_facility.trim()}</Text></View>
                      </View>
                    )}
                    {mission.hospital_service?.trim() && (
                      <View style={dsStyles.dsInfoRow}>
                        <Ionicons name="layers-outline" size={14} color="#64748B" />
                        <View style={{ flex: 1 }}><Text style={dsStyles.dsInfoLabel}>Service</Text><Text style={dsStyles.dsMainText}>{mission.hospital_service.trim()}</Text></View>
                      </View>
                    )}
                    {mission.doctor_name?.trim() && (
                      <View style={dsStyles.dsInfoRow}>
                        <Ionicons name="person-outline" size={14} color="#64748B" />
                        <View style={{ flex: 1 }}><Text style={dsStyles.dsInfoLabel}>Médecin</Text><Text style={dsStyles.dsMainText}>Dr {mission.doctor_name.trim()}</Text></View>
                      </View>
                    )}
                    {mission.notes_medical?.trim() && (
                      <View style={[dsStyles.dsInfoRow, { marginBottom: 0 }]}>
                        <Ionicons name="document-text-outline" size={14} color="#64748B" />
                        <View style={{ flex: 1 }}><Text style={dsStyles.dsInfoLabel}>Notes médicales</Text><Text style={dsStyles.dsMainText}>{mission.notes_medical.trim()}</Text></View>
                      </View>
                    )}
                  </View>
                </View>
              )}

              {/* Instructions accès */}
              {(mission.pickup_access_notes?.trim() || mission.dropoff_access_notes?.trim()) && (
                <View style={dsStyles.dsCard}>
                  <View style={dsStyles.dsCardHeader}>
                    <Ionicons name="enter-outline" size={14} color="#00796B" />
                    <Text style={dsStyles.dsCardTitle}>Instructions d'accès</Text>
                  </View>
                  <View style={dsStyles.dsCardBody}>
                    {mission.pickup_access_notes?.trim() && (
                      <View style={dsStyles.dsInfoRow}>
                        <Ionicons name="location-outline" size={14} color="#64748B" />
                        <View style={{ flex: 1 }}><Text style={dsStyles.dsInfoLabel}>Prise en charge</Text><Text style={dsStyles.dsMainText}>{mission.pickup_access_notes.trim()}</Text></View>
                      </View>
                    )}
                    {mission.dropoff_access_notes?.trim() && (
                      <View style={[dsStyles.dsInfoRow, { marginBottom: 0 }]}>
                        <Ionicons name="flag-outline" size={14} color="#64748B" />
                        <View style={{ flex: 1 }}><Text style={dsStyles.dsInfoLabel}>Destination</Text><Text style={dsStyles.dsMainText}>{mission.dropoff_access_notes.trim()}</Text></View>
                      </View>
                    )}
                  </View>
                </View>
              )}

              {/* Notes */}
              {mission.notes?.trim() && (
                <View style={dsStyles.dsCard}>
                  <View style={dsStyles.dsCardHeader}>
                    <Ionicons name="chatbox-ellipses-outline" size={14} color="#00796B" />
                    <Text style={dsStyles.dsCardTitle}>Notes</Text>
                  </View>
                  <View style={dsStyles.dsCardBody}>
                    <Text style={{ fontSize: 13, color: "#1E293B", lineHeight: 19 }}>{mission.notes.trim()}</Text>
                  </View>
                </View>
              )}

              {/* Entreprise */}
              {mission.company_name && (
                <View style={dsStyles.dsCard}>
                  <View style={dsStyles.dsCardHeader}>
                    <Ionicons name="business-outline" size={14} color="#00796B" />
                    <Text style={dsStyles.dsCardTitle}>Entreprise</Text>
                  </View>
                  <View style={dsStyles.dsCardBody}>
                    <Text style={dsStyles.dsMainText}>{mission.company_name}</Text>
                  </View>
                </View>
              )}
            </ScrollView>
          </View>
        </View>
      </Modal>

      {/* 7. MissionActionsDanger : Libérer / Annuler (uniquement ASSIGNED ou EN_ROUTE) */}
      {(normalizedStatus === "ASSIGNED" || normalizedStatus === "EN_ROUTE") && (
        <View style={[styles.actionsRowSecondary, isCompact && styles.actionsRowSecondaryCompact]}>
          <TouchableOpacity
            onPress={handleReleasePress}
            activeOpacity={0.7}
            style={styles.actionItemSecondary}
          >
            <Ionicons name="refresh" size={15} color="white" />
            <Text style={styles.actionLabel}>Libérer</Text>
          </TouchableOpacity>
          <TouchableOpacity
            onPress={() => setCancelModalVisible(true)}
            style={styles.actionItemDanger}
          >
            <Ionicons name="close-circle" size={15} color="white" />
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

      {/* Modal de libération de course */}
      <Modal
        visible={releaseModalVisible}
        transparent
        animationType="slide"
        onRequestClose={() => setReleaseModalVisible(false)}
      >
        <Pressable
          style={{
            flex: 1,
            backgroundColor: "rgba(0,0,0,0.45)",
            justifyContent: "flex-end",
          }}
          onPress={() => setReleaseModalVisible(false)}
        >
          <View
            style={{
              backgroundColor: "#fff",
              borderTopLeftRadius: 20,
              borderTopRightRadius: 20,
              paddingBottom: Platform.OS === "ios" ? 32 : 16,
              ...(Platform.OS === "web"
                ? { boxShadow: "0 -4px 24px rgba(0,0,0,0.12)" }
                : {
                    shadowColor: "#000",
                    shadowOffset: { width: 0, height: -4 },
                    shadowOpacity: 0.1,
                    shadowRadius: 16,
                    elevation: 12,
                  }),
            }}
            onStartShouldSetResponder={() => true}
            onTouchEnd={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <View
              style={{
                flexDirection: "row",
                alignItems: "center",
                justifyContent: "space-between",
                paddingHorizontal: 20,
                paddingTop: 20,
                paddingBottom: 14,
                borderBottomWidth: 1,
                borderBottomColor: "rgba(0,121,107,0.08)",
              }}
            >
              <View style={{ flexDirection: "row", alignItems: "center", gap: 10, flex: 1 }}>
                <View
                  style={{
                    width: 32,
                    height: 32,
                    borderRadius: 8,
                    backgroundColor: "rgba(0,121,107,0.08)",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Ionicons name="swap-horizontal" size={16} color="#00796B" />
                </View>
                <Text style={{ fontSize: 16, fontWeight: "600", color: "#1E293B", letterSpacing: -0.2 }}>
                  Libérer la course
                </Text>
              </View>
              <TouchableOpacity
                onPress={() => setReleaseModalVisible(false)}
                hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                style={{ width: 28, height: 28, borderRadius: 6, alignItems: "center", justifyContent: "center" }}
              >
                <Ionicons name="close" size={18} color="#94A3B8" />
              </TouchableOpacity>
            </View>

            {/* Body */}
            <View style={{ paddingHorizontal: 20, paddingTop: 16, paddingBottom: 8 }}>
              <Text style={{ fontSize: 14, color: "#1E293B", fontWeight: "500", lineHeight: 21, marginBottom: 10 }}>
                Libérer cette course ?
              </Text>
              <Text style={{ fontSize: 13, color: "#64748B", lineHeight: 19 }}>
                Elle sera réassignée à un autre chauffeur.
              </Text>

              <View
                style={{
                  flexDirection: "row",
                  alignItems: "center",
                  gap: 8,
                  marginTop: 14,
                  paddingVertical: 10,
                  paddingHorizontal: 12,
                  backgroundColor: "rgba(0,121,107,0.04)",
                  borderRadius: 10,
                  borderWidth: 1,
                  borderColor: "rgba(0,121,107,0.08)",
                }}
              >
                <Ionicons name="information-circle-outline" size={16} color="#00796B" />
                <Text style={{ fontSize: 12, color: "#64748B", flex: 1, lineHeight: 17 }}>
                  Sans facturation.
                </Text>
              </View>
            </View>

            {/* Footer */}
            <View
              style={{
                flexDirection: "row",
                gap: 10,
                paddingHorizontal: 20,
                paddingTop: 16,
              }}
            >
              <TouchableOpacity
                onPress={() => setReleaseModalVisible(false)}
                style={{
                  flex: 1,
                  height: 40,
                  borderRadius: 10,
                  alignItems: "center",
                  justifyContent: "center",
                  backgroundColor: "#f8fafc",
                  borderWidth: 1,
                  borderColor: "rgba(0,0,0,0.08)",
                }}
              >
                <Text style={{ fontSize: 14, fontWeight: "500", color: "#64748B" }}>Annuler</Text>
              </TouchableOpacity>
              <TouchableOpacity
                onPress={handleReleaseConfirm}
                style={{
                  flex: 1.5,
                  height: 40,
                  borderRadius: 10,
                  flexDirection: "row",
                  alignItems: "center",
                  justifyContent: "center",
                  backgroundColor: "#00796B",
                  gap: 6,
                }}
              >
                <Ionicons name="swap-horizontal-outline" size={15} color="#fff" />
                <Text style={{ fontSize: 14, fontWeight: "600", color: "#fff" }}>Libérer</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Pressable>
      </Modal>

    </View>
  );
};

// ✅ Attacher EmptyStateComponent comme propriété statique pour compatibilité
MissionCard.EmptyState = EmptyStateComponent;

export default MissionCard;
