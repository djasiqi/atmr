import React, { useState, useEffect, useRef } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  GestureResponderEvent,
  ViewStyle,
  Animated,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";

import { EnterpriseCard } from "./EnterpriseCard";
import { isCompletedStatus } from "@/utils/bookingStatus";
import { sendIngestEvent } from "@/src/config/telemetry";

type BadgeTone = "default" | "warning" | "danger" | "info" | "success";

type Badge = {
  label: string;
  tone?: BadgeTone;
};

export type RideSnippet = {
  id: string;
  time: string;
  client: string;
  pickup: string;
  dropoff: string;
  assignedTo?: string | null;
  status?: "unassigned" | "assigned" | "completed" | "return_completed" | "in_progress" | "en_route" | "pending"; // ✅ Ajout du statut pending pour les transferts
  badges?: Badge[];
  onPress?: (event: GestureResponderEvent) => void;
  onPrimaryAction?: (event: GestureResponderEvent) => void;
  onQuickAction?: (event: GestureResponderEvent) => void;
  primaryIcon?: string; // ✅ Icône personnalisée pour l'action principale
  quickIcon?: string; // ✅ Icône personnalisée pour l'action rapide
  primaryIconColor?: string; // ✅ Couleur personnalisée pour l'icône principale
  quickIconColor?: string; // ✅ Couleur personnalisée pour l'icône rapide
  footerActions?: React.ReactNode;
  showUndefinedIcon?: boolean;
  delayMinutes?: number | null; // ✅ Minutes de retard (positif) ou d'avance (négatif)
};

const palette = {
  time: "#00796B",
  timeUndefined: "#F59E0B",
  client: "#1E293B",
  chevron: "#94A3B8",
  badgeText: "#1E293B",
  badgeDefaultBg: "rgba(100,116,139,0.1)",
  badgeAssignedBg: "rgba(0,121,107,0.1)",
  badgeCompletedBg: "rgba(100,116,139,0.12)",
  badgeCompletedText: "#64748B",
  badgeWarningBg: "rgba(251,191,36,0.12)",
  badgeDangerBg: "rgba(239,68,68,0.1)",
  badgeInfoBg: "rgba(59,130,246,0.1)",
  badgeBorder: "rgba(0,121,107,0.1)",
  routeText: "#64748B",
  pickupIcon: "#00796B",
  dropoffIcon: "#00796B",
  chipBg: "rgba(0,121,107,0.07)",
  chipIcon: "#00796B",
  assignBg: "#00796B",
  assignIcon: "#FFFFFF",
  expandedDivider: "rgba(0,121,107,0.06)",
};

// ✅ Couleurs de statut alignées avec le frontend web (ReservationTable.module.css)
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
  const normalizedStatus = status.toLowerCase();
  return statusColors[normalizedStatus as keyof typeof statusColors] || statusColors.pending;
};

const BADGE_LIMIT = 10;

const formatBadge = (value: string) => {
  const trimmed = value.trim();
  if (!trimmed) return trimmed;

  if (trimmed.toLowerCase() === "non assignée") {
    return "Non Assigné";
  }

  const parts = trimmed.split(/\s+/);
  if (parts.length === 1) {
    return parts[0].slice(0, BADGE_LIMIT);
  }

  const first = parts[0];
  const second = parts[1] ?? "";
  const remaining = BADGE_LIMIT - first.length - 1;
  if (remaining <= 0) return first.slice(0, BADGE_LIMIT);

  const truncatedSecond = second.slice(0, Math.max(1, remaining));
  return `${first} ${truncatedSecond}`;
};

const toneStyle = (tone?: BadgeTone) => {
  switch (tone) {
    case "warning":
      return styles.badgeWarning;
    case "danger":
      return styles.badgeDanger;
    case "info":
      return styles.badgeInfo;
    default:
      return styles.badgeDefault;
  }
};

// ✅ Composant de texte défilant pour les retards
const ScrollingText: React.FC<{
  text: string;
  style?: ViewStyle;
  textStyle?: any;
}> = ({ text, style, textStyle }) => {
  const scrollX = useRef(new Animated.Value(0)).current;
  const [textWidth, setTextWidth] = useState(0);
  const [containerWidth, setContainerWidth] = useState(0);
  const shouldScroll = textWidth > containerWidth;

  useEffect(() => {
    if (shouldScroll) {
      const scrollDistance = textWidth - containerWidth + 20; // 20px de marge
      Animated.loop(
        Animated.sequence([
          Animated.delay(1000),
          Animated.timing(scrollX, {
            toValue: -scrollDistance,
            duration: 3000,
            useNativeDriver: true,
          }),
          Animated.delay(1000),
          Animated.timing(scrollX, {
            toValue: 0,
            duration: 3000,
            useNativeDriver: true,
          }),
        ])
      ).start();
    }
  }, [shouldScroll, textWidth, containerWidth, scrollX]);

  return (
    <View
      style={[styles.scrollingContainer, style]}
      onLayout={(e) => setContainerWidth(e.nativeEvent.layout.width)}
    >
      <Animated.View
        style={[
          styles.scrollingContent,
          shouldScroll && {
            transform: [{ translateX: scrollX }],
          },
        ]}
      >
        <Text
          style={textStyle}
          onLayout={(e) => setTextWidth(e.nativeEvent.layout.width)}
        >
          {text}
        </Text>
      </Animated.View>
    </View>
  );
};

export const RideSnippetCard: React.FC<{
  ride: RideSnippet;
  style?: ViewStyle;
  expanded?: boolean;
  onToggle?: () => void;
}> = ({ ride, style, expanded: controlledExpanded, onToggle }) => {
  const [internalExpanded, setInternalExpanded] = useState(false);

  // Utiliser l'état contrôlé si fourni, sinon utiliser l'état interne
  const expanded = controlledExpanded !== undefined ? controlledExpanded : internalExpanded;

  const toggleExpanded = () => {
    if (onToggle) {
      onToggle();
    } else {
      setInternalExpanded((prev) => !prev);
    }
  };

  return (
    <EnterpriseCard style={[styles.card, style]}>
      <TouchableOpacity
        style={styles.summaryRow}
        onPress={toggleExpanded}
        activeOpacity={0.85}
      >
        <View style={styles.timeContainer}>
          {ride.showUndefinedIcon ? (
            <Ionicons
              name="time-outline"
              size={18}
              color={palette.timeUndefined}
            />
          ) : (
            <Text style={styles.time}>{ride.time}</Text>
          )}
        </View>

        <Text style={styles.client} numberOfLines={1} ellipsizeMode="tail">
          {ride.client}
        </Text>

        <View style={styles.badgeContainer}>
          {(() => {
            // ✅ Normaliser le statut en minuscules pour éviter les problèmes de casse
            const normalizedStatus = ride.status ? String(ride.status).toLowerCase().trim() : undefined;

            // ✅ Obtenir les couleurs selon le statut (aligné avec le frontend web)
            const statusColors = getStatusColors(normalizedStatus);
            // ✅ P0-1: Utiliser la fonction de normalisation
            const isCompleted = isCompletedStatus(normalizedStatus);

            // ✅ Si terminée, toujours afficher les couleurs de statut (vert) SANS retard
            // ✅ Afficher le badge même si pas de chauffeur assigné (cas rare mais possible)
            // ✅ Ignorer complètement delayMinutes pour les courses terminées
            if (isCompleted) {
              return (
                <View
                  style={[
                    styles.badge,
                    {
                      backgroundColor: statusColors.bg,
                      borderColor: statusColors.text + "40", // 40 = 25% opacity en hex
                    },
                  ]}
                >
                  <Text
                    style={[
                      styles.badgeLabel,
                      {
                        color: statusColors.text,
                      },
                    ]}
                    numberOfLines={1}
                    ellipsizeMode="tail"
                  >
                    {ride.assignedTo ? formatBadge(ride.assignedTo) : "Terminée"}
                  </Text>
                </View>
              );
            }

            // ✅ Sinon, logique normale avec retard et couleurs de statut
            if (ride.assignedTo) {
              const delayMinutes = ride.delayMinutes ?? 0;
              const hasDelay = delayMinutes > 0;
              const isLongDelay = hasDelay && delayMinutes >= 15;
              // ✅ Raccourcir le texte de retard : utiliser seulement le prénom pour garder la même hauteur
              const shortName = hasDelay
                ? ride.assignedTo.split(' ')[0].toUpperCase() // Seulement le prénom
                : ride.assignedTo;
              const delayText = hasDelay
                ? `${shortName} ${delayMinutes}min`
                : formatBadge(ride.assignedTo);

              // ✅ Utiliser les couleurs de retard si présent, sinon les couleurs de statut
              const badgeBg = hasDelay
                ? isLongDelay
                  ? "#fee2e2" // --error-bg pour retard long (rouge)
                  : "#fef3c7" // --warning-bg pour retard court (jaune)
                : statusColors.bg;
              const badgeText = hasDelay
                ? isLongDelay
                  ? "#ef4444" // --error-primary pour retard long (rouge)
                  : "#f59e0b" // --warning-primary pour retard court (jaune)
                : statusColors.text;

              return (
                <View
                  style={[
                    styles.badge,
                    {
                      backgroundColor: badgeBg,
                      borderColor: badgeText + "40", // 40 = 25% opacity en hex
                    },
                    hasDelay && isLongDelay && styles.badgeLongDelay,
                    hasDelay && !isLongDelay && styles.badgeShortDelay,
                  ]}
                >
                  <Text
                    style={[
                      styles.badgeLabel,
                      {
                        color: badgeText,
                      },
                      hasDelay && isLongDelay && styles.badgeLabelLongDelay,
                      hasDelay && !isLongDelay && styles.badgeLabelShortDelay,
                    ]}
                    numberOfLines={1}
                    ellipsizeMode="tail"
                  >
                    {delayText}
                  </Text>
                </View>
              );
            }

            // ✅ Non assignée - utiliser les couleurs de statut "pending"
            return (
              <View
                style={[
                  styles.badge,
                  {
                    backgroundColor: statusColors.bg,
                    borderColor: statusColors.text + "40", // 40 = 25% opacity en hex
                  },
                ]}
              >
                <Text
                  style={[
                    styles.badgeLabel,
                    {
                      color: statusColors.text,
                    },
                  ]}
                  numberOfLines={1}
                  ellipsizeMode="tail"
                >
                  Non Assigné
                </Text>
              </View>
            );
          })()}
        </View>

        <View style={styles.chevronContainer}>
          <Ionicons
            name={expanded ? "chevron-up-outline" : "chevron-down-outline"}
            size={16}
            color={palette.chevron}
          />
        </View>
      </TouchableOpacity>

      {expanded && (
        <View style={styles.expandedContent}>
          <View style={styles.routeColumn}>
            <View style={styles.routeRow}>
              <View style={styles.routeIcon}>
                <Ionicons
                  name="location-outline"
                  size={16}
                  color={palette.pickupIcon}
                />
              </View>
              <Text style={styles.route} numberOfLines={1} ellipsizeMode="tail">
                {ride.pickup}
              </Text>
            </View>
            <View style={styles.routeDivider} />
            <View style={styles.routeRow}>
              <View style={styles.routeIcon}>
                <Ionicons
                  name="flag-outline"
                  size={16}
                  color={palette.dropoffIcon}
                />
              </View>
              <Text style={styles.route} numberOfLines={1} ellipsizeMode="tail">
                {ride.dropoff}
              </Text>
            </View>
          </View>

          {/* ✅ Désactiver les actions si la course est terminée */}
          {(() => {
            // ✅ Normaliser le statut pour vérifier si la course est terminée
            const normalizedStatus = ride.status ? String(ride.status).toLowerCase().trim() : undefined;
            // ✅ P0-1: Utiliser la fonction de normalisation
            const isCompleted = isCompletedStatus(normalizedStatus);

            (() => {
              try {
                sendIngestEvent({ location: 'RideSnippetCard.tsx:427', message: 'Quick actions rendering', data: { rideId: ride.id, hasOnQuickAction: !!ride.onQuickAction, hasOnPrimaryAction: !!ride.onPrimaryAction, isCompleted: isCompleted, assignedTo: ride.assignedTo, quickIcon: ride.quickIcon, primaryIcon: ride.primaryIcon, willShowActions: !!(ride.onQuickAction || ride.onPrimaryAction) && !isCompleted, willShowQuick: !!(ride.onQuickAction && (!ride.assignedTo || ride.quickIcon)) }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run8', hypothesisId: 'H3-H4' });
              } catch { }
            })();
            return (ride.onQuickAction || ride.onPrimaryAction) && !isCompleted ? (
              <View style={styles.expandedActions}>
                {ride.onQuickAction && (!ride.assignedTo || ride.quickIcon) ? (
                  <TouchableOpacity
                    style={[
                      styles.chipButton,
                      ride.quickIconColor && { backgroundColor: ride.quickIconColor + "15" }, // Bg semi-transparent
                    ]}
                    onPress={(e) => {
                      try {
                        sendIngestEvent({ location: 'RideSnippetCard.tsx:onQuickAction:click', message: 'Quick action button clicked', data: { rideId: ride.id, hasOnQuickAction: !!ride.onQuickAction }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run11', hypothesisId: 'H1' });
                      } catch { }
                      ride.onQuickAction?.(e);
                    }}
                  >
                    <Ionicons
                      name={(ride.quickIcon as any) || "flash-outline"}
                      size={16}
                      color={ride.quickIconColor || palette.chipIcon} // ✅ Couleur personnalisée
                    />
                  </TouchableOpacity>
                ) : null}
                {ride.onPrimaryAction ? (
                  <TouchableOpacity
                    style={[
                      styles.assignButton,
                      ride.primaryIconColor && { backgroundColor: ride.primaryIconColor }, // ✅ Couleur personnalisée
                    ]}
                    onPress={(e) => {
                      try {
                        sendIngestEvent({ location: 'RideSnippetCard.tsx:onPrimaryAction:click', message: 'Primary action button clicked', data: { rideId: ride.id, hasOnPrimaryAction: !!ride.onPrimaryAction }, timestamp: Date.now(), sessionId: 'debug-session', runId: 'run11', hypothesisId: 'H1' });
                      } catch { }
                      ride.onPrimaryAction?.(e);
                    }}
                  >
                    <Ionicons
                      name={(ride.primaryIcon as any) || "person-add-outline"}
                      size={18}
                      color={ride.primaryIconColor ? "#FFFFFF" : palette.assignIcon} // ✅ Blanc si couleur personnalisée
                    />
                  </TouchableOpacity>
                ) : null}
              </View>
            ) : null;
          })()}
        </View>
      )}
      {expanded && ride.footerActions ? (
        <View style={styles.footerActions}>{ride.footerActions}</View>
      ) : null}
    </EnterpriseCard>
  );
};

const styles = StyleSheet.create({
  card: {
    padding: 12,
  },
  summaryRow: {
    flexDirection: "row",
    alignItems: "center",
  },
  timeContainer: {
    width: 50,
    alignItems: "flex-start",
    marginRight: 10,
  },
  time: {
    color: palette.time,
    fontWeight: "700",
    fontSize: 15,
    letterSpacing: 0.2,
  },
  client: {
    color: palette.client,
    fontSize: 14,
    fontWeight: "600",
    width: 120,
    marginRight: 10,
  },
  chevronContainer: {
    width: 24,
    alignItems: "center",
    marginLeft: 2,
    marginRight: 12,
  },
  badgeContainer: {
    flex: 1,
    alignItems: "flex-end",
  },
  badge: {
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
    maxWidth: 130,
    minWidth: 60,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: palette.badgeBorder,
  },
  badgeLabel: {
    fontSize: 10,
    fontWeight: "700",
    color: palette.badgeText,
    letterSpacing: 0.3,
    textTransform: "uppercase",
    textAlign: "center",
  },
  badgeDefault: {
    backgroundColor: palette.badgeDefaultBg,
  },
  assignedBadge: {
    backgroundColor: palette.badgeAssignedBg,
  },
  badgeCompleted: {
    backgroundColor: palette.badgeCompletedBg,
    borderColor: "rgba(95,115,105,0.25)",
  },
  badgeLabelCompleted: {
    color: palette.badgeCompletedText,
  },
  badgeShortDelay: {
    backgroundColor: "rgba(251,191,36,0.15)",
    borderColor: "rgba(251,191,36,0.3)",
  },
  badgeLongDelay: {
    backgroundColor: "rgba(239,68,68,0.15)",
    borderColor: "rgba(239,68,68,0.3)",
  },
  badgeLabelShortDelay: {
    color: "#F59E0B",
  },
  badgeLabelLongDelay: {
    color: "#EF4444",
  },
  scrollingContainer: {
    overflow: "hidden",
    flex: 1,
  },
  scrollingContent: {
    flexDirection: "row",
  },
  badgeWarning: {
    backgroundColor: palette.badgeWarningBg,
  },
  badgeDanger: {
    backgroundColor: palette.badgeDangerBg,
  },
  badgeInfo: {
    backgroundColor: palette.badgeInfoBg,
  },
  chipButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: palette.chipBg,
    borderWidth: 1,
    borderColor: palette.badgeBorder,
    marginLeft: 10,
  },
  assignButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: palette.assignBg,
    marginLeft: 10,
  },
  routeRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 4,
  },
  routeIcon: {
    marginRight: 8,
  },
  routeDivider: {
    height: 1,
    backgroundColor: palette.expandedDivider,
    marginVertical: 6,
    marginLeft: 24,
  },
  route: {
    color: palette.routeText,
    fontSize: 13,
    flexShrink: 1,
    maxWidth: 180,
  },
  expandedContent: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 10,
  },
  routeColumn: {
    flex: 1,
    marginRight: 16,
  },
  expandedActions: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "flex-end",
    marginLeft: -10, // Compenser les marges des boutons
  },
  footerActions: {
    marginTop: 10,
  },
});
