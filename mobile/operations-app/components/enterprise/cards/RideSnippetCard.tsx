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

type BadgeTone = "default" | "warning" | "danger" | "info";

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
  status?: "unassigned" | "assigned" | "completed" | "return_completed" | "in_progress" | "en_route"; // ✅ Statut de la course
  badges?: Badge[];
  onPress?: (event: GestureResponderEvent) => void;
  onPrimaryAction?: (event: GestureResponderEvent) => void;
  onQuickAction?: (event: GestureResponderEvent) => void;
  primaryIcon?: string;
  footerActions?: React.ReactNode;
  showUndefinedIcon?: boolean;
  delayMinutes?: number | null; // ✅ Minutes de retard (positif) ou d'avance (négatif)
};

// ✅ Palette professionnelle cohérente avec le dashboard driver
const palette = {
  time: "#0A7F59",
  timeUndefined: "#F59E0B",
  client: "#15362B",
  chevron: "#91A59D",
  badgeText: "#15362B",
  badgeDefaultBg: "rgba(95,115,105,0.12)",
  badgeAssignedBg: "rgba(10,127,89,0.12)",
  badgeCompletedBg: "rgba(95,115,105,0.15)", // ✅ Gris pour terminé
  badgeCompletedText: "#5F7369", // ✅ Texte gris pour terminé
  badgeWarningBg: "rgba(251,191,36,0.12)",
  badgeDangerBg: "rgba(239,68,68,0.12)",
  badgeInfoBg: "rgba(59,130,246,0.12)",
  badgeBorder: "rgba(15,54,43,0.12)",
  routeText: "#5F7369",
  pickupIcon: "#0A7F59",
  dropoffIcon: "#0A7F59",
  chipBg: "rgba(10,127,89,0.08)",
  chipIcon: "#0A7F59",
  assignBg: "#0A7F59",
  assignIcon: "#FFFFFF",
  expandedDivider: "rgba(15,54,43,0.08)",
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
            const isCompleted = normalizedStatus === "completed" || normalizedStatus === "return_completed";
            
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
              const delayText = hasDelay
                ? `${ride.assignedTo.toUpperCase()} ${delayMinutes}min de retard`
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
                  {hasDelay ? (
                    <ScrollingText
                      text={delayText}
                      textStyle={[
                        styles.badgeLabel,
                        {
                          color: badgeText,
                        },
                        hasDelay && isLongDelay && styles.badgeLabelLongDelay,
                        hasDelay && !isLongDelay && styles.badgeLabelShortDelay,
                      ]}
                    />
                  ) : (
                    <Text
                      style={[
                        styles.badgeLabel,
                        {
                          color: badgeText,
                        },
                      ]}
                      numberOfLines={1}
                      ellipsizeMode="tail"
                    >
                      {formatBadge(ride.assignedTo)}
                    </Text>
                  )}
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
              <Ionicons
                name="location-outline"
                size={16}
                color={palette.pickupIcon}
              />
              <Text style={styles.route} numberOfLines={1} ellipsizeMode="tail">
                {ride.pickup}
              </Text>
            </View>
            <View style={styles.routeDivider} />
            <View style={styles.routeRow}>
              <Ionicons
                name="flag-outline"
                size={16}
                color={palette.dropoffIcon}
              />
              <Text style={styles.route} numberOfLines={1} ellipsizeMode="tail">
                {ride.dropoff}
              </Text>
            </View>
          </View>

          {/* ✅ Désactiver les actions si la course est terminée */}
          {(() => {
            // ✅ Normaliser le statut pour vérifier si la course est terminée
            const normalizedStatus = ride.status ? String(ride.status).toLowerCase().trim() : undefined;
            const isCompleted = normalizedStatus === "completed" || normalizedStatus === "return_completed";
            
            return (ride.onQuickAction || ride.onPrimaryAction) && !isCompleted ? (
              <View style={styles.expandedActions}>
                {ride.onQuickAction && !ride.assignedTo ? (
                  <TouchableOpacity
                    style={styles.chipButton}
                    onPress={ride.onQuickAction}
                  >
                    <Ionicons
                      name="flash-outline"
                      size={16}
                      color={palette.chipIcon}
                    />
                  </TouchableOpacity>
                ) : null}
                {ride.onPrimaryAction ? (
                  <TouchableOpacity
                    style={styles.assignButton}
                    onPress={ride.onPrimaryAction}
                  >
                    <Ionicons
                      name="person-add-outline"
                      size={18}
                      color={palette.assignIcon}
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
    padding: 16,
  },
  summaryRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 12,
  },
  timeContainer: {
    width: 54,
    alignItems: "flex-start",
  },
  time: {
    color: palette.time,
    fontWeight: "700",
    fontSize: 16,
    letterSpacing: 0.2,
  },
  client: {
    color: palette.client,
    fontSize: 16,
    fontWeight: "600",
    width: 130,
  },
  chevronContainer: {
    width: 24,
    alignItems: "center",
    marginLeft: 2,
  },
  badgeContainer: {
    flex: 1,
    alignItems: "flex-end",
  },
  badge: {
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 6,
    maxWidth: 140,
    minWidth: 80,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: palette.badgeBorder,
  },
  badgeLabel: {
    fontSize: 11,
    fontWeight: "600",
    color: palette.badgeText,
    letterSpacing: 0.3,
    textTransform: "uppercase",
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
  },
  assignButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: palette.assignBg,
  },
  routeRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    marginBottom: 4,
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
    marginTop: 14,
    gap: 16,
  },
  routeColumn: {
    flex: 1,
  },
  expandedActions: {
    flexDirection: "row",
    gap: 10,
    alignItems: "center",
    justifyContent: "flex-end",
  },
  footerActions: {
    marginTop: 12,
  },
});
