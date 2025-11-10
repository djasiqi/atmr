import React, { useState } from "react";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  GestureResponderEvent,
  ViewStyle,
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
  badges?: Badge[];
  onPress?: (event: GestureResponderEvent) => void;
  onPrimaryAction?: (event: GestureResponderEvent) => void;
  onQuickAction?: (event: GestureResponderEvent) => void;
  primaryIcon?: string;
  footerActions?: React.ReactNode;
  showUndefinedIcon?: boolean;
};

const palette = {
  time: "#79E0AE",
  timeUndefined: "#F6C158",
  client: "#F4FFFA",
  chevron: "rgba(184,214,198,0.55)",
  badgeText: "#052015",
  badgeDefaultBg: "rgba(30,185,128,0.18)",
  badgeAssignedBg: "#4ADE80",
  badgeWarningBg: "rgba(246,193,88,0.28)",
  badgeDangerBg: "rgba(241,104,104,0.28)",
  badgeInfoBg: "rgba(126,246,168,0.24)",
  badgeBorder: "rgba(78,214,160,0.4)",
  routeText: "rgba(214,236,224,0.86)",
  pickupIcon: "#79E0AE",
  dropoffIcon: "#9CF2C9",
  chipBg: "rgba(30,185,128,0.14)",
  chipIcon: "#79E0AE",
  assignBg: "#1EB980",
  assignIcon: "#052015",
  expandedDivider: "rgba(46,128,94,0.3)",
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

export const RideSnippetCard: React.FC<{
  ride: RideSnippet;
  style?: ViewStyle;
}> = ({ ride, style }) => {
  const [expanded, setExpanded] = useState(false);

  const toggleExpanded = () => setExpanded((prev) => !prev);

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

        <View style={styles.chevronContainer}>
          <Ionicons name="chevron-down" size={16} color={palette.chevron} />
        </View>

        <View style={styles.badgeContainer}>
          {ride.assignedTo ? (
            <View style={[styles.badge, styles.assignedBadge]}>
              <Text
                style={styles.badgeLabel}
                numberOfLines={1}
                ellipsizeMode="tail"
              >
                {formatBadge(ride.assignedTo)}
              </Text>
            </View>
          ) : (
            <View style={[styles.badge, styles.badgeDefault]}>
              <Text
                style={styles.badgeLabel}
                numberOfLines={1}
                ellipsizeMode="tail"
              >
                Non Assigné
              </Text>
            </View>
          )}
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

          {(ride.onQuickAction || ride.onPrimaryAction) && (
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
          )}
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
  },
  badgeContainer: {
    flex: 1,
    alignItems: "flex-end",
  },
  badge: {
    borderRadius: 12,
    paddingHorizontal: 10,
    paddingVertical: 6,
    maxWidth: 120,
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

export default RideSnippetCard;
