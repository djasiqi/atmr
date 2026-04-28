import type { ReactNode } from "react";
import { Pressable, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import dayjs from "dayjs";
import "dayjs/locale/fr";

import type { CompanyDispatchMission } from "../api/contracts";
import { E } from "../theme/enterpriseOpsTheme";
import { getEnterpriseStatusColors } from "../theme/enterpriseStatusColors";
import { isDispatchCompleted, isDispatchCancelled } from "../utils/companyDispatchStatus";
import { isPickupSentinel } from "../utils/pickupSentinel";
import { createShadow } from "../../../styles/shadowStyles";

dayjs.locale("fr");

const palette = {
  time: E.BRAND,
  timeUndefined: E.URGENT,
  client: E.TEXT,
  chevron: E.TEXT_MUTED,
  routeText: E.TEXT_SEC,
  pickupIcon: E.BRAND,
  dropoffIcon: E.BRAND,
} as const;

const BADGE_LIMIT = 10;

function formatBadge(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return trimmed;
  if (trimmed.toLowerCase() === "non assignée" || trimmed.toLowerCase() === "non assigné") {
    return "Non assigné";
  }
  const parts = trimmed.split(/\s+/);
  if (parts.length === 1) {
    return parts[0]!.slice(0, BADGE_LIMIT);
  }
  const first = parts[0] ?? "";
  const second = parts[1] ?? "";
  const remaining = BADGE_LIMIT - first.length - 1;
  if (remaining <= 0) return first.slice(0, BADGE_LIMIT);
  const truncatedSecond = second.slice(0, Math.max(1, remaining));
  return `${first} ${truncatedSecond}`;
}

type DispatchRideListCardProps = {
  mission: CompanyDispatchMission;
  expanded: boolean;
  onToggleExpand: () => void;
  /**
   * Remplace l’icône horloge « heure à définir » (souvent le bouton rond Urgence).
   * Si absent alors que l’heure est indéfinie, l’icône horloge est conservée.
   */
  timeSentinelAction?: ReactNode;
  /** Entre la pastille et le chevron, si présent. */
  priorityStrip?: ReactNode;
  /**
   * Si défini en mission active sans chauffeur : la pastille « Non assigné » devient un CTA
   * (fond / bordure ambre) en dehors du zone-tap « déplier », typiquement pour ouvrir l’assignation.
   */
  onUnassignedPress?: () => void;
  unassignedPressDisabled?: boolean;
  footer: ReactNode;
};

export function DispatchRideListCard({
  mission,
  expanded,
  onToggleExpand,
  timeSentinelAction,
  priorityStrip,
  onUnassignedPress,
  unassignedPressDisabled,
  footer,
}: DispatchRideListCardProps) {
  const hasSchedule = mission.scheduled_at && !isPickupSentinel(mission.scheduled_at);
  const pickupTime = hasSchedule ? dayjs(mission.scheduled_at).format("HH[h]mm") : "";
  const showTimeUndefined = !hasSchedule;
  const client = mission.client_name?.trim() || `Course #${mission.mission_id}`;

  const normStatus = mission.status ? String(mission.status).toLowerCase().trim() : undefined;
  const isCompleted = isDispatchCompleted(mission);
  const isCancelled = isDispatchCancelled(mission);

  const statusColors = getEnterpriseStatusColors(normStatus);

  const driverLabel = mission.driver_name ?? (mission.driver_id != null ? `Chauffeur #${mission.driver_id}` : null);
  const assignedTo = driverLabel;

  let delayMinutes: number | null = null;
  if (!isCompleted && !isCancelled && mission.driver_id && mission.scheduled_at && !isPickupSentinel(mission.scheduled_at)) {
    const scheduled = dayjs(mission.scheduled_at);
    if (scheduled.isValid() && scheduled.isBefore(dayjs())) {
      delayMinutes = Math.max(0, dayjs().diff(scheduled, "minute"));
    }
  }

  const unassignedCta = !isCompleted && !isCancelled && !assignedTo && onUnassignedPress != null;

  return (
    <View style={styles.card}>
      <View style={styles.summaryRow}>
        <View style={styles.timeContainer}>
          {showTimeUndefined ? (
            timeSentinelAction != null ? (
              timeSentinelAction
            ) : (
              <Ionicons name="time-outline" size={18} color={palette.timeUndefined} />
            )
          ) : (
            <Text style={styles.time}>{pickupTime}</Text>
          )}
        </View>
        {unassignedCta ? (
          <View style={styles.summaryMain}>
            <TouchableOpacity
              onPress={onToggleExpand}
              activeOpacity={0.85}
              style={styles.summaryTapSolo}
              accessibilityRole="button"
              accessibilityLabel="Afficher ou masquer le détail de la course"
            >
              <Text style={styles.client} numberOfLines={1} ellipsizeMode="tail">
                {client}
              </Text>
            </TouchableOpacity>
            <View style={styles.badgeContainer}>
              <Pressable
                onPress={onUnassignedPress}
                disabled={unassignedPressDisabled}
                style={({ pressed }) => [
                  styles.badgeUnassignedCta,
                  unassignedPressDisabled && styles.badgeCtaDisabled,
                  pressed && !unassignedPressDisabled && styles.badgeCtaPressed,
                ]}
                accessibilityRole="button"
                accessibilityLabel="Assigner un chauffeur"
                accessibilityState={{ disabled: unassignedPressDisabled === true }}
              >
                <Text style={styles.badgeUnassignedCtaLabel} numberOfLines={1} ellipsizeMode="tail">
                  Non assigné
                </Text>
              </Pressable>
            </View>
          </View>
        ) : (
          <TouchableOpacity
            onPress={onToggleExpand}
            activeOpacity={0.85}
            style={styles.summaryTap}
            accessibilityRole="button"
            accessibilityLabel="Afficher ou masquer le détail de la course"
          >
            <Text style={styles.client} numberOfLines={1} ellipsizeMode="tail">
              {client}
            </Text>
            <View style={styles.badgeContainer}>
              {isCompleted ? (
                <View
                  style={[
                    styles.badge,
                    { backgroundColor: statusColors.bg, borderColor: `${statusColors.text}40` },
                  ]}
                >
                  <Text style={[styles.badgeLabel, { color: statusColors.text }]} numberOfLines={1} ellipsizeMode="tail">
                    {assignedTo ? formatBadge(assignedTo) : "Terminée"}
                  </Text>
                </View>
              ) : assignedTo ? (
                (() => {
                  const d = delayMinutes ?? 0;
                  const hasDelay = d > 0;
                  const isLong = hasDelay && d >= 15;
                  const shortName = hasDelay
                    ? (assignedTo.split(" ")[0] ?? assignedTo).toUpperCase()
                    : assignedTo;
                  const delayText = hasDelay ? `${shortName} ${d}min` : formatBadge(assignedTo);
                  const bg = hasDelay
                    ? isLong
                      ? "#fee2e2"
                      : "#fef3c7"
                    : statusColors.bg;
                  const tx = hasDelay
                    ? isLong
                      ? "#ef4444"
                      : "#f59e0b"
                    : statusColors.text;
                  return (
                    <View
                      style={[
                        styles.badge,
                        { backgroundColor: bg, borderColor: `${tx}40` },
                        hasDelay && (isLong ? styles.badgeLongDelay : styles.badgeShortDelay),
                      ]}
                    >
                      <Text
                        style={[styles.badgeLabel, { color: tx }]}
                        numberOfLines={1}
                        ellipsizeMode="tail"
                      >
                        {delayText}
                      </Text>
                    </View>
                  );
                })()
              ) : (
                <View
                  style={[
                    styles.badge,
                    { backgroundColor: statusColors.bg, borderColor: `${statusColors.text}40` },
                  ]}
                >
                  <Text style={[styles.badgeLabel, { color: statusColors.text }]} numberOfLines={1} ellipsizeMode="tail">
                    {isCancelled ? "Annulée" : "Non assigné"}
                  </Text>
                </View>
              )}
            </View>
          </TouchableOpacity>
        )}
        {priorityStrip}
        <TouchableOpacity
          onPress={onToggleExpand}
          activeOpacity={0.85}
          style={styles.chevronContainer}
          accessibilityRole="button"
          accessibilityLabel={expanded ? "Replier" : "Déplier le détail"}
        >
          <Ionicons
            name={expanded ? "chevron-up-outline" : "chevron-down-outline"}
            size={16}
            color={palette.chevron}
          />
        </TouchableOpacity>
      </View>

      {expanded ? (
        <View>
          <View style={styles.expandedContent}>
            <View style={styles.routeColumn}>
              <View style={styles.routeRow}>
                <View style={styles.routeIcon}>
                  <Ionicons name="location-outline" size={16} color={palette.pickupIcon} />
                </View>
                <Text style={styles.route} numberOfLines={2} ellipsizeMode="tail">
                  {mission.pickup_label ?? "—"}
                </Text>
              </View>
              <View style={styles.routeDivider} />
              <View style={styles.routeRow}>
                <View style={styles.routeIcon}>
                  <Ionicons name="flag-outline" size={16} color={palette.dropoffIcon} />
                </View>
                <Text style={styles.route} numberOfLines={2} ellipsizeMode="tail">
                  {mission.dropoff_label ?? "—"}
                </Text>
              </View>
            </View>
          </View>
          {footer ? <View style={styles.footerSlot}>{footer}</View> : null}
        </View>
      ) : null}
    </View>
  );
}

const cardShadow = createShadow({
  shadowColor: "#000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

const styles = StyleSheet.create({
  card: {
    backgroundColor: E.CARD,
    borderRadius: 14,
    padding: 12,
    borderWidth: 1,
    borderColor: E.BORDER,
    ...cardShadow,
  },
  summaryRow: { flexDirection: "row", alignItems: "center", minWidth: 0 },
  /** Heure + client + pastille (repli / dépli) */
  summaryTap: { flex: 1, flexDirection: "row", alignItems: "center", minWidth: 0 },
  /** Client seul (tap) + pastille CTA « Non assigné » séparée, pour ne pas mélanger tap déplier / assigner */
  summaryMain: { flex: 1, flexDirection: "row", alignItems: "center", minWidth: 0 },
  summaryTapSolo: { flex: 1, minWidth: 0, marginRight: 0, flexDirection: "row", alignItems: "center" },
  timeContainer: { width: 50, minHeight: 32, marginRight: 6, alignItems: "center", justifyContent: "center" },
  time: { color: palette.time, fontWeight: "700", fontSize: 15, letterSpacing: 0.2 },
  client: { color: palette.client, fontSize: 14, fontWeight: "600", width: 110, marginRight: 6, flexShrink: 0 },
  chevronContainer: {
    width: 28,
    alignItems: "center",
    justifyContent: "center",
    marginLeft: 0,
  },
  badgeContainer: { flex: 1, minWidth: 0, alignItems: "flex-end" },
  badge: {
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 4,
    maxWidth: 130,
    minWidth: 60,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: E.BORDER,
  },
  badgeLabel: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "uppercase" as const,
    textAlign: "center" as const,
  },
  /** Pastille cliquable « Non assigné » (réf. ambre) */
  badgeUnassignedCta: {
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 5,
    maxWidth: 130,
    minWidth: 60,
    borderWidth: 1,
    borderColor: "rgba(245, 158, 11, 0.25)",
    backgroundColor: "#fef3c7",
    overflow: "hidden",
  },
  badgeCtaPressed: { opacity: 0.86 },
  badgeCtaDisabled: { opacity: 0.55 },
  badgeUnassignedCtaLabel: {
    fontSize: 10,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "uppercase" as const,
    textAlign: "center" as const,
    color: "#f59e0b",
  },
  badgeShortDelay: {},
  badgeLongDelay: {},
  routeRow: { flexDirection: "row", alignItems: "center", marginBottom: 4 },
  routeIcon: { marginRight: 8 },
  routeDivider: {
    height: 1,
    backgroundColor: "rgba(0,121,107,0.06)",
    marginVertical: 6,
    marginLeft: 24,
  },
  route: { color: palette.routeText, fontSize: 13, flex: 1, flexShrink: 1 },
  expandedContent: { marginTop: 10 },
  routeColumn: { width: "100%" },
  footerSlot: { marginTop: 10 },
});
