import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import type { LayoutChangeEvent } from "react-native";
import { AccessibilityInfo, Pressable, StyleSheet, TouchableOpacity, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import Animated, {
  cancelAnimation,
  Easing,
  useAnimatedStyle,
  useSharedValue,
  withRepeat,
  withTiming,
} from "react-native-reanimated";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";

import { AppText } from "../../../design/ui/AppText";
import type { CompanyDispatchMission } from "../api/contracts";
import { E } from "../theme/enterpriseOpsTheme";
import { getEnterpriseStatusColors } from "../theme/enterpriseStatusColors";
import {
  formatDispatchScheduledTime,
  pickupArrivalHintFr,
  uiForDispatchDelayMinutes,
} from "../utils/dispatchWebAlignment";
import { isDispatchCompleted, isDispatchCancelled } from "../utils/companyDispatchStatus";
import { isTimeUndefined } from "../utils/pickupSentinel";
import { buildIdentityFromMission } from "../utils/bookingIdentity";
import { createShadow } from "../../../styles/shadowStyles";

/** Même coque que `operations-app` `EnterpriseCard` + `RideSnippetCard`. */
const cardSurfaceShadow = createShadow({
  shadowColor: "#000000",
  shadowOffset: { width: 0, height: 2 },
  shadowOpacity: 0.04,
  shadowRadius: 8,
  elevation: 2,
});

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

const BADGE_MARQUEE_W = 80;
const BADGE_MARQUEE_H = 28;
/** Espace entre les deux copies — boucle linéaire sans à-coup. */
const MARQUEE_SEGMENT_GAP = 14;
/** Vitesse constante (px/s) pour un défilement lisible et régulier. */
const MARQUEE_PX_PER_SEC = 36;

type MarqueeRowContentProps = {
  isCritical: boolean;
  accent: string;
  delayMinutes: number;
  driverLabel: string;
  onSegmentLayout?: (e: LayoutChangeEvent) => void;
};

function MarqueeBadgeRowInner({
  isCritical,
  accent,
  delayMinutes,
  driverLabel,
  onSegmentLayout,
}: MarqueeRowContentProps) {
  return (
    <View
      style={[styles.badgeScrollInnerRow, styles.badgeMarqueeSegmentPad, styles.badgeMarqueeSegmentShrink]}
      onLayout={onSegmentLayout}
    >
      {isCritical ? (
        <Ionicons name="warning-outline" size={10} color={accent} accessibilityElementsHidden />
      ) : null}
      <AppText variant="caption" style={[styles.badgeDelayMinutes, { color: accent }]} numberOfLines={1}>
        {`+${delayMinutes}min`}
      </AppText>
      <AppText variant="caption" style={[styles.badgeLabelInline, { color: accent }]} numberOfLines={1}>
        {driverLabel}
      </AppText>
    </View>
  );
}

type DispatchDelayMarqueeBadgeProps = {
  isCritical: boolean;
  accent: string;
  delayMinutes: number;
  driverLabel: string;
  backgroundColor: string;
  borderColor: string;
  accessibilityLabel: string;
  isLongDelayStyle: boolean;
};

/** Pastille ~80×28 : défilement linéaire en boucle (double segment) si débordement. */
function DispatchDelayMarqueeBadge({
  isCritical,
  accent,
  delayMinutes,
  driverLabel,
  backgroundColor,
  borderColor,
  accessibilityLabel,
  isLongDelayStyle,
}: DispatchDelayMarqueeBadgeProps) {
  const [viewportW, setViewportW] = useState(0);
  const [segmentW, setSegmentW] = useState(0);
  const [reduceMotion, setReduceMotion] = useState(false);
  const translateX = useSharedValue(0);

  useEffect(() => {
    let alive = true;
    void AccessibilityInfo.isReduceMotionEnabled().then((v) => {
      if (alive) setReduceMotion(v);
    });
    const sub = AccessibilityInfo.addEventListener("reduceMotionChanged", setReduceMotion);
    return () => {
      alive = false;
      sub.remove();
    };
  }, []);

  const overflowPx = segmentW > 0 && viewportW > 0 ? segmentW - viewportW : 0;
  const needsMarquee =
    !reduceMotion && viewportW > 0 && segmentW > 0 && overflowPx > 0.5;
  const loopDistance = needsMarquee ? segmentW + MARQUEE_SEGMENT_GAP : 0;

  useEffect(() => {
    cancelAnimation(translateX);
    if (loopDistance <= 0) {
      translateX.value = 0;
      return;
    }
    translateX.value = 0;
    const durationMs = Math.min(20_000, Math.max(3_200, (loopDistance / MARQUEE_PX_PER_SEC) * 1000));
    translateX.value = withRepeat(
      withTiming(-loopDistance, { duration: durationMs, easing: Easing.linear }),
      -1,
      false,
    );
    return () => {
      cancelAnimation(translateX);
    };
  }, [loopDistance]);

  const rowAnimatedStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: translateX.value }],
  }));

  const rowProps = { isCritical, accent, delayMinutes, driverLabel };

  return (
    <View
      style={[
        styles.badgeShellMarqueeOuter,
        { backgroundColor, borderColor },
        isLongDelayStyle ? styles.badgeLongDelay : styles.badgeShortDelay,
      ]}
      accessibilityLabel={accessibilityLabel}
    >
      <View
        style={styles.badgeMarqueeClip}
        onLayout={(e: LayoutChangeEvent) => setViewportW(e.nativeEvent.layout.width)}
      >
        {needsMarquee ? (
          <Animated.View
            collapsable={false}
            style={[styles.badgeMarqueeTrack, rowAnimatedStyle]}
            accessibilityElementsHidden
            importantForAccessibility="no-hide-descendants"
          >
            <MarqueeBadgeRowInner
              {...rowProps}
              onSegmentLayout={(e: LayoutChangeEvent) => setSegmentW(e.nativeEvent.layout.width)}
            />
            <View style={styles.badgeMarqueeBetweenGap} />
            <MarqueeBadgeRowInner {...rowProps} />
          </Animated.View>
        ) : (
          <MarqueeBadgeRowInner
            {...rowProps}
            onSegmentLayout={(e: LayoutChangeEvent) => setSegmentW(e.nativeEvent.layout.width)}
          />
        )}
      </View>
    </View>
  );
}

type DispatchRideListCardProps = {
  mission: CompanyDispatchMission;
  /**
   * Minutes de retard alignées tableau web (tout retard **> 0**, paliers 1–5 min = léger).
   * `undefined` tant que les retards ne sont pas chargés (pas de badge).
   */
  bookingDelayPickupMinutes?: number | null;
  /** ISO ETA pickup (après fusion live + snapshot delays). */
  bookingPickupEtaIso?: string | null;
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
  bookingDelayPickupMinutes,
  bookingPickupEtaIso,
  expanded,
  onToggleExpand,
  timeSentinelAction,
  priorityStrip,
  onUnassignedPress,
  unassignedPressDisabled,
  footer,
}: DispatchRideListCardProps) {
  const hasSchedule = mission.scheduled_at && !isTimeUndefined(mission);
  const pickupTime = hasSchedule ? formatDispatchScheduledTime(mission.scheduled_at) : "";
  const showTimeUndefined = !hasSchedule;
  const missionIdentity = buildIdentityFromMission(mission);
  const client = missionIdentity.passengerLabel?.trim() || `Course #${mission.mission_id}`;
  const sourceLine = missionIdentity.source?.name?.trim() || null;

  const normStatus = mission.status ? String(mission.status).toLowerCase().trim() : undefined;
  const isCompleted = isDispatchCompleted(mission);
  const isCancelled = isDispatchCancelled(mission);

  const statusColors = getEnterpriseStatusColors(normStatus);

  const driverLabel = mission.driver_name ?? (mission.driver_id != null ? `Chauffeur #${mission.driver_id}` : null);

  const pickupEtaStatuses = new Set(["accepted", "assigned", "en_route"]);
  const etaHintIso =
    hasSchedule &&
    bookingPickupEtaIso?.trim() &&
    normStatus &&
    pickupEtaStatuses.has(normStatus)
      ? bookingPickupEtaIso.trim()
      : null;
  const pickupEtaUi = etaHintIso ? pickupArrivalHintFr(etaHintIso) : null;
  const assignedTo = driverLabel;
  /** `undefined` : retards pas encore chargés. `null` ou nombre : aligné carte dispatch web (minutes > 0). */
  const webPickupDelayMin =
    bookingDelayPickupMinutes === undefined
      ? undefined
      : typeof bookingDelayPickupMinutes === "number" && bookingDelayPickupMinutes > 0
        ? Math.round(bookingDelayPickupMinutes)
        : null;

  const delayUi =
    !isCancelled &&
    webPickupDelayMin != null &&
    webPickupDelayMin > 0
      ? uiForDispatchDelayMinutes(webPickupDelayMin)
      : null;

  const unassignedCta = !isCompleted && !isCancelled && !assignedTo && onUnassignedPress != null;

  return (
    <View
      style={[
        styles.card,
        delayUi ? { borderLeftWidth: 3, borderLeftColor: delayUi.stripeColor } : null,
      ]}
    >
      <View style={styles.summaryRow}>
        <View style={styles.timeContainer}>
          {showTimeUndefined ? (
            timeSentinelAction != null ? (
              timeSentinelAction
            ) : (
              <Ionicons name="time-outline" size={18} color={palette.timeUndefined} />
            )
          ) : (
            <>
              <AppText variant="caption" style={[styles.time, delayUi ? { color: delayUi.timeColor } : null]}>
                {pickupTime}
              </AppText>
              {pickupEtaUi ? (
                <AppText
                  variant="caption"
                  style={styles.etaHint}
                  numberOfLines={1}
                  accessibilityLabel={pickupEtaUi.accessibility}
                >
                  {pickupEtaUi.text}
                </AppText>
              ) : null}
            </>
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
              <AppText variant="body" style={styles.client} numberOfLines={1} ellipsizeMode="tail">
                {client}
              </AppText>
              {sourceLine ? (
                <AppText variant="caption" style={styles.sourceLine} numberOfLines={1} ellipsizeMode="tail">
                  {sourceLine}
                </AppText>
              ) : null}
            </TouchableOpacity>
            <View style={styles.badgeContainer}>
              <Pressable
                onPress={onUnassignedPress}
                disabled={unassignedPressDisabled}
                style={({ pressed }) => [
                  styles.badgeShell,
                  styles.badgeUnassignedCta,
                  unassignedPressDisabled && styles.badgeCtaDisabled,
                  pressed && !unassignedPressDisabled && styles.badgeCtaPressed,
                ]}
                accessibilityRole="button"
                accessibilityLabel="Assigner un chauffeur"
                accessibilityState={{ disabled: unassignedPressDisabled === true }}
              >
                <AppText
                  variant="caption"
                  style={styles.badgeUnassignedCtaLabel}
                  numberOfLines={1}
                  ellipsizeMode="tail"
                >
                  Non assigné
                </AppText>
              </Pressable>
            </View>
          </View>
        ) : (
          <View style={styles.summaryMain}>
            <TouchableOpacity
              onPress={onToggleExpand}
              activeOpacity={0.85}
              style={styles.summaryTapSolo}
              accessibilityRole="button"
              accessibilityLabel="Afficher ou masquer le détail de la course"
            >
              <AppText variant="body" style={styles.client} numberOfLines={1} ellipsizeMode="tail">
                {client}
              </AppText>
              {sourceLine ? (
                <AppText variant="caption" style={styles.sourceLine} numberOfLines={1} ellipsizeMode="tail">
                  {sourceLine}
                </AppText>
              ) : null}
            </TouchableOpacity>
            <View style={styles.badgeContainer}>
              {isCompleted ? (
                <View
                  style={[
                    styles.badgeShell,
                    { backgroundColor: statusColors.bg, borderColor: `${statusColors.text}40` },
                  ]}
                >
                  <AppText
                    variant="caption"
                    style={[styles.badgeLabel, { color: statusColors.text }]}
                    numberOfLines={1}
                    ellipsizeMode="tail"
                  >
                    {assignedTo ? formatBadge(assignedTo) : "Terminée"}
                  </AppText>
                </View>
              ) : assignedTo ? (
                (() => {
                  const hasMinutes =
                    webPickupDelayMin !== undefined && webPickupDelayMin !== null;
                  const d = hasMinutes ? webPickupDelayMin : 0;
                  const delayBadgeUi =
                    hasMinutes && d > 0 ? uiForDispatchDelayMinutes(d) : null;
                  const showDelay = delayBadgeUi != null;
                  const isCritical = delayBadgeUi?.severity === "critical";
                  const bg = showDelay ? delayBadgeUi.badgeBg : statusColors.bg;
                  const badgeBorderColor = showDelay ? delayBadgeUi.badgeBorder : `${statusColors.text}40`;
                  const accent = showDelay ? delayBadgeUi.timeColor : statusColors.text;
                  const driverLabelShort = formatBadge(assignedTo);
                  const driverMarqueeText =
                    (assignedTo ?? "").trim().length > 0 ? (assignedTo ?? "").trim() : driverLabelShort;
                  const a11yDelayBadge = showDelay
                    ? `${isCritical ? "Retard critique. " : ""}+${d} minutes. ${driverMarqueeText}`
                    : driverLabelShort;
                  return showDelay ? (
                    <DispatchDelayMarqueeBadge
                      isCritical={isCritical}
                      accent={accent}
                      delayMinutes={d}
                      driverLabel={driverMarqueeText}
                      backgroundColor={bg}
                      borderColor={badgeBorderColor}
                      accessibilityLabel={a11yDelayBadge}
                      isLongDelayStyle={isCritical}
                    />
                  ) : (
                    <View
                      style={[
                        styles.badgeShell,
                        { backgroundColor: bg, borderColor: badgeBorderColor },
                      ]}
                    >
                      <AppText
                        variant="caption"
                        style={[styles.badgeLabel, { color: accent }]}
                        numberOfLines={1}
                        ellipsizeMode="tail"
                      >
                        {driverLabelShort}
                      </AppText>
                    </View>
                  );
                })()
              ) : (
                <View
                  style={[
                    styles.badgeShell,
                    { backgroundColor: statusColors.bg, borderColor: `${statusColors.text}40` },
                  ]}
                >
                  <AppText
                    variant="caption"
                    style={[styles.badgeLabel, { color: statusColors.text }]}
                    numberOfLines={1}
                    ellipsizeMode="tail"
                  >
                    {isCancelled ? "Annulée" : "Non assigné"}
                  </AppText>
                </View>
              )}
            </View>
          </View>
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
                <AppText variant="caption" style={styles.route} numberOfLines={2} ellipsizeMode="tail">
                  {mission.pickup_label ?? "—"}
                </AppText>
              </View>
              <View style={styles.routeDivider} />
              <View style={styles.routeRow}>
                <View style={styles.routeIcon}>
                  <Ionicons name="flag-outline" size={16} color={palette.dropoffIcon} />
                </View>
                <AppText variant="caption" style={styles.route} numberOfLines={2} ellipsizeMode="tail">
                  {mission.dropoff_label ?? "—"}
                </AppText>
              </View>
            </View>
          </View>
          {footer ? <View style={styles.footerSlot}>{footer}</View> : null}
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: E.CARD,
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: E.BORDER,
    ...cardSurfaceShadow,
  },
  summaryRow: { flexDirection: "row", alignItems: "center", minWidth: 0 },
  /** Passager seul (tap) + pastille séparée — évite ScrollView / CTA imbriqués dans le tap déplier */
  summaryMain: { flex: 1, flexDirection: "row", alignItems: "center", minWidth: 0 },
  summaryTapSolo: { flex: 1, minWidth: 0, marginRight: 0, flexDirection: "column", justifyContent: "center" },
  timeContainer: {
    width: 54,
    minHeight: 32,
    marginRight: 10,
    alignItems: "flex-start",
    justifyContent: "center",
    gap: 2,
  },
  time: { color: palette.time, fontWeight: "700", fontSize: FONT_SIZE.px15, letterSpacing: 0.2 },
  etaHint: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "600",
    color: E.TEXT_MUTED,
    maxWidth: 54,
  },
  client: { color: palette.client, fontWeight: "600", fontSize: FONT_SIZE.px14, flexShrink: 1 },
  sourceLine: { color: palette.routeText, fontSize: FONT_SIZE.px11, marginTop: 1, flexShrink: 1 },
  chevronContainer: {
    width: 24,
    alignItems: "center",
    justifyContent: "center",
    marginLeft: 2,
    marginRight: 0,
  },
  badgeContainer: { flex: 1, minWidth: 0, alignItems: "flex-end" },
  /** Toutes les pastilles résumé : même gabarit web (≈ padding inline ~6px, 10px / lh 16, 80×28). */
  badgeShell: {
    width: BADGE_MARQUEE_W,
    height: BADGE_MARQUEE_H,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: E.BORDER,
    overflow: "hidden",
    justifyContent: "center",
    alignItems: "stretch",
    paddingHorizontal: 6,
  },
  badgeDelayMinutes: {
    fontSize: FONT_SIZE.px10,
    lineHeight: 16,
    fontWeight: "700" as const,
    letterSpacing: 0.2,
    textTransform: "none" as const,
  },
  badgeLabel: {
    width: "100%",
    fontSize: FONT_SIZE.px10,
    lineHeight: 16,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "uppercase" as const,
    textAlign: "center" as const,
  },
  /** Couleurs CTA ambre uniquement (`badgeShell` fournit la taille). */
  badgeUnassignedCta: {
    backgroundColor: "#fef3c7",
    borderColor: "rgba(245, 158, 11, 0.25)",
  },
  badgeCtaPressed: { opacity: 0.86 },
  badgeCtaDisabled: { opacity: 0.55 },
  badgeUnassignedCtaLabel: {
    width: "100%",
    fontSize: FONT_SIZE.px10,
    lineHeight: 16,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "uppercase" as const,
    textAlign: "center" as const,
    color: "#f59e0b",
  },
  badgeShortDelay: {},
  badgeLongDelay: {},
  /** Même cible 80×28 que `badgeShell` ; pas de padding sur la coque (géré dans la ligne marquee). */
  badgeShellMarqueeOuter: {
    width: BADGE_MARQUEE_W,
    height: BADGE_MARQUEE_H,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: E.BORDER,
    overflow: "hidden",
    justifyContent: "center",
  },
  /** `alignItems: flex-start` : sinon la ligne est étirée à la largeur du clip et segmentW ≈ viewportW → pas de marquee. */
  badgeMarqueeClip: {
    flex: 1,
    width: "100%",
    overflow: "hidden",
    justifyContent: "center",
    alignItems: "flex-start",
  },
  badgeMarqueeTrack: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    flexWrap: "nowrap" as const,
    alignSelf: "flex-start",
  },
  badgeMarqueeBetweenGap: {
    width: MARQUEE_SEGMENT_GAP,
    flexShrink: 0,
  },
  badgeMarqueeSegmentPad: {
    paddingHorizontal: 6,
  },
  badgeMarqueeSegmentShrink: {
    alignSelf: "flex-start",
    flexShrink: 0,
  },
  badgeScrollInnerRow: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    flexWrap: "nowrap" as const,
    gap: 4,
  },
  badgeLabelInline: {
    fontSize: FONT_SIZE.px10,
    fontWeight: "700",
    letterSpacing: 0.3,
    textTransform: "uppercase" as const,
    lineHeight: 16,
    textAlign: "center" as const,
    flexShrink: 0,
  },
  routeRow: { flexDirection: "row", alignItems: "center", marginBottom: 4 },
  routeIcon: { marginRight: 8 },
  routeDivider: {
    height: 1,
    backgroundColor: "rgba(0,121,107,0.06)",
    marginVertical: 6,
    marginLeft: 24,
  },
  route: { color: palette.routeText, fontSize: FONT_SIZE.px13, lineHeight: 18, flex: 1, flexShrink: 1 },
  expandedContent: { marginTop: 10 },
  routeColumn: { width: "100%" },
  /** Aligné `RideSnippetCard` `footerActions`. */
  footerSlot: { marginTop: 10 },
});
