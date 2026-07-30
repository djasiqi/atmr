import { Pressable, StyleSheet, View } from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { AppText } from "../../../design/ui/AppText";
import { useAccessibilityScale } from "../../../design/responsive";
import { FONT_SIZE } from "../../../design/responsive/typographyTokens";
import { createShadow } from "../../../styles/shadowStyles";
import { resolveDriverStatusForUx, getDriverStatusUx } from "../statusDictionary";
import type { DriverMission, DriverMissionStatus, DriverTransitionStatus } from "../types";
import {
  getBadgeStatusLabel,
  getClientBirthDateDisplay,
  getMissionClientDisplayName,
  getScheduledWhenDisplay,
} from "../domain/missionDisplay";
import { formatMissionPickupTime } from "../domain/missionMetrics";
import {
  getDropoffHints,
  getPickupHints,
  inferTripDirection,
  type HintItem,
  type MissionHintLike,
} from "../domain/missionHints";
import { MissionProgressStepper } from "./MissionProgressStepper";
import { RouteVerticalDashedLine } from "./RouteVerticalDashedLine";
import { useMissionRouteMetrics } from "../hooks/useMissionRouteMetrics";
import { useDynamicEtaQuery } from "../hooks";
import { useMissionLocationStale } from "../hooks/useMissionLocationStale";
import { useSession } from "../../../core/sessionProvider";
import {
  getCallablePhoneFromMission,
  openNavigation,
  safeCall,
} from "../utils/missionContact";
import { D, dashboardSoftShadow, missionActiveCardShadow } from "../theme/driverDashboardTheme";

type FocusSide = "pickup" | "dropoff" | null;

function resolveFocusSide(statusKey: DriverMissionStatus, terminal: boolean): FocusSide {
  if (terminal) return null;
  if (statusKey === "IN_PROGRESS") return "dropoff";
  return "pickup";
}

function focusHeaderLabel(focus: FocusSide, statusKey: DriverMissionStatus): string {
  if (focus === "dropoff") return "À L'ARRIVÉE";
  if (statusKey === "ARRIVED") return "À LA PRISE EN CHARGE";
  return "POUR LA PRISE EN CHARGE";
}

type Props = {
  mission: DriverMission;
  pending?: boolean;
  onMissionTransition?: (target: DriverTransitionStatus) => void;
  onMissionRelease?: () => void;
  onOpenDetails?: () => void;
  onOpenChat?: () => void;
};

const cardShadow = createShadow(missionActiveCardShadow);
const softSurfaceShadow = createShadow(missionActiveCardShadow);
const actionShadow = createShadow(dashboardSoftShadow);

const FORWARD_TRANSITION_PRIORITY: DriverTransitionStatus[] = [
  "EN_ROUTE",
  "ARRIVED",
  "IN_PROGRESS",
  "COMPLETED",
];

function conciseAddressLine(s: string | null | undefined): string {
  return s?.trim() || "—";
}

function navigationDestination(mission: DriverMission, statusKey: DriverMissionStatus): string {
  const pickup = String(mission.pickup_location ?? "").trim();
  const dropoff = String(mission.dropoff_location ?? "").trim();
  if (statusKey === "IN_PROGRESS") return dropoff || pickup;
  return pickup || dropoff;
}

function transitionLabel(target: DriverTransitionStatus): string {
  switch (target) {
    case "EN_ROUTE":
      return "EN ROUTE";
    case "ARRIVED":
      return "ARRIVÉ";
    case "IN_PROGRESS":
      return "À BORD";
    case "COMPLETED":
      return "TERMINER";
    default:
      return target;
  }
}

type MetricTileProps = {
  label: string;
  value: string;
  icon: keyof typeof Ionicons.glyphMap;
  showDivider?: boolean;
  stacked?: boolean;
  /** Mode 3 colonnes : labels plus courts / centrés, moins de wrap agressif. */
  compactLabel?: boolean;
};

function MetricTile({ label, value, icon, showDivider, stacked, compactLabel }: MetricTileProps) {
  return (
    <>
      {showDivider ? (
        <View
          style={[styles.metricDivider, stacked && styles.metricDividerStacked]}
          accessibilityElementsHidden
        />
      ) : null}
      <View style={[styles.metricTile, stacked && styles.metricTileStacked]}>
        <View style={[styles.metricHead, compactLabel && styles.metricHeadCompact]}>
          <Ionicons name={icon} size={11} color={D.brand} accessibilityElementsHidden />
          <AppText
            variant="caption"
            scaleRole="chrome"
            style={[styles.metricLabel, compactLabel && styles.metricLabelCompact]}
            numberOfLines={2}
          >
            {label}
          </AppText>
        </View>
        <AppText
          variant="label"
          scaleRole="chrome"
          style={[styles.metricValue, stacked && styles.metricValueStacked]}
          numberOfLines={1}
        >
          {value}
        </AppText>
      </View>
    </>
  );
}

type QuickActionProps = {
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
  onPress?: () => void;
  disabled?: boolean;
  stacked?: boolean;
};

function QuickAction({ label, icon, onPress, disabled, stacked }: QuickActionProps) {
  if (!onPress) return null;
  return (
    <Pressable
      onPress={onPress}
      disabled={disabled}
      accessibilityRole="button"
      accessibilityLabel={label}
      style={({ pressed }) => [
        styles.quickAction,
        stacked && styles.quickActionStacked,
        disabled && styles.quickActionDisabled,
        pressed && styles.pressed,
      ]}
    >
      <Ionicons name={icon} size={13} color={D.brand} />
      <AppText
        variant="caption"
        scaleRole="chrome"
        style={styles.quickActionLabel}
        numberOfLines={1}
      >
        {label}
      </AppText>
    </Pressable>
  );
}

export function DashboardActiveMission({
  mission,
  pending = false,
  onMissionTransition,
  onMissionRelease,
}: Props) {
  const { can } = useSession();
  const { isVeryLargeText, fontScale } = useAccessibilityScale();
  /** KPI + actions rapides sur une ligne ; empilement seulement si police vraiment extrême. */
  const stackMetrics = fontScale >= 1.75;
  const statusUx = getDriverStatusUx(mission.status);
  const statusKey = resolveDriverStatusForUx(mission.status);
  const locationStale = useMissionLocationStale(mission.id, statusKey);
  const etaQuery = useDynamicEtaQuery(mission.id, { missionStatus: statusKey });
  const routeMetrics = useMissionRouteMetrics(mission, {
    etaMinutes: etaQuery.data?.eta_minutes,
    etaSnapshot: etaQuery.data ?? null,
  });
  const pickup = (mission.pickup_location as string | null | undefined) ?? null;
  const dropoff = (mission.dropoff_location as string | null | undefined) ?? null;
  const phone = getCallablePhoneFromMission(mission);
  const dest = navigationDestination(mission, statusKey);

  const canMutate = can("mission:update_status") && typeof onMissionTransition === "function";
  const showTransitionActions = canMutate && !statusUx.terminal;
  const forwardTransition = FORWARD_TRANSITION_PRIORITY.find((t) =>
    statusUx.nextTransitions.includes(t)
  );
  const canRelease = Boolean(
    can("mission:update_status") &&
      typeof onMissionRelease === "function" &&
      (statusKey === "ASSIGNED" || statusKey === "EN_ROUTE")
  );
  const showGps = !statusUx.terminal && dest.length > 0;
  const showCall = Boolean(phone) && !statusUx.terminal;

  const showNavigation = showGps && dest.length > 0;
  const showCallAction = showCall && Boolean(phone);
  const showQuickActions = showNavigation || showCallAction || canRelease;

  const clientTitle = getMissionClientDisplayName(mission);
  const birthDateDisplay = getClientBirthDateDisplay(mission);
  const scheduledWhen = getScheduledWhenDisplay(mission);
  const badgeLabel = getBadgeStatusLabel(statusKey).toUpperCase();

  const focusSide = resolveFocusSide(statusKey, statusUx.terminal);
  const tripDirection = inferTripDirection(mission as unknown as MissionHintLike);
  const directionLabel =
    tripDirection === "outbound" ? "ALLER" : tripDirection === "return" ? "RETOUR" : null;
  const hints: HintItem[] = !statusUx.terminal
    ? statusKey === "IN_PROGRESS"
      ? getDropoffHints(mission as unknown as MissionHintLike)
      : getPickupHints(mission as unknown as MissionHintLike)
    : [];
  const focusHeader = focusHeaderLabel(focusSide, statusKey);
  const pickupActive = focusSide === "pickup";
  const dropoffActive = focusSide === "dropoff";

  return (
    <View style={styles.root} accessibilityLabel={`Mission active ${mission.id}`}>
      <View style={styles.infoCard}>
        <View style={styles.cardHeader}>
          <View style={styles.cardHeaderLeft}>
            <Ionicons name="bookmark-outline" size={15} color={D.brand} accessibilityElementsHidden />
            <AppText variant="caption" style={styles.cardEyebrow}>
              MISSION ACTIVE
            </AppText>
          </View>
          <View style={styles.headerBadges}>
            {locationStale ? (
              <View style={styles.staleBadge}>
                <AppText variant="caption" style={styles.staleBadgeText} numberOfLines={1}>
                  NON LOCALISÉ
                </AppText>
              </View>
            ) : null}
            <View style={styles.statusBadge}>
              <AppText variant="caption" style={styles.statusBadgeText} numberOfLines={1}>
                {badgeLabel}
              </AppText>
            </View>
          </View>
        </View>

        <View style={styles.clientBlock}>
          <AppText variant="screenTitle" style={styles.clientName} numberOfLines={2}>
            {clientTitle}
          </AppText>
          {birthDateDisplay ? (
            <AppText variant="caption" style={styles.clientBirthDate}>
              {birthDateDisplay}
            </AppText>
          ) : null}
        </View>

        {scheduledWhen ? (
          <View style={styles.whenRow}>
            <Ionicons name="time-outline" size={15} color={D.textMuted} accessibilityElementsHidden />
            <AppText variant="caption" style={styles.whenText} numberOfLines={2}>
              <AppText variant="caption" style={styles.whenPrefix}>
                Heure prévue{" "}
              </AppText>
              {scheduledWhen}
            </AppText>
          </View>
        ) : null}

        <View style={styles.routeTimeline}>
          <View style={[styles.routeRow, pickupActive && styles.routeRowActive]}>
            <View style={styles.routeMarkerCol}>
              <View style={[styles.routeDot, pickupActive && styles.routeDotActive]} />
              <RouteVerticalDashedLine height={34} />
            </View>
            <View style={styles.routeTextCol}>
              <View style={styles.routeKeyRow}>
                <AppText
                  variant="caption"
                  style={[styles.routeKey, pickupActive && styles.routeKeyActive]}
                >
                  DÉPART
                </AppText>
                {pickupActive ? (
                  <AppText variant="caption" style={styles.routeFocusTag} numberOfLines={1}>
                    ÉTAPE EN COURS
                  </AppText>
                ) : null}
              </View>
              <AppText
                variant="body"
                style={[styles.routeAddress, pickupActive && styles.routeAddressActive]}
                numberOfLines={isVeryLargeText ? undefined : 3}
              >
                {conciseAddressLine(pickup)}
              </AppText>
            </View>
          </View>
          <View style={[styles.routeRow, dropoffActive && styles.routeRowActive]}>
            <View style={styles.routeMarkerCol}>
              <View
                style={[styles.routeFlagWrap, dropoffActive && styles.routeFlagWrapActive]}
              >
                <Ionicons
                  name="flag"
                  size={13}
                  color={dropoffActive ? D.brand : D.flag}
                />
              </View>
            </View>
            <View style={[styles.routeTextCol, styles.routeTextColLast]}>
              <View style={styles.routeKeyRow}>
                <AppText
                  variant="caption"
                  style={[styles.routeKey, dropoffActive && styles.routeKeyActive]}
                >
                  ARRIVÉE
                </AppText>
                {dropoffActive ? (
                  <AppText variant="caption" style={styles.routeFocusTag} numberOfLines={1}>
                    ÉTAPE EN COURS
                  </AppText>
                ) : null}
              </View>
              <AppText
                variant="body"
                style={[styles.routeAddress, dropoffActive && styles.routeAddressActive]}
                numberOfLines={isVeryLargeText ? undefined : 3}
              >
                {conciseAddressLine(dropoff)}
              </AppText>
            </View>
          </View>
        </View>

        {focusSide && hints.length > 0 ? (
          <View
            style={styles.hintsBlock}
            accessibilityLabel={`Informations ${focusHeader.toLowerCase()}`}
          >
            <View style={styles.hintsHeaderRow}>
              <Ionicons
                name={focusSide === "dropoff" ? "flag" : "navigate-outline"}
                size={12}
                color={D.brand}
                accessibilityElementsHidden
              />
              <AppText variant="caption" style={styles.hintsHeader} numberOfLines={1}>
                {focusHeader}
              </AppText>
              {directionLabel ? (
                <View style={styles.directionPill} accessibilityElementsHidden>
                  <AppText variant="caption" style={styles.directionPillText} numberOfLines={1}>
                    {directionLabel}
                  </AppText>
                </View>
              ) : null}
            </View>
            <View style={styles.hintsList}>
              {hints.map((hint, idx) => (
                <View key={`${hint.label}-${idx}`} style={styles.hintRow}>
                  <View style={styles.hintIconWrap} accessibilityElementsHidden>
                    <Ionicons name={hint.icon} size={13} color={D.brand} />
                  </View>
                  <AppText variant="caption" style={styles.hintText} numberOfLines={2}>
                    <AppText variant="caption" style={styles.hintKey}>
                      {hint.label} :{" "}
                    </AppText>
                    {hint.value}
                  </AppText>
                </View>
              ))}
            </View>
          </View>
        ) : null}

        <View style={styles.metricsDivider} accessibilityElementsHidden />
        <View style={[styles.metricsRow, stackMetrics && styles.metricsRowStacked]}>
          <MetricTile
            label={routeMetrics.distanceMetricLabel}
            value={routeMetrics.distanceLabel}
            icon="trail-sign-outline"
            stacked={stackMetrics}
            compactLabel={!stackMetrics}
          />
          <MetricTile
            label={routeMetrics.durationMetricLabel}
            value={routeMetrics.durationLabel}
            icon="timer-outline"
            showDivider
            stacked={stackMetrics}
            compactLabel={!stackMetrics}
          />
          <MetricTile
            label={stackMetrics ? "HEURE PRISE EN CHARGE" : "PRISE EN CHARGE"}
            value={formatMissionPickupTime(mission)}
            icon="calendar-outline"
            showDivider
            stacked={stackMetrics}
            compactLabel={!stackMetrics}
          />
        </View>
      </View>

      <View style={styles.controlsBlock}>
        <MissionProgressStepper
          mission={mission}
          etaSnapshot={etaQuery.data ?? null}
          remainingDistanceKm={routeMetrics.distanceKm}
        />

        {showTransitionActions && forwardTransition ? (
          <Pressable
            onPress={() => onMissionTransition?.(forwardTransition)}
            disabled={pending}
            accessibilityRole="button"
            accessibilityLabel={transitionLabel(forwardTransition)}
            style={({ pressed }) => [
              styles.primaryCta,
              pending && styles.disabledOpacity,
              pressed && styles.pressed,
            ]}
          >
            <Ionicons name="play" size={12} color="#FFFFFF" accessibilityElementsHidden />
            <AppText variant="label" style={styles.primaryCtaLabel}>
              {transitionLabel(forwardTransition)}
            </AppText>
          </Pressable>
        ) : null}

        {showQuickActions ? (
          <View style={[styles.quickActionsRow, stackMetrics && styles.quickActionsRowStacked]}>
            {showNavigation ? (
              <QuickAction
                label="NAVIGATION"
                icon="navigate-outline"
                onPress={() => void openNavigation(dest)}
                disabled={pending}
                stacked={stackMetrics}
              />
            ) : null}
            {showCallAction ? (
              <QuickAction
                label="APPELER"
                icon="call-outline"
                onPress={() => void safeCall(phone!)}
                disabled={pending}
                stacked={stackMetrics}
              />
            ) : null}
            {canRelease ? (
              <QuickAction
                label="LIBÉRER"
                icon="refresh-outline"
                onPress={() => onMissionRelease?.()}
                disabled={pending}
                stacked={stackMetrics}
              />
            ) : null}
          </View>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    alignSelf: "stretch",
    gap: 14,
  },
  infoCard: {
    backgroundColor: D.cardBg,
    borderWidth: 0,
    borderRadius: D.cardRadius,
    paddingHorizontal: 16,
    paddingTop: 15,
    paddingBottom: 14,
    gap: 11,
    ...cardShadow,
  },
  controlsBlock: {
    gap: 8,
  },
  cardHeader: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  cardHeaderLeft: {
    flexDirection: "row",
    alignItems: "center",
    gap: 7,
    flexShrink: 1,
    minWidth: 0,
  },
  cardEyebrow: {
    color: D.brand,
    fontWeight: "800",
    letterSpacing: 0.55,
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
    flexShrink: 1,
    minWidth: 0,
  },
  headerBadges: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    flexShrink: 1,
    minWidth: 0,
  },
  staleBadge: {
    backgroundColor: "rgba(220, 38, 38, 0.1)",
    borderWidth: 1,
    borderColor: "rgba(220, 38, 38, 0.35)",
    borderRadius: 999,
    paddingHorizontal: 9,
    paddingVertical: 4,
  },
  staleBadgeText: {
    color: "#B91C1C",
    fontWeight: "800",
    fontSize: FONT_SIZE.px10,
    letterSpacing: 0.35,
  },
  statusBadge: {
    backgroundColor: D.assignedBadgeBg,
    borderWidth: 1,
    borderColor: D.assignedBadgeBorder,
    borderRadius: 999,
    paddingHorizontal: 11,
    paddingVertical: 5,
  },
  statusBadgeText: {
    color: D.assignedBadgeText,
    fontWeight: "800",
    fontSize: FONT_SIZE.px10,
    letterSpacing: 0.45,
  },
  clientBlock: {
    gap: 3,
    marginTop: 2,
    minWidth: 0,
    flexShrink: 1,
  },
  clientName: {
    color: D.text,
    fontSize: FONT_SIZE.px23,
    fontWeight: "800",
    lineHeight: 29,
    letterSpacing: -0.2,
    flexShrink: 1,
    minWidth: 0,
  },
  clientBirthDate: {
    color: D.textMuted,
    fontWeight: "600",
    fontSize: FONT_SIZE.px13,
    lineHeight: 17,
    marginTop: 2,
  },
  whenRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 7,
  },
  whenText: {
    color: D.textSub,
    fontWeight: "600",
    fontSize: FONT_SIZE.px12,
    lineHeight: 17,
    flex: 1,
    flexShrink: 1,
    minWidth: 0,
  },
  whenPrefix: {
    color: D.textMuted,
    fontWeight: "600",
    fontSize: FONT_SIZE.px12,
    lineHeight: 17,
  },
  routeTimeline: {
    paddingTop: 2,
    gap: 0,
  },
  routeRow: {
    flexDirection: "row",
    gap: 11,
    borderRadius: 10,
    paddingVertical: 4,
    paddingHorizontal: 6,
    marginHorizontal: -6,
  },
  routeRowActive: {
    backgroundColor: "rgba(0, 121, 107, 0.07)",
  },
  routeMarkerCol: {
    width: 16,
    alignItems: "center",
    paddingTop: 2,
  },
  routeDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: D.brand,
  },
  routeDotActive: {
    width: 12,
    height: 12,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: D.brand,
    backgroundColor: D.cardBg,
  },
  routeFlagWrap: {
    width: 16,
    height: 16,
    alignItems: "center",
    justifyContent: "center",
  },
  routeFlagWrapActive: {
    width: 18,
    height: 18,
    borderRadius: 9,
    backgroundColor: "rgba(0, 121, 107, 0.12)",
  },
  routeTextCol: {
    flex: 1,
    flexShrink: 1,
    minWidth: 0,
    paddingBottom: 12,
    gap: 3,
  },
  routeTextColLast: {
    paddingBottom: 0,
  },
  routeKeyRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    flexWrap: "wrap",
  },
  routeKey: {
    color: D.routeLabel,
    fontWeight: "800",
    letterSpacing: 0.45,
    fontSize: FONT_SIZE.px10,
    lineHeight: 13,
  },
  routeKeyActive: {
    color: D.brand,
  },
  routeFocusTag: {
    color: D.brand,
    fontWeight: "800",
    letterSpacing: 0.5,
    fontSize: FONT_SIZE.px10,
    lineHeight: 13,
    paddingHorizontal: 6,
    paddingVertical: 1,
    borderRadius: 6,
    backgroundColor: "rgba(0, 121, 107, 0.13)",
  },
  routeAddress: {
    color: D.routeText,
    fontWeight: "500",
    fontSize: FONT_SIZE.px14,
    lineHeight: 20,
    flexShrink: 1,
    minWidth: 0,
  },
  routeAddressActive: {
    color: D.text,
    fontWeight: "700",
  },
  hintsBlock: {
    backgroundColor: "rgba(0, 121, 107, 0.07)",
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    gap: 8,
    marginTop: 6,
  },
  hintsHeaderRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
  },
  hintsHeader: {
    color: D.brand,
    fontWeight: "800",
    letterSpacing: 0.55,
    fontSize: FONT_SIZE.px10,
    lineHeight: 13,
    flex: 1,
    flexShrink: 1,
    minWidth: 0,
  },
  directionPill: {
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: 999,
    backgroundColor: D.cardBg,
    borderWidth: 1,
    borderColor: "rgba(0, 121, 107, 0.22)",
  },
  directionPillText: {
    color: D.brand,
    fontWeight: "800",
    letterSpacing: 0.5,
    fontSize: FONT_SIZE.px10,
    lineHeight: 12,
  },
  hintsList: {
    gap: 5,
  },
  hintRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 7,
  },
  hintIconWrap: {
    width: 16,
    height: 16,
    alignItems: "center",
    justifyContent: "center",
    paddingTop: 1,
  },
  hintText: {
    color: D.textSub,
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
    fontWeight: "500",
    flex: 1,
    flexShrink: 1,
    minWidth: 0,
  },
  hintKey: {
    color: D.text,
    fontWeight: "700",
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
  },
  metricsDivider: {
    height: StyleSheet.hairlineWidth,
    backgroundColor: D.metricDivider,
    marginTop: 2,
  },
  metricsRow: {
    flexDirection: "row",
    alignItems: "stretch",
    paddingTop: 5,
    paddingBottom: 1,
    gap: 0,
  },
  metricsRowStacked: {
    flexDirection: "column",
    alignItems: "stretch",
    gap: 8,
  },
  metricDivider: {
    width: StyleSheet.hairlineWidth,
    backgroundColor: D.metricDivider,
    alignSelf: "stretch",
  },
  metricDividerStacked: {
    width: "100%",
    height: StyleSheet.hairlineWidth,
    alignSelf: "stretch",
  },
  metricTile: {
    flex: 1,
    flexShrink: 1,
    minWidth: 0,
    alignItems: "center",
    justifyContent: "flex-start",
    gap: 2,
    paddingHorizontal: 4,
  },
  metricTileStacked: {
    flexGrow: 0,
    width: "100%",
    alignItems: "flex-start",
    paddingVertical: 2,
  },
  metricHead: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 2,
    flexWrap: "wrap",
    width: "100%",
    minWidth: 0,
    flexShrink: 1,
  },
  metricHeadCompact: {
    flexWrap: "nowrap",
  },
  metricLabel: {
    flexShrink: 1,
    minWidth: 0,
    color: D.textMuted,
    fontWeight: "700",
    fontSize: FONT_SIZE.px7,
    letterSpacing: 0.15,
    textAlign: "center",
    textTransform: "uppercase",
    lineHeight: 10,
  },
  metricLabelCompact: {
    lineHeight: 11,
  },
  metricValue: {
    color: D.text,
    fontWeight: "800",
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    textAlign: "center",
    flexShrink: 1,
    minWidth: 0,
    width: "100%",
  },
  metricValueStacked: {
    textAlign: "left",
    width: undefined,
  },
  primaryCta: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 5,
    minHeight: 40,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 10,
    backgroundColor: D.brandDark,
    ...actionShadow,
  },
  primaryCtaLabel: {
    color: "#FFFFFF",
    fontWeight: "800",
    fontSize: FONT_SIZE.px13,
    letterSpacing: 0.7,
    flexShrink: 1,
    minWidth: 0,
  },
  quickActionsRow: {
    flexDirection: "row",
    alignItems: "stretch",
    gap: 6,
    flexWrap: "nowrap",
  },
  quickActionsRowStacked: {
    flexDirection: "column",
    alignItems: "stretch",
    flexWrap: "wrap",
  },
  quickAction: {
    flex: 1,
    flexShrink: 1,
    minWidth: 0,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    minHeight: 36,
    backgroundColor: D.cardBg,
    borderWidth: 0,
    borderRadius: 8,
    gap: 4,
    paddingVertical: 6,
    paddingHorizontal: 6,
    ...softSurfaceShadow,
  },
  quickActionStacked: {
    flexGrow: 0,
    flex: 0,
    width: "100%",
  },
  quickActionDisabled: {
    opacity: 0.38,
  },
  quickActionLabel: {
    flexShrink: 1,
    minWidth: 0,
    color: D.textSub,
    fontWeight: "700",
    fontSize: FONT_SIZE.px9,
    letterSpacing: 0.1,
    textAlign: "center",
  },
  disabledOpacity: { opacity: 0.55 },
  pressed: { opacity: 0.9, transform: [{ scale: 0.985 }] },
});
