import { useEffect, useMemo, useRef, useState } from "react";
import {
  Animated,
  LayoutAnimation,
  Pressable,
  StyleSheet,
  View,
  useColorScheme,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import { AppText, Screen, useAppViewport } from "../../../src/design/responsive";
import { Motion, MotionEasing } from "../../../src/design/navigation/navigationMotion";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import {
  useActiveDriverContextId,
  useDriverCompanyBookingsTodayQuery,
  useDriverMissionsListFocusResync,
  useDriverMissionsQuery,
  useDriverStatusTransition,
  useDynamicEtaQuery,
} from "../../../src/features/driver/hooks";
import { useMissionRouteMetrics } from "../../../src/features/driver/hooks/useMissionRouteMetrics";
import {
  getDriverStatusUx,
  normalizeDriverMissionStatus,
  resolveDriverStatusForUx,
} from "../../../src/features/driver/statusDictionary";
import type { DriverMission, DriverTransitionStatus } from "../../../src/features/driver/types";
import { MissionLiveTrackingDisclosureModal } from "../../../src/features/driver/components/MissionLiveTrackingDisclosureModal";
import { useMissionLiveTrackingGuard } from "../../../src/features/driver/hooks/useMissionLiveTrackingGuard";
import { requiresLiveTrackingPermission } from "../../../src/features/driver/services/missionLiveTrackingEligibility";
import { missionActiveCardShadow } from "../../../src/features/driver/theme/driverDashboardTheme";
import { getCallablePhoneFromMission, openNavigation, safeCall } from "../../../src/features/driver/utils/missionContact";
import { createShadow } from "../../../src/styles/shadowStyles";
import {
  driverHasConfirmedPickupTime,
  driverHasScheduledPickupTime,
  formatDriverScheduleTimeLabel,
} from "../../../src/features/driver/utils/pickupScheduling";

const softCardShadow = createShadow(missionActiveCardShadow);

type TopTab = "mine" | "company";
const TAB_COUNT = 2;
const TAB_INSET = 3;

type DispatchTheme = {
  bg: string;
  headerBg: string;
  headerBorder: string;
  filterBg: string;
  filterBorder: string;
  title: string;
  subtitle: string;
  metricsBg: string;
  metricsBorder: string;
  tabsBg: string;
  tabIndicatorBg: string;
  tabIdle: string;
  tabActive: string;
  cardBg: string;
  cardBorder: string;
  cardDoneBg: string;
  cardDoneBorder: string;
  bodyText: string;
  mutedText: string;
  sectionText: string;
  divider: string;
  secondaryActionBg: string;
  secondaryActionBorder: string;
  secondaryActionText: string;
};

function buildDispatchTheme(isDark: boolean): DispatchTheme {
  if (isDark) {
    return {
      bg: "#0F172A",
      headerBg: "#1E293B",
      headerBorder: "rgba(148,163,184,0.22)",
      filterBg: "rgba(15,23,42,0.55)",
      filterBorder: "rgba(148,163,184,0.28)",
      title: "#E2E8F0",
      subtitle: "#94A3B8",
      metricsBg: "rgba(30,41,59,0.86)",
      metricsBorder: "rgba(148,163,184,0.18)",
      tabsBg: "rgba(30,41,59,0.82)",
      tabIndicatorBg: "rgba(71,85,105,0.7)",
      tabIdle: "#94A3B8",
      tabActive: "#E2E8F0",
      cardBg: "rgba(30,41,59,0.9)",
      cardBorder: "rgba(148,163,184,0.2)",
      cardDoneBg: "rgba(51,65,85,0.92)",
      cardDoneBorder: "rgba(148,163,184,0.26)",
      bodyText: "#E2E8F0",
      mutedText: "#94A3B8",
      sectionText: "#CBD5E1",
      divider: "rgba(148,163,184,0.25)",
      secondaryActionBg: "rgba(51,65,85,0.72)",
      secondaryActionBorder: "rgba(148,163,184,0.24)",
      secondaryActionText: "#E2E8F0",
    };
  }
  return {
    bg: "#F5F7F6",
    headerBg: "#FFFFFF",
    headerBorder: "rgba(15,23,42,0.06)",
    filterBg: "rgba(255,255,255,0.72)",
    filterBorder: "rgba(15,23,42,0.08)",
    title: "#0F172A",
    subtitle: "#475569",
    metricsBg: "#FFFFFF",
    metricsBorder: "rgba(15,23,42,0.06)",
    tabsBg: "#E9EEF3",
    tabIndicatorBg: "#FFFFFF",
    tabIdle: "#64748B",
    tabActive: "#0F172A",
    cardBg: "#FFFFFF",
    cardBorder: "rgba(15, 23, 42, 0.07)",
    cardDoneBg: "#F2F4F7",
    cardDoneBorder: "rgba(148, 163, 184, 0.35)",
    bodyText: "#334155",
    mutedText: "#64748B",
    sectionText: "#0F172A",
    divider: "rgba(148,163,184,0.2)",
    secondaryActionBg: "#FFFFFF",
    secondaryActionBorder: "rgba(15,23,42,0.12)",
    secondaryActionText: "#0F172A",
  };
}

import { FONT_SIZE } from "../../../src/design/responsive/typographyTokens";

function missionSortTs(mission: DriverMission): number {
  if (!driverHasScheduledPickupTime(mission)) return Number.MAX_SAFE_INTEGER;
  const ts = Date.parse(String(mission.scheduled_time ?? ""));
  return Number.isFinite(ts) ? ts : Number.MAX_SAFE_INTEGER;
}

function sortMissions(a: DriverMission, b: DriverMission): number {
  const delta = missionSortTs(a) - missionSortTs(b);
  if (delta !== 0) return delta;
  return a.id - b.id;
}

function missionIsToday(mission: DriverMission, now: Date): boolean {
  const key = normalizeDriverMissionStatus(mission.status);
  if (key === "EN_ROUTE" || key === "ARRIVED" || key === "IN_PROGRESS") return true;
  if (key === "CANCELLED" || key === "FAILED" || key === "NO_SHOW" || key === "REASSIGNED") return false;
  const raw = mission.scheduled_time;
  if (!raw || !String(raw).trim()) return true;
  const ts = Date.parse(String(raw));
  if (!Number.isFinite(ts)) return true;
  const d = new Date(ts);
  return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate();
}

function whenLabel(mission: DriverMission): string {
  return formatDriverScheduleTimeLabel(mission);
}

function normalizePersonDisplayName(value: unknown): string {
  const name = typeof value === "string" ? value.trim() : "";
  if (!name) return "";
  const cleaned = name.replace(/\bnone\b/gi, "").replace(/\s+/g, " ").trim();
  return cleaned;
}

function missionClientName(mission: DriverMission): string {
  const direct = normalizePersonDisplayName(mission.client_name);
  if (direct) return direct;
  const nested = normalizePersonDisplayName((mission.client as { full_name?: unknown } | null | undefined)?.full_name);
  if (nested) return nested;
  return `Course #${mission.id}`;
}

function missionDriverName(mission: DriverMission): string {
  const direct = normalizePersonDisplayName(mission.driver_name);
  if (direct) return direct;
  const nested = normalizePersonDisplayName((mission.driver as { full_name?: unknown } | null | undefined)?.full_name);
  if (nested) return nested;
  return "Non assigne";
}

function missionStatusBucket(status: string): "todo" | "progress" | "done" {
  const key = normalizeDriverMissionStatus(status);
  if (key === "COMPLETED") return "done";
  if (key === "EN_ROUTE" || key === "ARRIVED" || key === "IN_PROGRESS") return "progress";
  return "todo";
}

function todayLabel(now: Date): string {
  return now.toLocaleDateString("fr-CH", { weekday: "long", day: "2-digit", month: "long" });
}

function missionStatusColor(status: string): string {
  const key = normalizeDriverMissionStatus(status);
  switch (key) {
    case "ASSIGNED":
      return "#14B8A6";
    case "EN_ROUTE":
      return "#3B82F6";
    case "ARRIVED":
      return "#F59E0B";
    case "IN_PROGRESS":
      return "#8B5CF6";
    case "COMPLETED":
      return "#94A3B8";
    default:
      return "#EF4444";
  }
}

function nextTransitionTarget(status: string): DriverTransitionStatus | null {
  const key = normalizeDriverMissionStatus(status);
  if (key === "ASSIGNED") return "EN_ROUTE";
  if (key === "EN_ROUTE") return "ARRIVED";
  if (key === "ARRIVED" || key === "IN_PROGRESS") return "COMPLETED";
  return null;
}

function compactEta(mission: DriverMission): string {
  const rawMission = mission as Record<string, unknown>;
  const minuteCandidates = [
    mission.eta_minutes,
    mission.estimated_duration_min,
    mission.duration_minutes,
    rawMission.duration_in_minutes,
    rawMission.eta,
    rawMission.eta_minutes,
    rawMission.duration_minutes,
    rawMission.route_duration_minutes,
    rawMission.travel_time_minutes,
  ];
  for (const candidate of minuteCandidates) {
    const value = typeof candidate === "number" ? candidate : Number(candidate);
    if (Number.isFinite(value) && value > 0) return `${Math.round(value)} min`;
  }

  const secondCandidates = [
    rawMission.duration_seconds,
    rawMission.estimated_duration_seconds,
    rawMission.route_duration_seconds,
    rawMission.travel_time_seconds,
  ];
  for (const candidate of secondCandidates) {
    const seconds = typeof candidate === "number" ? candidate : Number(candidate);
    if (Number.isFinite(seconds) && seconds > 0) return `${Math.round(seconds / 60)} min`;
  }

  return "-- min";
}

function compactDistance(mission: DriverMission): string {
  const rawMission = mission as Record<string, unknown>;
  const kmCandidates = [
    mission.distance_km,
    mission.distanceKm,
    mission.route_distance_km,
    rawMission.distance_km,
    rawMission.route_distance_km,
    rawMission.estimated_distance_km,
    rawMission.travel_distance_km,
  ];
  for (const candidate of kmCandidates) {
    const value = typeof candidate === "number" ? candidate : Number(candidate);
    if (Number.isFinite(value) && value > 0) return `${value.toFixed(1)} km`;
  }

  const meterCandidates = [
    rawMission.distance_meters,
    rawMission.estimated_distance_meters,
    rawMission.route_distance_meters,
    rawMission.travel_distance_meters,
  ];
  for (const candidate of meterCandidates) {
    const meters = typeof candidate === "number" ? candidate : Number(candidate);
    if (Number.isFinite(meters) && meters > 0) return `${(meters / 1000).toFixed(1)} km`;
  }

  return "-- km";
}

function compactAddressLabel(address: string): string {
  const normalized = address.replace(/\s+/g, " ").trim();
  const head = normalized.split(",")[0]?.trim() ?? normalized;
  return head
    .replace(/^Chemin\b/i, "Ch")
    .replace(/^Route\b/i, "Rte")
    .replace(/^Avenue\b/i, "Av")
    .replace(/^Boulevard\b/i, "Bd")
    .replace(/^Foyer de jour pour personnes Ã¢gÃ©es\s+/i, "Foyer de jour ")
    .replace(/\bAux\s+/i, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function isLateMission(mission: DriverMission, nowTs: number): boolean {
  const bucket = missionStatusBucket(String(mission.status ?? ""));
  if (bucket === "done") return false;
  if (!driverHasConfirmedPickupTime(mission)) return false;
  return missionSortTs(mission) < nowTs;
}

function isInstitutionTransport(mission: DriverMission): boolean {
  const raw = mission as Record<string, unknown>;

  const booleanLikeKeys = [
    "is_institution_transport",
    "is_institutional_transport",
    "institutional",
    "is_hospital_transport",
    "is_hospitalized",
    "is_medical_institution",
  ];
  for (const key of booleanLikeKeys) {
    const value = raw[key];
    if (value === true || value === 1 || value === "1") return true;
    if (typeof value === "string" && value.trim().toLowerCase() === "true") return true;
  }

  const idLikeKeys = ["institution_id", "hospital_id", "facility_id", "medical_facility_id"];
  for (const key of idLikeKeys) {
    const value = raw[key];
    if (typeof value === "number" && Number.isFinite(value) && value > 0) return true;
    if (typeof value === "string" && value.trim().length > 0 && value.trim().toLowerCase() !== "none") return true;
  }

  const typeHints = [
    raw.transport_type,
    raw.ride_type,
    raw.mission_type,
    raw.booking_type,
    raw.context_type,
    raw.client_type,
    mission.pickup_location,
    mission.dropoff_location,
  ]
    .filter((v): v is string => typeof v === "string" && v.trim().length > 0)
    .join(" ")
    .toLowerCase();

  return (
    typeHints.includes("institution") ||
    typeHints.includes("hospital") ||
    typeHints.includes("hopital") ||
    typeHints.includes("clinique") ||
    typeHints.includes("clinic") ||
    typeHints.includes("foyer") ||
    typeHints.includes("ems") ||
    typeHints.includes("etablissement")
  );
}

function personLabelForMission(mission: DriverMission): string {
  return isInstitutionTransport(mission) ? "Patient" : "Client";
}

function extractDriverScopedId(contextId: string | null): string | null {
  if (!contextId || !contextId.startsWith("driver:")) return null;
  const value = contextId.slice("driver:".length).trim();
  return value.length > 0 ? value : null;
}

function missionAssignedDriverId(mission: DriverMission): string | null {
  const raw = mission as Record<string, unknown>;
  const nestedDriver = mission.driver as { id?: unknown } | null | undefined;
  const candidates = [raw.driver_id, raw.assigned_driver_id, raw.driverId, nestedDriver?.id];
  for (const candidate of candidates) {
    if (candidate == null) continue;
    const value = String(candidate).trim();
    if (value.length > 0 && value.toLowerCase() !== "none") return value;
  }
  return null;
}

function isAssignedToActiveDriver(mission: DriverMission, activeDriverId: string | null): boolean {
  if (!activeDriverId) return false;
  const missionDriverId = missionAssignedDriverId(mission);
  if (!missionDriverId) return false;
  return missionDriverId === activeDriverId;
}

function MissionAccordionCard({
  mission,
  allowExpand,
  expanded,
  onToggle,
  onStart,
  showDriver,
  late,
  theme,
}: {
  mission: DriverMission;
  allowExpand: boolean;
  expanded: boolean;
  onToggle?: () => void;
  onStart: (mission: DriverMission) => void;
  showDriver: boolean;
  late: boolean;
  theme: DispatchTheme;
}) {
  const status = String(mission.status ?? "");
  const done = missionStatusBucket(status) === "done";
  const ux = getDriverStatusUx(String(mission.status ?? ""));
  const statusColor = missionStatusColor(status);
  const pickup = mission.pickup_location?.trim() || "Depart non defini";
  const destination = mission.dropoff_location?.trim() || "Destination non definie";
  const pickupCompact = compactAddressLabel(pickup);
  const destinationCompact = compactAddressLabel(destination);
  const canCall = Boolean(getCallablePhoneFromMission(mission));
  const statusKey = resolveDriverStatusForUx(mission.status);
  const etaQuery = useDynamicEtaQuery(done ? null : mission.id, { missionStatus: statusKey });
  const routeMetrics = useMissionRouteMetrics(mission, {
    etaMinutes: etaQuery.data?.eta_minutes,
    etaSnapshot: etaQuery.data ?? null,
  });
  const etaText =
    routeMetrics.durationLabel !== "â€”" ? routeMetrics.durationLabel : compactEta(mission);
  const distanceText =
    routeMetrics.distanceLabel !== "â€”" ? routeMetrics.distanceLabel : compactDistance(mission);
  const expandAnim = useRef(new Animated.Value(expanded ? 1 : 0)).current;

  useEffect(() => {
    Animated.timing(expandAnim, {
      toValue: expanded ? 1 : 0,
      duration: Motion.detail,
      easing: MotionEasing,
      useNativeDriver: false,
    }).start();
  }, [expandAnim, expanded]);

  const content = (
    <View style={[styles.card, { backgroundColor: theme.cardBg }, done && styles.cardDone, done && { backgroundColor: theme.cardDoneBg }]}>
      <View style={styles.cardBody}>
        <View style={styles.cardTop}>
          <AppText variant="sectionTitle" style={[styles.when, { color: theme.title }, done && styles.whenDone, done && { color: theme.mutedText }]}>
            {whenLabel(mission)}
          </AppText>
          <View style={styles.statusBadges}>
            {late ? (
              <View style={styles.lateBadge}>
                <AppText variant="label" style={styles.lateBadgeText}>
                  Retard
                </AppText>
              </View>
            ) : null}
            <View style={[styles.badge, done && styles.badgeDone, { borderColor: `${statusColor}55`, backgroundColor: `${statusColor}1F` }]}>
              <AppText variant="label" style={[styles.badgeText, done && styles.badgeTextDone]} numberOfLines={1}>
                {ux.label}
              </AppText>
            </View>
          </View>
        </View>

        <AppText variant="label" style={[styles.client, { color: theme.sectionText }, done && styles.clientDone, done && { color: theme.mutedText }]} numberOfLines={1}>
          {missionClientName(mission)}
        </AppText>

        <View style={styles.routeRow}>
          <Ionicons name="ellipse" size={8} color={done ? "#94A3B8" : statusColor} />
          <AppText variant="body" style={[styles.route, { color: theme.bodyText }, done && styles.routeDone, done && { color: theme.mutedText }]} numberOfLines={1}>
            {pickupCompact} {" â†’ "} {destinationCompact}
          </AppText>
        </View>

        {allowExpand ? (
          <View style={styles.metaAndHintRow}>
            <View style={styles.metaRow}>
              <View style={styles.metaItem}>
                <Ionicons name="car-outline" size={13} color={done ? "#94A3B8" : "#334155"} />
                <AppText variant="caption" style={[styles.metaText, { color: theme.mutedText }, done && styles.routeDone, done && { color: theme.mutedText }]}>
                  {etaText}
                </AppText>
              </View>
              <View style={styles.metaItem}>
                <Ionicons name="navigate-outline" size={13} color={done ? "#94A3B8" : "#334155"} />
                <AppText variant="caption" style={[styles.metaText, { color: theme.mutedText }, done && styles.routeDone, done && { color: theme.mutedText }]}>
                  {distanceText}
                </AppText>
              </View>
            </View>

            <View style={styles.expandHint}>
              <Ionicons name={expanded ? "chevron-up-outline" : "chevron-down-outline"} size={14} color="#64748B" />
              <AppText variant="caption" style={[styles.expandHintText, { color: theme.mutedText }]}>
                {expanded ? "Masquer details" : "Voir details"}
              </AppText>
            </View>
          </View>
        ) : null}

        {showDriver ? (
          <AppText variant="caption" style={[styles.driverLine, { color: theme.sectionText }, done && styles.driverLineDone, done && { color: theme.mutedText }]} numberOfLines={1}>
            Chauffeur : {missionDriverName(mission)}
          </AppText>
        ) : null}

        {allowExpand ? (
          <Animated.View
            pointerEvents={expanded ? "auto" : "none"}
            style={[
              styles.expandAnimatedWrap,
              {
                opacity: expandAnim,
                maxHeight: expandAnim.interpolate({
                  inputRange: [0, 1],
                  outputRange: [0, 520],
                }),
                transform: [
                  {
                    translateY: expandAnim.interpolate({
                      inputRange: [0, 1],
                      outputRange: [-8, 0],
                    }),
                  },
                ],
              },
            ]}
          >
            <View style={[styles.expandArea, { borderTopColor: theme.divider }]}>
            <AppText variant="label" style={[styles.sectionLabel, { color: theme.mutedText }]}>
              {personLabelForMission(mission)}
            </AppText>
            <View style={styles.detailRow}>
              <AppText variant="body" style={[styles.sectionValue, styles.detailRowText, { color: theme.sectionText }]}>
                {missionClientName(mission)}
              </AppText>
              {canCall ? (
                <Pressable
                  style={({ pressed }) => [styles.detailIconButton, { borderColor: theme.secondaryActionBorder, backgroundColor: theme.secondaryActionBg }, pressed && styles.actionPressed]}
                  onPress={() => {
                    const phone = getCallablePhoneFromMission(mission);
                    if (!phone) return;
                    void safeCall(phone);
                  }}
                  accessibilityRole="button"
                >
                  <Ionicons name="call-outline" size={15} color={theme.secondaryActionText} />
                </Pressable>
              ) : null}
            </View>

            <View style={styles.timelineRow}>
              <View style={styles.timelineTrack}>
                <View style={styles.dotStart} />
                <View style={styles.trackLine} />
                <View style={styles.dotEnd} />
              </View>
              <View style={styles.timelineTexts}>
                <AppText variant="label" style={[styles.sectionLabel, { color: theme.mutedText }]}>
                  Prise en charge
                </AppText>
                <View style={styles.detailRow}>
                  <AppText variant="body" style={[styles.sectionValue, styles.detailRowText, { color: theme.sectionText }]}>
                    {pickup}
                  </AppText>
                  <Pressable
                    style={({ pressed }) => [styles.detailIconButton, { borderColor: theme.secondaryActionBorder, backgroundColor: theme.secondaryActionBg }, pressed && styles.actionPressed]}
                    onPress={() => void openNavigation(pickup)}
                    accessibilityRole="button"
                  >
                    <Ionicons name="navigate-outline" size={15} color={theme.secondaryActionText} />
                  </Pressable>
                </View>
                <AppText variant="caption" style={[styles.timelineEta, { color: theme.mutedText }]}>
                  {etaText}
                </AppText>
                <AppText variant="label" style={[styles.sectionLabel, styles.sectionGap, { color: theme.mutedText }]}>
                  Destination
                </AppText>
                <View style={styles.detailRow}>
                  <AppText variant="body" style={[styles.sectionValue, styles.detailRowText, { color: theme.sectionText }]}>
                    {destination}
                  </AppText>
                  <Pressable
                    style={({ pressed }) => [styles.detailIconButton, { borderColor: theme.secondaryActionBorder, backgroundColor: theme.secondaryActionBg }, pressed && styles.actionPressed]}
                    onPress={() => void openNavigation(destination)}
                    accessibilityRole="button"
                  >
                    <Ionicons name="navigate-outline" size={15} color={theme.secondaryActionText} />
                  </Pressable>
                </View>
              </View>
            </View>

            <AppText variant="label" style={[styles.sectionLabel, styles.sectionGap, { color: theme.mutedText }]}>
              Notes
            </AppText>
            <AppText variant="bodyMuted" style={[styles.sectionValue, { color: theme.mutedText }]}>
              {typeof mission.notes === "string" && mission.notes.trim() ? mission.notes.trim() : "Aucune note"}
            </AppText>

            </View>
          </Animated.View>
        ) : null}
      </View>
    </View>
  );

  if (!allowExpand || !onToggle) return content;

  return (
    <Pressable
      onPress={onToggle}
      style={({ pressed }) => [styles.pressable, pressed && styles.pressablePressed]}
      android_ripple={{ color: "rgba(100,116,139,0.12)" }}
    >
      {content}
    </Pressable>
  );
}

export default function DriverTripsScreen() {
  const { width } = useAppViewport();
  const compact = width < 380;
  const isDark = useColorScheme() === "dark";
  const theme = useMemo(() => buildDispatchTheme(isDark), [isDark]);
  const referenceNowTs = useMemo(() => Date.now(), []);
  const now = useMemo(() => new Date(referenceNowTs), [referenceNowTs]);
  const activeDriverContextId = useActiveDriverContextId();
  const activeDriverId = useMemo(() => extractDriverScopedId(activeDriverContextId), [activeDriverContextId]);
  useDriverMissionsListFocusResync();
  const mineQuery = useDriverMissionsQuery();
  const companyQuery = useDriverCompanyBookingsTodayQuery();
  const statusTransition = useDriverStatusTransition();
  const liveTrackingGuard = useMissionLiveTrackingGuard();
  const [tab, setTab] = useState<TopTab>("mine");
  const [expandedMissionId, setExpandedMissionId] = useState<number | null>(null);
  const [tabsWidth, setTabsWidth] = useState(0);
  const tabAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(tabAnim, {
      toValue: tab === "mine" ? 0 : 1,
      duration: Motion.page,
      easing: MotionEasing,
      useNativeDriver: true,
    }).start();
  }, [tab, tabAnim]);

  const mine = useMemo(() => {
    return (mineQuery.data ?? []).filter((m) => missionIsToday(m, now)).sort(sortMissions);
  }, [mineQuery.data, now]);
  const company = useMemo(() => {
    return (companyQuery.data ?? []).filter((m) => missionIsToday(m, now)).sort(sortMissions);
  }, [companyQuery.data, now]);

  const mineWithAssignedDone = useMemo(() => {
    const assignedDoneFromCompany = company.filter(
      (mission) =>
        missionStatusBucket(String(mission.status ?? "")) === "done" &&
        isAssignedToActiveDriver(mission, activeDriverId),
    );
    if (assignedDoneFromCompany.length === 0) return mine;
    const byId = new Map<number, DriverMission>();
    for (const mission of mine) byId.set(mission.id, mission);
    for (const mission of assignedDoneFromCompany) byId.set(mission.id, mission);
    return Array.from(byId.values()).sort(sortMissions);
  }, [mine, company, activeDriverId]);

  const list = tab === "mine" ? mineWithAssignedDone : company;
  const stats = useMemo(() => {
    let todo = 0;
    let progress = 0;
    let done = 0;
    for (const mission of list) {
      const bucket = missionStatusBucket(String(mission.status ?? ""));
      if (bucket === "done") done += 1;
      else if (bucket === "progress") progress += 1;
      else todo += 1;
    }
    return { total: list.length, done, progress, todo };
  }, [list]);
  const missionSections = useMemo(() => {
    const todo = list.filter((m) => missionStatusBucket(String(m.status ?? "")) === "todo");
    const progress = list.filter((m) => missionStatusBucket(String(m.status ?? "")) === "progress");
    const done = list.filter((m) => missionStatusBucket(String(m.status ?? "")) === "done");
    return [
      { key: "todo", label: "A effectuer", items: todo },
      { key: "progress", label: "En cours", items: progress },
      { key: "done", label: "Terminees", items: done },
    ] as const;
  }, [list]);

  const loading = tab === "mine" ? mineQuery.isLoading : companyQuery.isLoading;
  const error = tab === "mine" ? mineQuery.isError : companyQuery.isError;
  const tabSegmentWidth = Math.max(0, (tabsWidth - TAB_INSET * 2) / TAB_COUNT);
  const indicatorX = tabAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [0, tabSegmentWidth],
  });

  function toggleExpand(id: number) {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setExpandedMissionId((prev) => (prev === id ? null : id));
  }

  function onStartMission(mission: DriverMission) {
    const target = nextTransitionTarget(String(mission.status ?? ""));
    if (!target) return;
    const proceed = () => {
      statusTransition.mutate({ missionId: mission.id, targetStatus: target });
    };
    if (requiresLiveTrackingPermission(target)) {
      liveTrackingGuard.guardTransition({
        missionId: mission.id,
        target,
        onProceed: proceed,
      });
      return;
    }
    proceed();
  }

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          withHorizontalPadding={false}
          pageTransition={false}
          extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
          contentContainerStyle={[styles.page, compact && styles.pageCompact]}
          backgroundColor={theme.bg}
        >
          <View style={[styles.headerShell, { backgroundColor: theme.headerBg }]}>
            <View style={styles.header}>
              <AppText variant="screenTitle" style={[styles.title, { color: theme.title }]}>
                Courses du jour
              </AppText>
              <AppText variant="bodyMuted" style={[styles.subtitle, { color: theme.subtitle }]}>
                {todayLabel(now)}
              </AppText>
            </View>
            <View style={styles.headerRight}>
              <View style={styles.liveBadge}>
                <View style={styles.liveDot} />
                <AppText variant="caption" style={[styles.liveText, { color: theme.subtitle }]}>
                  Live
                </AppText>
              </View>
            </View>
          </View>

          <View style={styles.metrics}>
            <View style={[styles.metricCard, { backgroundColor: theme.metricsBg }]}>
              <View style={styles.metricRow}>
                <View style={[styles.metricIconBubble, { backgroundColor: "rgba(20,184,166,0.14)" }]}>
                  <Ionicons name="calendar-outline" size={13} color="#14B8A6" />
                </View>
                <View style={styles.metricTextCol}>
                  <AppText variant="caption" style={[styles.metricLabel, { color: theme.mutedText }]}>
                    A effectuer
                  </AppText>
                  <AppText variant="sectionTitle" style={[styles.metricValue, { color: theme.title }]}>
                    {stats.todo}
                  </AppText>
                </View>
              </View>
            </View>
            <View style={[styles.metricCard, { backgroundColor: theme.metricsBg }]}>
              <View style={styles.metricRow}>
                <View style={[styles.metricIconBubble, { backgroundColor: "rgba(59,130,246,0.14)" }]}>
                  <Ionicons name="pulse-outline" size={13} color="#3B82F6" />
                </View>
                <View style={styles.metricTextCol}>
                  <AppText variant="caption" style={[styles.metricLabel, { color: theme.mutedText }]}>
                    En cours
                  </AppText>
                  <AppText variant="sectionTitle" style={[styles.metricValue, { color: theme.title }]}>
                    {stats.progress}
                  </AppText>
                </View>
              </View>
            </View>
            <View style={[styles.metricCard, { backgroundColor: theme.metricsBg }]}>
              <View style={styles.metricRow}>
                <View style={[styles.metricIconBubble, { backgroundColor: "rgba(148,163,184,0.18)" }]}>
                  <Ionicons name="checkmark-circle-outline" size={13} color="#94A3B8" />
                </View>
                <View style={styles.metricTextCol}>
                  <AppText variant="caption" style={[styles.metricLabel, { color: theme.mutedText }]}>
                    Terminees
                  </AppText>
                  <AppText variant="sectionTitle" style={[styles.metricValue, { color: theme.title }]}>
                    {stats.done}
                  </AppText>
                </View>
              </View>
            </View>
          </View>

          <View style={[styles.tabs, { backgroundColor: theme.tabsBg }]} onLayout={(e) => setTabsWidth(e.nativeEvent.layout.width)}>
            <Animated.View
              pointerEvents="none"
              style={[
                styles.tabIndicator,
                {
                  width: tabSegmentWidth,
                  transform: [{ translateX: indicatorX }],
                  backgroundColor: theme.tabIndicatorBg,
                },
              ]}
            />
            <Pressable
              onPress={() => setTab("mine")}
              style={styles.tab}
              accessibilityRole="button"
            >
              <AppText variant="label" style={[styles.tabLabel, { color: theme.tabIdle }, tab === "mine" && styles.tabLabelActive, tab === "mine" && { color: theme.tabActive }]}>
                Mes courses
              </AppText>
            </Pressable>
            <Pressable
              onPress={() => setTab("company")}
              style={styles.tab}
              accessibilityRole="button"
            >
              <AppText variant="label" style={[styles.tabLabel, { color: theme.tabIdle }, tab === "company" && styles.tabLabelActive, tab === "company" && { color: theme.tabActive }]}>
                Entreprise (jour)
              </AppText>
            </Pressable>
          </View>

          {loading ? <AppText variant="bodyMuted" style={[styles.info, { color: theme.mutedText }]}>Chargement...</AppText> : null}
          {error ? <AppText variant="error" style={styles.error}>Erreur de chargement des courses.</AppText> : null}

          {!loading && !error && list.length === 0 ? (
            <AppText variant="bodyMuted" style={[styles.info, { color: theme.mutedText }]}>
              Aucune course pour aujourd&apos;hui.
            </AppText>
          ) : null}

          <View style={styles.list}>
            {missionSections.map((section) =>
              section.items.length > 0 ? (
                <View key={section.key} style={styles.sectionBlock}>
                  <View style={styles.sectionHeader}>
                    <AppText variant="label" style={[styles.sectionTitle, { color: theme.tabActive }]}>
                      {section.label}
                    </AppText>
                    <View style={[styles.sectionCountPill, { backgroundColor: theme.tabsBg }]}>
                      <AppText variant="caption" style={[styles.sectionCountText, { color: theme.mutedText }]}>
                        {section.items.length}
                      </AppText>
                    </View>
                  </View>
                  <View style={styles.sectionList}>
                    {section.items.map((mission) => (
                      <MissionAccordionCard
                        key={mission.id}
                        mission={mission}
                        allowExpand={tab === "mine"}
                        expanded={expandedMissionId === mission.id}
                        onToggle={tab === "mine" ? () => toggleExpand(mission.id) : undefined}
                        onStart={onStartMission}
                        showDriver={tab === "company"}
                        late={isLateMission(mission, referenceNowTs)}
                        theme={theme}
                      />
                    ))}
                  </View>
                </View>
              ) : null,
            )}
          </View>
        </Screen>
        <MissionLiveTrackingDisclosureModal
          visible={liveTrackingGuard.disclosureVisible}
          pending={liveTrackingGuard.disclosurePending}
          showOpenSettings={liveTrackingGuard.showOpenSettings}
          onCancel={liveTrackingGuard.onDisclosureCancel}
          onContinue={liveTrackingGuard.onDisclosureContinue}
          onOpenSettings={liveTrackingGuard.onDisclosureOpenSettings}
        />
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: { paddingHorizontal: 16, paddingTop: 10, paddingBottom: 18, gap: 10 },
  pageCompact: { paddingHorizontal: 12 },
  headerShell: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    borderRadius: 18,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderWidth: 0,
    overflow: "hidden",
    ...softCardShadow,
  },
  header: { flex: 1, gap: 2, backgroundColor: "transparent" },
  title: { color: "#0F172A", fontSize: FONT_SIZE.px21, lineHeight: 25, fontWeight: "700" },
  subtitle: { color: "#475569", textTransform: "capitalize", fontSize: FONT_SIZE.px12, lineHeight: 15, fontWeight: "500", letterSpacing: 0.1 },
  headerRight: { flexDirection: "row", alignItems: "center", gap: 6 },
  liveBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
    borderRadius: 999,
    paddingHorizontal: 7,
    paddingVertical: 3,
    backgroundColor: "rgba(20,184,166,0.14)",
    borderWidth: 1,
    borderColor: "rgba(20,184,166,0.28)",
  },
  liveDot: { width: 6, height: 6, borderRadius: 999, backgroundColor: "#14B8A6" },
  liveText: { fontWeight: "600", letterSpacing: 0.15, fontSize: FONT_SIZE.px12, lineHeight: 14 },
  filterButton: {
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: "center",
    justifyContent: "center",
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.08)",
    backgroundColor: "rgba(255,255,255,0.72)",
  },
  metrics: { flexDirection: "row", gap: 6 },
  metricCard: {
    flex: 1,
    minWidth: 0,
    height: 56,
    borderRadius: 18,
    backgroundColor: "#FFFFFF",
    borderWidth: 0,
    paddingVertical: 6,
    paddingHorizontal: 9,
    alignItems: "flex-start",
    justifyContent: "center",
    gap: 1,
    ...softCardShadow,
  },
  metricRow: { flexDirection: "row", alignItems: "center", gap: 7, flex: 1 },
  metricIconBubble: {
    width: 20,
    height: 20,
    borderRadius: 10,
    alignItems: "center",
    justifyContent: "center",
  },
  metricTextCol: { flex: 1, minWidth: 0, justifyContent: "center", gap: 0 },
  metricValue: { color: "#0F172A", fontWeight: "700", fontSize: FONT_SIZE.px16, lineHeight: 18, marginTop: 1 },
  metricLabel: { color: "#64748B", textTransform: "uppercase", letterSpacing: 0.15, fontSize: FONT_SIZE.px9, lineHeight: 11 },
  tabs: {
    backgroundColor: "#E9EEF3",
    borderRadius: 14,
    padding: TAB_INSET,
    flexDirection: "row",
    position: "relative",
    overflow: "hidden",
  },
  tabIndicator: {
    position: "absolute",
    left: TAB_INSET,
    top: TAB_INSET,
    bottom: TAB_INSET,
    borderRadius: 10,
    backgroundColor: "#FFFFFF",
  },
  tab: { flex: 1, minHeight: 36, borderRadius: 10, alignItems: "center", justifyContent: "center", zIndex: 1 },
  tabLabel: { color: "#64748B", fontSize: FONT_SIZE.px12, lineHeight: 15, fontWeight: "600" },
  tabLabelActive: { color: "#0F172A", fontWeight: "700" },
  companyHint: { color: "#475569" },
  info: { color: "#64748B" },
  error: { color: "#B42318" },
  list: { gap: 10 },
  pressable: { borderRadius: 24 },
  pressablePressed: { transform: [{ scale: 0.995 }], opacity: 0.94 },
  card: {
    flexDirection: "row",
    flex: 1,
    backgroundColor: "#FFFFFF",
    borderWidth: 0,
    borderRadius: 24,
    paddingVertical: 12,
    paddingHorizontal: 12,
    ...softCardShadow,
  },
  cardBody: { flex: 1, gap: 8 },
  cardDone: { backgroundColor: "#F2F4F7" },
  cardTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 8 },
  statusBadges: { flexDirection: "row", alignItems: "center", gap: 6 },
  client: { color: "#0F172A" },
  clientDone: { color: "#667085" },
  badge: {
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  badgeDone: {},
  badgeText: { fontSize: FONT_SIZE.px11, lineHeight: 14 },
  badgeTextDone: { color: "#667085" },
  lateBadge: {
    borderWidth: 1,
    borderColor: "rgba(239,68,68,0.45)",
    backgroundColor: "rgba(239,68,68,0.12)",
    borderRadius: 999,
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  lateBadgeText: { color: "#DC2626", fontSize: FONT_SIZE.px11, lineHeight: 14, fontWeight: "700" },
  when: { color: "#0F172A" },
  whenDone: { color: "#7C8592" },
  routeRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  route: { color: "#334155", flex: 1 },
  routeDone: { color: "#7C8592" },
  metaAndHintRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 8 },
  metaRow: { flexDirection: "row", gap: 12, flexShrink: 1 },
  metaItem: { flexDirection: "row", alignItems: "center", gap: 4 },
  metaText: { color: "#475569" },
  driverLine: { color: "#163A34", fontWeight: "600" },
  driverLineDone: { color: "#667085" },
  expandHint: { flexDirection: "row", alignItems: "center", gap: 4, marginLeft: 8, flexShrink: 0 },
  expandHintText: { color: "#64748B" },
  expandAnimatedWrap: { overflow: "hidden" },
  expandArea: {
    marginTop: 4,
    borderTopWidth: 1,
    borderTopColor: "rgba(148,163,184,0.2)",
    paddingTop: 10,
    gap: 6,
  },
  sectionLabel: { color: "#64748B", textTransform: "uppercase", letterSpacing: 0.3 },
  sectionValue: { color: "#0F172A" },
  detailRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  detailRowText: { flex: 1 },
  detailIconButton: {
    width: 30,
    height: 30,
    borderRadius: 15,
    borderWidth: 1,
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
  },
  sectionGap: { marginTop: 6 },
  timelineRow: { flexDirection: "row", gap: 10, marginTop: 2 },
  timelineTrack: { width: 14, alignItems: "center" },
  dotStart: { width: 8, height: 8, borderRadius: 999, backgroundColor: "#14B8A6", marginTop: 4 },
  trackLine: { width: 2, flex: 1, minHeight: 26, backgroundColor: "#CBD5E1", marginVertical: 4 },
  dotEnd: { width: 9, height: 9, borderRadius: 999, borderWidth: 2, borderColor: "#3B82F6", backgroundColor: "#FFFFFF", marginBottom: 2 },
  timelineTexts: { flex: 1 },
  timelineEta: { color: "#64748B", marginTop: 2 },
  actionsRow: { flexDirection: "row", gap: 8, marginTop: 8 },
  actionPrimary: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 5,
    borderRadius: 999,
    backgroundColor: "#0F766E",
    paddingHorizontal: 12,
    paddingVertical: 9,
    flex: 1,
  },
  actionPrimaryText: { color: "#FFFFFF" },
  actionSecondary: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 5,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.12)",
    backgroundColor: "#FFFFFF",
    paddingHorizontal: 10,
    paddingVertical: 9,
    flex: 1,
  },
  actionSecondaryText: { color: "#0F172A" },
  actionDisabled: { opacity: 0.45 },
  actionPressed: { transform: [{ scale: 0.985 }], opacity: 0.9 },
  sectionBlock: { gap: 8 },
  sectionHeader: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  sectionTitle: { letterSpacing: 0.2 },
  sectionCountPill: { minWidth: 28, borderRadius: 999, paddingHorizontal: 8, paddingVertical: 3, alignItems: "center" },
  sectionCountText: { fontWeight: "700" },
  sectionList: { gap: 10 },
});

