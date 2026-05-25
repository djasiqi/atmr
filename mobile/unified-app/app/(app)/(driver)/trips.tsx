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
import { missionActiveCardShadow } from "../../../src/features/driver/theme/driverDashboardTheme";
import { getCallablePhoneFromMission, openNavigation, safeCall } from "../../../src/features/driver/utils/missionContact";
import { createShadow } from "../../../src/styles/shadowStyles";
import { FONT_SIZE } from "../../../src/design/responsive/typographyTokens";

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

function isUndefinedMissionTime(raw: unknown): boolean {
  if (typeof raw !== "string" || raw.trim().length === 0) return true;
  const ts = Date.parse(raw);
  if (!Number.isFinite(ts)) return true;
  const date = new Date(ts);
  return date.getHours() === 0 && date.getMinutes() === 0;
}

function missionSortTs(mission: DriverMission): number {
  if (isUndefinedMissionTime(mission.scheduled_time)) return Number.MAX_SAFE_INTEGER;
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

function whenLabel(raw: unknown): string {
  if (isUndefinedMissionTime(raw)) return "Heure à definir";
  const ts = Date.parse(String(raw ?? ""));
  if (!Number.isFinite(ts)) return "Heure à definir";
  return new Date(ts).toLocaleString("fr-CH", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
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
    .replace(/^Foyer de jour pour personnes âgées\s+/i, "Foyer de jour ")
    .replace(/\bAux\s+/i, "")
    .replace(/\s{2,}/g, " ")
    .trim();
}

function isLateMission(mission: DriverMission, nowTs: number): boolean {
  const bucket = missionStatusBucket(String(mission.status ?? ""));
  if (bucket === "done") return false;
  if (isUndefinedMissionTime(mission.scheduled_time)) return false;
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
    routeMetrics.durationLabel !== "—" ? routeMetrics.durationLabel : compactEta(mission);
  const distanceText =
    routeMetrics.distanceLabel !== "—" ? routeMetrics.distanceLabel : compactDistance(mission);
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
            {whenLabel(mission.scheduled_time)}
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
            {pickupCompact} {" → "} {destinationCompact}
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
    statusTransition.mutate({ missionId: mission.id, targetStatus: target });
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

/*
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Animated,
  LayoutAnimation,
  Platform,
  Pressable,
  StyleSheet,
  UIManager,
  View,
  useWindowDimensions,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import {
  useDriverCompanyBookingsTodayQuery,
  useDriverMissionsListFocusResync,
  useDriverMissionsQuery,
} from "../../../src/features/driver/hooks";
import type { DriverMission } from "../../../src/features/driver/types";
import { getDriverStatusUx, normalizeDriverMissionStatus } from "../../../src/features/driver/statusDictionary";
import { AppText, Screen } from "../../../src/design/responsive";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import { getCallablePhoneFromMission, openNavigation, safeCall } from "../../../src/features/driver/utils/missionContact";

const BG = "#F4F7F9";
const PAD = 16;
const TAB_COUNT = 2;
type TopTab = "mine" | "company";

if (Platform.OS === "android" && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

function isTodayMission(m: DriverMission, now: Date): boolean {
  const key = normalizeDriverMissionStatus(m.status);
  if (key === "EN_ROUTE" || key === "ARRIVED" || key === "IN_PROGRESS") return true;
  if (key === "CANCELLED" || key === "NO_SHOW" || key === "FAILED" || key === "REASSIGNED") return false;
  const raw = m.scheduled_time;
  if (!raw || !String(raw).trim()) return true;
  const t = Date.parse(String(raw));
  if (!Number.isFinite(t)) return true;
  const d = new Date(t);
  return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth() && d.getDate() === now.getDate();
}

function undefinedTime(raw: unknown): boolean {
  if (typeof raw !== "string" || raw.trim().length === 0) return true;
  const t = Date.parse(raw);
  if (!Number.isFinite(t)) return true;
  const d = new Date(t);
  return d.getHours() === 0 && d.getMinutes() === 0;
}

function missionSortTs(m: DriverMission): number {
  if (undefinedTime(m.scheduled_time)) return Number.MAX_SAFE_INTEGER;
  const t = Date.parse(String(m.scheduled_time));
  return Number.isFinite(t) ? t : Number.MAX_SAFE_INTEGER;
}

function sortMissions(a: DriverMission, b: DriverMission): number {
  const delta = missionSortTs(a) - missionSortTs(b);
  if (delta !== 0) return delta;
  return a.id - b.id;
}

function whenLabel(raw: unknown): string {
  if (undefinedTime(raw)) return "Heure à définir";
  const t = Date.parse(String(raw));
  if (!Number.isFinite(t)) return "Heure à définir";
  return new Date(t).toLocaleString("fr-CH", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function todayLabel(d: Date): string {
  return d.toLocaleDateString("fr-CH", { weekday: "long", day: "2-digit", month: "long" });
}

function missionName(m: DriverMission): string {
  const direct = typeof m.client_name === "string" ? m.client_name.trim() : "";
  if (direct) return direct;
  const nested = m.client as { full_name?: unknown } | null | undefined;
  if (nested?.full_name && String(nested.full_name).trim()) return String(nested.full_name).trim();
  return `Course #${m.id}`;
}

function missionNote(m: DriverMission): string {
  const fields = [m.notes, m.special_instructions, m.comment, m.comments];
  for (const f of fields) if (typeof f === "string" && f.trim()) return f.trim();
  return "Aucune note";
}

function bucket(status: string): "todo" | "progress" | "done" {
  const key = normalizeDriverMissionStatus(status);
  if (key === "COMPLETED") return "done";
  if (key === "EN_ROUTE" || key === "ARRIVED" || key === "IN_PROGRESS") return "progress";
  return "todo";
}

function tone(status: string) {
  const key = normalizeDriverMissionStatus(status);
  switch (key) {
    case "ASSIGNED":
      return { bg: "rgba(20,184,166,0.12)", border: "rgba(20,184,166,0.35)", text: "#0F766E" };
    case "EN_ROUTE":
      return { bg: "rgba(59,130,246,0.12)", border: "rgba(59,130,246,0.35)", text: "#1D4ED8" };
    case "ARRIVED":
      return { bg: "rgba(245,158,11,0.12)", border: "rgba(245,158,11,0.35)", text: "#B45309" };
    case "IN_PROGRESS":
      return { bg: "rgba(139,92,246,0.12)", border: "rgba(139,92,246,0.35)", text: "#6D28D9" };
    case "COMPLETED":
      return { bg: "rgba(148,163,184,0.16)", border: "rgba(148,163,184,0.35)", text: "#475569" };
    default:
      return { bg: "rgba(239,68,68,0.12)", border: "rgba(239,68,68,0.35)", text: "#B91C1C" };
  }
}

function MissionAccordion({
  mission,
  expanded,
  onToggle,
  allowExpand,
}: {
  mission: DriverMission;
  expanded: boolean;
  onToggle?: () => void;
  allowExpand: boolean;
}) {
  const ux = getDriverStatusUx(String(mission.status ?? ""));
  const sTone = tone(String(mission.status ?? ""));
  const pickup = mission.pickup_location?.trim() || "Adresse départ non définie";
  const dropoff = mission.dropoff_location?.trim() || "Adresse destination non définie";
  const canCall = Boolean(getCallablePhoneFromMission(mission));

  const content = (
    <View style={styles.cardInner}>
      <View style={styles.rowTop}>
        <AppText style={styles.when}>{whenLabel(mission.scheduled_time)}</AppText>
        <View style={[styles.statusBadge, { backgroundColor: sTone.bg, borderColor: sTone.border }]}>
          <AppText style={[styles.statusBadgeText, { color: sTone.text }]} numberOfLines={1}>
            {ux.label}
          </AppText>
        </View>
      </View>
      <AppText style={styles.client} numberOfLines={1}>
        {missionName(mission)}
      </AppText>
      <AppText style={styles.route} numberOfLines={1}>
        {pickup} -> {dropoff}
      </AppText>

      {allowExpand ? (
        <View style={styles.hintRow}>
          <Ionicons name={expanded ? "chevron-up-outline" : "chevron-down-outline"} size={14} color="#334155" />
          <AppText style={styles.hintText}>{expanded ? "Réduire" : "Voir plus"}</AppText>
        </View>
      ) : null}

      {allowExpand && expanded ? (
        <View style={styles.expandBlock}>
          <View style={styles.timelineRow}>
            <View style={styles.timelineCol}>
              <View style={styles.dotStart} />
              <View style={styles.timelineLine} />
              <View style={styles.dotEnd} />
            </View>
            <View style={styles.timelineTextCol}>
              <AppText style={styles.sectionLabel}>PRISE EN CHARGE</AppText>
              <AppText style={styles.sectionValue}>{pickup}</AppText>
              <AppText style={[styles.sectionLabel, styles.sectionOffset]}>DESTINATION</AppText>
              <AppText style={styles.sectionValue}>{dropoff}</AppText>
            </View>
          </View>
          <AppText style={[styles.sectionLabel, styles.sectionOffset]}>NOTES</AppText>
          <AppText style={styles.sectionValue}>{missionNote(mission)}</AppText>
          <View style={styles.actions}>
            <Pressable style={styles.actionPrimary}>
              <Ionicons name="play-outline" size={14} color="#fff" />
              <AppText style={styles.actionPrimaryText}>Démarrer</AppText>
            </Pressable>
            <Pressable
              disabled={!canCall}
              style={[styles.actionSecondary, !canCall && styles.disabled]}
              onPress={() => {
                const phone = getCallablePhoneFromMission(mission);
                if (!phone) return;
                void safeCall(phone);
              }}
            >
              <Ionicons name="call-outline" size={14} color="#163A34" />
              <AppText style={styles.actionSecondaryText}>Appeler</AppText>
            </Pressable>
            <Pressable style={styles.actionSecondary} onPress={() => void openNavigation(dropoff)}>
              <Ionicons name="navigate-outline" size={14} color="#163A34" />
              <AppText style={styles.actionSecondaryText}>Itinéraire</AppText>
            </Pressable>
          </View>
        </View>
      ) : null}
    </View>
  );

  if (!onToggle) return <View style={styles.card}>{content}</View>;
  return (
    <Pressable onPress={onToggle} style={({ pressed }) => [styles.card, pressed && styles.cardPressed]}>
      {content}
    </Pressable>
  );
}

/*
export default function DriverTripsScreen() {
  const { width } = useWindowDimensions();
  const compact = width < 380;
  const qMine = useDriverMissionsQuery();
  const qCompany = useDriverCompanyBookingsTodayQuery();
  useDriverMissionsListFocusResync();

  const [tab, setTab] = useState<TopTab>("mine");
  const [expandedMap, setExpandedMap] = useState<Record<string, boolean>>({});
  const [tabsW, setTabsW] = useState(0);
  const indicator = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.spring(indicator, { toValue: tab === "mine" ? 0 : 1, useNativeDriver: true, speed: 16, bounciness: 7 }).start();
  }, [indicator, tab]);

  const mine = useMemo(() => {
    const now = new Date();
    return (qMine.data ?? []).filter((m) => isTodayMission(m, now)).sort(sortMissions);
  }, [qMine.data]);
  const company = useMemo(() => {
    const now = new Date();
    return (qCompany.data ?? []).filter((m) => isTodayMission(m, now)).sort(sortMissions);
  }, [qCompany.data]);

  const shown = tab === "mine" ? mine : company;
  const stats = useMemo(() => {
    let todo = 0;
    let progress = 0;
    let done = 0;
    for (const m of shown) {
      const b = bucket(String(m.status ?? ""));
      if (b === "done") done += 1;
      else if (b === "progress") progress += 1;
      else todo += 1;
    }
    return { todo, progress, done };
  }, [shown]);

  const tx = indicator.interpolate({ inputRange: [0, 1], outputRange: [0, Math.max(0, tabsW / TAB_COUNT)] });

  function toggle(id: number) {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    const key = `${tab}:${id}`;
    setExpandedMap((p) => ({ ...p, [key]: !p[key] }));
  }

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          backgroundColor={BG}
          withHorizontalPadding={false}
          extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
          contentContainerStyle={[styles.page, compact && styles.pageCompact]}
        >
          <View style={styles.header}>
            <View>
              <AppText style={styles.headerTitle}>Courses du jour</AppText>
              <AppText style={styles.headerDate}>{todayLabel(new Date())}</AppText>
            </View>
            <Pressable style={styles.filterBtn}>
              <Ionicons name="options-outline" size={18} color="#0F172A" />
            </Pressable>
          </View>

          <View style={styles.statsRow}>
            <View style={styles.statsCard}>
              <AppText style={styles.statsValue}>{stats.todo}</AppText>
              <AppText style={styles.statsLabel}>À effectuer</AppText>
            </View>
            <View style={styles.statsCard}>
              <AppText style={styles.statsValue}>{stats.progress}</AppText>
              <AppText style={styles.statsLabel}>En cours</AppText>
            </View>
            <View style={styles.statsCard}>
              <AppText style={styles.statsValue}>{stats.done}</AppText>
              <AppText style={styles.statsLabel}>Terminées</AppText>
            </View>
          </View>

          <View style={styles.tabsWrap} onLayout={(e) => setTabsW(e.nativeEvent.layout.width)}>
            {tabsW > 0 ? (
              <Animated.View style={[styles.tabIndicator, { width: tabsW / TAB_COUNT, transform: [{ translateX: tx }] }]} />
            ) : null}
            <Pressable style={styles.tabBtn} onPress={() => setTab("mine")}>
              <AppText style={[styles.tabText, tab === "mine" && styles.tabTextActive]}>Mes courses</AppText>
            </Pressable>
            <Pressable style={styles.tabBtn} onPress={() => setTab("company")}>
              <AppText style={[styles.tabText, tab === "company" && styles.tabTextActive]}>Entreprise (jour)</AppText>
            </Pressable>
          </View>

          {tab === "mine" && qMine.isLoading ? <AppText style={styles.info}>Chargement des courses…</AppText> : null}
          {tab === "company" && qCompany.isLoading ? <AppText style={styles.info}>Chargement du planning entreprise…</AppText> : null}

          <View style={styles.list}>
            {shown.map((m) => {
              const key = `${tab}:${m.id}`;
              return (
                <MissionAccordion
                  key={key}
                  mission={m}
                  expanded={Boolean(expandedMap[key])}
                  onToggle={tab === "mine" ? () => toggle(m.id) : undefined}
                  allowExpand={tab === "mine"}
                />
              );
            })}
          </View>

          {!qMine.isLoading && tab === "mine" && shown.length === 0 ? <AppText style={styles.info}>Aucune course du jour.</AppText> : null}
          {!qCompany.isLoading && tab === "company" && shown.length === 0 ? (
            <AppText style={styles.info}>Aucune course entreprise du jour.</AppText>
          ) : null}
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: { paddingHorizontal: PAD, paddingTop: PAD, paddingBottom: 24, gap: 12 },
  pageCompact: { paddingHorizontal: 12, paddingTop: 12, gap: 10 },
  header: {
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 14,
    backgroundColor: "#ECF3F6",
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.06)",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  headerTitle: { color: "#0F172A", fontSize: FONT_SIZE.px24, lineHeight: 28, fontWeight: "800" },
  headerDate: { color: "#475569", fontSize: FONT_SIZE.px13, lineHeight: 18, marginTop: 2, textTransform: "capitalize" },
  filterBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.08)",
    backgroundColor: "rgba(255,255,255,0.72)",
    alignItems: "center",
    justifyContent: "center",
  },
  statsRow: { flexDirection: "row", gap: 8 },
  statsCard: {
    flex: 1,
    backgroundColor: "#fff",
    borderRadius: 18,
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.05)",
  },
  statsValue: { color: "#0F172A", fontSize: FONT_SIZE.px22, lineHeight: 24, fontWeight: "800" },
  statsLabel: { color: "#64748B", fontSize: FONT_SIZE.px12, lineHeight: 16, fontWeight: "600", marginTop: 2 },
  tabsWrap: {
    position: "relative",
    flexDirection: "row",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.08)",
    backgroundColor: "rgba(255,255,255,0.86)",
    overflow: "hidden",
  },
  tabIndicator: { position: "absolute", top: 0, bottom: 0, left: 0, backgroundColor: "#0A8F7A", borderRadius: 12 },
  tabBtn: { flex: 1, minHeight: 42, alignItems: "center", justifyContent: "center", paddingHorizontal: 10 },
  tabText: { color: "#334155", fontSize: FONT_SIZE.px13, lineHeight: 16, fontWeight: "600" },
  tabTextActive: { color: "#fff" },
  list: { gap: 10 },
  card: {
    borderRadius: 20,
    backgroundColor: "#fff",
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.06)",
    shadowColor: "#0F172A",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.06,
    shadowRadius: 20,
    elevation: 2,
  },
  cardPressed: { opacity: 0.93 },
  cardInner: { padding: 14 },
  rowTop: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 8 },
  when: { color: "#0F172A", fontSize: FONT_SIZE.px13, lineHeight: 16, fontWeight: "700", flexShrink: 1 },
  statusBadge: { borderWidth: 1, borderRadius: 999, paddingHorizontal: 8, paddingVertical: 2, maxWidth: "52%" },
  statusBadgeText: { fontSize: FONT_SIZE.px11, lineHeight: 14, fontWeight: "700" },
  client: { marginTop: 8, color: "#111827", fontSize: FONT_SIZE.px16, lineHeight: 20, fontWeight: "700" },
  route: { marginTop: 4, color: "#334155", fontSize: FONT_SIZE.px13, lineHeight: 17, fontWeight: "500" },
  hintRow: { marginTop: 8, flexDirection: "row", alignItems: "center", gap: 4 },
  hintText: { color: "#334155", fontSize: FONT_SIZE.px12, lineHeight: 16, fontWeight: "600" },
  expandBlock: { marginTop: 10, paddingTop: 10, borderTopWidth: 1, borderTopColor: "rgba(148,163,184,0.3)", gap: 8 },
  timelineRow: { flexDirection: "row", gap: 10 },
  timelineCol: { width: 16, alignItems: "center", paddingTop: 2 },
  dotStart: { width: 8, height: 8, borderRadius: 4, backgroundColor: "#14B8A6" },
  timelineLine: { width: 2, flex: 1, minHeight: 28, backgroundColor: "rgba(148,163,184,0.55)", marginVertical: 3, borderRadius: 999 },
  dotEnd: { width: 10, height: 10, borderRadius: 5, borderWidth: 2, borderColor: "#0A8F7A", backgroundColor: "#fff" },
  timelineTextCol: { flex: 1, minWidth: 0 },
  sectionLabel: { color: "#64748B", fontSize: FONT_SIZE.px11, lineHeight: 14, fontWeight: "700", letterSpacing: 0.4 },
  sectionOffset: { marginTop: 8 },
  sectionValue: { color: "#0F172A", fontSize: FONT_SIZE.px13, lineHeight: 17, fontWeight: "500" },
  actions: { marginTop: 4, flexDirection: "row", gap: 8, flexWrap: "wrap" },
  actionPrimary: {
    minHeight: 34,
    borderRadius: 999,
    paddingHorizontal: 12,
    backgroundColor: "#0A8F7A",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
  },
  actionSecondary: {
    minHeight: 34,
    borderRadius: 999,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: "rgba(22,58,52,0.2)",
    backgroundColor: "#fff",
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
  },
  actionPrimaryText: { color: "#fff", fontSize: FONT_SIZE.px12, lineHeight: 15, fontWeight: "700" },
  actionSecondaryText: { color: "#163A34", fontSize: FONT_SIZE.px12, lineHeight: 15, fontWeight: "700" },
  disabled: { opacity: 0.45 },
  info: { color: "#64748B", fontSize: FONT_SIZE.px13, lineHeight: 18 },
  error: { color: "#B42318", fontSize: FONT_SIZE.px13, lineHeight: 18, fontWeight: "600" },
});
/*
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Animated,
  LayoutAnimation,
  Platform,
  Pressable,
  StyleSheet,
  UIManager,
  View,
  useWindowDimensions,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import {
  useDriverCompanyBookingsTodayQuery,
  useDriverMissionsListFocusResync,
  useDriverMissionsQuery,
} from "../../../src/features/driver/hooks";
import type { DriverMission } from "../../../src/features/driver/types";
import { getDriverStatusUx, normalizeDriverMissionStatus } from "../../../src/features/driver/statusDictionary";
import { AppText, Screen } from "../../../src/design/responsive";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import { getCallablePhoneFromMission, openNavigation, safeCall } from "../../../src/features/driver/utils/missionContact";

const PAGE_BG = "#F4F7F9";
const PAGE_PAD = 16;
const TAB_COUNT = 2;

if (Platform.OS === "android" && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

type TopTab = "mine" | "company";

function isMissionShownOnDeviceLocalDay(m: DriverMission, now: Date): boolean {
  const key = normalizeDriverMissionStatus(m.status);
  if (key === "EN_ROUTE" || key === "ARRIVED" || key === "IN_PROGRESS") return true;
  if (key === "CANCELLED" || key === "NO_SHOW" || key === "FAILED" || key === "REASSIGNED") return false;

  const raw = m.scheduled_time;
  if (raw == null || String(raw).trim() === "") return true;
  const t = Date.parse(String(raw));
  if (!Number.isFinite(t)) return true;
  const scheduled = new Date(t);
  return (
    scheduled.getFullYear() === now.getFullYear() &&
    scheduled.getMonth() === now.getMonth() &&
    scheduled.getDate() === now.getDate()
  );
}

function isUndefinedScheduledTime(raw: unknown): boolean {
  if (typeof raw !== "string" || raw.trim().length === 0) return true;
  const t = Date.parse(raw);
  if (!Number.isFinite(t)) return true;
  const d = new Date(t);
  return d.getHours() === 0 && d.getMinutes() === 0;
}

function getMissionSortTimestamp(m: DriverMission): number {
  const raw = m.scheduled_time;
  if (isUndefinedScheduledTime(raw)) return Number.MAX_SAFE_INTEGER;
  const t = Date.parse(String(raw));
  return Number.isFinite(t) ? t : Number.MAX_SAFE_INTEGER;
}

function sortMissionsForDay(a: DriverMission, b: DriverMission): number {
  const byTime = getMissionSortTimestamp(a) - getMissionSortTimestamp(b);
  if (byTime !== 0) return byTime;
  return a.id - b.id;
}

function formatHeaderDate(d: Date): string {
  return d.toLocaleDateString("fr-CH", { weekday: "long", day: "2-digit", month: "long" });
}

function formatMissionWhen(raw: unknown): string {
  if (isUndefinedScheduledTime(raw)) return "Heure à définir";
  const t = Date.parse(String(raw));
  if (!Number.isFinite(t)) return "Heure à définir";
  return new Date(t).toLocaleString("fr-CH", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getMissionClientName(mission: DriverMission): string {
  const direct = typeof mission.client_name === "string" ? mission.client_name.trim() : "";
  if (direct) return direct;
  const nested = mission.client as { full_name?: unknown } | null | undefined;
  if (nested?.full_name && String(nested.full_name).trim()) return String(nested.full_name).trim();
  return `Course #${mission.id}`;
}

function getNotes(mission: DriverMission): string {
  const candidates = [mission.notes, mission.special_instructions, mission.comment, mission.comments];
  for (const value of candidates) {
    if (typeof value === "string" && value.trim().length > 0) return value.trim();
  }
  return "Aucune note";
}

function missionStateBucket(status: string): "todo" | "inProgress" | "done" {
  const key = normalizeDriverMissionStatus(status);
  if (key === "COMPLETED") return "done";
  if (key === "EN_ROUTE" || key === "ARRIVED" || key === "IN_PROGRESS") return "inProgress";
  return "todo";
}

function getStatusTone(status: string) {
  const key = normalizeDriverMissionStatus(status);
  switch (key) {
    case "ASSIGNED":
      return { bg: "rgba(20,184,166,0.12)", border: "rgba(20,184,166,0.35)", text: "#0F766E" };
    case "EN_ROUTE":
      return { bg: "rgba(59,130,246,0.12)", border: "rgba(59,130,246,0.35)", text: "#1D4ED8" };
    case "ARRIVED":
      return { bg: "rgba(245,158,11,0.12)", border: "rgba(245,158,11,0.35)", text: "#B45309" };
    case "IN_PROGRESS":
      return { bg: "rgba(139,92,246,0.12)", border: "rgba(139,92,246,0.35)", text: "#6D28D9" };
    case "COMPLETED":
      return { bg: "rgba(148,163,184,0.16)", border: "rgba(148,163,184,0.35)", text: "#475569" };
    default:
      return { bg: "rgba(239,68,68,0.12)", border: "rgba(239,68,68,0.35)", text: "#B91C1C" };
  }
}

function MissionCard({
  mission,
  expanded,
  onToggle,
  compact,
  showExpanded,
}: {
  mission: DriverMission;
  expanded: boolean;
  onToggle?: () => void;
  compact: boolean;
  showExpanded: boolean;
}) {
  const statusUx = getDriverStatusUx(String(mission.status ?? ""));
  const tone = getStatusTone(String(mission.status ?? ""));
  const client = getMissionClientName(mission);
  const when = formatMissionWhen(mission.scheduled_time);
  const pickup = mission.pickup_location?.trim() || "Adresse de départ non définie";
  const destination = mission.dropoff_location?.trim() || "Adresse de destination non définie";
  const canCall = Boolean(getCallablePhoneFromMission(mission));

  const body = (
    <View style={styles.cardInner}>
      <View style={styles.rowTop}>
        <AppText style={styles.when}>{when}</AppText>
        <View style={[styles.statusBadge, { backgroundColor: tone.bg, borderColor: tone.border }]}>
          <AppText style={[styles.statusBadgeText, { color: tone.text }]} numberOfLines={1}>
            {statusUx.label}
          </AppText>
        </View>
      </View>
      <AppText style={styles.client} numberOfLines={1}>
        {client}
      </AppText>
      <AppText style={styles.route} numberOfLines={1}>
        {pickup} -> {destination}
      </AppText>

      {showExpanded && expanded ? (
        <View style={styles.expanded}>
          <View style={styles.timelineRow}>
            <View style={styles.timelineCol}>
              <View style={styles.dotPickup} />
              <View style={styles.timelineLine} />
              <View style={styles.dotDropoff} />
            </View>
            <View style={styles.timelineContent}>
              <AppText style={styles.sectionLabel}>PRISE EN CHARGE</AppText>
              <AppText style={styles.sectionValue}>{pickup}</AppText>
              <AppText style={[styles.sectionLabel, styles.sectionLabelOffset]}>DESTINATION</AppText>
              <AppText style={styles.sectionValue}>{destination}</AppText>
            </View>
          </View>

          <AppText style={[styles.sectionLabel, styles.sectionLabelOffset]}>NOTES</AppText>
          <AppText style={styles.sectionValue}>{getNotes(mission)}</AppText>

          <View style={styles.actions}>
            <Pressable
              style={({ pressed }) => [styles.actionPrimary, pressed && styles.actionPressed]}
              onPress={() => Alert.alert("Action", "Utilisez l'écran mission pour changer l'état.")}
            >
              <Ionicons name="play-outline" size={14} color="#FFFFFF" />
              <AppText style={styles.actionPrimaryText}>Démarrer</AppText>
            </Pressable>
            <Pressable
              disabled={!canCall}
              style={({ pressed }) => [
                styles.actionSecondary,
                !canCall && styles.actionDisabled,
                pressed && styles.actionPressed,
              ]}
              onPress={() => {
                const phone = getCallablePhoneFromMission(mission);
                if (!phone) return;
                void safeCall(phone);
              }}
            >
              <Ionicons name="call-outline" size={14} color="#163A34" />
              <AppText style={styles.actionSecondaryText}>Appeler</AppText>
            </Pressable>
            <Pressable
              style={({ pressed }) => [styles.actionSecondary, pressed && styles.actionPressed]}
              onPress={() => void openNavigation(destination)}
            >
              <Ionicons name="navigate-outline" size={14} color="#163A34" />
              <AppText style={styles.actionSecondaryText}>Itinéraire</AppText>
            </Pressable>
          </View>
        </View>
      ) : null}

      {showExpanded ? (
        <View style={styles.expandHintRow}>
          <Ionicons name={expanded ? "chevron-up-outline" : "chevron-down-outline"} size={14} color="#334155" />
          <AppText style={styles.expandHint}>{expanded ? "Réduire" : "Voir plus"}</AppText>
        </View>
      ) : null}
    </View>
  );

  if (!onToggle) {
    return <View style={[styles.card, compact && styles.cardCompact]}>{body}</View>;
  }

  return (
    <Pressable
      onPress={onToggle}
      style={({ pressed }) => [styles.card, compact && styles.cardCompact, pressed && styles.cardPressed]}
      accessibilityRole="button"
      accessibilityLabel={`Mission ${mission.id}`}
    >
      {body}
    </Pressable>
  );
}

/*
export default function DriverTripsScreen() {
  const { width } = useWindowDimensions();
  const compact = width < 380;
  const missionsQuery = useDriverMissionsQuery();
  const companyDayQuery = useDriverCompanyBookingsTodayQuery();
  useDriverMissionsListFocusResync();

  const [topTab, setTopTab] = useState<TopTab>("mine");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [tabsWidth, setTabsWidth] = useState(0);
  const indicator = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.spring(indicator, {
      toValue: topTab === "mine" ? 0 : 1,
      useNativeDriver: true,
      speed: 16,
      bounciness: 7,
    }).start();
  }, [indicator, topTab]);

  const myMissions = useMemo(() => {
    const now = new Date();
    return (missionsQuery.data ?? []).filter((m) => isMissionShownOnDeviceLocalDay(m, now)).sort(sortMissionsForDay);
  }, [missionsQuery.data]);

  const companyMissions = useMemo(() => {
    const now = new Date();
    return (companyDayQuery.data ?? []).filter((m) => isMissionShownOnDeviceLocalDay(m, now)).sort(sortMissionsForDay);
  }, [companyDayQuery.data]);

  const shown = topTab === "mine" ? myMissions : companyMissions;

  const summary = useMemo(() => {
    let todo = 0;
    let inProgress = 0;
    let done = 0;
    for (const mission of shown) {
      const bucket = missionStateBucket(String(mission.status ?? ""));
      if (bucket === "done") done += 1;
      else if (bucket === "inProgress") inProgress += 1;
      else todo += 1;
    }
    return { todo, inProgress, done };
  }, [shown]);

  function toggleMission(id: number) {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    const key = `${topTab}:${id}`;
    setExpanded((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  const translateX = indicator.interpolate({
    inputRange: [0, 1],
    outputRange: [0, Math.max(0, tabsWidth / TAB_COUNT)],
  });

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          backgroundColor={PAGE_BG}
          withHorizontalPadding={false}
          extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
          contentContainerStyle={[styles.page, compact && styles.pageCompact]}
        >
          <View style={styles.header}>
            <View>
              <AppText style={styles.headerTitle}>Courses du jour</AppText>
              <AppText style={styles.headerDate}>{formatHeaderDate(new Date())}</AppText>
            </View>
            <Pressable style={styles.filterBtn} accessibilityRole="button" accessibilityLabel="Filtrer">
              <Ionicons name="options-outline" size={18} color="#0F172A" />
            </Pressable>
          </View>

          <View style={styles.summaryRow}>
            <View style={styles.summaryCard}>
              <AppText style={styles.summaryValue}>{summary.todo}</AppText>
              <AppText style={styles.summaryLabel}>À effectuer</AppText>
            </View>
            <View style={styles.summaryCard}>
              <AppText style={styles.summaryValue}>{summary.inProgress}</AppText>
              <AppText style={styles.summaryLabel}>En cours</AppText>
            </View>
            <View style={styles.summaryCard}>
              <AppText style={styles.summaryValue}>{summary.done}</AppText>
              <AppText style={styles.summaryLabel}>Terminées</AppText>
            </View>
          </View>

          <View style={styles.tabs} onLayout={(e) => setTabsWidth(e.nativeEvent.layout.width)}>
            {tabsWidth > 0 ? (
              <Animated.View style={[styles.tabIndicator, { width: tabsWidth / TAB_COUNT, transform: [{ translateX }] }]} />
            ) : null}
            <Pressable style={styles.tabBtn} onPress={() => setTopTab("mine")}>
              <AppText style={[styles.tabText, topTab === "mine" && styles.tabTextActive]}>Mes courses</AppText>
            </Pressable>
            <Pressable style={styles.tabBtn} onPress={() => setTopTab("company")}>
              <AppText style={[styles.tabText, topTab === "company" && styles.tabTextActive]}>Entreprise (jour)</AppText>
            </Pressable>
          </View>

          {topTab === "mine" && missionsQuery.isLoading ? <AppText style={styles.info}>Chargement des courses…</AppText> : null}
          {topTab === "company" && companyDayQuery.isLoading ? (
            <AppText style={styles.info}>Chargement du planning entreprise…</AppText>
          ) : null}
          {topTab === "mine" && missionsQuery.error ? (
            <AppText style={styles.error}>
              {missionsQuery.error instanceof Error ? missionsQuery.error.message : "Erreur chargement courses."}
            </AppText>
          ) : null}
          {topTab === "company" && companyDayQuery.error ? (
            <AppText style={styles.error}>
              {companyDayQuery.error instanceof Error ? companyDayQuery.error.message : "Erreur chargement planning entreprise."}
            </AppText>
          ) : null}

          <View style={styles.stack}>
            {shown.map((mission) => {
              const key = `${topTab}:${mission.id}`;
              return (
                <MissionCard
                  key={key}
                  mission={mission}
                  expanded={Boolean(expanded[key])}
                  onToggle={topTab === "mine" ? () => toggleMission(mission.id) : undefined}
                  compact={compact}
                  showExpanded={topTab === "mine"}
                />
              );
            })}
          </View>

          {!missionsQuery.isLoading && topTab === "mine" && shown.length === 0 ? (
            <AppText style={styles.info}>Aucune course du jour sur cet appareil.</AppText>
          ) : null}
          {!companyDayQuery.isLoading && topTab === "company" && shown.length === 0 ? (
            <AppText style={styles.info}>Aucune course entreprise du jour pour ce fuseau local.</AppText>
          ) : null}
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: { paddingHorizontal: PAGE_PAD, paddingTop: PAGE_PAD, paddingBottom: 24, gap: 12 },
  pageCompact: { paddingHorizontal: 12, paddingTop: 12, gap: 10 },
  header: {
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 14,
    backgroundColor: "#ECF3F6",
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.06)",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  headerTitle: { color: "#0F172A", fontSize: FONT_SIZE.px24, lineHeight: 28, fontWeight: "800" },
  headerDate: { color: "#475569", fontSize: FONT_SIZE.px13, lineHeight: 18, marginTop: 2, textTransform: "capitalize" },
  filterBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.08)",
    backgroundColor: "rgba(255,255,255,0.72)",
    alignItems: "center",
    justifyContent: "center",
  },
  summaryRow: { flexDirection: "row", gap: 8 },
  summaryCard: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    borderRadius: 18,
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.05)",
    shadowColor: "#0F172A",
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.06,
    shadowRadius: 16,
    elevation: 2,
  },
  summaryValue: { color: "#0F172A", fontSize: FONT_SIZE.px22, lineHeight: 24, fontWeight: "800" },
  summaryLabel: { color: "#64748B", fontSize: FONT_SIZE.px12, lineHeight: 16, fontWeight: "600", marginTop: 2 },
  tabs: {
    position: "relative",
    flexDirection: "row",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.08)",
    backgroundColor: "rgba(255,255,255,0.86)",
    overflow: "hidden",
  },
  tabIndicator: { position: "absolute", left: 0, top: 0, bottom: 0, backgroundColor: "#0A8F7A", borderRadius: 12 },
  tabBtn: { flex: 1, minHeight: 42, alignItems: "center", justifyContent: "center", paddingHorizontal: 10 },
  tabText: { color: "#334155", fontSize: FONT_SIZE.px13, lineHeight: 16, fontWeight: "600" },
  tabTextActive: { color: "#FFFFFF" },
  stack: { gap: 10 },
  card: {
    borderRadius: 20,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.06)",
    shadowColor: "#0F172A",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.06,
    shadowRadius: 20,
    elevation: 2,
  },
  cardCompact: {},
  cardPressed: { opacity: 0.93 },
  cardInner: { padding: 14 },
  rowTop: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 8 },
  when: { color: "#0F172A", fontSize: FONT_SIZE.px13, lineHeight: 16, fontWeight: "700", flexShrink: 1 },
  statusBadge: { borderWidth: 1, borderRadius: 999, paddingHorizontal: 8, paddingVertical: 2, maxWidth: "52%" },
  statusBadgeText: { fontSize: FONT_SIZE.px11, lineHeight: 14, fontWeight: "700" },
  client: { marginTop: 8, color: "#111827", fontSize: FONT_SIZE.px16, lineHeight: 20, fontWeight: "700" },
  route: { marginTop: 4, color: "#334155", fontSize: FONT_SIZE.px13, lineHeight: 17, fontWeight: "500" },
  expanded: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: "rgba(148,163,184,0.3)",
    gap: 8,
  },
  timelineRow: { flexDirection: "row", alignItems: "stretch", gap: 10 },
  timelineCol: { width: 16, alignItems: "center", paddingTop: 2 },
  dotPickup: { width: 8, height: 8, borderRadius: 4, backgroundColor: "#14B8A6" },
  timelineLine: { width: 2, flex: 1, minHeight: 28, backgroundColor: "rgba(148,163,184,0.55)", marginVertical: 3, borderRadius: 999 },
  dotDropoff: { width: 10, height: 10, borderRadius: 5, borderWidth: 2, borderColor: "#0A8F7A", backgroundColor: "#FFFFFF" },
  timelineContent: { flex: 1, minWidth: 0 },
  sectionLabel: { color: "#64748B", fontSize: FONT_SIZE.px11, lineHeight: 14, fontWeight: "700", letterSpacing: 0.4 },
  sectionLabelOffset: { marginTop: 8 },
  sectionValue: { color: "#0F172A", fontSize: FONT_SIZE.px13, lineHeight: 17, fontWeight: "500" },
  actions: { marginTop: 4, flexDirection: "row", gap: 8, flexWrap: "wrap" },
  actionPrimary: {
    minHeight: 34,
    borderRadius: 999,
    paddingHorizontal: 12,
    backgroundColor: "#0A8F7A",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 6,
  },
  actionSecondary: {
    minHeight: 34,
    borderRadius: 999,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: "rgba(22,58,52,0.2)",
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 6,
  },
  actionPrimaryText: { color: "#FFFFFF", fontSize: FONT_SIZE.px12, lineHeight: 15, fontWeight: "700" },
  actionSecondaryText: { color: "#163A34", fontSize: FONT_SIZE.px12, lineHeight: 15, fontWeight: "700" },
  actionDisabled: { opacity: 0.45 },
  actionPressed: { opacity: 0.85 },
  expandHintRow: { marginTop: 8, flexDirection: "row", alignItems: "center", gap: 4 },
  expandHint: { color: "#334155", fontSize: FONT_SIZE.px12, lineHeight: 16, fontWeight: "600" },
  info: { color: "#64748B", fontSize: FONT_SIZE.px13, lineHeight: 18 },
  error: { color: "#B42318", fontSize: FONT_SIZE.px13, lineHeight: 18, fontWeight: "600" },
});
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Animated,
  LayoutAnimation,
  Platform,
  Pressable,
  StyleSheet,
  UIManager,
  View,
  useWindowDimensions,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import {
  useDriverCompanyBookingsTodayQuery,
  useDriverMissionsListFocusResync,
  useDriverMissionsQuery,
} from "../../../src/features/driver/hooks";
import type { DriverMission } from "../../../src/features/driver/types";
import {
  getDriverStatusUx,
  normalizeDriverMissionStatus,
} from "../../../src/features/driver/statusDictionary";
import { AppText, Screen } from "../../../src/design/responsive";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import {
  getCallablePhoneFromMission,
  openNavigation,
  safeCall,
} from "../../../src/features/driver/utils/missionContact";

const PAGE_BG = "#F4F7F9";
const PAGE_PAD = 16;
const TAB_COUNT = 2;

if (Platform.OS === "android" && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

type TopTab = "mine" | "company";

function isMissionShownOnDeviceLocalDay(m: DriverMission, now: Date): boolean {
  const key = normalizeDriverMissionStatus(m.status);
  if (key === "EN_ROUTE" || key === "ARRIVED" || key === "IN_PROGRESS") return true;
  if (key === "CANCELLED" || key === "NO_SHOW" || key === "FAILED" || key === "REASSIGNED") return false;

  const raw = m.scheduled_time;
  if (raw == null || String(raw).trim() === "") return true;
  const t = Date.parse(String(raw));
  if (!Number.isFinite(t)) return true;
  const scheduled = new Date(t);

  return (
    scheduled.getFullYear() === now.getFullYear() &&
    scheduled.getMonth() === now.getMonth() &&
    scheduled.getDate() === now.getDate()
  );
}

function isUndefinedScheduledTime(raw: unknown): boolean {
  if (typeof raw !== "string" || raw.trim().length === 0) return true;
  const t = Date.parse(raw);
  if (!Number.isFinite(t)) return true;
  const d = new Date(t);
  return d.getHours() === 0 && d.getMinutes() === 0;
}

function getMissionSortTimestamp(m: DriverMission): number {
  const raw = m.scheduled_time;
  if (isUndefinedScheduledTime(raw)) return Number.MAX_SAFE_INTEGER;
  const t = Date.parse(String(raw));
  return Number.isFinite(t) ? t : Number.MAX_SAFE_INTEGER;
}

function sortMissionsForDay(a: DriverMission, b: DriverMission): number {
  const byTime = getMissionSortTimestamp(a) - getMissionSortTimestamp(b);
  if (byTime !== 0) return byTime;
  return a.id - b.id;
}

function formatHeaderDate(d: Date): string {
  return d.toLocaleDateString("fr-CH", {
    weekday: "long",
    day: "2-digit",
    month: "long",
  });
}

function formatMissionWhen(raw: unknown): string {
  if (isUndefinedScheduledTime(raw)) return "Heure à définir";
  const t = Date.parse(String(raw));
  if (!Number.isFinite(t)) return "Heure à définir";
  return new Date(t).toLocaleString("fr-CH", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getNotes(mission: DriverMission): string {
  const values = [mission.notes, mission.special_instructions, mission.comment, mission.comments];
  for (const value of values) {
    if (typeof value === "string" && value.trim().length > 0) return value.trim();
  }
  return "Aucune note";
}

function getStatusTone(status: string) {
  const key = normalizeDriverMissionStatus(status);
  switch (key) {
    case "ASSIGNED":
      return { bg: "rgba(20,184,166,0.12)", border: "rgba(20,184,166,0.35)", text: "#0F766E" };
    case "EN_ROUTE":
      return { bg: "rgba(59,130,246,0.12)", border: "rgba(59,130,246,0.35)", text: "#1D4ED8" };
    case "ARRIVED":
      return { bg: "rgba(245,158,11,0.12)", border: "rgba(245,158,11,0.35)", text: "#B45309" };
    case "IN_PROGRESS":
      return { bg: "rgba(139,92,246,0.12)", border: "rgba(139,92,246,0.35)", text: "#6D28D9" };
    case "COMPLETED":
      return { bg: "rgba(148,163,184,0.16)", border: "rgba(148,163,184,0.35)", text: "#475569" };
    default:
      return { bg: "rgba(239,68,68,0.12)", border: "rgba(239,68,68,0.35)", text: "#B91C1C" };
  }
}

function missionStateBucket(status: string): "todo" | "inProgress" | "done" {
  const key = normalizeDriverMissionStatus(status);
  if (key === "COMPLETED") return "done";
  if (key === "EN_ROUTE" || key === "ARRIVED" || key === "IN_PROGRESS") return "inProgress";
  return "todo";
}

function getMissionClientName(mission: DriverMission): string {
  const direct = typeof mission.client_name === "string" ? mission.client_name.trim() : "";
  if (direct) return direct;
  const nested = mission.client as { full_name?: unknown } | null | undefined;
  if (nested?.full_name && String(nested.full_name).trim()) return String(nested.full_name).trim();
  return `Course #${mission.id}`;
}

function getMissionEtaDistance(mission: DriverMission): { eta: string; distance: string } {
  const etaMin =
    typeof mission.eta_minutes === "number" && Number.isFinite(mission.eta_minutes)
      ? mission.eta_minutes
      : null;
  const distanceKm =
    typeof mission.distance_km === "number" && Number.isFinite(mission.distance_km)
      ? mission.distance_km
      : null;

  return {
    eta: etaMin != null ? `${Math.max(0, Math.round(etaMin))} min` : "--",
    distance: distanceKm != null ? `${distanceKm.toFixed(1)} km` : "--",
  };
}

function MissionAccordionCard({
  mission,
  expanded,
  onToggle,
  compact = false,
  allowActions = false,
}: {
  mission: DriverMission;
  expanded: boolean;
  onToggle: () => void;
  compact?: boolean;
  allowActions?: boolean;
}) {
  const statusUx = getDriverStatusUx(String(mission.status ?? ""));
  const statusTone = getStatusTone(String(mission.status ?? ""));
  const when = formatMissionWhen(mission.scheduled_time);
  const client = getMissionClientName(mission);
  const pickup = mission.pickup_location?.trim() || "Adresse de départ non définie";
  const destination = mission.dropoff_location?.trim() || "Adresse de destination non définie";
  const notes = getNotes(mission);
  const { eta, distance } = getMissionEtaDistance(mission);
  const canCall = Boolean(getCallablePhoneFromMission(mission));

  return (
    <Pressable
      onPress={onToggle}
      style={({ pressed }) => [
        styles.missionCard,
        compact && styles.missionCardCompact,
        pressed && styles.missionCardPressed,
      ]}
      accessibilityRole="button"
      accessibilityLabel={`Mission ${mission.id}, ${expanded ? "replier" : "déplier"}`}
    >
      <View style={styles.compactTopRow}>
        <AppText style={styles.timeText}>{when}</AppText>
        <View style={[styles.statusPill, { backgroundColor: statusTone.bg, borderColor: statusTone.border }]}>
          <AppText style={[styles.statusPillText, { color: statusTone.text }]} numberOfLines={1}>
            {statusUx.label}
          </AppText>
        </View>
      </View>

      <AppText style={styles.clientName} numberOfLines={1}>
        {client}
      </AppText>
      <AppText style={styles.routeText} numberOfLines={1}>
        {`${pickup} -> ${destination}`}
      </AppText>

      <View style={styles.metaRow}>
        <View style={styles.metaItem}>
          <Ionicons name="car-outline" size={13} color="#475569" />
          <AppText style={styles.metaText}>{eta}</AppText>
        </View>
        <View style={styles.metaItem}>
          <Ionicons name="navigate-outline" size={13} color="#475569" />
          <AppText style={styles.metaText}>{distance}</AppText>
        </View>
        <View style={styles.metaItem}>
          <Ionicons name={expanded ? "chevron-up-outline" : "chevron-down-outline"} size={14} color="#334155" />
        </View>
      </View>

      {expanded ? (
        <View style={styles.expandedBlock}>
          <View style={styles.timelineRow}>
            <View style={styles.timelineCol}>
              <View style={styles.timelineDot} />
              <View style={styles.timelineLine} />
              <View style={styles.timelineDotDestination} />
            </View>
            <View style={styles.timelineContent}>
              <AppText style={styles.sectionLabel}>PRISE EN CHARGE</AppText>
              <AppText style={styles.sectionValue}>{pickup}</AppText>
              <AppText style={styles.sectionEta}>{eta}</AppText>
              <AppText style={[styles.sectionLabel, styles.sectionSpacing]}>DESTINATION</AppText>
              <AppText style={styles.sectionValue}>{destination}</AppText>
            </View>
          </View>

          <AppText style={[styles.sectionLabel, styles.sectionSpacing]}>NOTES</AppText>
          <AppText style={styles.sectionValue}>{notes}</AppText>

          {allowActions ? (
            <View style={styles.actionsRow}>
              <Pressable
                style={({ pressed }) => [styles.actionPillPrimary, pressed && styles.actionPressed]}
                onPress={() => Alert.alert("Action rapide", "Passez la mission en cours depuis l'écran mission.")}
              >
                <Ionicons name="play-outline" size={14} color="#FFFFFF" />
                <AppText style={styles.actionLabelPrimary}>Démarrer</AppText>
              </Pressable>
              <Pressable
                style={({ pressed }) => [
                  styles.actionPillSecondary,
                  !canCall && styles.actionDisabled,
                  pressed && styles.actionPressed,
                ]}
                disabled={!canCall}
                onPress={() => {
                  const phone = getCallablePhoneFromMission(mission);
                  if (!phone) return;
                  void safeCall(phone);
                }}
              >
                <Ionicons name="call-outline" size={14} color="#163A34" />
                <AppText style={styles.actionLabelSecondary}>Appeler</AppText>
              </Pressable>
              <Pressable
                style={({ pressed }) => [styles.actionPillSecondary, pressed && styles.actionPressed]}
                onPress={() => void openNavigation(destination)}
              >
                <Ionicons name="navigate-outline" size={14} color="#163A34" />
                <AppText style={styles.actionLabelSecondary}>Itinéraire</AppText>
              </Pressable>
            </View>
          ) : null}
        </View>
      ) : null}
    </Pressable>
  );
}

export default function DriverTripsScreen() {
  const { width } = useWindowDimensions();
  const isCompactMobile = width < 380;
  const missionsQuery = useDriverMissionsQuery();
  const companyDayQuery = useDriverCompanyBookingsTodayQuery();
  useDriverMissionsListFocusResync();

  const [topTab, setTopTab] = useState<TopTab>("mine");
  const [expandedByTab, setExpandedByTab] = useState<Record<string, boolean>>({});
  const [tabBarWidth, setTabBarWidth] = useState(0);
  const tabIndicator = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.spring(tabIndicator, {
      toValue: topTab === "mine" ? 0 : 1,
      useNativeDriver: true,
      speed: 16,
      bounciness: 7,
    }).start();
  }, [tabIndicator, topTab]);

  const myDayMissions = useMemo(() => {
    const list = missionsQuery.data ?? [];
    const now = new Date();
    return list.filter((m) => isMissionShownOnDeviceLocalDay(m, now)).sort(sortMissionsForDay);
  }, [missionsQuery.data]);

  const companyDayMissions = useMemo(() => {
    const list = companyDayQuery.data ?? [];
    const now = new Date();
    return list.filter((m) => isMissionShownOnDeviceLocalDay(m, now)).sort(sortMissionsForDay);
  }, [companyDayQuery.data]);

  const shownMissions = topTab === "mine" ? myDayMissions : companyDayMissions;
  const summary = useMemo(() => {
    let todo = 0;
    let inProgress = 0;
    let done = 0;
    for (const mission of shownMissions) {
      const bucket = missionStateBucket(String(mission.status ?? ""));
      if (bucket === "done") done += 1;
      else if (bucket === "inProgress") inProgress += 1;
      else todo += 1;
    }
    return { todo, inProgress, done };
  }, [shownMissions]);

  const indicatorTranslate = tabIndicator.interpolate({
    inputRange: [0, 1],
    outputRange: [0, Math.max(0, tabBarWidth / TAB_COUNT)],
  });

  function toggleExpanded(id: number) {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    const key = `${topTab}:${id}`;
    setExpandedByTab((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          backgroundColor={PAGE_BG}
          withHorizontalPadding={false}
          extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
          contentContainerStyle={[styles.page, isCompactMobile && styles.pageCompact]}
        >
          <View style={styles.headerCard}>
            <View>
              <AppText style={styles.headerTitle}>Courses du jour</AppText>
              <AppText style={styles.headerSubtitle}>{formatHeaderDate(new Date())}</AppText>
            </View>
            <Pressable style={styles.filterBtn} accessibilityRole="button" accessibilityLabel="Filtrer les courses">
              <Ionicons name="options-outline" size={18} color="#0f172a" />
            </Pressable>
          </View>

          <View style={styles.summaryRow}>
            <View style={styles.summaryCard}>
              <AppText style={styles.summaryValue}>{summary.todo}</AppText>
              <AppText style={styles.summaryLabel}>À effectuer</AppText>
            </View>
            <View style={styles.summaryCard}>
              <AppText style={styles.summaryValue}>{summary.inProgress}</AppText>
              <AppText style={styles.summaryLabel}>En cours</AppText>
            </View>
            <View style={styles.summaryCard}>
              <AppText style={styles.summaryValue}>{summary.done}</AppText>
              <AppText style={styles.summaryLabel}>Terminées</AppText>
            </View>
          </View>

          <View style={styles.tabsWrap} onLayout={(e) => setTabBarWidth(e.nativeEvent.layout.width)}>
            {tabBarWidth > 0 ? (
              <Animated.View
                pointerEvents="none"
                style={[
                  styles.tabIndicator,
                  { width: tabBarWidth / TAB_COUNT, transform: [{ translateX: indicatorTranslate }] },
                ]}
              />
            ) : null}
            <Pressable style={styles.tabBtn} onPress={() => setTopTab("mine")}>
              <AppText style={[styles.tabLabel, topTab === "mine" && styles.tabLabelActive]}>Mes courses</AppText>
            </Pressable>
            <Pressable style={styles.tabBtn} onPress={() => setTopTab("company")}>
              <AppText style={[styles.tabLabel, topTab === "company" && styles.tabLabelActive]}>
                Entreprise (jour)
              </AppText>
            </Pressable>
          </View>

          {topTab === "mine" && missionsQuery.isLoading ? (
            <AppText style={styles.infoText}>Chargement des courses…</AppText>
          ) : null}
          {topTab === "company" && companyDayQuery.isLoading ? (
            <AppText style={styles.infoText}>Chargement du planning entreprise…</AppText>
          ) : null}
          {topTab === "mine" && missionsQuery.error ? (
            <AppText style={styles.errorText}>
              {missionsQuery.error instanceof Error ? missionsQuery.error.message : "Erreur chargement courses."}
            </AppText>
          ) : null}
          {topTab === "company" && companyDayQuery.error ? (
            <AppText style={styles.errorText}>
              {companyDayQuery.error instanceof Error
                ? companyDayQuery.error.message
                : "Erreur chargement planning entreprise."}
            </AppText>
          ) : null}

          <View style={styles.missionsStack}>
            {shownMissions.map((mission) => {
              const expanded = Boolean(expandedByTab[`${topTab}:${mission.id}`]);
              return (
                <MissionAccordionCard
                  key={`${topTab}-${mission.id}`}
                  mission={mission}
                  expanded={expanded}
                  onToggle={() => toggleExpanded(mission.id)}
                  compact={isCompactMobile}
                  allowActions={topTab === "mine"}
                />
              );
            })}
          </View>

          {!missionsQuery.isLoading && topTab === "mine" && shownMissions.length === 0 ? (
            <AppText style={styles.infoText}>
              Aucune course du jour sur cet appareil. Les missions hors jour ne sont pas affichées.
            </AppText>
          ) : null}
          {!companyDayQuery.isLoading && topTab === "company" && shownMissions.length === 0 ? (
            <AppText style={styles.infoText}>
              Aucune course entreprise du jour pour ce fuseau local.
            </AppText>
          ) : null}
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: {
    paddingHorizontal: PAGE_PAD,
    paddingTop: PAGE_PAD,
    paddingBottom: 24,
    gap: 12,
  },
  pageCompact: {
    paddingHorizontal: 12,
    paddingTop: 12,
    gap: 10,
  },
  headerCard: {
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 14,
    backgroundColor: "#ECF3F6",
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.06)",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  headerTitle: {
    color: "#0F172A",
    fontSize: FONT_SIZE.px24,
    lineHeight: 28,
    fontWeight: "800",
  },
  headerSubtitle: {
    color: "#475569",
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
    marginTop: 2,
    textTransform: "capitalize",
  },
  filterBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.08)",
    backgroundColor: "rgba(255,255,255,0.72)",
    alignItems: "center",
    justifyContent: "center",
  },
  summaryRow: {
    flexDirection: "row",
    gap: 8,
  },
  summaryCard: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    borderRadius: 18,
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.05)",
    shadowColor: "#0F172A",
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.06,
    shadowRadius: 16,
    elevation: 2,
  },
  summaryValue: {
    color: "#0F172A",
    fontSize: FONT_SIZE.px22,
    lineHeight: 24,
    fontWeight: "800",
  },
  summaryLabel: {
    color: "#64748B",
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    fontWeight: "600",
    marginTop: 2,
  },
  tabsWrap: {
    position: "relative",
    flexDirection: "row",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.08)",
    backgroundColor: "rgba(255,255,255,0.86)",
    overflow: "hidden",
  },
  tabIndicator: {
    position: "absolute",
    left: 0,
    top: 0,
    bottom: 0,
    backgroundColor: "#0A8F7A",
    borderRadius: 12,
  },
  tabBtn: {
    flex: 1,
    minHeight: 42,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 10,
  },
  tabLabel: {
    color: "#334155",
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    fontWeight: "600",
  },
  tabLabelActive: {
    color: "#FFFFFF",
  },
  missionsStack: {
    gap: 10,
  },
  missionCard: {
    borderRadius: 20,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.06)",
    padding: 14,
    shadowColor: "#0F172A",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.06,
    shadowRadius: 20,
    elevation: 2,
  },
  missionCardCompact: {
    padding: 12,
  },
  missionCardPressed: {
    opacity: 0.92,
    transform: [{ scale: 0.995 }],
  },
  compactTopRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  timeText: {
    color: "#0F172A",
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    fontWeight: "700",
    flexShrink: 1,
  },
  statusPill: {
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 8,
    paddingVertical: 2,
    maxWidth: "52%",
  },
  statusPillText: {
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
    fontWeight: "700",
  },
  clientName: {
    marginTop: 8,
    color: "#111827",
    fontSize: FONT_SIZE.px16,
    lineHeight: 20,
    fontWeight: "700",
  },
  routeText: {
    marginTop: 4,
    color: "#334155",
    fontSize: FONT_SIZE.px13,
    lineHeight: 17,
    fontWeight: "500",
  },
  metaRow: {
    marginTop: 8,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  metaItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
  },
  metaText: {
    color: "#475569",
    fontSize: FONT_SIZE.px12,
    lineHeight: 15,
    fontWeight: "600",
  },
  expandedBlock: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: "rgba(148,163,184,0.3)",
    gap: 8,
  },
  timelineRow: {
    flexDirection: "row",
    alignItems: "stretch",
    gap: 10,
  },
  timelineCol: {
    width: 16,
    alignItems: "center",
    paddingTop: 2,
  },
  timelineDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "#14B8A6",
  },
  timelineLine: {
    width: 2,
    flex: 1,
    minHeight: 28,
    backgroundColor: "rgba(148,163,184,0.55)",
    marginVertical: 3,
    borderRadius: 999,
  },
  timelineDotDestination: {
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: "#0A8F7A",
    backgroundColor: "#FFFFFF",
  },
  timelineContent: {
    flex: 1,
    minWidth: 0,
  },
  sectionLabel: {
    color: "#64748B",
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
    fontWeight: "700",
    letterSpacing: 0.4,
  },
  sectionValue: {
    color: "#0F172A",
    fontSize: FONT_SIZE.px13,
    lineHeight: 17,
    fontWeight: "500",
  },
  sectionEta: {
    color: "#475569",
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    marginTop: 2,
  },
  sectionSpacing: {
    marginTop: 8,
  },
  actionsRow: {
    marginTop: 4,
    flexDirection: "row",
    gap: 8,
    flexWrap: "wrap",
  },
  actionPillPrimary: {
    minHeight: 34,
    borderRadius: 999,
    paddingHorizontal: 12,
    backgroundColor: "#0A8F7A",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 6,
  },
  actionPillSecondary: {
    minHeight: 34,
    borderRadius: 999,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: "rgba(22,58,52,0.2)",
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 6,
  },
  actionLabelPrimary: {
    color: "#FFFFFF",
    fontSize: FONT_SIZE.px12,
    lineHeight: 15,
    fontWeight: "700",
  },
  actionLabelSecondary: {
    color: "#163A34",
    fontSize: FONT_SIZE.px12,
    lineHeight: 15,
    fontWeight: "700",
  },
  actionDisabled: {
    opacity: 0.45,
  },
  actionPressed: {
    opacity: 0.85,
  },
  infoText: {
    color: "#64748B",
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
  },
  errorText: {
    color: "#B42318",
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
    fontWeight: "600",
  },
});
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Animated,
  LayoutAnimation,
  Platform,
  Pressable,
  StyleSheet,
  UIManager,
  View,
  useWindowDimensions,
} from "react-native";
import { Ionicons } from "@expo/vector-icons";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import {
  useDriverCompanyBookingsTodayQuery,
  useDriverMissionsListFocusResync,
  useDriverMissionsQuery,
} from "../../../src/features/driver/hooks";
import type { DriverMission } from "../../../src/features/driver/types";
import {
  getDriverStatusUx,
  normalizeDriverMissionStatus,
} from "../../../src/features/driver/statusDictionary";
import { AppText, Screen } from "../../../src/design/responsive";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";
import {
  getCallablePhoneFromMission,
  openNavigation,
  safeCall,
} from "../../../src/features/driver/utils/missionContact";

const PAGE_BG = "#F4F7F9";
const PAGE_PAD = 16;
const TAB_COUNT = 2;

if (Platform.OS === "android" && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

type TopTab = "mine" | "company";

function isMissionShownOnDeviceLocalDay(m: DriverMission, now: Date): boolean {
  const key = normalizeDriverMissionStatus(m.status);
  if (key === "EN_ROUTE" || key === "ARRIVED" || key === "IN_PROGRESS") return true;
  if (key === "CANCELLED" || key === "NO_SHOW" || key === "FAILED" || key === "REASSIGNED") return false;

  const raw = m.scheduled_time;
  if (raw == null || String(raw).trim() === "") return true;
  const t = Date.parse(String(raw));
  if (!Number.isFinite(t)) return true;
  const scheduled = new Date(t);

  return (
    scheduled.getFullYear() === now.getFullYear() &&
    scheduled.getMonth() === now.getMonth() &&
    scheduled.getDate() === now.getDate()
  );
}

function isUndefinedScheduledTime(raw: unknown): boolean {
  if (typeof raw !== "string" || raw.trim().length === 0) return true;
  const t = Date.parse(raw);
  if (!Number.isFinite(t)) return true;
  const d = new Date(t);
  return d.getHours() === 0 && d.getMinutes() === 0;
}

function getMissionSortTimestamp(m: DriverMission): number {
  const raw = m.scheduled_time;
  if (isUndefinedScheduledTime(raw)) return Number.MAX_SAFE_INTEGER;
  const t = Date.parse(String(raw));
  return Number.isFinite(t) ? t : Number.MAX_SAFE_INTEGER;
}

function sortMissionsForDay(a: DriverMission, b: DriverMission): number {
  const byTime = getMissionSortTimestamp(a) - getMissionSortTimestamp(b);
  if (byTime !== 0) return byTime;
  return a.id - b.id;
}

function formatHeaderDate(d: Date): string {
  return d.toLocaleDateString("fr-CH", {
    weekday: "long",
    day: "2-digit",
    month: "long",
  });
}

function formatMissionWhen(raw: unknown): string {
  if (isUndefinedScheduledTime(raw)) return "Heure à définir";
  const t = Date.parse(String(raw));
  if (!Number.isFinite(t)) return "Heure à définir";
  return new Date(t).toLocaleString("fr-CH", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function getNotes(mission: DriverMission): string {
  const values = [mission.notes, mission.special_instructions, mission.comment, mission.comments];
  for (const value of values) {
    if (typeof value === "string" && value.trim().length > 0) return value.trim();
  }
  return "Aucune note";
}

function getStatusTone(status: string) {
  const key = normalizeDriverMissionStatus(status);
  switch (key) {
    case "ASSIGNED":
      return { bg: "rgba(20,184,166,0.12)", border: "rgba(20,184,166,0.35)", text: "#0F766E" };
    case "EN_ROUTE":
      return { bg: "rgba(59,130,246,0.12)", border: "rgba(59,130,246,0.35)", text: "#1D4ED8" };
    case "ARRIVED":
      return { bg: "rgba(245,158,11,0.12)", border: "rgba(245,158,11,0.35)", text: "#B45309" };
    case "IN_PROGRESS":
      return { bg: "rgba(139,92,246,0.12)", border: "rgba(139,92,246,0.35)", text: "#6D28D9" };
    case "COMPLETED":
      return { bg: "rgba(148,163,184,0.16)", border: "rgba(148,163,184,0.35)", text: "#475569" };
    default:
      return { bg: "rgba(239,68,68,0.12)", border: "rgba(239,68,68,0.35)", text: "#B91C1C" };
  }
}

function missionStateBucket(status: string): "todo" | "inProgress" | "done" {
  const key = normalizeDriverMissionStatus(status);
  if (key === "COMPLETED") return "done";
  if (key === "EN_ROUTE" || key === "ARRIVED" || key === "IN_PROGRESS") return "inProgress";
  return "todo";
}

function getMissionClientName(mission: DriverMission): string {
  const direct = typeof mission.client_name === "string" ? mission.client_name.trim() : "";
  if (direct) return direct;
  const nested = mission.client as { full_name?: unknown } | null | undefined;
  if (nested?.full_name && String(nested.full_name).trim()) return String(nested.full_name).trim();
  return `Course #${mission.id}`;
}

function getMissionEtaDistance(mission: DriverMission): { eta: string; distance: string } {
  const etaMin =
    typeof mission.eta_minutes === "number" && Number.isFinite(mission.eta_minutes)
      ? mission.eta_minutes
      : null;
  const distanceKm =
    typeof mission.distance_km === "number" && Number.isFinite(mission.distance_km)
      ? mission.distance_km
      : null;

  return {
    eta: etaMin != null ? `${Math.max(0, Math.round(etaMin))} min` : "--",
    distance: distanceKm != null ? `${distanceKm.toFixed(1)} km` : "--",
  };
}

function MissionAccordionCard({
  mission,
  expanded,
  onToggle,
  compact = false,
  allowActions = false,
}: {
  mission: DriverMission;
  expanded: boolean;
  onToggle: () => void;
  compact?: boolean;
  allowActions?: boolean;
}) {
  const statusUx = getDriverStatusUx(String(mission.status ?? ""));
  const statusTone = getStatusTone(String(mission.status ?? ""));
  const when = formatMissionWhen(mission.scheduled_time);
  const client = getMissionClientName(mission);
  const pickup = mission.pickup_location?.trim() || "Adresse de départ non définie";
  const destination = mission.dropoff_location?.trim() || "Adresse de destination non définie";
  const notes = getNotes(mission);
  const { eta, distance } = getMissionEtaDistance(mission);
  const canCall = Boolean(getCallablePhoneFromMission(mission));

  return (
    <Pressable
      onPress={onToggle}
      style={({ pressed }) => [
        styles.missionCard,
        compact && styles.missionCardCompact,
        pressed && styles.missionCardPressed,
      ]}
      accessibilityRole="button"
      accessibilityLabel={`Mission ${mission.id}, ${expanded ? "replier" : "déplier"}`}
    >
      <View style={styles.compactTopRow}>
        <AppText style={styles.timeText}>{when}</AppText>
        <View style={[styles.statusPill, { backgroundColor: statusTone.bg, borderColor: statusTone.border }]}>
          <AppText style={[styles.statusPillText, { color: statusTone.text }]} numberOfLines={1}>
            {statusUx.label}
          </AppText>
        </View>
      </View>

      <AppText style={styles.clientName} numberOfLines={1}>
        {client}
      </AppText>
      <AppText style={styles.routeText} numberOfLines={1}>
        {`${pickup} -> ${destination}`}
      </AppText>

      <View style={styles.metaRow}>
        <View style={styles.metaItem}>
          <Ionicons name="car-outline" size={13} color="#475569" />
          <AppText style={styles.metaText}>{eta}</AppText>
        </View>
        <View style={styles.metaItem}>
          <Ionicons name="navigate-outline" size={13} color="#475569" />
          <AppText style={styles.metaText}>{distance}</AppText>
        </View>
        <View style={styles.metaItem}>
          <Ionicons name={expanded ? "chevron-up-outline" : "chevron-down-outline"} size={14} color="#334155" />
        </View>
      </View>

      {expanded ? (
        <View style={styles.expandedBlock}>
          <View style={styles.timelineRow}>
            <View style={styles.timelineCol}>
              <View style={styles.timelineDot} />
              <View style={styles.timelineLine} />
              <View style={styles.timelineDotDestination} />
            </View>
            <View style={styles.timelineContent}>
              <AppText style={styles.sectionLabel}>PRISE EN CHARGE</AppText>
              <AppText style={styles.sectionValue}>{pickup}</AppText>
              <AppText style={styles.sectionEta}>{eta}</AppText>
              <AppText style={[styles.sectionLabel, styles.sectionSpacing]}>DESTINATION</AppText>
              <AppText style={styles.sectionValue}>{destination}</AppText>
            </View>
          </View>

          <AppText style={[styles.sectionLabel, styles.sectionSpacing]}>NOTES</AppText>
          <AppText style={styles.sectionValue}>{notes}</AppText>

          {allowActions ? (
            <View style={styles.actionsRow}>
              <Pressable
                style={({ pressed }) => [styles.actionPillPrimary, pressed && styles.actionPressed]}
                onPress={() => Alert.alert("Action rapide", "Passez la mission en cours depuis l'écran mission.")}
              >
                <Ionicons name="play-outline" size={14} color="#FFFFFF" />
                <AppText style={styles.actionLabelPrimary}>Démarrer</AppText>
              </Pressable>
              <Pressable
                style={({ pressed }) => [
                  styles.actionPillSecondary,
                  !canCall && styles.actionDisabled,
                  pressed && styles.actionPressed,
                ]}
                disabled={!canCall}
                onPress={() => {
                  const phone = getCallablePhoneFromMission(mission);
                  if (!phone) return;
                  void safeCall(phone);
                }}
              >
                <Ionicons name="call-outline" size={14} color="#163A34" />
                <AppText style={styles.actionLabelSecondary}>Appeler</AppText>
              </Pressable>
              <Pressable
                style={({ pressed }) => [styles.actionPillSecondary, pressed && styles.actionPressed]}
                onPress={() => void openNavigation(destination)}
              >
                <Ionicons name="navigate-outline" size={14} color="#163A34" />
                <AppText style={styles.actionLabelSecondary}>Itinéraire</AppText>
              </Pressable>
            </View>
          ) : null}
        </View>
      ) : null}
    </Pressable>
  );
}

export default function DriverTripsScreen() {
  const { width } = useWindowDimensions();
  const isCompactMobile = width < 380;
  const missionsQuery = useDriverMissionsQuery();
  const companyDayQuery = useDriverCompanyBookingsTodayQuery();
  useDriverMissionsListFocusResync();

  const [topTab, setTopTab] = useState<TopTab>("mine");
  const [expandedByTab, setExpandedByTab] = useState<Record<string, boolean>>({});
  const [tabBarWidth, setTabBarWidth] = useState(0);
  const tabIndicator = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.spring(tabIndicator, {
      toValue: topTab === "mine" ? 0 : 1,
      useNativeDriver: true,
      speed: 16,
      bounciness: 7,
    }).start();
  }, [tabIndicator, topTab]);

  const myDayMissions = useMemo(() => {
    const list = missionsQuery.data ?? [];
    const now = new Date();
    return list.filter((m) => isMissionShownOnDeviceLocalDay(m, now)).sort(sortMissionsForDay);
  }, [missionsQuery.data]);

  const companyDayMissions = useMemo(() => {
    const list = companyDayQuery.data ?? [];
    const now = new Date();
    return list.filter((m) => isMissionShownOnDeviceLocalDay(m, now)).sort(sortMissionsForDay);
  }, [companyDayQuery.data]);

  const shownMissions = topTab === "mine" ? myDayMissions : companyDayMissions;
  const summary = useMemo(() => {
    let todo = 0;
    let inProgress = 0;
    let done = 0;
    for (const mission of shownMissions) {
      const bucket = missionStateBucket(String(mission.status ?? ""));
      if (bucket === "done") done += 1;
      else if (bucket === "inProgress") inProgress += 1;
      else todo += 1;
    }
    return { todo, inProgress, done };
  }, [shownMissions]);

  const indicatorTranslate = tabIndicator.interpolate({
    inputRange: [0, 1],
    outputRange: [0, Math.max(0, tabBarWidth / TAB_COUNT)],
  });

  function toggleExpanded(id: number) {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    const key = `${topTab}:${id}`;
    setExpandedByTab((prev) => ({ ...prev, [key]: !prev[key] }));
  }

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          backgroundColor={PAGE_BG}
          withHorizontalPadding={false}
          extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
          contentContainerStyle={[styles.page, isCompactMobile && styles.pageCompact]}
        >
          <View style={styles.headerCard}>
            <View>
              <AppText style={styles.headerTitle}>Courses du jour</AppText>
              <AppText style={styles.headerSubtitle}>{formatHeaderDate(new Date())}</AppText>
            </View>
            <Pressable style={styles.filterBtn} accessibilityRole="button" accessibilityLabel="Filtrer les courses">
              <Ionicons name="options-outline" size={18} color="#0f172a" />
            </Pressable>
          </View>

          <View style={styles.summaryRow}>
            <View style={styles.summaryCard}>
              <AppText style={styles.summaryValue}>{summary.todo}</AppText>
              <AppText style={styles.summaryLabel}>À effectuer</AppText>
            </View>
            <View style={styles.summaryCard}>
              <AppText style={styles.summaryValue}>{summary.inProgress}</AppText>
              <AppText style={styles.summaryLabel}>En cours</AppText>
            </View>
            <View style={styles.summaryCard}>
              <AppText style={styles.summaryValue}>{summary.done}</AppText>
              <AppText style={styles.summaryLabel}>Terminées</AppText>
            </View>
          </View>

          <View style={styles.tabsWrap} onLayout={(e) => setTabBarWidth(e.nativeEvent.layout.width)}>
            {tabBarWidth > 0 ? (
              <Animated.View
                pointerEvents="none"
                style={[
                  styles.tabIndicator,
                  { width: tabBarWidth / TAB_COUNT, transform: [{ translateX: indicatorTranslate }] },
                ]}
              />
            ) : null}
            <Pressable style={styles.tabBtn} onPress={() => setTopTab("mine")}>
              <AppText style={[styles.tabLabel, topTab === "mine" && styles.tabLabelActive]}>Mes courses</AppText>
            </Pressable>
            <Pressable style={styles.tabBtn} onPress={() => setTopTab("company")}>
              <AppText style={[styles.tabLabel, topTab === "company" && styles.tabLabelActive]}>
                Entreprise (jour)
              </AppText>
            </Pressable>
          </View>

          {topTab === "mine" && missionsQuery.isLoading ? (
            <AppText style={styles.infoText}>Chargement des courses…</AppText>
          ) : null}
          {topTab === "company" && companyDayQuery.isLoading ? (
            <AppText style={styles.infoText}>Chargement du planning entreprise…</AppText>
          ) : null}
          {topTab === "mine" && missionsQuery.error ? (
            <AppText style={styles.errorText}>
              {missionsQuery.error instanceof Error ? missionsQuery.error.message : "Erreur chargement courses."}
            </AppText>
          ) : null}
          {topTab === "company" && companyDayQuery.error ? (
            <AppText style={styles.errorText}>
              {companyDayQuery.error instanceof Error
                ? companyDayQuery.error.message
                : "Erreur chargement planning entreprise."}
            </AppText>
          ) : null}

          <View style={styles.missionsStack}>
            {shownMissions.map((mission) => {
              const expanded = Boolean(expandedByTab[`${topTab}:${mission.id}`]);
              return (
                <MissionAccordionCard
                  key={`${topTab}-${mission.id}`}
                  mission={mission}
                  expanded={expanded}
                  onToggle={() => toggleExpanded(mission.id)}
                  compact={isCompactMobile}
                  allowActions={topTab === "mine"}
                />
              );
            })}
          </View>

          {!missionsQuery.isLoading && topTab === "mine" && shownMissions.length === 0 ? (
            <AppText style={styles.infoText}>
              Aucune course du jour sur cet appareil. Les missions hors jour ne sont pas affichées.
            </AppText>
          ) : null}
          {!companyDayQuery.isLoading && topTab === "company" && shownMissions.length === 0 ? (
            <AppText style={styles.infoText}>
              Aucune course entreprise du jour pour ce fuseau local.
            </AppText>
          ) : null}
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: {
    paddingHorizontal: PAGE_PAD,
    paddingTop: PAGE_PAD,
    paddingBottom: 24,
    gap: 12,
  },
  pageCompact: {
    paddingHorizontal: 12,
    paddingTop: 12,
    gap: 10,
  },
  headerCard: {
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 14,
    backgroundColor: "#ECF3F6",
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.06)",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  headerTitle: {
    color: "#0F172A",
    fontSize: FONT_SIZE.px24,
    lineHeight: 28,
    fontWeight: "800",
  },
  headerSubtitle: {
    color: "#475569",
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
    marginTop: 2,
    textTransform: "capitalize",
  },
  filterBtn: {
    width: 38,
    height: 38,
    borderRadius: 19,
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.08)",
    backgroundColor: "rgba(255,255,255,0.72)",
    alignItems: "center",
    justifyContent: "center",
  },
  summaryRow: {
    flexDirection: "row",
    gap: 8,
  },
  summaryCard: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    borderRadius: 18,
    paddingVertical: 12,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.05)",
    shadowColor: "#0F172A",
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.06,
    shadowRadius: 16,
    elevation: 2,
  },
  summaryValue: {
    color: "#0F172A",
    fontSize: FONT_SIZE.px22,
    lineHeight: 24,
    fontWeight: "800",
  },
  summaryLabel: {
    color: "#64748B",
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    fontWeight: "600",
    marginTop: 2,
  },
  tabsWrap: {
    position: "relative",
    flexDirection: "row",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "rgba(15,23,42,0.08)",
    backgroundColor: "rgba(255,255,255,0.86)",
    overflow: "hidden",
  },
  tabIndicator: {
    position: "absolute",
    left: 0,
    top: 0,
    bottom: 0,
    backgroundColor: "#0A8F7A",
    borderRadius: 12,
  },
  tabBtn: {
    flex: 1,
    minHeight: 42,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 10,
  },
  tabLabel: {
    color: "#334155",
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    fontWeight: "600",
  },
  tabLabelActive: {
    color: "#FFFFFF",
  },
  missionsStack: {
    gap: 10,
  },
  missionCard: {
    borderRadius: 20,
    backgroundColor: "#FFFFFF",
    borderWidth: 1,
    borderColor: "rgba(15, 23, 42, 0.06)",
    padding: 14,
    shadowColor: "#0F172A",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.06,
    shadowRadius: 20,
    elevation: 2,
  },
  missionCardCompact: {
    padding: 12,
  },
  missionCardPressed: {
    opacity: 0.92,
    transform: [{ scale: 0.995 }],
  },
  compactTopRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  timeText: {
    color: "#0F172A",
    fontSize: FONT_SIZE.px13,
    lineHeight: 16,
    fontWeight: "700",
    flexShrink: 1,
  },
  statusPill: {
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 8,
    paddingVertical: 2,
    maxWidth: "52%",
  },
  statusPillText: {
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
    fontWeight: "700",
  },
  clientName: {
    marginTop: 8,
    color: "#111827",
    fontSize: FONT_SIZE.px16,
    lineHeight: 20,
    fontWeight: "700",
  },
  routeText: {
    marginTop: 4,
    color: "#334155",
    fontSize: FONT_SIZE.px13,
    lineHeight: 17,
    fontWeight: "500",
  },
  metaRow: {
    marginTop: 8,
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
  },
  metaItem: {
    flexDirection: "row",
    alignItems: "center",
    gap: 4,
  },
  metaText: {
    color: "#475569",
    fontSize: FONT_SIZE.px12,
    lineHeight: 15,
    fontWeight: "600",
  },
  expandedBlock: {
    marginTop: 10,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: "rgba(148,163,184,0.3)",
    gap: 8,
  },
  timelineRow: {
    flexDirection: "row",
    alignItems: "stretch",
    gap: 10,
  },
  timelineCol: {
    width: 16,
    alignItems: "center",
    paddingTop: 2,
  },
  timelineDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: "#14B8A6",
  },
  timelineLine: {
    width: 2,
    flex: 1,
    minHeight: 28,
    backgroundColor: "rgba(148,163,184,0.55)",
    marginVertical: 3,
    borderRadius: 999,
  },
  timelineDotDestination: {
    width: 10,
    height: 10,
    borderRadius: 5,
    borderWidth: 2,
    borderColor: "#0A8F7A",
    backgroundColor: "#FFFFFF",
  },
  timelineContent: {
    flex: 1,
    minWidth: 0,
  },
  sectionLabel: {
    color: "#64748B",
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
    fontWeight: "700",
    letterSpacing: 0.4,
  },
  sectionValue: {
    color: "#0F172A",
    fontSize: FONT_SIZE.px13,
    lineHeight: 17,
    fontWeight: "500",
  },
  sectionEta: {
    color: "#475569",
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    marginTop: 2,
  },
  sectionSpacing: {
    marginTop: 8,
  },
  actionsRow: {
    marginTop: 4,
    flexDirection: "row",
    gap: 8,
    flexWrap: "wrap",
  },
  actionPillPrimary: {
    minHeight: 34,
    borderRadius: 999,
    paddingHorizontal: 12,
    backgroundColor: "#0A8F7A",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 6,
  },
  actionPillSecondary: {
    minHeight: 34,
    borderRadius: 999,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: "rgba(22,58,52,0.2)",
    backgroundColor: "#FFFFFF",
    alignItems: "center",
    justifyContent: "center",
    flexDirection: "row",
    gap: 6,
  },
  actionLabelPrimary: {
    color: "#FFFFFF",
    fontSize: FONT_SIZE.px12,
    lineHeight: 15,
    fontWeight: "700",
  },
  actionLabelSecondary: {
    color: "#163A34",
    fontSize: FONT_SIZE.px12,
    lineHeight: 15,
    fontWeight: "700",
  },
  actionDisabled: {
    opacity: 0.45,
  },
  actionPressed: {
    opacity: 0.85,
  },
  infoText: {
    color: "#64748B",
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
  },
  errorText: {
    color: "#B42318",
    fontSize: FONT_SIZE.px13,
    lineHeight: 18,
    fontWeight: "600",
  },
});
import { useMemo, useState } from "react";
import { Pressable, StyleSheet, View, useWindowDimensions } from "react-native";
import { useRouter } from "expo-router";
import { DriverContextGuard, PermissionGuard } from "../../../src/core/guards";
import {
  useDriverCompanyBookingsTodayQuery,
  useDriverMissionsListFocusResync,
  useDriverMissionsQuery,
} from "../../../src/features/driver/hooks";
import type { DriverMission } from "../../../src/features/driver/types";
import {
  getDriverStatusUx,
  normalizeDriverMissionStatus,
} from "../../../src/features/driver/statusDictionary";
import { AppText, brandSurfaceSoft, Screen } from "../../../src/design/responsive";
import { DRIVER_FLOATING_TAB_SCROLL_PADDING } from "../../../src/features/driver/navigation/DriverFloatingTabBar";

const BRAND = "#0A8F7A";
const BORDER = "rgba(145, 165, 157, 0.45)";
const PAGE_PAD = 20;

type TopTab = "mine" | "company";
type DayMissionBucket = "todo" | "done";

/**
 * Filtre « jour » (mes courses + entreprise) :
 * - calendrier : même jour local que l’appareil ;
 * - pas de courses « figées » dans le passé : si le créneau planifié est déjà passé
 *   et que la mission n’est pas en cours (en route / arrivé / en transport), on masque.
 * /
function isMissionShownOnDeviceLocalDay(m: DriverMission, now: Date): boolean {
  const key = normalizeDriverMissionStatus(m.status);
  if (key === "EN_ROUTE" || key === "ARRIVED" || key === "IN_PROGRESS") {
    return true;
  }
  if (key === "CANCELLED" || key === "NO_SHOW" || key === "FAILED" || key === "REASSIGNED") {
    return false;
  }

  const raw = m.scheduled_time;
  if (raw == null || String(raw).trim() === "") return true;

  const t = Date.parse(String(raw));
  if (!Number.isFinite(t)) return true;
  const scheduled = new Date(t);

  const sameLocalDay =
    scheduled.getFullYear() === now.getFullYear() &&
    scheduled.getMonth() === now.getMonth() &&
    scheduled.getDate() === now.getDate();
  if (!sameLocalDay) return false;
  return true;
}

function getMissionDayBucket(m: DriverMission): DayMissionBucket {
  return normalizeDriverMissionStatus(m.status) === "COMPLETED" ? "done" : "todo";
}

function isUndefinedScheduledTime(raw: unknown): boolean {
  if (typeof raw !== "string" || raw.trim().length === 0) return true;
  const t = Date.parse(raw);
  if (!Number.isFinite(t)) return true;
  const d = new Date(t);
  return d.getHours() === 0 && d.getMinutes() === 0;
}

function getMissionSortTimestamp(m: DriverMission): number {
  const raw = m.scheduled_time;
  if (isUndefinedScheduledTime(raw)) return Number.MAX_SAFE_INTEGER;
  const t = Date.parse(String(raw));
  return Number.isFinite(t) ? t : Number.MAX_SAFE_INTEGER;
}

function sortMissionsForDay(a: DriverMission, b: DriverMission): number {
  const byTime = getMissionSortTimestamp(a) - getMissionSortTimestamp(b);
  if (byTime !== 0) return byTime;
  return a.id - b.id;
}

function formatMissionWhen(raw: unknown): string {
  if (typeof raw !== "string" || raw.trim().length === 0) return "Heure non définie";
  const t = Date.parse(raw);
  if (!Number.isFinite(t)) return "Heure non définie";
  const d = new Date(t);
  // Certaines courses arrivent à 00:00 pour signifier « heure à définir ».
  if (d.getHours() === 0 && d.getMinutes() === 0) return "Heure à définir";
  return d.toLocaleString("fr-CH", {
    weekday: "short",
    day: "2-digit",
    month: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function CompanyDayMissionCard({
  mission,
  compact = false,
}: {
  mission: DriverMission;
  compact?: boolean;
}) {
  const ux = getDriverStatusUx(String(mission.status ?? ""));
  const statusKey = normalizeDriverMissionStatus(mission.status);
  const when = formatMissionWhen(mission.scheduled_time);
  const client =
    typeof mission.client_name === "string" && mission.client_name.trim()
      ? mission.client_name.trim()
      : `Course #${mission.id}`;
  const statusTone =
    statusKey === "COMPLETED"
      ? coStyles.companyStatusDone
      : statusKey === "CANCELLED" || statusKey === "FAILED" || statusKey === "NO_SHOW"
        ? coStyles.companyStatusAlert
        : coStyles.companyStatusTodo;

  return (
    <View
      accessibilityLabel={`Course entreprise ${mission.id}`}
      style={[
        coStyles.companyCard,
        compact && coStyles.companyCardCompact,
        getMissionDayBucket(mission) === "done" && coStyles.companyCardDone,
      ]}
    >
      <View style={coStyles.companyContent}>
        <View style={coStyles.companyHeader}>
          <AppText
            variant="label"
            style={[coStyles.companyTitle, compact && coStyles.companyTitleCompact]}
            numberOfLines={1}
          >
            {client}
          </AppText>
        </View>
        <View style={coStyles.companyMetaRow}>
          <AppText style={[coStyles.companyStatusBadge, statusTone]} numberOfLines={1}>
            {ux.label}
          </AppText>
          <AppText variant="bodyMuted" style={coStyles.companyMeta} numberOfLines={1}>
            {when}
          </AppText>
        </View>
      </View>
    </View>
  );
}

function DayMissionCompactCard({
  mission,
  compact = false,
  onOpenDetail,
}: {
  mission: DriverMission;
  compact?: boolean;
  onOpenDetail: () => void;
}) {
  const ux = getDriverStatusUx(String(mission.status ?? ""));
  const statusKey = normalizeDriverMissionStatus(mission.status);
  const client =
    typeof mission.client_name === "string" && mission.client_name.trim()
      ? mission.client_name.trim()
      : `Course #${mission.id}`;
  const when = formatMissionWhen(mission.scheduled_time);
  const isDone = getMissionDayBucket(mission) === "done";
  const statusTone =
    statusKey === "COMPLETED"
      ? styles.compactStatusDone
      : statusKey === "CANCELLED" || statusKey === "FAILED" || statusKey === "NO_SHOW"
        ? styles.compactStatusAlert
        : styles.compactStatusTodo;

  return (
    <Pressable
      onPress={onOpenDetail}
      accessibilityRole="button"
      accessibilityLabel={`Ouvrir le détail de la course ${mission.id}`}
      style={({ pressed }) => [
        styles.compactCard,
        compact && styles.compactCardSmall,
        isDone && styles.compactCardDone,
        pressed && styles.compactCardPressed,
      ]}
    >
      <View style={styles.compactContent}>
        <View style={styles.compactHeader}>
          <AppText style={styles.compactTitle} numberOfLines={1}>
            {client}
          </AppText>
          <AppText style={styles.compactChevron}>{">"}</AppText>
        </View>
        <View style={styles.compactMetaRow}>
          <AppText style={[styles.compactStatusBadge, statusTone]} numberOfLines={1}>
            {ux.label}
          </AppText>
          <AppText variant="bodyMuted" style={styles.compactWhen} numberOfLines={1}>
            {when}
          </AppText>
        </View>
      </View>
    </Pressable>
  );
}

const coStyles = StyleSheet.create({
  companyCard: {
    padding: 0,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.35)",
    backgroundColor: "#fff",
    overflow: "hidden",
  },
  companyContent: { padding: 12, gap: 6 },
  companyTitle: { fontWeight: "700", color: "#0f172a" },
  companyMeta: { color: "#64748b", fontSize: FONT_SIZE.px12, lineHeight: 16 },
  companyLine: { color: "#334155", fontSize: FONT_SIZE.px13, lineHeight: 18, fontWeight: "500" },
  companyCardCompact: {},
  companyCardDone: { backgroundColor: "#f8fafc" },
  companyCardPressed: { opacity: 0.93 },
  companyHeader: { flexDirection: "row", alignItems: "center" },
  companyMetaRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", gap: 8 },
  companyStatusBadge: {
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 8,
    paddingVertical: 2,
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
    fontWeight: "700",
    overflow: "hidden",
  },
  companyStatusTodo: {
    color: "#0A8F7A",
    borderColor: "rgba(10, 143, 122, 0.35)",
    backgroundColor: "rgba(10, 143, 122, 0.08)",
  },
  companyStatusDone: {
    color: "#475569",
    borderColor: "rgba(100, 116, 139, 0.35)",
    backgroundColor: "rgba(100, 116, 139, 0.08)",
  },
  companyStatusAlert: {
    color: "#B42318",
    borderColor: "rgba(180, 35, 24, 0.35)",
    backgroundColor: "rgba(180, 35, 24, 0.08)",
  },
  companyTitleCompact: { fontSize: FONT_SIZE.px14, lineHeight: 18 },
  driverLine: { color: "#163A34", fontWeight: "600", marginTop: 1, fontSize: FONT_SIZE.px12, lineHeight: 16 },
});

/*
export default function DriverTripsScreen() {
  const router = useRouter();
  const { width } = useWindowDimensions();
  const isCompactMobile = width < 380;
  const missionsQuery = useDriverMissionsQuery();
  const companyDayQuery = useDriverCompanyBookingsTodayQuery();
  useDriverMissionsListFocusResync();

  const [topTab, setTopTab] = useState<TopTab>("mine");

  const myDayMissions = useMemo(() => {
    const list = missionsQuery.data ?? [];
    const now = new Date();
    return list.filter((m) => isMissionShownOnDeviceLocalDay(m, now)).sort(sortMissionsForDay);
  }, [missionsQuery.data]);
  const myDayTodoMissions = useMemo(
    () => myDayMissions.filter((m) => getMissionDayBucket(m) === "todo"),
    [myDayMissions]
  );
  const myDayDoneMissions = useMemo(
    () => myDayMissions.filter((m) => getMissionDayBucket(m) === "done"),
    [myDayMissions]
  );
  const allMyCount = missionsQuery.data?.length ?? 0;

  const companyDayMissionsFiltered = useMemo(() => {
    const list = companyDayQuery.data ?? [];
    const now = new Date();
    return list.filter((m) => isMissionShownOnDeviceLocalDay(m, now)).sort(sortMissionsForDay);
  }, [companyDayQuery.data]);
  const companyDayTodoMissions = useMemo(
    () => companyDayMissionsFiltered.filter((m) => getMissionDayBucket(m) === "todo"),
    [companyDayMissionsFiltered]
  );
  const companyDayDoneMissions = useMemo(
    () => companyDayMissionsFiltered.filter((m) => getMissionDayBucket(m) === "done"),
    [companyDayMissionsFiltered]
  );
  const allCompanyCount = companyDayQuery.data?.length ?? 0;

  function pushTripDetail(
    missionId: number,
    mission: Pick<
      DriverMission,
      "pickup_location" | "dropoff_location" | "status" | "scheduled_time" | "client_name" | "driver_name"
    >,
    source: "active" | "company_day"
  ) {
    router.push({
      pathname: `/(app)/(driver)/trips/${missionId}` as any,
      params: {
        pickup: String(mission.pickup_location ?? ""),
        dropoff: String(mission.dropoff_location ?? ""),
        status: String(mission.status ?? ""),
        scheduled: String(mission.scheduled_time ?? ""),
        client: String(mission.client_name ?? ""),
        driver: String(mission.driver_name ?? ""),
        source,
      },
    } as any);
  }

  return (
    <DriverContextGuard>
      <PermissionGuard permission="mission:read">
        <Screen
          scroll
          backgroundColor={brandSurfaceSoft}
          withHorizontalPadding={false}
          extraScrollBottomPadding={DRIVER_FLOATING_TAB_SCROLL_PADDING}
          contentContainerStyle={[styles.page, isCompactMobile && styles.pageCompact]}
        >
          <AppText variant="sectionTitle" style={[styles.title, isCompactMobile && styles.titleCompact]}>
            Courses du jour
          </AppText>

          <View style={styles.mainTabs}>
            <Pressable
              onPress={() => setTopTab("mine")}
              style={[styles.mainTab, isCompactMobile && styles.mainTabCompact, topTab === "mine" && styles.mainTabActive]}
            >
              <AppText
                style={[
                  styles.mainTabLabel,
                  isCompactMobile && styles.mainTabLabelCompact,
                  topTab === "mine" && styles.mainTabLabelActive,
                ]}
                numberOfLines={2}
              >
                Mes courses du jour
              </AppText>
            </Pressable>
            <Pressable
              onPress={() => setTopTab("company")}
              style={[styles.mainTab, isCompactMobile && styles.mainTabCompact, topTab === "company" && styles.mainTabActive]}
            >
              <AppText
                style={[
                  styles.mainTabLabel,
                  isCompactMobile && styles.mainTabLabelCompact,
                  topTab === "company" && styles.mainTabLabelActive,
                ]}
                numberOfLines={2}
              >
                Entreprise (jour)
              </AppText>
            </Pressable>
          </View>

          {topTab === "mine" ? (
            <>
              {missionsQuery.isLoading ? (
                <AppText variant="bodyMuted" style={styles.muted}>
                  Chargement des courses…
                </AppText>
              ) : null}
              {missionsQuery.error ? (
                <AppText variant="error" style={styles.error}>
                  {missionsQuery.error instanceof Error
                    ? missionsQuery.error.message
                    : "Erreur chargement courses."}
                </AppText>
              ) : null}

              {myDayTodoMissions.length > 0 ? (
                <View style={styles.groupBlock}>
                  <AppText variant="label" style={styles.groupTitle}>
                    À effectuer
                  </AppText>
                  {myDayTodoMissions.map((mission) => (
                    <DayMissionCompactCard
                      key={mission.id}
                      mission={mission}
                      compact={isCompactMobile}
                      onOpenDetail={() => pushTripDetail(mission.id, mission, "active")}
                    />
                  ))}
                </View>
              ) : null}

              {myDayDoneMissions.length > 0 ? (
                <View style={styles.groupBlock}>
                  <AppText variant="label" style={styles.groupTitle}>
                    Effectuées
                  </AppText>
                  {myDayDoneMissions.map((mission) => (
                    <DayMissionCompactCard
                      key={mission.id}
                      mission={mission}
                      compact={isCompactMobile}
                      onOpenDetail={() => pushTripDetail(mission.id, mission, "active")}
                    />
                  ))}
                </View>
              ) : null}

              {!missionsQuery.isLoading && myDayMissions.length === 0 && allMyCount === 0 ? (
                <AppText variant="bodyMuted" style={styles.muted}>
                  Aucune course assignée pour le moment.
                </AppText>
              ) : null}
              {!missionsQuery.isLoading && myDayMissions.length === 0 && allMyCount > 0 ? (
                <AppText variant="bodyMuted" style={styles.muted}>
                  Aucune course assignée prévue aujourd’hui (date de l’appareil). Les courses
                  d’autres jours ne s’affichent pas dans cet onglet.
                </AppText>
              ) : null}
            </>
          ) : (
            <>
              <AppText variant="bodyMuted" style={styles.companyIntro}>
                Planning entreprise du jour avec accès au détail de chaque course (y compris les
                collègues).
              </AppText>
              {companyDayQuery.isLoading ? (
                <AppText variant="bodyMuted" style={styles.muted}>
                  Chargement du planning entreprise…
                </AppText>
              ) : null}
              {companyDayQuery.error ? (
                <AppText variant="error" style={styles.error}>
                  {companyDayQuery.error instanceof Error
                    ? companyDayQuery.error.message
                    : "Erreur chargement planning entreprise."}
                </AppText>
              ) : null}
              {!companyDayQuery.isLoading && !companyDayQuery.error && companyDayTodoMissions.length > 0 ? (
                <View style={styles.groupBlock}>
                  <AppText variant="label" style={styles.groupTitle}>
                    À effectuer
                  </AppText>
                  {companyDayTodoMissions.map((mission) => (
                    <CompanyDayMissionCard
                      key={mission.id}
                      mission={mission}
                      compact={isCompactMobile}
                    />
                  ))}
                </View>
              ) : null}
              {!companyDayQuery.isLoading && !companyDayQuery.error && companyDayDoneMissions.length > 0 ? (
                <View style={styles.groupBlock}>
                  <AppText variant="label" style={styles.groupTitle}>
                    Effectuées
                  </AppText>
                  {companyDayDoneMissions.map((mission) => (
                    <CompanyDayMissionCard
                      key={mission.id}
                      mission={mission}
                      compact={isCompactMobile}
                    />
                  ))}
                </View>
              ) : null}
              {!companyDayQuery.isLoading &&
              !companyDayQuery.error &&
              companyDayMissionsFiltered.length === 0 &&
              allCompanyCount === 0 ? (
                <AppText variant="bodyMuted" style={styles.muted}>
                  Aucune course entreprise pour la fenêtre du jour côté serveur.
                </AppText>
              ) : null}
              {!companyDayQuery.isLoading &&
              !companyDayQuery.error &&
              companyDayMissionsFiltered.length === 0 &&
              allCompanyCount > 0 ? (
                <AppText variant="bodyMuted" style={styles.muted}>
                  Aucune course entreprise ne correspond à aujourd’hui sur cet appareil (fuseau
                  local). Les entrées hors jour ne s’affichent pas.
                </AppText>
              ) : null}
            </>
          )}
        </Screen>
      </PermissionGuard>
    </DriverContextGuard>
  );
}

const styles = StyleSheet.create({
  page: {
    paddingHorizontal: PAGE_PAD,
    paddingTop: PAGE_PAD,
    gap: 12,
    paddingBottom: 28,
  },
  pageCompact: {
    paddingHorizontal: 14,
    paddingTop: 14,
    gap: 10,
  },
  title: {
    color: "#0f172a",
  },
  titleCompact: {
    fontSize: FONT_SIZE.px20,
    lineHeight: 25,
  },
  mainTabs: {
    flexDirection: "row",
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
    marginHorizontal: -4,
  },
  mainTab: {
    flex: 1,
    minHeight: 48,
    paddingVertical: 12,
    paddingHorizontal: 8,
    alignItems: "center",
    justifyContent: "center",
    borderBottomWidth: 3,
    borderBottomColor: "transparent",
  },
  mainTabCompact: {
    minHeight: 46,
    paddingVertical: 10,
    paddingHorizontal: 6,
  },
  mainTabActive: {
    borderBottomColor: BRAND,
  },
  mainTabLabel: {
    fontSize: FONT_SIZE.px14,
    lineHeight: 18,
    fontWeight: "600",
    color: "#64748b",
    textAlign: "center",
  },
  mainTabLabelCompact: {
    fontSize: FONT_SIZE.px13,
    lineHeight: 17,
  },
  mainTabLabelActive: {
    color: "#163A34",
  },
  block: {
    gap: 6,
  },
  groupBlock: {
    gap: 8,
  },
  groupTitle: {
    color: "#163A34",
    fontWeight: "700",
  },
  compactCard: {
    padding: 0,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "rgba(145, 165, 157, 0.35)",
    backgroundColor: "#fff",
    overflow: "hidden",
  },
  compactCardSmall: {},
  compactCardDone: {
    backgroundColor: "#f8fafc",
  },
  compactCardPressed: {
    opacity: 0.93,
  },
  compactContent: {
    padding: 12,
    gap: 6,
  },
  compactHeader: {
    flexDirection: "row",
    gap: 8,
    alignItems: "center",
  },
  compactTitle: {
    flex: 1,
    color: "#0f172a",
    fontWeight: "700",
    fontSize: FONT_SIZE.px14,
    lineHeight: 18,
  },
  compactChevron: {
    color: "#64748b",
    fontSize: FONT_SIZE.px16,
    lineHeight: 16,
    fontWeight: "700",
  },
  compactMetaRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 8,
  },
  compactStatusBadge: {
    borderWidth: 1,
    borderRadius: 999,
    paddingHorizontal: 8,
    paddingVertical: 2,
    fontSize: FONT_SIZE.px11,
    lineHeight: 14,
    fontWeight: "700",
    overflow: "hidden",
  },
  compactStatusTodo: {
    color: "#0A8F7A",
    borderColor: "rgba(10, 143, 122, 0.35)",
    backgroundColor: "rgba(10, 143, 122, 0.08)",
  },
  compactStatusDone: {
    color: "#475569",
    borderColor: "rgba(100, 116, 139, 0.35)",
    backgroundColor: "rgba(100, 116, 139, 0.08)",
  },
  compactStatusAlert: {
    color: "#B42318",
    borderColor: "rgba(180, 35, 24, 0.35)",
    backgroundColor: "rgba(180, 35, 24, 0.08)",
  },
  compactWhen: {
    color: "#64748b",
    fontSize: FONT_SIZE.px12,
    lineHeight: 16,
    flexShrink: 1,
    textAlign: "right",
  },
  compactRouteSingleLine: {
    color: "#334155",
    fontSize: FONT_SIZE.px13,
    lineHeight: 17,
    fontWeight: "500",
  },
  muted: {
    color: "#64748b",
    lineHeight: 20,
  },
  error: {
    color: "#B42318",
  },
  companyIntro: {
    color: "#475569",
    lineHeight: 20,
    marginBottom: 2,
  },
});
*/
