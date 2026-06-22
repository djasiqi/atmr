import type { CompanyDriverLiveLocation, CompanyDispatchMission } from "../api/contracts";
import type { CompanyInboxNotification } from "../api/companyInboxApi";
import type { DashboardViewModel } from "./dispatchDashboardPresentation";
import {
  buildOperationalLiveFeed,
  pickDelayedMissionForFeed,
  shouldSkipAlertLineForSticky,
  type DashboardLiveActivityItem,
  type DashboardLiveActivityKind,
  type DashboardLiveActivityTimeKind,
} from "./companyDashboardLiveFeed";
import {
  conciseRouteSegment,
  formatMissionScheduleTimeLabel,
  formatMissionTime,
  formatMissionDateTimeShort,
  formatEtaLabel,
  resolveMissionUiStatus,
  toEpoch,
  type MissionUiStatus,
} from "./companyDashboardMissionUi";

export type {
  DashboardLiveActivityItem,
  DashboardLiveActivityKind,
  DashboardLiveActivityTimeKind,
};

export type DashboardCompactStat = {
  key: "assign_pending" | "in_progress" | "delayed" | "drivers_available";
  label: string;
  value: string;
  accentColor: string;
};

export type DashboardLiveOverlay = {
  activeDrivers: number;
  missionsInProgress: number;
  delayedMissions: number;
};

export type DashboardDelayedMissionCard = {
  missionId: number;
  delayMinutes: number;
  scheduledTimeLabel: string;
  clientName: string;
  routeLabel: string;
  driverLabel: string;
};

export type DashboardCompactMissionRow = {
  missionId: number;
  timeLabel: string;
  clientName: string;
  routeLabel: string;
  etaLabel: string | null;
  status: MissionUiStatus;
};

export type DashboardQuickAction = {
  key: "create" | "assign" | "urgent" | "search";
  label: string;
  icon: "add-circle-outline" | "person-add-outline" | "warning-outline" | "search-outline";
};

export type DashboardStickyAlertLine = {
  id: string;
  text: string;
  severity: "error" | "warning";
};

export type CompanyDashboardUiModel = {
  isLive: boolean;
  liveOverlay: DashboardLiveOverlay;
  compactStats: DashboardCompactStat[];
  quickActions: DashboardQuickAction[];
  liveActivity: DashboardLiveActivityItem[];
  stickyAlertLines: DashboardStickyAlertLine[];
  delayedMission: DashboardDelayedMissionCard | null;
  upcomingMissions: DashboardCompactMissionRow[];
  showEmptyMissions: boolean;
};

export type BuildCompanyDashboardUiInput = {
  isLive: boolean;
  missions: CompanyDispatchMission[];
  drivers: CompanyDriverLiveLocation[];
  missionsPending: number;
  missionsInProgress: number;
  delayedBookings: number;
  driversAvailable: number;
  presentationView: DashboardViewModel;
  alertTexts: { id: string; text: string; isError: boolean }[];
  inboxNotifications?: CompanyInboxNotification[];
  selectedDateIso: string;
  loading: boolean;
};

function buildDelayedCard(mission: CompanyDispatchMission): DashboardDelayedMissionCard {
  const delay = Number(mission.assignment_pickup_delay_minutes);
  const delayMinutes = Number.isFinite(delay) && delay > 0 ? Math.round(delay) : 12;
  const client = mission.client_name?.trim() || "Client";
  const route = `${conciseRouteSegment(mission.pickup_label, 28)} → ${conciseRouteSegment(mission.dropoff_label, 28)}`;
  const driver =
    mission.driver_name?.trim() ||
    (mission.driver_id != null ? `Chauffeur #${mission.driver_id}` : "Non assigné");
  return {
    missionId: mission.mission_id,
    delayMinutes,
    scheduledTimeLabel: formatMissionTime(mission.scheduled_at),
    clientName: client,
    routeLabel: route,
    driverLabel: driver,
  };
}

const UPCOMING_WINDOW_MS = 60 * 60 * 1000;

const upcomingRowsMemo = new WeakMap<CompanyDispatchMission[], DashboardCompactMissionRow[]>();
const liveFeedMemo = new WeakMap<object, ReturnType<typeof buildOperationalLiveFeed>>();

function mapUpcomingRows(missions: CompanyDispatchMission[]): DashboardCompactMissionRow[] {
  const now = Date.now();
  return missions
    .filter((m) => m.status !== "completed" && m.status !== "cancelled")
    .filter((m) => {
      const t = toEpoch(m.scheduled_at);
      return t != null && t >= now && t - now <= UPCOMING_WINDOW_MS;
    })
    .sort((a, b) => toEpoch(a.scheduled_at) - toEpoch(b.scheduled_at))
    .slice(0, 6)
    .map((m) => ({
      missionId: m.mission_id,
      timeLabel: formatMissionScheduleTimeLabel(m),
      clientName: m.client_name?.trim() || "Invité",
      routeLabel: `${conciseRouteSegment(m.pickup_label, 22)} → ${conciseRouteSegment(m.dropoff_label, 22)}`,
      etaLabel: formatEtaLabel(m),
      status: resolveMissionUiStatus(m),
    }));
}

export function mapUpcomingRowsMemoized(
  missions: CompanyDispatchMission[]
): DashboardCompactMissionRow[] {
  const cached = upcomingRowsMemo.get(missions);
  if (cached) return cached;
  const rows = mapUpcomingRows(missions);
  upcomingRowsMemo.set(missions, rows);
  return rows;
}

function buildOperationalLiveFeedMemoized(
  input: Parameters<typeof buildOperationalLiveFeed>[0]
): DashboardLiveActivityItem[] {
  const cacheKey = {
    missions: input.missions,
    drivers: input.drivers,
    alertTexts: input.alertTexts,
    inboxNotifications: input.inboxNotifications,
    selectedDateIso: input.selectedDateIso,
  };
  const cached = liveFeedMemo.get(cacheKey);
  if (cached) return cached;
  const feed = buildOperationalLiveFeed(input);
  liveFeedMemo.set(cacheKey, feed);
  return feed;
}

function isDelayedMissionInFeed(items: DashboardLiveActivityItem[]): boolean {
  return items.some((item) => item.kind === "mission_delayed");
}

export function buildCompanyDashboardUiModel(input: BuildCompanyDashboardUiInput): CompanyDashboardUiModel {
  const activeDrivers = input.drivers.filter((d) => d.location_status !== "offline").length;
  const delayedMissionEntity = pickDelayedMissionForFeed(input.missions);
  const liveActivity = buildOperationalLiveFeedMemoized({
    missions: input.missions,
    drivers: input.drivers,
    alertTexts: input.alertTexts,
    inboxNotifications: input.inboxNotifications,
    selectedDateIso: input.selectedDateIso,
  });
  const stickyDelayedMission =
    delayedMissionEntity && !isDelayedMissionInFeed(liveActivity)
      ? buildDelayedCard(delayedMissionEntity)
      : null;
  const stickyAlertLines = input.presentationView.alertLines
    .filter((line) => !shouldSkipAlertLineForSticky(line))
    .map((line) => ({
      id: line.id,
      text: line.text,
      severity: line.severity,
    }));

  return {
    isLive: input.isLive,
    liveOverlay: {
      activeDrivers: activeDrivers || input.drivers.length,
      missionsInProgress: input.missionsInProgress,
      delayedMissions: input.delayedBookings,
    },
    compactStats: [
      {
        key: "assign_pending",
        label: "À assigner",
        value: String(input.missionsPending),
        accentColor: "#F59E0B",
      },
      {
        key: "in_progress",
        label: "En cours",
        value: String(input.missionsInProgress),
        accentColor: "#3B82F6",
      },
      {
        key: "delayed",
        label: "En retard",
        value: String(input.delayedBookings),
        accentColor: "#EF4444",
      },
      {
        key: "drivers_available",
        label: "Disponibles",
        value: String(input.driversAvailable),
        accentColor: "#00796B",
      },
    ],
    quickActions: [
      { key: "create", label: "Créer mission", icon: "add-circle-outline" },
      { key: "assign", label: "Assigner", icon: "person-add-outline" },
      { key: "urgent", label: "Urgences", icon: "warning-outline" },
      { key: "search", label: "Rechercher", icon: "search-outline" },
    ],
    liveActivity,
    stickyAlertLines,
    delayedMission: stickyDelayedMission,
    upcomingMissions: mapUpcomingRowsMemoized(input.missions),
    showEmptyMissions: input.missions.length === 0 && !input.loading,
  };
}

export function formatDelayedMissionScheduleFull(iso: string | null | undefined): string {
  return formatMissionDateTimeShort(iso);
}
