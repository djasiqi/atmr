import type { CompanyDriverLiveLocation, CompanyDispatchMission } from "../api/contracts";
import type { CompanyInboxNotification } from "../api/companyInboxApi";
import {
  conciseRouteSegment,
  formatMissionTime,
  IN_FLIGHT_MISSION_STATUSES,
  isMissionDelayed,
  toEpoch,
} from "./companyDashboardMissionUi";

const SWISS_TZ = "Europe/Zurich";
const MAX_FEED_ITEMS = 4;

export type DashboardLiveActivityKind =
  | "mission_delayed"
  | "inbox_event"
  | "network_alert"
  | "mission_active"
  | "driver_available"
  | "empty_state";

export type DashboardLiveActivityTimeKind = "instant" | "scheduled" | "received_at" | "day_summary";

export type DashboardLiveActivityItem = {
  id: string;
  kind: DashboardLiveActivityKind;
  message: string;
  detail?: string;
  timeCaption: string;
  timeKind: DashboardLiveActivityTimeKind;
  isDelayed: boolean;
  missionId?: number;
  /** Compatibilité composants legacy */
  timeLabel: string;
};

export type BuildOperationalLiveFeedInput = {
  missions: CompanyDispatchMission[];
  drivers: CompanyDriverLiveLocation[];
  alertTexts: { id: string; text: string; isError: boolean }[];
  inboxNotifications?: CompanyInboxNotification[];
  selectedDateIso: string;
  nowMs?: number;
  maxItems?: number;
};

function zonedDateParts(epochMs: number): { year: number; month: number; day: number } {
  const parts = new Intl.DateTimeFormat("en-CA", {
    timeZone: SWISS_TZ,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).formatToParts(new Date(epochMs));
  const year = Number(parts.find((p) => p.type === "year")?.value);
  const month = Number(parts.find((p) => p.type === "month")?.value);
  const day = Number(parts.find((p) => p.type === "day")?.value);
  return { year, month, day };
}

function isoDateInZurich(epochMs: number): string {
  const { year, month, day } = zonedDateParts(epochMs);
  return `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
}

function scheduledCaption(scheduledAt: string | null | undefined): string {
  const time = formatMissionTime(scheduledAt);
  return time === "—" ? "Horaire inconnu" : `Prévu ${time}`;
}

function formatReceivedCaption(createdAt: string, nowMs: number): { caption: string; timeKind: DashboardLiveActivityTimeKind } {
  const createdMs = toEpoch(createdAt);
  if (createdMs <= 0) {
    return { caption: "Reçu récemment", timeKind: "received_at" };
  }
  const ageMs = Math.max(0, nowMs - createdMs);
  if (ageMs < 2 * 60_000) {
    return { caption: "À l'instant", timeKind: "instant" };
  }
  if (ageMs < 60 * 60_000) {
    const minutes = Math.max(1, Math.floor(ageMs / 60_000));
    return { caption: `Il y a ${minutes} min`, timeKind: "instant" };
  }
  return { caption: `Reçu ${formatMissionTime(createdAt)}`, timeKind: "received_at" };
}

function withTimeLabel(item: Omit<DashboardLiveActivityItem, "timeLabel">): DashboardLiveActivityItem {
  return { ...item, timeLabel: item.timeCaption };
}

export function pickDelayedMissionForFeed(
  missions: CompanyDispatchMission[],
  nowMs = Date.now()
): CompanyDispatchMission | null {
  const candidates = missions
    .filter((m) => m.status !== "completed" && m.status !== "cancelled")
    .filter((m) => isMissionDelayed(m, nowMs))
    .sort((a, b) => toEpoch(a.scheduled_at) - toEpoch(b.scheduled_at));
  return candidates[0] ?? null;
}

export function shouldSkipAlertLineForFeed(alertText: string, hasDelayedMissionRow: boolean): boolean {
  if (!hasDelayedMissionRow) return false;
  return isDelayCountSynthesisAlertText(alertText);
}

/** Synthèse réseau du type « 2 retard(s) signalé(s) sur le réseau » — jamais sur le sticky. */
export function shouldSkipAlertLineForSticky(alert: { id: string; text: string }): boolean {
  if (alert.id === "delayed") return true;
  return isDelayCountSynthesisAlertText(alert.text);
}

function isDelayCountSynthesisAlertText(alertText: string): boolean {
  const normalized = alertText.trim().toLowerCase();
  if (/\bretard\b/.test(normalized) && /\d+/.test(normalized)) return true;
  if (/\bdelay(ed)?\b/.test(normalized) && /\d+/.test(normalized)) return true;
  return false;
}

function extractMissionIdFromNotification(notification: CompanyInboxNotification): number | undefined {
  const meta = notification.metadata;
  if (!meta || typeof meta !== "object") return undefined;
  const raw = meta.mission_id ?? meta.booking_id ?? meta.bookingId;
  if (typeof raw === "number" && Number.isFinite(raw)) return raw;
  if (typeof raw === "string") {
    const parsed = Number.parseInt(raw, 10);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function buildDelayedMissionItem(mission: CompanyDispatchMission): DashboardLiveActivityItem {
  const delay = Number(mission.assignment_pickup_delay_minutes);
  const delayMinutes = Number.isFinite(delay) && delay > 0 ? Math.round(delay) : null;
  const client = mission.client_name?.trim() || "Client";
  const driver =
    mission.driver_name?.trim() ||
    (mission.driver_id != null ? `Chauffeur #${mission.driver_id}` : "Non assigné");
  const route = `${conciseRouteSegment(mission.pickup_label, 24)} → ${conciseRouteSegment(mission.dropoff_label, 24)}`;
  const detailParts = [driver, route];
  if (delayMinutes != null) detailParts.push(`+${delayMinutes} min`);

  return withTimeLabel({
    id: `mission-delayed-${mission.mission_id}`,
    kind: "mission_delayed",
    message: `Retard — ${client}`,
    detail: detailParts.join(" · "),
    timeCaption: scheduledCaption(mission.scheduled_at),
    timeKind: "scheduled",
    isDelayed: true,
    missionId: mission.mission_id,
  });
}

function buildActiveMissionItem(mission: CompanyDispatchMission): DashboardLiveActivityItem {
  const client = mission.client_name?.trim() || "Client";
  const driver = mission.driver_name?.trim();
  return withTimeLabel({
    id: `mission-active-${mission.mission_id}`,
    kind: "mission_active",
    message: driver ? `${driver} en course` : "Course en cours",
    detail: client,
    timeCaption: scheduledCaption(mission.scheduled_at),
    timeKind: "scheduled",
    isDelayed: false,
    missionId: mission.mission_id,
  });
}

function buildAlertItem(alert: { id: string; text: string; isError: boolean }): DashboardLiveActivityItem {
  return withTimeLabel({
    id: `alert-${alert.id}`,
    kind: "network_alert",
    message: alert.text,
    timeCaption: "Synthèse",
    timeKind: "day_summary",
    isDelayed: /retard/i.test(alert.text),
  });
}

function buildDriverAvailableItem(drivers: CompanyDriverLiveLocation[]): DashboardLiveActivityItem {
  if (drivers.length === 1) {
    const d = drivers[0];
    const name =
      d.driver_name?.trim() || d.full_name?.trim() || `Chauffeur #${d.driver_id}`;
    return withTimeLabel({
      id: `driver-${d.driver_id}`,
      kind: "driver_available",
      message: `${name} est disponible`,
      timeCaption: "À l'instant",
      timeKind: "instant",
      isDelayed: false,
    });
  }
  return withTimeLabel({
    id: "drivers-available",
    kind: "driver_available",
    message: `${drivers.length} chauffeurs disponibles`,
    timeCaption: "À l'instant",
    timeKind: "instant",
    isDelayed: false,
  });
}

function buildInboxItems(
  notifications: CompanyInboxNotification[],
  selectedDateIso: string,
  nowMs: number,
  limit: number
): DashboardLiveActivityItem[] {
  const dayItems = notifications
    .filter((n) => {
      const createdMs = toEpoch(n.created_at);
      if (createdMs <= 0) return false;
      return isoDateInZurich(createdMs) === selectedDateIso;
    })
    .sort((a, b) => toEpoch(b.created_at) - toEpoch(a.created_at))
    .slice(0, limit);

  return dayItems.map((n) => {
    const { caption, timeKind } = formatReceivedCaption(n.created_at, nowMs);
    const title = n.title?.trim();
    const body = n.message?.trim();
    const message = title && body && title !== body ? title : title || body || "Notification";
    const detail = title && body && title !== body ? body : undefined;
    return withTimeLabel({
      id: `inbox-${n.id}`,
      kind: "inbox_event",
      message,
      detail,
      timeCaption: caption,
      timeKind,
      isDelayed: /retard|delay|late/i.test(`${title ?? ""} ${body ?? ""}`),
      missionId: extractMissionIdFromNotification(n),
    });
  });
}

function feedKindPriority(kind: DashboardLiveActivityKind): number {
  switch (kind) {
    case "mission_delayed":
      return 100;
    case "inbox_event":
      return 90;
    case "network_alert":
      return 70;
    case "mission_active":
      return 60;
    case "driver_available":
      return 30;
    case "empty_state":
    default:
      return 0;
  }
}

function sortFeedItems(items: DashboardLiveActivityItem[]): DashboardLiveActivityItem[] {
  return [...items].sort((a, b) => feedKindPriority(b.kind) - feedKindPriority(a.kind));
}

export function buildOperationalLiveFeed(input: BuildOperationalLiveFeedInput): DashboardLiveActivityItem[] {
  const nowMs = input.nowMs ?? Date.now();
  const maxItems = input.maxItems ?? MAX_FEED_ITEMS;
  const items: DashboardLiveActivityItem[] = [];
  const seenMissionIds = new Set<number>();

  const delayedMission = pickDelayedMissionForFeed(input.missions, nowMs);
  if (delayedMission) {
    items.push(buildDelayedMissionItem(delayedMission));
    seenMissionIds.add(delayedMission.mission_id);
  }

  for (const inboxItem of buildInboxItems(
    input.inboxNotifications ?? [],
    input.selectedDateIso,
    nowMs,
    2
  )) {
    if (inboxItem.missionId != null && seenMissionIds.has(inboxItem.missionId)) continue;
    items.push(inboxItem);
    if (inboxItem.missionId != null) seenMissionIds.add(inboxItem.missionId);
  }

  for (const alert of input.alertTexts.slice(0, 3)) {
    if (shouldSkipAlertLineForFeed(alert.text, delayedMission != null)) continue;
    items.push(buildAlertItem(alert));
  }

  const inProgress = input.missions
    .filter((m) => IN_FLIGHT_MISSION_STATUSES.includes(m.status))
    .sort((a, b) => toEpoch(a.scheduled_at) - toEpoch(b.scheduled_at));

  let activeCount = 0;
  for (const mission of inProgress) {
    if (seenMissionIds.has(mission.mission_id)) continue;
    if (activeCount >= 2) break;
    items.push(buildActiveMissionItem(mission));
    seenMissionIds.add(mission.mission_id);
    activeCount += 1;
  }

  const availableDrivers = input.drivers.filter((d) => !d.mission_id);
  if (availableDrivers.length > 0) {
    items.push(buildDriverAvailableItem(availableDrivers));
  }

  if (items.length === 0) {
    items.push(
      withTimeLabel({
        id: "empty-state",
        kind: "empty_state",
        message: "Rien à signaler pour cette journée",
        detail: "La flotte et les courses planifiées sont à jour",
        timeCaption: "État actuel",
        timeKind: "day_summary",
        isDelayed: false,
      })
    );
  }

  return sortFeedItems(items).slice(0, maxItems);
}
