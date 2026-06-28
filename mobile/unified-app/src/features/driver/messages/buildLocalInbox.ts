import type { DriverMission } from "../types";
import type { HubChatMessage, MessageHubThread } from "./types";
import {
  MESSAGE_HUB_THREAD_DISPATCH,
  MESSAGE_HUB_THREAD_SUPPORT,
  MESSAGE_HUB_THREAD_TEAM,
  directThreadId,
} from "./contracts";

const TERMINAL_STATUSES = new Set([
  "COMPLETED",
  "CANCELLED",
  "CANCELED",
  "REASSIGNED",
  "NO_SHOW",
  "FAILED",
]);

function missionThreadId(bookingId: number): string {
  return `mission:${bookingId}`;
}

function isTerminalMission(mission: DriverMission): boolean {
  const status = String(mission.status ?? "").toUpperCase();
  return TERMINAL_STATUSES.has(status);
}

function missionIdsWithDiscussion(legacyMessages?: HubChatMessage[]): Set<number> {
  const ids = new Set<number>();
  (legacyMessages ?? []).forEach((m) => {
    const tid = m.thread_id;
    if (typeof tid === "string" && tid.startsWith("mission:")) {
      const parsed = Number.parseInt(tid.slice("mission:".length), 10);
      if (Number.isFinite(parsed)) ids.add(parsed);
    }
    if (typeof m.booking_id === "number" && Number.isFinite(m.booking_id)) {
      ids.add(m.booking_id);
    }
  });
  return ids;
}

function missionTitle(mission: DriverMission): string {
  const name =
    (typeof mission.client_name === "string" && mission.client_name.trim()) ||
    (typeof (mission as Record<string, unknown>).patient_name === "string"
      ? String((mission as Record<string, unknown>).patient_name)
      : "");
  return name.trim() || `Mission #${mission.id}`;
}

function summarizeLegacyMessages(messages: HubChatMessage[]): {
  dispatchUnread: number;
  teamUnread: number;
  dispatchLast: HubChatMessage | null;
  teamLast: HubChatMessage | null;
} {
  let dispatchUnread = 0;
  let teamUnread = 0;
  let dispatchLast: HubChatMessage | null = null;
  let teamLast: HubChatMessage | null = null;

  const touch = (target: "dispatch" | "team", m: HubChatMessage) => {
    const cur = target === "dispatch" ? dispatchLast : teamLast;
    if (!cur || Date.parse(m.timestamp) > Date.parse(cur.timestamp)) {
      if (target === "dispatch") dispatchLast = m;
      else teamLast = m;
    }
  };

  messages.forEach((m) => {
    const role = String(m.sender_role ?? "").toUpperCase();
    const isDirect = m.receiver_id != null;
    if (isDirect) return;
    if (role === "COMPANY") {
      if (m.is_read === false) dispatchUnread += 1;
      touch("dispatch", m);
      return;
    }
    if (role === "DRIVER") {
      if (m.is_read === false) teamUnread += 1;
      touch("team", m);
    }
  });

  return { dispatchUnread, teamUnread, dispatchLast, teamLast };
}

/** Fils DM dérivés de l'historique legacy (receiver_id). */
export function buildColleagueThreadsFromLegacy(
  messages: HubChatMessage[],
  myUserId: number | null
): MessageHubThread[] {
  if (myUserId == null) return [];
  const peers = new Map<number, { last: HubChatMessage; unread: number }>();

  messages.forEach((m) => {
    const sender = m.sender_id != null ? Number(m.sender_id) : null;
    const receiver = m.receiver_id != null ? Number(m.receiver_id) : null;
    if (sender == null || receiver == null) return;
    let peer: number | null = null;
    if (sender === myUserId && receiver !== myUserId) peer = receiver;
    else if (receiver === myUserId && sender !== myUserId) peer = sender;
    if (peer == null) return;

    const existing = peers.get(peer);
    const unreadAdd =
      receiver === myUserId && m.is_read === false ? 1 : 0;
    if (!existing) {
      peers.set(peer, { last: m, unread: unreadAdd });
      return;
    }
    if (Date.parse(m.timestamp) > Date.parse(existing.last.timestamp)) {
      peers.set(peer, {
        last: m,
        unread: existing.unread + unreadAdd,
      });
    } else {
      peers.set(peer, {
        last: existing.last,
        unread: existing.unread + unreadAdd,
      });
    }
  });

  return [...peers.entries()].map(([peerUserId, meta]) => ({
    thread_id: directThreadId(peerUserId),
    section: "colleagues" as const,
    title: meta.last.sender_name?.trim() || `Collègue #${peerUserId}`,
    subtitle: "Message direct",
    peer_user_id: peerUserId,
    booking_id: null,
    status: null,
    unread_count: meta.unread,
    priority: "normal" as const,
    last_message_preview: meta.last.content?.trim() || "Conversation",
    last_message_at: meta.last.timestamp,
  }));
}

/** Inbox minimale toujours utilisable (mission active + équipe + dispatch + support). */
export function buildLocalInboxThreads(
  missions: DriverMission[] | undefined,
  legacyMessages?: HubChatMessage[],
  myUserId?: number | null
): MessageHubThread[] {
  const list = Array.isArray(missions) ? missions : [];
  const legacy = legacyMessages?.length ? summarizeLegacyMessages(legacyMessages) : null;
  const discussedMissionIds = missionIdsWithDiscussion(legacyMessages);
  const threads: MessageHubThread[] = [];

  const activeMission =
    list.find((m) => !isTerminalMission(m) && String(m.status ?? "").toUpperCase() !== "ASSIGNED") ??
    list.find((m) => !isTerminalMission(m));

  if (activeMission && discussedMissionIds.has(activeMission.id)) {
    const missionMsgs = (legacyMessages ?? []).filter(
      (m) =>
        m.thread_id === missionThreadId(activeMission.id) ||
        m.booking_id === activeMission.id
    );
    const last = missionMsgs.reduce<HubChatMessage | null>((acc, m) => {
      if (!acc || Date.parse(m.timestamp) > Date.parse(acc.timestamp)) return m;
      return acc;
    }, null);
    threads.push({
      thread_id: missionThreadId(activeMission.id),
      section: "mission_active",
      title: missionTitle(activeMission),
      subtitle: `Mission #${activeMission.id}`,
      booking_id: activeMission.id,
      status: String(activeMission.status ?? ""),
      scheduled_time:
        typeof activeMission.scheduled_time === "string" ? activeMission.scheduled_time : null,
      pickup_location:
        typeof activeMission.pickup_location === "string" ? activeMission.pickup_location : null,
      dropoff_location:
        typeof activeMission.dropoff_location === "string" ? activeMission.dropoff_location : null,
      unread_count: missionMsgs.filter((m) => m.is_read === false).length,
      priority: "normal",
      last_message_preview: last?.content?.trim() || "Conversation mission",
      last_message_at: last?.timestamp ?? null,
    });
  }

  threads.push({
    thread_id: MESSAGE_HUB_THREAD_TEAM,
    section: "team",
    title: "Équipe chauffeurs",
    subtitle: "Canal groupe · tous les collègues",
    booking_id: null,
    status: null,
    unread_count: legacy?.teamUnread ?? 0,
    priority: legacy && legacy.teamUnread > 0 ? "important" : "normal",
    last_message_preview:
      legacy?.teamLast?.content?.trim() ?? "Échanges entre chauffeurs",
    last_message_at: legacy?.teamLast?.timestamp ?? null,
  });

  threads.push({
    thread_id: MESSAGE_HUB_THREAD_DISPATCH,
    section: "dispatch",
    title: "Dispatch",
    subtitle: "Exploitation & régulation",
    booking_id: null,
    status: null,
    unread_count: legacy?.dispatchUnread ?? 0,
    priority: legacy && legacy.dispatchUnread > 0 ? "important" : "normal",
    last_message_preview:
      legacy?.dispatchLast?.content?.trim() ?? "Canal exploitation — appuyer pour écrire",
    last_message_at: legacy?.dispatchLast?.timestamp ?? null,
  });

  threads.push({
    thread_id: MESSAGE_HUB_THREAD_SUPPORT,
    section: "support",
    title: "Support",
    subtitle: "Assistance LIRIE",
    booking_id: null,
    status: null,
    unread_count: 0,
    priority: "normal",
    last_message_preview: "Questions techniques ou compte",
    last_message_at: null,
  });

  if (legacyMessages?.length && myUserId != null) {
    threads.push(...buildColleagueThreadsFromLegacy(legacyMessages, myUserId));
  }

  return threads;
}

/** Masque les fils mission sans échange réel (filet de sécurité après fusion API / local). */
export function filterMissionThreadsWithDiscussion(
  threads: MessageHubThread[]
): MessageHubThread[] {
  return threads.filter((t) => {
    if (!t.thread_id.startsWith("mission:")) return true;
    return Boolean(t.last_message_at);
  });
}

export function mergeInboxThreads(
  apiThreads: MessageHubThread[] | undefined,
  localThreads: MessageHubThread[]
): MessageHubThread[] {
  const map = new Map<string, MessageHubThread>();
  localThreads.forEach((t) => map.set(t.thread_id, { ...t }));
  (apiThreads ?? []).forEach((t) => {
    const existing = map.get(t.thread_id);
    if (!existing) {
      map.set(t.thread_id, t);
      return;
    }
    const existingTs = Date.parse(existing.last_message_at ?? "") || 0;
    const incomingTs = Date.parse(t.last_message_at ?? "") || 0;
    if (incomingTs >= existingTs) {
      map.set(t.thread_id, { ...existing, ...t });
    } else {
      map.set(t.thread_id, { ...t, ...existing });
    }
  });

  const merged = [...map.values()];
  const order = (section: string) => {
    if (section === "mission_active") return 0;
    if (section === "urgent") return 1;
    if (section === "dispatch") return 2;
    if (section === "team") return 3;
    if (section === "colleagues") return 4;
    if (section === "support") return 5;
    if (section === "archives") return 6;
    return 7;
  };
  merged.sort((a, b) => {
    const sa = order(String(a.section));
    const sb = order(String(b.section));
    if (sa !== sb) return sa - sb;
    const ta = a.last_message_at ? Date.parse(a.last_message_at) : 0;
    const tb = b.last_message_at ? Date.parse(b.last_message_at) : 0;
    return tb - ta;
  });
  return filterMissionThreadsWithDiscussion(merged);
}
