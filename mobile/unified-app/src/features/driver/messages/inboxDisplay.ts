import type { MessageHubThread } from "./types";

export type InboxTab = "all" | "missions" | "contacts";

const MISSION_SECTIONS = new Set(["mission_active", "archives", "urgent"]);
const CONTACT_SECTIONS = new Set(["dispatch", "team", "support", "colleagues", "drivers"]);

export function countUnreadForTab(threads: MessageHubThread[], tab: InboxTab): number {
  return threads
    .filter((t) => threadMatchesTab(t, tab))
    .reduce((sum, t) => sum + (t.unread_count ?? 0), 0);
}

export function threadMatchesTab(thread: MessageHubThread, tab: InboxTab): boolean {
  if (tab === "all") return true;
  const section = String(thread.section ?? "");
  if (tab === "missions") {
    return MISSION_SECTIONS.has(section) || thread.booking_id != null;
  }
  return CONTACT_SECTIONS.has(section);
}

/** Ordre fixe des canaux système quand pas encore de message horodaté. */
const CANONICAL_THREAD_ORDER: Record<string, number> = {
  dispatch: 10,
  team: 20,
  support: 30,
};

function canonicalRank(threadId: string): number {
  return CANONICAL_THREAD_ORDER[threadId] ?? 1000;
}

export function sortThreadsByRecent(threads: MessageHubThread[]): MessageHubThread[] {
  return [...threads].sort((a, b) => {
    const ta = a.last_message_at ? Date.parse(a.last_message_at) : 0;
    const tb = b.last_message_at ? Date.parse(b.last_message_at) : 0;
    if (tb !== ta) return tb - ta;
    const unreadDelta = (b.unread_count ?? 0) - (a.unread_count ?? 0);
    if (unreadDelta !== 0) return unreadDelta;
    const rankDelta = canonicalRank(a.thread_id) - canonicalRank(b.thread_id);
    if (rankDelta !== 0) return rankDelta;
    return a.thread_id.localeCompare(b.thread_id);
  });
}

export function formatInboxTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = Date.parse(iso);
  if (!Number.isFinite(d)) return "";
  const now = new Date();
  const msg = new Date(d);
  const sameDay =
    now.getFullYear() === msg.getFullYear() &&
    now.getMonth() === msg.getMonth() &&
    now.getDate() === msg.getDate();
  if (sameDay) {
    return msg.toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit" });
  }
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  const isYesterday =
    yesterday.getFullYear() === msg.getFullYear() &&
    yesterday.getMonth() === msg.getMonth() &&
    yesterday.getDate() === msg.getDate();
  if (isYesterday) return "Hier";
  return msg.toLocaleDateString("fr-FR", { day: "numeric", month: "short" });
}

export type ThreadDisplayLines = {
  headline: string;
  subline: string | null;
  avatarKind: "mission" | "dispatch" | "support" | "group" | "person";
  showPin: boolean;
};

export function getThreadDisplayLines(thread: MessageHubThread): ThreadDisplayLines {
  const section = String(thread.section ?? "");
  if (thread.booking_id != null || section === "mission_active" || section === "archives") {
    return {
      headline: `Mission #${thread.booking_id ?? "—"}`,
      subline: thread.title?.trim() || null,
      avatarKind: "mission",
      showPin: section === "mission_active",
    };
  }

  const tid = String(thread.thread_id ?? "");

  if (section === "dispatch" || tid === "dispatch") {
    return {
      headline: thread.title?.trim() || "Dispatch",
      subline: thread.subtitle?.trim() || "Exploitation & régulation",
      avatarKind: "dispatch",
      showPin: false,
    };
  }

  if (section === "support" || tid === "support") {
    return {
      headline: "Support LIRIE",
      subline: thread.subtitle ?? "Assistance LIRIE",
      avatarKind: "support",
      showPin: false,
    };
  }

  if (section === "team" || tid === "team") {
    return {
      headline: "Équipe chauffeurs",
      subline: thread.subtitle ?? "Canal groupe",
      avatarKind: "group",
      showPin: false,
    };
  }

  if (tid.startsWith("company_driver:")) {
    return {
      headline: thread.title?.trim() || "Exploitation",
      subline: thread.subtitle ?? "Conversation privée",
      avatarKind: "dispatch",
      showPin: false,
    };
  }

  if (section === "drivers" || tid.startsWith("company_driver:")) {
    return {
      headline: thread.title?.trim() || "Chauffeur",
      subline: thread.subtitle ?? "Conversation privée",
      avatarKind: "person",
      showPin: false,
    };
  }

  if (section === "colleagues" || thread.peer_user_id != null) {
    return {
      headline: thread.title?.trim() || "Collègue",
      subline: thread.subtitle ?? "Message direct",
      avatarKind: "person",
      showPin: false,
    };
  }

  return {
    headline: thread.title?.trim() || "Conversation",
    subline: thread.subtitle ?? null,
    avatarKind: "person",
    showPin: false,
  };
}
