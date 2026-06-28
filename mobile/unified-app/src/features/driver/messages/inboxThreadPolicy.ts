import type { MessageHubThread } from "./types";
import type { InboxTab } from "./inboxDisplay";

const MISSION_SECTIONS = new Set(["mission_active", "archives", "urgent"]);

/** Fils liés à une mission (chat par course). */
export function isMissionThread(thread: MessageHubThread): boolean {
  const tid = String(thread.thread_id ?? "");
  const section = String(thread.section ?? "");
  if (tid.startsWith("mission:")) return true;
  if (thread.booking_id != null) return true;
  return MISSION_SECTIONS.has(section);
}

/** Mission terminée / archivée — masquée sur mobile pour éviter la liste infinie. */
export function isArchivedMissionThread(thread: MessageHubThread): boolean {
  if (!isMissionThread(thread)) return false;
  if (thread.section === "archives") return true;
  if (thread.section === "mission_active" || thread.section === "urgent") return false;
  return thread.thread_id.startsWith("mission:");
}

/** Supprime les chats mission archivés renvoyés par l’API / legacy. */
export function applyMobileInboxThreadPolicy(threads: MessageHubThread[]): MessageHubThread[] {
  return threads.filter((t) => !isArchivedMissionThread(t));
}

/** Filtre affichage par onglet (mobile simplifié). */
export function filterThreadsForMobileTab(
  threads: MessageHubThread[],
  tab: InboxTab
): MessageHubThread[] {
  if (tab === "all") {
    return threads.filter((t) => !isMissionThread(t));
  }
  if (tab === "missions") {
    return threads.filter(
      (t) => t.section === "mission_active" || (t.section === "urgent" && isMissionThread(t))
    );
  }
  return threads.filter((t) => !isMissionThread(t));
}

const PINNED_THREAD_ORDER: Record<string, number> = {
  team: 0,
  dispatch: 1,
  support: 2,
};

function pinnedRank(threadId: string): number {
  return PINNED_THREAD_ORDER[threadId] ?? 100;
}

/** Tri inbox mobile : canal équipe en tête, puis dispatch, puis récence. */
export function sortThreadsForMobileInbox(threads: MessageHubThread[]): MessageHubThread[] {
  return [...threads].sort((a, b) => {
    const pinDelta = pinnedRank(a.thread_id) - pinnedRank(b.thread_id);
    if (pinDelta !== 0) return pinDelta;

    const ta = a.last_message_at ? Date.parse(a.last_message_at) : 0;
    const tb = b.last_message_at ? Date.parse(b.last_message_at) : 0;
    if (tb !== ta) return tb - ta;

    const unreadDelta = (b.unread_count ?? 0) - (a.unread_count ?? 0);
    if (unreadDelta !== 0) return unreadDelta;

    return a.thread_id.localeCompare(b.thread_id);
  });
}
