import type { MessageHubThread } from "./types";

/** Une entrée par `thread_id` — garde la conversation la plus récente. */
export function dedupeMessageHubThreads(threads: MessageHubThread[]): MessageHubThread[] {
  const byThread = new Map<string, MessageHubThread>();
  for (const thread of threads) {
    const key = thread.thread_id;
    const existing = byThread.get(key);
    if (!existing) {
      byThread.set(key, thread);
      continue;
    }
    const tTs = Date.parse(thread.last_message_at ?? "") || 0;
    const eTs = Date.parse(existing.last_message_at ?? "") || 0;
    if (tTs > eTs) {
      byThread.set(key, thread);
      continue;
    }
    if (tTs === eTs && key === "dispatch") {
      const tCid = thread.conversation_id ?? 0;
      const eCid = existing.conversation_id ?? 0;
      if (tCid > eCid) {
        byThread.set(key, thread);
      }
      continue;
    }
    if (tTs === eTs && (thread.unread_count ?? 0) > (existing.unread_count ?? 0)) {
      byThread.set(key, thread);
      continue;
    }
    if (existing.section === "urgent" && thread.section !== "urgent") {
      byThread.set(key, thread);
    }
  }
  return [...byThread.values()];
}

export function inboxThreadListKey(thread: MessageHubThread): string {
  return thread.thread_id;
}
