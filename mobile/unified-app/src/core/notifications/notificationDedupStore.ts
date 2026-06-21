/** TTL cross-canal Expo/FCM (redelivery typique < 45s). */
export const NOTIFICATION_CROSS_CHANNEL_DEDUP_TTL_MS = 45_000;
const DEDUP_TTL_MS = NOTIFICATION_CROSS_CHANNEL_DEDUP_TTL_MS;
export const MAX_DEDUP_ENTRIES = 1000;

type DedupEntry = {
  expiresAt: number;
};

const entries = new Map<string, DedupEntry>();

function evictExpired(now: number): void {
  for (const [key, entry] of entries) {
    if (entry.expiresAt <= now) {
      entries.delete(key);
    }
  }
}

function evictOldest(count: number): void {
  if (count <= 0) return;
  const sorted = [...entries.entries()].sort((a, b) => a[1].expiresAt - b[1].expiresAt);
  for (let i = 0; i < Math.min(count, sorted.length); i += 1) {
    entries.delete(sorted[i][0]);
  }
}

export function buildNotificationDedupKey(input: {
  dedupeKey?: string | null;
  eventId?: string | null;
  notificationId?: string | null;
  missionId?: number | null;
  type?: string | null;
}): string {
  if (input.dedupeKey) return input.dedupeKey;
  if (input.eventId) return `event:${input.eventId}`;
  if (input.notificationId) return `notif:${input.notificationId}`;
  if (input.missionId != null && input.type) return `fallback:${input.type}:${input.missionId}`;
  if (input.notificationId) return `notif:${input.notificationId}`;
  return `anon:${Date.now()}:${Math.random()}`;
}

/** Returns true when the event was already handled and should be skipped. */
export function markNotificationHandled(dedupKey: string): boolean {
  const now = Date.now();
  evictExpired(now);
  const existing = entries.get(dedupKey);
  if (existing && existing.expiresAt > now) {
    return true;
  }
  if (entries.size >= MAX_DEDUP_ENTRIES) {
    evictOldest(entries.size - MAX_DEDUP_ENTRIES + 1);
  }
  entries.set(dedupKey, { expiresAt: now + DEDUP_TTL_MS });
  return false;
}

export function resetNotificationDedupStoreForTests(): void {
  entries.clear();
}
