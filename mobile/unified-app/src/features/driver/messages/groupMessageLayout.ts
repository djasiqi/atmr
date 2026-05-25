import type { HubChatMessage } from "./types";

const GROUP_GAP_MS = 5 * 60 * 1000;

const SENDER_COLORS = [
  "#7C3AED",
  "#0D9488",
  "#DB2777",
  "#2563EB",
  "#D97706",
  "#059669",
  "#DC2626",
  "#4F46E5",
];

export function senderColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i += 1) h = (h + name.charCodeAt(i) * (i + 1)) % 997;
  return SENDER_COLORS[h % SENDER_COLORS.length];
}

export function messageSenderKey(msg: HubChatMessage, ownUserId: string | null): string {
  if (ownUserId != null && msg.sender_id != null && String(msg.sender_id) === ownUserId) {
    return "__self__";
  }
  if (msg.sender_id != null) return String(msg.sender_id);
  return String(msg.sender_name ?? msg.sender_role ?? "unknown");
}

export type GroupMessageMeta = {
  showAvatar: boolean;
  showSenderName: boolean;
  isFirstInGroup: boolean;
  isLastInGroup: boolean;
};

export function buildGroupMessageMeta(
  messages: HubChatMessage[],
  ownUserId: string | null
): Map<string, GroupMessageMeta> {
  const map = new Map<string, GroupMessageMeta>();
  for (let i = 0; i < messages.length; i += 1) {
    const msg = messages[i];
    if (msg.message_type === "system") continue;

    const prev = i > 0 ? messages[i - 1] : null;
    const next = i < messages.length - 1 ? messages[i + 1] : null;
    const key = messageSenderKey(msg, ownUserId);
    const prevKey =
      prev && prev.message_type !== "system" ? messageSenderKey(prev, ownUserId) : null;
    const nextKey =
      next && next.message_type !== "system" ? messageSenderKey(next, ownUserId) : null;

    const samePrev =
      prevKey === key &&
      prev != null &&
      withinGroupWindow(prev.timestamp, msg.timestamp) &&
      dayKeyFromIso(prev.timestamp) === dayKeyFromIso(msg.timestamp);

    const sameNext =
      nextKey === key &&
      next != null &&
      withinGroupWindow(msg.timestamp, next.timestamp) &&
      dayKeyFromIso(msg.timestamp) === dayKeyFromIso(next.timestamp);

    const isOwn = key === "__self__";
    map.set(String(msg.id), {
      showAvatar: !isOwn && !samePrev,
      showSenderName: !isOwn && !samePrev,
      isFirstInGroup: !samePrev,
      isLastInGroup: !sameNext,
    });
  }
  return map;
}

function withinGroupWindow(a: string, b: string): boolean {
  const ta = Date.parse(a);
  const tb = Date.parse(b);
  if (!Number.isFinite(ta) || !Number.isFinite(tb)) return false;
  return Math.abs(tb - ta) <= GROUP_GAP_MS;
}

function dayKeyFromIso(iso: string): string {
  const d = Date.parse(iso);
  if (!Number.isFinite(d)) return "";
  const msg = new Date(d);
  return `${msg.getFullYear()}-${msg.getMonth()}-${msg.getDate()}`;
}

export function initialsFromName(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return name.slice(0, 2).toUpperCase() || "?";
}

export function avatarColor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i += 1) h = (h + name.charCodeAt(i) * (i + 1)) % 997;
  const palette = ["#6366f1", "#0D9488", "#D97706", "#7C3AED", "#DB2777", "#2563EB"];
  return palette[h % palette.length];
}
