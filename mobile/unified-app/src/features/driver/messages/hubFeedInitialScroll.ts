import type { HubChatMessage } from "./types";

export type HubFeedScrollItem =
  | { kind: "date"; id: string }
  | { kind: "message"; id: string; message: HubChatMessage };

export type HubFeedScrollPlan =
  | { mode: "end" }
  | { mode: "index"; feedIndex: number; messageId: string };

export type HubFeedInitialAnchorMode = "first_unread" | "latest";

/** Message entrant non lu (pas les nôtres, pas les messages système). */
export function isIncomingUnreadMessage(
  message: HubChatMessage,
  ownSenderId: string | null
): boolean {
  if (message.message_type === "system") return false;
  if (message.is_read === true) return false;
  if (ownSenderId != null && message.sender_id != null) {
    if (String(message.sender_id) === String(ownSenderId)) return false;
  }
  return true;
}

export function resolveHubFeedScrollPlan(
  messages: HubChatMessage[],
  feedItems: HubFeedScrollItem[],
  ownSenderId: string | null,
  anchorMode: HubFeedInitialAnchorMode = "first_unread"
): HubFeedScrollPlan {
  if (anchorMode === "latest") {
    return { mode: "end" };
  }
  const firstUnread = messages.find((m) => isIncomingUnreadMessage(m, ownSenderId));
  if (!firstUnread) {
    return { mode: "end" };
  }
  const messageId = String(firstUnread.id);
  const feedIndex = feedItems.findIndex(
    (item) => item.kind === "message" && String(item.message.id) === messageId
  );
  if (feedIndex < 0) {
    return { mode: "end" };
  }
  return { mode: "index", feedIndex, messageId };
}
