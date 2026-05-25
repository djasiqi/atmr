import {
  isIncomingUnreadMessage,
  resolveHubFeedScrollPlan,
  type HubFeedScrollItem,
} from "./hubFeedInitialScroll";
import type { HubChatMessage } from "./types";

function msg(partial: Partial<HubChatMessage> & { id: number }): HubChatMessage {
  return {
    id: partial.id,
    content: partial.content ?? "hi",
    timestamp: partial.timestamp ?? "2026-05-20T10:00:00Z",
    sender_id: partial.sender_id ?? 2,
    is_read: partial.is_read,
    message_type: partial.message_type,
  };
}

describe("hubFeedInitialScroll", () => {
  it("detects incoming unread from others only", () => {
    expect(isIncomingUnreadMessage(msg({ id: 1, is_read: false, sender_id: 2 }), "5")).toBe(true);
    expect(isIncomingUnreadMessage(msg({ id: 2, is_read: true, sender_id: 2 }), "5")).toBe(false);
    expect(isIncomingUnreadMessage(msg({ id: 3, is_read: false, sender_id: 5 }), "5")).toBe(false);
    expect(isIncomingUnreadMessage(msg({ id: 4, is_read: false, message_type: "system" }), "5")).toBe(
      false
    );
  });

  it("scrolls to first unread feed index when present", () => {
    const messages = [
      msg({ id: 1, is_read: true, sender_id: 2 }),
      msg({ id: 2, is_read: false, sender_id: 2 }),
      msg({ id: 3, is_read: false, sender_id: 2 }),
    ];
    const feedItems: HubFeedScrollItem[] = [
      { kind: "date", id: "date-1" },
      { kind: "message", id: "1", message: messages[0] },
      { kind: "message", id: "2", message: messages[1] },
      { kind: "message", id: "3", message: messages[2] },
    ];
    const plan = resolveHubFeedScrollPlan(messages, feedItems, "5");
    expect(plan.mode).toBe("index");
    if (plan.mode === "index") {
      expect(plan.feedIndex).toBe(2);
      expect(plan.messageId).toBe("2");
    }
  });

  it("falls back to end when all read", () => {
    const messages = [msg({ id: 1, is_read: true }), msg({ id: 2, is_read: true })];
    const feedItems: HubFeedScrollItem[] = messages.map((m) => ({
      kind: "message" as const,
      id: String(m.id),
      message: m,
    }));
    expect(resolveHubFeedScrollPlan(messages, feedItems, "5").mode).toBe("end");
  });
});
