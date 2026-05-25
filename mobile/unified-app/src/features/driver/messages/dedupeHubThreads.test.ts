import { dedupeMessageHubThreads, inboxThreadListKey } from "./dedupeHubThreads";
import type { MessageHubThread } from "./types";

describe("dedupeMessageHubThreads", () => {
  it("fusionne les doublons dispatch", () => {
    const threads: MessageHubThread[] = [
      {
        thread_id: "dispatch",
        section: "dispatch",
        title: "A",
        unread_count: 0,
        priority: "normal",
        conversation_id: 1,
      },
      {
        thread_id: "dispatch",
        section: "dispatch",
        title: "B",
        last_message_at: "2026-05-19T12:00:00Z",
        unread_count: 1,
        priority: "normal",
        conversation_id: 2,
      },
    ];
    const out = dedupeMessageHubThreads(threads);
    expect(out).toHaveLength(1);
    expect(out[0]?.title).toBe("B");
  });
});

describe("inboxThreadListKey", () => {
  it("utilise thread_id", () => {
    expect(
      inboxThreadListKey({
        thread_id: "dispatch",
        section: "dispatch",
        title: "D",
        unread_count: 0,
        priority: "normal",
        conversation_id: 42,
      })
    ).toBe("dispatch");
  });
});
