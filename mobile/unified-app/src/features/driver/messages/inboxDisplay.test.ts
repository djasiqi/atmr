import { getThreadDisplayLines, sortThreadsByRecent } from "./inboxDisplay";
import type { MessageHubThread } from "./types";

function thread(
  thread_id: string,
  last_message_at: string | null = null
): MessageHubThread {
  return {
    thread_id,
    section: thread_id === "dispatch" ? "dispatch" : "team",
    title: thread_id,
    unread_count: 0,
    priority: "normal",
    last_message_at,
  };
}

describe("getThreadDisplayLines", () => {
  it("uses custom dispatch title from hub thread", () => {
    const lines = getThreadDisplayLines({
      thread_id: "dispatch",
      section: "dispatch",
      title: "Emmenez-Moi",
      subtitle: "Gestion réservée à l'exploitation",
      unread_count: 0,
      priority: "normal",
    });
    expect(lines.headline).toBe("Emmenez-Moi");
    expect(lines.subline).toBe("Gestion réservée à l'exploitation");
  });

  it("falls back to Dispatch when title is empty", () => {
    const lines = getThreadDisplayLines({
      thread_id: "dispatch",
      section: "dispatch",
      title: "",
      unread_count: 0,
      priority: "normal",
    });
    expect(lines.headline).toBe("Dispatch");
  });
});

describe("sortThreadsByRecent", () => {
  it("keeps canonical order when no timestamps", () => {
    const sorted = sortThreadsByRecent([
      thread("support"),
      thread("team"),
      thread("dispatch"),
    ]);
    expect(sorted.map((t) => t.thread_id)).toEqual(["dispatch", "team", "support"]);
  });

  it("sorts by timestamp when present", () => {
    const sorted = sortThreadsByRecent([
      thread("team", "2026-05-19T10:00:00Z"),
      thread("dispatch", "2026-05-19T12:00:00Z"),
    ]);
    expect(sorted[0]?.thread_id).toBe("dispatch");
  });
});
