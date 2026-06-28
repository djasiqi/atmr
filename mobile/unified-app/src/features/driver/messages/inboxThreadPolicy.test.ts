import {
  applyMobileInboxThreadPolicy,
  filterThreadsForMobileTab,
  isMissionThread,
  sortThreadsForMobileInbox,
} from "./inboxThreadPolicy";
import type { MessageHubThread } from "./types";

function mission(id: number, section: MessageHubThread["section"]): MessageHubThread {
  return {
    thread_id: `mission:${id}`,
    section,
    title: `Mission ${id}`,
    booking_id: id,
    unread_count: 0,
    priority: "normal",
    last_message_at: "2026-06-01T10:00:00Z",
  };
}

describe("inboxThreadPolicy", () => {
  it("détecte les fils mission", () => {
    expect(isMissionThread(mission(42, "mission_active"))).toBe(true);
    expect(isMissionThread({ thread_id: "team", section: "team", title: "Équipe", unread_count: 0, priority: "normal" })).toBe(
      false
    );
  });

  it("retire les missions archivées", () => {
    const filtered = applyMobileInboxThreadPolicy([
      mission(1, "archives"),
      mission(2, "mission_active"),
      { thread_id: "team", section: "team", title: "Équipe", unread_count: 0, priority: "normal" },
    ]);
    expect(filtered.map((t) => t.thread_id)).toEqual(["mission:2", "team"]);
  });

  it("masque les missions dans l’onglet TOUTES", () => {
    const filtered = filterThreadsForMobileTab(
      [mission(2, "mission_active"), { thread_id: "team", section: "team", title: "Équipe", unread_count: 0, priority: "normal" }],
      "all"
    );
    expect(filtered.map((t) => t.thread_id)).toEqual(["team"]);
  });

  it("ne garde que la mission active dans MISSIONS", () => {
    const filtered = filterThreadsForMobileTab(
      [mission(1, "archives"), mission(2, "mission_active")],
      "missions"
    );
    expect(filtered.map((t) => t.thread_id)).toEqual(["mission:2"]);
  });

  it("priorise le canal équipe", () => {
    const sorted = sortThreadsForMobileInbox([
      { thread_id: "dispatch", section: "dispatch", title: "Dispatch", unread_count: 0, priority: "normal", last_message_at: "2026-06-02T10:00:00Z" },
      { thread_id: "team", section: "team", title: "Équipe", unread_count: 0, priority: "normal", last_message_at: "2026-06-01T10:00:00Z" },
    ]);
    expect(sorted[0]?.thread_id).toBe("team");
  });
});
