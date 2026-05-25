import {
  buildColleagueThreadsFromLegacy,
  buildLocalInboxThreads,
  filterMissionThreadsWithDiscussion,
  mergeInboxThreads,
} from "./buildLocalInbox";
import type { HubChatMessage } from "./types";
import type { DriverMission } from "../types";

describe("buildLocalInboxThreads", () => {
  it("always exposes team, dispatch and support", () => {
    const threads = buildLocalInboxThreads([]);
    const ids = threads.map((t) => t.thread_id);
    expect(ids).toContain("team");
    expect(ids).toContain("dispatch");
    expect(ids).toContain("support");
  });

  it("keeps fresher dispatch row when api sends duplicate thread_id", () => {
    const local = buildLocalInboxThreads([]);
    const api = [
      {
        thread_id: "dispatch",
        section: "dispatch" as const,
        title: "Dispatch",
        conversation_id: 1,
        unread_count: 0,
        priority: "normal" as const,
        last_message_at: "2026-05-20T19:07:53+02:00",
        last_message_preview: "Je suis disponible",
      },
      {
        thread_id: "dispatch",
        section: "dispatch" as const,
        title: "Dispatch",
        conversation_id: 18,
        unread_count: 1,
        priority: "urgent" as const,
        last_message_at: "2026-05-20T19:40:13+02:00",
        last_message_preview: "⚠ Patient absent",
      },
    ];
    const merged = mergeInboxThreads(api, local);
    const dispatch = merged.find((t) => t.thread_id === "dispatch");
    expect(dispatch?.conversation_id).toBe(18);
    expect(dispatch?.last_message_preview).toBe("⚠ Patient absent");
  });

  it("merges api threads over local defaults", () => {
    const local = buildLocalInboxThreads([]);
    const api = [
      {
        thread_id: "dispatch",
        section: "dispatch" as const,
        title: "Dispatch",
        unread_count: 3,
        priority: "urgent" as const,
        last_message_preview: "Nouvelle mission",
      },
    ];
    const merged = mergeInboxThreads(api, local);
    const dispatch = merged.find((t) => t.thread_id === "dispatch");
    expect(dispatch?.unread_count).toBe(3);
    expect(dispatch?.last_message_preview).toBe("Nouvelle mission");
  });

  it("builds colleague threads from legacy DMs", () => {
    const legacy: HubChatMessage[] = [
      {
        id: 1,
        sender_id: 10,
        receiver_id: 20,
        content: "Salut",
        sender_role: "DRIVER",
        timestamp: "2026-05-19T10:00:00Z",
        message_type: "text",
        priority: "normal",
      },
    ];
    const threads = buildColleagueThreadsFromLegacy(legacy, 10);
    expect(threads).toHaveLength(1);
    expect(threads[0]?.thread_id).toBe("direct:20");
  });

  it("does not add mission thread without messages", () => {
    const missions: DriverMission[] = [
      { id: 42, status: "EN_ROUTE", client_name: "Catherine BRONNIMANN" },
    ];
    const threads = buildLocalInboxThreads(missions);
    expect(threads.some((t) => t.thread_id === "mission:42")).toBe(false);
  });

  it("adds mission thread when legacy has discussion", () => {
    const missions: DriverMission[] = [
      { id: 42, status: "EN_ROUTE", client_name: "Catherine BRONNIMANN" },
    ];
    const legacy: HubChatMessage[] = [
      {
        id: 1,
        sender_id: 1,
        receiver_id: null,
        content: "Patient prêt",
        thread_id: "mission:42",
        booking_id: 42,
        timestamp: "2026-05-19T10:00:00Z",
        message_type: "text",
        priority: "normal",
      },
    ];
    const threads = buildLocalInboxThreads(missions, legacy);
    expect(threads.some((t) => t.thread_id === "mission:42")).toBe(true);
  });

  it("filters merged mission threads without last_message_at", () => {
    const filtered = filterMissionThreadsWithDiscussion([
      {
        thread_id: "mission:1",
        section: "mission_active",
        title: "Mission",
        unread_count: 0,
        priority: "normal",
        last_message_at: null,
      },
      {
        thread_id: "team",
        section: "team",
        title: "Équipe",
        unread_count: 0,
        priority: "normal",
        last_message_at: null,
      },
    ]);
    expect(filtered.map((t) => t.thread_id)).toEqual(["team"]);
  });
});
