import { describe, expect, it } from "@jest/globals";
import { groupThreadsBySection } from "./groupThreads";
import type { MessageHubThread } from "./types";

describe("groupThreadsBySection", () => {
  it("groups threads by operational section", () => {
    const threads: MessageHubThread[] = [
      {
        thread_id: "mission:1",
        section: "mission_active",
        title: "Patient A",
        unread_count: 2,
        priority: "normal",
      },
      {
        thread_id: "dispatch",
        section: "dispatch",
        title: "Dispatch",
        unread_count: 1,
        priority: "important",
      },
    ];
    const grouped = groupThreadsBySection(threads);
    expect(grouped.mission_active).toHaveLength(1);
    expect(grouped.dispatch).toHaveLength(1);
  });
});
