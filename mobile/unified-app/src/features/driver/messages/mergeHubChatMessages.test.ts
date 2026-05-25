import { describe, expect, it } from "@jest/globals";
import type { HubChatMessage } from "./types";
import { mergeHubChatMessageLists, upsertHubChatMessage } from "./mergeHubChatMessages";

const optimistic = (id: string, content: string): HubChatMessage => ({
  id,
  _localId: id,
  sender_id: 1,
  content,
  sender_role: "DRIVER",
  timestamp: "2026-05-20T17:08:00.000Z",
});

const confirmed = (id: number, localId: string, content: string): HubChatMessage => ({
  id,
  _localId: localId,
  sender_id: 1,
  content,
  sender_role: "DRIVER",
  timestamp: "2026-05-20T17:08:01.000Z",
});

describe("mergeHubChatMessages", () => {
  it("replaces optimistic message when server echo arrives", () => {
    const pending = optimistic("local-1", "Test 1");
    const server = confirmed(42, "local-1", "Test 1");

    const merged = upsertHubChatMessage([pending], server);
    expect(merged).toHaveLength(1);
    expect(merged[0]?.id).toBe(42);
  });

  it("merges query cache and live without duplicate", () => {
    const pending = optimistic("local-2", "J ai besoin d aide");
    const server = confirmed(99, "local-2", "J ai besoin d aide");

    const merged = mergeHubChatMessageLists([server], [pending, server]);
    expect(merged).toHaveLength(1);
    expect(merged[0]?.id).toBe(99);
  });
});
