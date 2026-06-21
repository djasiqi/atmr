import { beforeEach, describe, expect, it } from "@jest/globals";
import {
  buildNotificationDedupKey,
  markNotificationHandled,
  resetNotificationDedupStoreForTests,
} from "./notificationDedupStore";

describe("notificationDedupStore", () => {
  beforeEach(() => {
    resetNotificationDedupStoreForTests();
  });

  it("deduplicates by event id", () => {
    const key = buildNotificationDedupKey({ eventId: "evt-1" });
    expect(markNotificationHandled(key)).toBe(false);
    expect(markNotificationHandled(key)).toBe(true);
  });

  it("priorise dedupe_key explicite", () => {
    const key = buildNotificationDedupKey({
      dedupeKey: "event:evt-explicit",
      eventId: "other",
    });
    expect(key).toBe("event:evt-explicit");
  });

  it("utilise une clé stable pour booking_assigned malgré event_id différent", () => {
    const key = buildNotificationDedupKey({
      eventId: "evt-a",
      missionId: 730,
      type: "booking_assigned",
    });
    expect(key).toBe("booking:730:event:assigned");
  });
});
