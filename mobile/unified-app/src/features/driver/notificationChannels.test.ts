import { describe, expect, it } from "@jest/globals";
import { resolveDriverNotificationContract } from "./notificationChannels";

describe("resolveDriverNotificationContract", () => {
  it("maps mission cancelled to urgent max priority", () => {
    const resolved = resolveDriverNotificationContract("mission_cancelled");
    expect(resolved.channelId).toBe("urgent");
    expect(resolved.priority).toBe("MAX");
    expect(resolved.silent).toBe(false);
  });

  it("maps silent refresh to silent behavior", () => {
    const resolved = resolveDriverNotificationContract("mission_refresh");
    expect(resolved.channelId).toBe("silent");
    expect(resolved.silent).toBe(true);
    expect(resolved.action).toBe("sync_only");
  });

  it("falls back unknown events to mission updates", () => {
    const resolved = resolveDriverNotificationContract("unknown_event");
    expect(resolved.channelId).toBe("mission_updates");
  });
});
