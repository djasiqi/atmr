import { describe, expect, it } from "@jest/globals";
import {
  resolveFlushDelayMs,
  resolveGpsEventFlushPriority,
} from "./gpsFlushScheduler";
import { REALTIME_FLUSH_MS } from "./gpsFlushConstants";

describe("gpsFlushScheduler", () => {
  it("resolveGpsEventFlushPriority maps critical events", () => {
    expect(resolveGpsEventFlushPriority("company_socket_reconnected")).toBe("critical");
    expect(resolveGpsEventFlushPriority("driver_live_state_update")).toBe("critical");
    expect(
      resolveGpsEventFlushPriority("driver_location_update", { immediate: true })
    ).toBe("critical");
  });

  it("maps observability-only GPS to background lane", () => {
    expect(
      resolveGpsEventFlushPriority("driver_location_update", { observabilityOnly: true })
    ).toBe("background");
  });

  it("resolveFlushDelayMs uses P0 pipeline when lanes disabled", () => {
    expect(resolveFlushDelayMs("critical", false)).toBe(0);
    expect(resolveFlushDelayMs("visible", false)).toBe(REALTIME_FLUSH_MS);
    expect(resolveFlushDelayMs("background", false)).toBe(REALTIME_FLUSH_MS);
  });

  it("resolveFlushDelayMs orders critical < visible < background when lanes enabled", () => {
    const critical = resolveFlushDelayMs("critical", true);
    const visible = resolveFlushDelayMs("visible", true);
    const background = resolveFlushDelayMs("background", true);
    expect(critical).toBe(0);
    expect(visible).toBeLessThan(background);
    expect(visible).toBe(REALTIME_FLUSH_MS);
  });
});
