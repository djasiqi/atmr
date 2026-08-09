import { computeFixAgeMs, WATCH_STALE_MS } from "./driverTrackingFixAge";

describe("driverTrackingFixAge", () => {
  it("refuse un fix watch vieux de plus de 25 s (âge max timestamp/watch)", () => {
    const now = 1_000_000;
    const fresh = computeFixAgeMs({ timestamp: now - 5_000 }, now - 5_000, now);
    expect(fresh).toBeLessThan(WATCH_STALE_MS);
    const staleByGpsTs = computeFixAgeMs({ timestamp: now - 30_000 }, now - 1_000, now);
    expect(staleByGpsTs).toBeGreaterThan(WATCH_STALE_MS);
    const staleByWatch = computeFixAgeMs({ timestamp: now - 1_000 }, now - 30_000, now);
    expect(staleByWatch).toBeGreaterThan(WATCH_STALE_MS);
    expect(computeFixAgeMs({ timestamp: now - WATCH_STALE_MS }, now, now)).toBe(WATCH_STALE_MS);
  });
});
