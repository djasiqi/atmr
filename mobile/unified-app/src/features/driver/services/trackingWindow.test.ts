import { describe, expect, it, afterEach } from "@jest/globals";
import { zonedWallClockToUtcDate } from "./businessTime";
import {
  getMsUntilNextWindowEdge,
  getNextTrackingWindowEdge,
  getTrackingWindowConfig,
  isWithinTrackingWindow,
} from "./trackingWindow";

const CONFIG_07_19 = { startHour: 7, endHour: 19 };

/** Instant absolu = horloge murale Europe/Zurich. */
function zurichAt(
  year: number,
  month: number,
  day: number,
  hour: number,
  minute = 0,
  second = 0
): Date {
  return zonedWallClockToUtcDate(year, month, day, hour, minute, second);
}

describe("trackingWindow (Europe/Zurich)", () => {
  describe("isWithinTrackingWindow", () => {
    it("is true at the open boundary 07:00 Zurich", () => {
      expect(isWithinTrackingWindow(zurichAt(2026, 4, 8, 7, 0), CONFIG_07_19)).toBe(true);
    });

    it("is true mid-day at 12:30 Zurich", () => {
      expect(isWithinTrackingWindow(zurichAt(2026, 4, 8, 12, 30), CONFIG_07_19)).toBe(
        true
      );
    });

    it("is true at 18:59:59 Zurich", () => {
      expect(isWithinTrackingWindow(zurichAt(2026, 4, 8, 18, 59, 59), CONFIG_07_19)).toBe(
        true
      );
    });

    it("is false at 19:00 Zurich", () => {
      expect(isWithinTrackingWindow(zurichAt(2026, 4, 8, 19, 0), CONFIG_07_19)).toBe(
        false
      );
    });

    it("is false at 19:30 Zurich", () => {
      expect(isWithinTrackingWindow(zurichAt(2026, 4, 8, 19, 30), CONFIG_07_19)).toBe(
        false
      );
    });

    it("is false at 02:00 Zurich", () => {
      expect(isWithinTrackingWindow(zurichAt(2026, 4, 8, 2, 0), CONFIG_07_19)).toBe(
        false
      );
    });

    it("is false at 06:59 Zurich", () => {
      expect(isWithinTrackingWindow(zurichAt(2026, 4, 8, 6, 59), CONFIG_07_19)).toBe(
        false
      );
    });

    it("same absolute instant: result independent of local TZ interpretation", () => {
      // 11 août 2026 10:00 Zurich = 08:00Z (été)
      const instant = new Date("2026-08-11T08:00:00.000Z");
      expect(isWithinTrackingWindow(instant, CONFIG_07_19)).toBe(true);
      // 11 août 2026 20:00 Zurich = 18:00Z → hors fenêtre
      const evening = new Date("2026-08-11T18:00:00.000Z");
      expect(isWithinTrackingWindow(evening, CONFIG_07_19)).toBe(false);
    });
  });

  describe("getNextTrackingWindowEdge — absolute UTC", () => {
    it("11 août 2026 19:00 Zurich → close edge of that day = 17:00Z", () => {
      // Juste avant 19:00 : next = close 19:00 = 17:00Z
      const beforeClose = zurichAt(2026, 8, 11, 18, 59, 0);
      const edge = getNextTrackingWindowEdge(beforeClose, CONFIG_07_19);
      expect(edge.type).toBe("close");
      expect(edge.at.toISOString()).toBe("2026-08-11T17:00:00.000Z");
    });

    it("11 janvier 2026 18:59 Zurich → close edge = 18:00Z", () => {
      const beforeClose = zurichAt(2026, 1, 11, 18, 59, 0);
      const edge = getNextTrackingWindowEdge(beforeClose, CONFIG_07_19);
      expect(edge.type).toBe("close");
      expect(edge.at.toISOString()).toBe("2026-01-11T18:00:00.000Z");
    });

    it("returns open edge next morning after close", () => {
      const afterClose = zurichAt(2026, 8, 11, 19, 0, 0);
      const edge = getNextTrackingWindowEdge(afterClose, CONFIG_07_19);
      expect(edge.type).toBe("open");
      expect(edge.at.toISOString()).toBe("2026-08-12T05:00:00.000Z"); // 07:00 Zurich été
    });

    it("returns same-day open when early morning", () => {
      const early = zurichAt(2026, 8, 11, 3, 0, 0);
      const edge = getNextTrackingWindowEdge(early, CONFIG_07_19);
      expect(edge.type).toBe("open");
      expect(edge.at.toISOString()).toBe("2026-08-11T05:00:00.000Z");
    });
  });

  describe("getMsUntilNextWindowEdge", () => {
    it("returns at least 60s minimum bound", () => {
      const now = zurichAt(2026, 4, 8, 19, 0);
      const ms = getMsUntilNextWindowEdge(now, CONFIG_07_19);
      expect(ms).toBeGreaterThanOrEqual(60_000);
    });

    it("returns ~6h30 to close at 19:00 from 12:30 Zurich", () => {
      const now = zurichAt(2026, 4, 8, 12, 30);
      const ms = getMsUntilNextWindowEdge(now, CONFIG_07_19);
      expect(ms).toBe(6 * 60 * 60_000 + 30 * 60_000);
    });
  });

  describe("getTrackingWindowConfig — bornes figées", () => {
    const ORIGINAL_ENV = { ...process.env };
    afterEach(() => {
      process.env = { ...ORIGINAL_ENV };
    });

    it("always returns 07/19", () => {
      delete process.env.EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_START_HOUR;
      delete process.env.EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_END_HOUR;
      expect(getTrackingWindowConfig()).toEqual({ startHour: 7, endHour: 19 });
    });

    it("ignores custom env hours (no silent divergence)", () => {
      process.env.EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_START_HOUR = "6";
      process.env.EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_END_HOUR = "20";
      expect(getTrackingWindowConfig()).toEqual({ startHour: 7, endHour: 19 });
    });
  });
});
