import { describe, expect, it } from "@jest/globals";
import {
  getMsUntilNextWindowEdge,
  getNextTrackingWindowEdge,
  getTrackingWindowConfig,
  isWithinTrackingWindow,
} from "./trackingWindow";

const CONFIG_07_19 = { startHour: 7, endHour: 19 };

function dateAt(year: number, month: number, day: number, hour: number, minute = 0): Date {
  return new Date(year, month, day, hour, minute, 0, 0);
}

describe("trackingWindow", () => {
  describe("isWithinTrackingWindow", () => {
    it("is true at the open boundary 07:00", () => {
      expect(isWithinTrackingWindow(dateAt(2026, 4, 8, 7, 0), CONFIG_07_19)).toBe(true);
    });

    it("is true mid-day at 12:30", () => {
      expect(isWithinTrackingWindow(dateAt(2026, 4, 8, 12, 30), CONFIG_07_19)).toBe(true);
    });

    it("is true at 18:59 (last minute open)", () => {
      expect(isWithinTrackingWindow(dateAt(2026, 4, 8, 18, 59), CONFIG_07_19)).toBe(true);
    });

    it("is false at 19:00 sharp (close boundary excluded)", () => {
      expect(isWithinTrackingWindow(dateAt(2026, 4, 8, 19, 0), CONFIG_07_19)).toBe(false);
    });

    it("is false at 19:30 (mission may still run via mission pipeline)", () => {
      expect(isWithinTrackingWindow(dateAt(2026, 4, 8, 19, 30), CONFIG_07_19)).toBe(false);
    });

    it("is false at 02:00 night", () => {
      expect(isWithinTrackingWindow(dateAt(2026, 4, 8, 2, 0), CONFIG_07_19)).toBe(false);
    });

    it("is false at 06:59 (one minute before open)", () => {
      expect(isWithinTrackingWindow(dateAt(2026, 4, 8, 6, 59), CONFIG_07_19)).toBe(false);
    });
  });

  describe("getNextTrackingWindowEdge", () => {
    it("returns the same-day close edge when window is open", () => {
      const now = dateAt(2026, 4, 8, 12, 30);
      const edge = getNextTrackingWindowEdge(now, CONFIG_07_19);
      expect(edge.type).toBe("close");
      expect(edge.at.getHours()).toBe(19);
      expect(edge.at.getMinutes()).toBe(0);
      expect(edge.at.getDate()).toBe(8);
    });

    it("returns the same-day open edge when window is closed in early morning", () => {
      const now = dateAt(2026, 4, 8, 3, 0);
      const edge = getNextTrackingWindowEdge(now, CONFIG_07_19);
      expect(edge.type).toBe("open");
      expect(edge.at.getHours()).toBe(7);
      expect(edge.at.getDate()).toBe(8);
    });

    it("returns the next-day open edge when window just closed", () => {
      const now = dateAt(2026, 4, 8, 22, 0);
      const edge = getNextTrackingWindowEdge(now, CONFIG_07_19);
      expect(edge.type).toBe("open");
      expect(edge.at.getHours()).toBe(7);
      expect(edge.at.getDate()).toBe(9);
    });

    it("returns close edge at exactly 07:00 (we are open and next bord is 19:00 same day)", () => {
      const now = dateAt(2026, 4, 8, 7, 0);
      const edge = getNextTrackingWindowEdge(now, CONFIG_07_19);
      expect(edge.type).toBe("close");
      expect(edge.at.getHours()).toBe(19);
      expect(edge.at.getDate()).toBe(8);
    });
  });

  describe("getMsUntilNextWindowEdge", () => {
    it("returns at least 60s minimum bound to avoid setTimeout(0) loops", () => {
      const now = dateAt(2026, 4, 8, 19, 0);
      const ms = getMsUntilNextWindowEdge(new Date(now.getTime()), CONFIG_07_19);
      expect(ms).toBeGreaterThanOrEqual(60_000);
    });

    it("returns ~6h30 to close at 19:00 from 12:30", () => {
      const now = dateAt(2026, 4, 8, 12, 30);
      const ms = getMsUntilNextWindowEdge(now, CONFIG_07_19);
      expect(ms).toBe(6 * 60 * 60_000 + 30 * 60_000);
    });
  });

  describe("getTrackingWindowConfig", () => {
    const ORIGINAL_ENV = { ...process.env };
    afterEach(() => {
      process.env = { ...ORIGINAL_ENV };
    });

    it("falls back to defaults 07/19 without env vars", () => {
      delete process.env.EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_START_HOUR;
      delete process.env.EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_END_HOUR;
      expect(getTrackingWindowConfig()).toEqual({ startHour: 7, endHour: 19 });
    });

    it("respects custom hours from env", () => {
      process.env.EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_START_HOUR = "6";
      process.env.EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_END_HOUR = "20";
      expect(getTrackingWindowConfig()).toEqual({ startHour: 6, endHour: 20 });
    });

    it("falls back when env config is invalid (end <= start)", () => {
      process.env.EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_START_HOUR = "20";
      process.env.EXPO_PUBLIC_DRIVER_TRACKING_WINDOW_END_HOUR = "10";
      expect(getTrackingWindowConfig()).toEqual({ startHour: 7, endHour: 19 });
    });
  });
});
