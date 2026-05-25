import { describe, expect, it } from "@jest/globals";
import {
  cockpitLiveStatusLabel,
  resolveCockpitLiveStatus,
} from "./cockpitLiveStatus";

describe("cockpitLiveStatus", () => {
  it("maps transport healthy to LIVE", () => {
    expect(resolveCockpitLiveStatus("healthy", { realtimeSocketExpected: true })).toBe(
      "connected"
    );
  });

  it("maps transport reconnecting without degrading pill", () => {
    expect(resolveCockpitLiveStatus("reconnecting", { realtimeSocketExpected: true })).toBe(
      "reconnecting"
    );
    expect(cockpitLiveStatusLabel("reconnecting")).toBe("Reconnexion…");
  });

  it("maps idle without socket expected to http fallback", () => {
    expect(resolveCockpitLiveStatus("idle", { realtimeSocketExpected: false })).toBe(
      "http_fallback"
    );
  });

  it("maps failed transport to offline", () => {
    expect(resolveCockpitLiveStatus("failed")).toBe("offline");
  });
});
