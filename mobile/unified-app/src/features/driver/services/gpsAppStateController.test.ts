import { describe, expect, it, beforeEach } from "@jest/globals";
import {
  GPS_OSCILLATION_MAX_TRANSITIONS,
  buildGpsControllerDecisionKey,
  invalidateGpsControllerDecision,
  isDirectResumeNativeStartReason,
  isGpsOscillationOpen,
  recordGpsControllerTransition,
  resetGpsAppStateControllerForTests,
  resetGpsOscillationOnTrustedSignal,
  resolveGpsControllerForeground,
  resolveGpsMissionStartHold,
  shouldApplyGpsControllerDecision,
  shouldIgnoreAppStateForGps,
} from "./gpsAppStateController";

describe("gpsAppStateController", () => {
  beforeEach(() => {
    resetGpsAppStateControllerForTests();
  });

  it("Android ignore AppState pour le GPS ; iOS le conserve", () => {
    expect(shouldIgnoreAppStateForGps("android")).toBe(true);
    expect(shouldIgnoreAppStateForGps("ios")).toBe(false);
  });

  it("Android : le premier plan processus l’emporte sur AppState transitoire", () => {
    expect(
      resolveGpsControllerForeground({
        platform: "android",
        appState: "background",
        processForeground: true,
      })
    ).toBe(true);
    expect(
      resolveGpsControllerForeground({
        platform: "android",
        appState: "active",
        processForeground: false,
      })
    ).toBe(false);
  });

  it("iOS : AppState reste l’autorité processus", () => {
    expect(
      resolveGpsControllerForeground({
        platform: "ios",
        appState: "background",
        processForeground: true,
      })
    ).toBe(false);
    expect(
      resolveGpsControllerForeground({
        platform: "ios",
        appState: "active",
        processForeground: false,
      })
    ).toBe(true);
  });

  it("app_resume ne démarre jamais le service directement", () => {
    expect(isDirectResumeNativeStartReason("app_resume")).toBe(true);
    expect(isDirectResumeNativeStartReason("app_resume_pending")).toBe(true);
    expect(isDirectResumeNativeStartReason("ensure_manager_state")).toBe(false);
    expect(isDirectResumeNativeStartReason("mission_started")).toBe(false);
  });

  it("pending / awaiting_start / owner nul : 0 démarrage mission", () => {
    expect(
      resolveGpsMissionStartHold({
        snapshot: { status: "pending" },
        bridgeMissionId: null,
        nativeOwnerPresent: true,
        presenceWindow: false,
      }).reason
    ).toBe("mission_snapshot_pending");

    expect(
      resolveGpsMissionStartHold({
        snapshot: { status: "resolved_mission", missionId: 45711 },
        bridgeMissionId: null,
        nativeOwnerPresent: true,
        presenceWindow: false,
      }).reason
    ).toBe("mission_snapshot_awaiting_start");

    expect(
      resolveGpsMissionStartHold({
        snapshot: { status: "resolved_none" },
        bridgeMissionId: null,
        nativeOwnerPresent: false,
        presenceWindow: false,
      }).reason
    ).toBe("native_owner_absent");

    expect(
      resolveGpsMissionStartHold({
        snapshot: { status: "resolved_mission", missionId: 45711 },
        bridgeMissionId: 45711,
        nativeOwnerPresent: true,
        presenceWindow: false,
      }).blocked
    ).toBe(false);
  });

  it("décision identique = no-op ; clé différente = appliquée", () => {
    const key = buildGpsControllerDecisionKey(["hold", "awaiting_start", 45711]);
    expect(shouldApplyGpsControllerDecision(key)).toBe(true);
    expect(shouldApplyGpsControllerDecision(key)).toBe(false);
    invalidateGpsControllerDecision();
    expect(shouldApplyGpsControllerDecision(key)).toBe(true);
  });

  it("coupe-circuit : trop de start/stop dans la fenêtre", () => {
    const t0 = 1_000_000;
    for (let i = 0; i < GPS_OSCILLATION_MAX_TRANSITIONS - 1; i += 1) {
      expect(recordGpsControllerTransition(i % 2 === 0 ? "start" : "stop", t0 + i * 10)).toBe(
        "ok"
      );
    }
    expect(
      recordGpsControllerTransition("start", t0 + GPS_OSCILLATION_MAX_TRANSITIONS * 10)
    ).toBe("tripped");
    expect(isGpsOscillationOpen(t0 + 50)).toBe(true);
    resetGpsOscillationOnTrustedSignal();
    expect(isGpsOscillationOpen(t0 + 50)).toBe(false);
  });
});
