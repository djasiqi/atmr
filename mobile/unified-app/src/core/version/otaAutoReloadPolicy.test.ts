import { afterEach, describe, expect, it } from "@jest/globals";

import {
  __resetOtaAutoReloadMissionGuardForTests,
  hasActiveDriverMissionStatus,
  isOtaAutoReloadMissionBlocking,
  setOtaAutoReloadMissionBlocking,
} from "./otaAutoReloadMissionGuard";
import { evaluateOtaAutoReload } from "./otaAutoReloadPolicy";

describe("otaAutoReloadPolicy", () => {
  afterEach(() => {
    __resetOtaAutoReloadMissionGuardForTests();
  });

  it("autorise le reload quand update pending et garde-fous OK", () => {
    const result = evaluateOtaAutoReload({
      updatesEnabled: true,
      isDev: false,
      featureEnabled: true,
      appState: "active",
      missionBlocking: false,
      reloadConsumedThisSession: false,
      startupReady: true,
      isUpdatePending: true,
    });

    expect(result).toEqual({ allowed: true, deferReason: null });
  });

  it("diffère si mission chauffeur active", () => {
    setOtaAutoReloadMissionBlocking(true);

    const result = evaluateOtaAutoReload({
      updatesEnabled: true,
      isDev: false,
      featureEnabled: true,
      appState: "active",
      reloadConsumedThisSession: false,
      startupReady: true,
      isUpdatePending: true,
    });

    expect(result.allowed).toBe(false);
    expect(result.deferReason).toBe("active_mission");
    expect(isOtaAutoReloadMissionBlocking()).toBe(true);
  });

  it("ignore en dev et si flag désactivé", () => {
    expect(
      evaluateOtaAutoReload({
        updatesEnabled: true,
        isDev: true,
        featureEnabled: true,
        appState: "active",
        reloadConsumedThisSession: false,
        startupReady: true,
        isUpdatePending: true,
      }).deferReason
    ).toBe("dev");

    expect(
      evaluateOtaAutoReload({
        updatesEnabled: true,
        isDev: false,
        featureEnabled: false,
        appState: "active",
        reloadConsumedThisSession: false,
        startupReady: true,
        isUpdatePending: true,
      }).deferReason
    ).toBe("disabled");
  });

  it("bloque un second reload dans la même session", () => {
    const result = evaluateOtaAutoReload({
      updatesEnabled: true,
      isDev: false,
      featureEnabled: true,
      appState: "active",
      reloadConsumedThisSession: true,
      startupReady: true,
      isUpdatePending: true,
    });

    expect(result.deferReason).toBe("already_reloaded_session");
  });
});

describe("otaAutoReloadMissionGuard", () => {
  afterEach(() => {
    __resetOtaAutoReloadMissionGuardForTests();
  });

  it("détecte les statuts mission actifs", () => {
    expect(hasActiveDriverMissionStatus("IN_PROGRESS")).toBe(true);
    expect(hasActiveDriverMissionStatus("COMPLETED")).toBe(false);
    expect(hasActiveDriverMissionStatus(null)).toBe(false);
  });
});
