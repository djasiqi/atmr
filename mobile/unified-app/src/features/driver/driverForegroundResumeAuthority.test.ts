import { describe, expect, it, beforeEach } from "@jest/globals";
import {
  armDriverForegroundResumeAfterSessionReady,
  disarmDriverForegroundResumeAuthority,
  emitDriverForegroundAppStateForTests,
  emitDriverProcessForegroundForTests,
  emitDriverStartedActivityCountForTests,
  emitDriverWindowFocusForTests,
  getDriverResumeEpoch,
  resetDriverForegroundResumeAuthorityForTests,
  setDriverResumeAuthorityPlatformForTests,
  subscribeDriverForegroundResume,
  subscribeDriverProcessForeground,
  tryClaimDriverResumeWork,
  wasDriverForegroundResumeRecent,
} from "./driverForegroundResumeAuthority";

describe("driverForegroundResumeAuthority", () => {
  beforeEach(() => {
    resetDriverForegroundResumeAuthorityForTests();
  });

  it("reste à epoch 0 tant que SESSION_READY n’a pas armé le resume", () => {
    disarmDriverForegroundResumeAuthority();
    const epochs: number[] = [];
    const stop = subscribeDriverForegroundResume((nextEpoch) => {
      epochs.push(nextEpoch);
    });
    emitDriverProcessForegroundForTests(false);
    emitDriverProcessForegroundForTests(true);
    expect(epochs).toEqual([]);
    expect(getDriverResumeEpoch()).toBe(0);
    armDriverForegroundResumeAfterSessionReady();
    expect(getDriverResumeEpoch()).toBe(0);
    emitDriverProcessForegroundForTests(true);
    expect(epochs).toEqual([]);
    emitDriverProcessForegroundForTests(false);
    emitDriverProcessForegroundForTests(true);
    expect(epochs).toEqual([1]);
    stop();
  });

  it("crée un epoch seulement sur process background → foreground", () => {
    const epochs: number[] = [];
    const stop = subscribeDriverForegroundResume((nextEpoch) => {
      epochs.push(nextEpoch);
    });
    emitDriverProcessForegroundForTests(false);
    emitDriverProcessForegroundForTests(true);
    emitDriverProcessForegroundForTests(true);
    expect(epochs).toHaveLength(1);
    expect(getDriverResumeEpoch()).toBe(1);
    expect(wasDriverForegroundResumeRecent(5_000)).toBe(true);
    expect(tryClaimDriverResumeWork("runtime", 1)).toBe(true);
    expect(tryClaimDriverResumeWork("runtime", 1)).toBe(false);
    expect(tryClaimDriverResumeWork("resync", 1)).toBe(true);
    emitDriverProcessForegroundForTests(false);
    emitDriverProcessForegroundForTests(true);
    expect(epochs).toEqual([1, 2]);
    stop();
  });

  it("iOS : AppState background → active crée un epoch processus", () => {
    setDriverResumeAuthorityPlatformForTests("ios");
    const epochs: number[] = [];
    const stop = subscribeDriverForegroundResume((nextEpoch) => {
      epochs.push(nextEpoch);
    });
    emitDriverForegroundAppStateForTests("active");
    expect(getDriverResumeEpoch()).toBe(0);
    emitDriverForegroundAppStateForTests("background");
    emitDriverForegroundAppStateForTests("active");
    expect(epochs).toEqual([1]);
    expect(getDriverResumeEpoch()).toBe(1);
    stop();
  });

  it("Android : AppState / onHostPause ne crée pas d’epoch", () => {
    setDriverResumeAuthorityPlatformForTests("android");
    const epochs: number[] = [];
    const stop = subscribeDriverForegroundResume((nextEpoch) => {
      epochs.push(nextEpoch);
    });
    emitDriverForegroundAppStateForTests("inactive");
    emitDriverForegroundAppStateForTests("background");
    emitDriverForegroundAppStateForTests("active");
    emitDriverForegroundAppStateForTests("background");
    emitDriverForegroundAppStateForTests("active");
    expect(epochs).toEqual([]);
    expect(getDriverResumeEpoch()).toBe(0);
    emitDriverProcessForegroundForTests(false);
    emitDriverProcessForegroundForTests(true);
    expect(epochs).toEqual([1]);
    stop();
  });

  it("Android : window_focus / blur ne créent pas d’epoch", () => {
    setDriverResumeAuthorityPlatformForTests("android");
    const epochs: number[] = [];
    const stop = subscribeDriverForegroundResume((nextEpoch) => {
      epochs.push(nextEpoch);
    });
    emitDriverWindowFocusForTests(false);
    emitDriverWindowFocusForTests(true);
    emitDriverWindowFocusForTests(false);
    emitDriverWindowFocusForTests(true);
    expect(epochs).toEqual([]);
    expect(getDriverResumeEpoch()).toBe(0);
    stop();
  });

  it("Android : overlay STARTED 1→2→1 ne crée pas d’epoch ; 0→1 oui", () => {
    setDriverResumeAuthorityPlatformForTests("android");
    const epochs: number[] = [];
    const stop = subscribeDriverForegroundResume((nextEpoch) => {
      epochs.push(nextEpoch);
    });
    emitDriverStartedActivityCountForTests(1);
    emitDriverStartedActivityCountForTests(2);
    emitDriverStartedActivityCountForTests(1);
    expect(epochs).toEqual([]);
    expect(getDriverResumeEpoch()).toBe(0);
    emitDriverStartedActivityCountForTests(0);
    emitDriverStartedActivityCountForTests(1);
    expect(epochs).toEqual([1]);
    expect(getDriverResumeEpoch()).toBe(1);
    stop();
  });

  it("notifie le premier plan processus même sans nouvel epoch", () => {
    const foregrounds: boolean[] = [];
    const stop = subscribeDriverProcessForeground((next) => {
      foregrounds.push(next);
    });
    emitDriverProcessForegroundForTests(true);
    emitDriverProcessForegroundForTests(false);
    emitDriverProcessForegroundForTests(false);
    emitDriverProcessForegroundForTests(true);
    expect(foregrounds).toEqual([false, true]);
    stop();
  });
});
