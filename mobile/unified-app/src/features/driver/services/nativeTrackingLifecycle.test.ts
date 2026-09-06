import { afterEach, beforeEach, describe, expect, it, jest } from "@jest/globals";
import { AppState } from "react-native";
import {
  ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED,
  NATIVE_FOREGROUND_STABLE_MS,
  __resetNativeTrackingLifecycleForTests,
  canAttemptNativeStartNow,
  getNativeLifecycleInFlight,
  getNativeLifecyclePhase,
  getNativeLifecycleSnapshot,
  notifyNativeLifecycleAppState,
  requestNativeRecover,
  requestNativeStart,
  requestNativeStop,
  type NativeStartRunResult,
} from "./nativeTrackingLifecycle";

function defer<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

describe("nativeTrackingLifecycle (P0-A)", () => {
  beforeEach(() => {
    jest.useFakeTimers();
    __resetNativeTrackingLifecycleForTests();
    (AppState as { currentState: string }).currentState = "active";
    notifyNativeLifecycleAppState("active");
    // Rendre le FG immédiatement stable pour les tests hors scénario debounce
    jest.advanceTimersByTime(NATIVE_FOREGROUND_STABLE_MS + 10);
  });

  afterEach(() => {
    __resetNativeTrackingLifecycleForTests();
    jest.useRealTimers();
  });

  it("sérialise START : un seul run natif à la fois (2e coalescé)", async () => {
    const first = defer<NativeStartRunResult>();
    let runs = 0;

    const p1 = requestNativeStart({
      reason: "t1",
      run: async () => {
        runs += 1;
        return first.promise;
      },
    });

    const p2 = requestNativeStart({
      reason: "t2",
      run: async () => {
        runs += 1;
        return { ok: true, nativeStarted: true, invokedNativeStart: true };
      },
    });

    await Promise.resolve();
    expect(getNativeLifecyclePhase()).toBe("STARTING");
    expect(getNativeLifecycleInFlight()).toEqual({ start_in_flight: 1, stop_in_flight: 0 });
    expect(runs).toBe(1);

    first.resolve({ ok: true, nativeStarted: true, invokedNativeStart: true });
    const [r1, r2] = await Promise.all([p1, p2]);

    expect(r1.outcome).toBe("running");
    // 2e start pendant STARTING partage la même promesse in-flight
    expect(r2.outcome).toBe("running");
    expect(runs).toBe(1);
    expect(getNativeLifecyclePhase()).toBe("RUNNING");
    expect(getNativeLifecycleInFlight()).toEqual({ start_in_flight: 0, stop_in_flight: 0 });
  });

  it("n'autorise jamais start_in_flight ∧ stop_in_flight simultanés", async () => {
    const startGate = defer<NativeStartRunResult>();
    const stopGate = defer<{ ok: boolean; nativeStopped: boolean }>();

    const startP = requestNativeStart({
      reason: "start",
      run: async () => startGate.promise,
    });

    await Promise.resolve();
    expect(getNativeLifecycleInFlight()).toEqual({ start_in_flight: 1, stop_in_flight: 0 });

    let stopRun = 0;
    const stopP = requestNativeStop({
      reason: "stop",
      run: async () => {
        stopRun += 1;
        // Pendant STOP, start ne doit plus être in-flight
        expect(getNativeLifecycleInFlight().start_in_flight).toBe(0);
        expect(getNativeLifecycleInFlight().stop_in_flight).toBe(1);
        return stopGate.promise;
      },
    });

    // STOP est pending tant que START n'est pas résolu
    expect(stopRun).toBe(0);
    expect(getNativeLifecycleInFlight()).toEqual({ start_in_flight: 1, stop_in_flight: 0 });

    startGate.resolve({ ok: true, nativeStarted: true, invokedNativeStart: true });
    await startP;
    await Promise.resolve();
    await Promise.resolve();

    expect(stopRun).toBe(1);
    expect(getNativeLifecyclePhase()).toBe("STOPPING");
    expect(getNativeLifecycleInFlight()).toEqual({ start_in_flight: 0, stop_in_flight: 1 });

    stopGate.resolve({ ok: true, nativeStopped: true });
    await stopP;
    expect(getNativeLifecyclePhase()).toBe("STOPPED");
  });

  it("exécute START pending seulement après STOP resolved", async () => {
    const stopGate = defer<{ ok: boolean; nativeStopped: boolean }>();
    let startRuns = 0;

    // Établir RUNNING d'abord
    await requestNativeStart({
      reason: "seed",
      run: async () => ({ ok: true, nativeStarted: true, invokedNativeStart: true }),
    });

    const stopP = requestNativeStop({
      reason: "stop",
      run: async () => stopGate.promise,
    });
    await Promise.resolve();

    const startP = requestNativeStart({
      reason: "after_stop",
      run: async () => {
        startRuns += 1;
        expect(getNativeLifecyclePhase()).not.toBe("STOPPING");
        return { ok: true, nativeStarted: true, invokedNativeStart: true };
      },
    });

    expect(startRuns).toBe(0);
    stopGate.resolve({ ok: true, nativeStopped: true });
    await stopP;
    await startP;
    expect(startRuns).toBe(1);
    expect(getNativeLifecyclePhase()).toBe("RUNNING");
  });

  it("coalesce RECOVER pendant STARTING et l'exécute après", async () => {
    const startGate = defer<NativeStartRunResult>();
    let recoverRuns = 0;

    const startP = requestNativeStart({
      reason: "start",
      run: async () => startGate.promise,
    });
    await Promise.resolve();

    const recoverP = requestNativeRecover({
      reason: "anti_zombie",
      run: async () => {
        recoverRuns += 1;
        return { ok: true, nativeStarted: true, invokedNativeStart: true };
      },
    });

    expect(recoverRuns).toBe(0);
    expect(getNativeLifecycleSnapshot().recover_pending).toBe(true);

    startGate.resolve({ ok: true, nativeStarted: true, invokedNativeStart: true });
    await startP;
    await recoverP;
    expect(recoverRuns).toBe(1);
  });

  it("ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED → BLOCKED sans retry immédiat", async () => {
    const r1 = await requestNativeStart({
      reason: "bg_flip",
      run: async () => ({
        ok: false,
        nativeStarted: false,
        invokedNativeStart: true,
        errorCode: ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED,
        errorMessage: "not allowed",
      }),
    });

    expect(r1.outcome).toBe("blocked_foreground");
    expect(getNativeLifecyclePhase()).toBe("BLOCKED_FOREGROUND_REQUIRED");
    expect(canAttemptNativeStartNow()).toBe(false);

    let runs = 0;
    const r2 = await requestNativeStart({
      reason: "spam",
      run: async () => {
        runs += 1;
        return { ok: true, nativeStarted: true, invokedNativeStart: true };
      },
    });
    expect(r2.outcome).toBe("deferred_blocked");
    expect(runs).toBe(0);
  });

  it("un simple flip background→active ne débloque pas (debounce FG stable)", async () => {
    await requestNativeStart({
      reason: "block",
      run: async () => ({
        ok: false,
        nativeStarted: false,
        invokedNativeStart: true,
        errorCode: ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED,
      }),
    });
    expect(getNativeLifecyclePhase()).toBe("BLOCKED_FOREGROUND_REQUIRED");

    // Simuler oscillation : BG puis FG immédiat (sans attendre stabilité)
    notifyNativeLifecycleAppState("background");
    notifyNativeLifecycleAppState("active");

    let runs = 0;
    const deferred = await requestNativeRecover({
      reason: "flip_recover",
      run: async () => {
        runs += 1;
        return { ok: true, nativeStarted: true, invokedNativeStart: true };
      },
    });
    expect(deferred.outcome).toBe("deferred_blocked");
    expect(runs).toBe(0);
    expect(getNativeLifecycleSnapshot().foreground_stable).toBe(false);

    // Après fenêtre de stabilité + backoff initial (2s)
    jest.advanceTimersByTime(NATIVE_FOREGROUND_STABLE_MS + 2000 + 50);
    await Promise.resolve();
    await Promise.resolve();

    // L'intention mémorisée doit finir par s'exécuter une fois FG stable
    expect(runs).toBe(1);
    expect(getNativeLifecyclePhase()).toBe("RUNNING");
  });

  it("STOP concurrent pendant STARTING : jamais stop_in_flight=1 avec start_in_flight=1", async () => {
    const observed: { start: 0 | 1; stop: 0 | 1 }[] = [];
    const startGate = defer<NativeStartRunResult>();

    const unsub = (() => {
      const id = setInterval(() => {
        const f = getNativeLifecycleInFlight();
        observed.push({ start: f.start_in_flight, stop: f.stop_in_flight });
      }, 1);
      return () => clearInterval(id);
    })();

    const startP = requestNativeStart({
      reason: "s",
      run: async () => startGate.promise,
    });
    const stopP = requestNativeStop({
      reason: "x",
      run: async () => ({ ok: true, nativeStopped: true }),
    });

    await Promise.resolve();
    startGate.resolve({ ok: true, nativeStarted: true, invokedNativeStart: true });
    await Promise.all([startP, stopP]);
    unsub();

    for (const sample of observed) {
      expect(sample.start === 1 && sample.stop === 1).toBe(false);
    }
  });
});
