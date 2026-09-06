import { AppState, Platform, type AppStateStatus } from "react-native";
import { noteDriverAppStateForAttribution } from "./driverLifecycleAttribution";

/**
 * DRIVER-RUNTIME-01C-A — React lifecycle ≠ process foreground lifecycle.
 *
 * resumeEpoch += 1 UNIQUEMENT SI :
 *   PROCESS_FOREGROUND false → true
 *   (startedActivityCount 0 → 1 après avoir réellement connu 0)
 *
 * PAS SI :
 *   ReactHost.onHostPause / onHostResume / onUserLeaveHint
 *   AppState / window focus / Activity unique
 *   GrantPermissionsActivity au-dessus de MainActivity
 *
 * Android : compteur natif STARTED = autorité. Le reste = télémétrie.
 * iOS     : AppState reste le cycle processus.
 */

export type DriverResumeWorkKind = "runtime" | "resync" | "fcm";

type ResumeListener = (epoch: number) => void;
type ProcessForegroundListener = (foreground: boolean) => void;

type NativeStartedCountModule = {
  getStartedActivityCount?: () => number;
  addListener?: (
    eventName: string,
    listener: (event: { count?: number }) => void
  ) => { remove: () => void };
};

let epoch = 0;
let resumeArmed = false;
let processForeground = true;
let previousState: AppStateStatus = AppState.currentState;
let lastResumeAtMs = 0;
let startedActivityCount = 1;
let platformOverrideForTests: string | null = null;
const subscriptions: { remove: () => void }[] = [];
const listeners = new Set<ResumeListener>();
const processForegroundListeners = new Set<ProcessForegroundListener>();
const claimed = new Map<DriverResumeWorkKind, number>();

function resumePlatform(): string {
  return platformOverrideForTests ?? Platform?.OS ?? "ios";
}

function usesNativeStartedCountAuthority(): boolean {
  return resumePlatform() === "android";
}

function getNativeStartedCountModule(): NativeStartedCountModule | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const core = require("expo-modules-core") as {
      requireOptionalNativeModule?: (name: string) => NativeStartedCountModule | null;
    };
    if (typeof core.requireOptionalNativeModule !== "function") return null;
    return core.requireOptionalNativeModule("DriverProcessLifecycle");
  } catch {
    return null;
  }
}

function notifyEpochListeners(): void {
  const current = epoch;
  for (const listener of listeners) {
    listener(current);
  }
}

function notifyProcessForegroundListeners(): void {
  const current = processForeground;
  for (const listener of processForegroundListeners) {
    listener(current);
  }
}

function applyProcessForeground(next: boolean, source: string): boolean {
  if (!resumeArmed) {
    if (processForeground === next) return false;
    processForeground = next;
    notifyProcessForegroundListeners();
    return false;
  }
  if (processForeground === next) return false;
  if (processForeground && !next) {
    processForeground = false;
    notifyProcessForegroundListeners();
    return false;
  }
  processForeground = true;
  epoch += 1;
  lastResumeAtMs = Date.now();
  notifyEpochListeners();
  notifyProcessForegroundListeners();
  noteDriverAppStateForAttribution({
    prev: previousState,
    next: previousState,
    createdEpoch: true,
    resumeArmed,
    resumeEpoch: epoch,
    processForeground,
    source,
    startedActivityCount,
  });
  return true;
}

function applyStartedActivityCount(count: number, source: string): void {
  startedActivityCount = Math.max(0, count);
  const createdEpoch = applyProcessForeground(startedActivityCount > 0, source);
  if (createdEpoch) return;
  noteDriverAppStateForAttribution({
    prev: previousState,
    next: previousState,
    createdEpoch: false,
    resumeArmed,
    resumeEpoch: epoch,
    processForeground,
    source,
    startedActivityCount,
  });
}

function handleAppStateChange(next: AppStateStatus): void {
  const prev = previousState;
  previousState = next;
  let createdEpoch = false;
  if (!usesNativeStartedCountAuthority()) {
    if (next === "active") {
      createdEpoch = applyProcessForeground(true, "ios_app_state");
    } else {
      applyProcessForeground(false, "ios_app_state");
    }
  }
  noteDriverAppStateForAttribution({
    prev,
    next,
    createdEpoch,
    resumeArmed,
    resumeEpoch: epoch,
    processForeground,
    source: usesNativeStartedCountAuthority() ? "android_app_state_ignored" : "ios_app_state",
    startedActivityCount,
  });
}

function handleWindowBlur(): void {
  noteDriverAppStateForAttribution({
    prev: previousState,
    next: previousState,
    createdEpoch: false,
    resumeArmed,
    resumeEpoch: epoch,
    processForeground,
    source: "window_blur_telemetry",
    startedActivityCount,
  });
}

function handleWindowFocus(): void {
  noteDriverAppStateForAttribution({
    prev: previousState,
    next: previousState,
    createdEpoch: false,
    resumeArmed,
    resumeEpoch: epoch,
    processForeground,
    source: "window_focus_telemetry",
    startedActivityCount,
  });
}

function ensureLifecycleSubscriptions(): void {
  if (subscriptions.length > 0) return;
  previousState = AppState.currentState;
  if (typeof AppState.addEventListener === "function") {
    subscriptions.push(AppState.addEventListener("change", handleAppStateChange));
    try {
      subscriptions.push(AppState.addEventListener("blur", handleWindowBlur));
      subscriptions.push(AppState.addEventListener("focus", handleWindowFocus));
    } catch {
      /* focus/blur absents */
    }
  }
  if (usesNativeStartedCountAuthority()) {
    const native = getNativeStartedCountModule();
    if (native) {
      try {
        const current = native.getStartedActivityCount?.();
        if (typeof current === "number") {
          applyStartedActivityCount(current, "android_started_count_seed");
        }
      } catch {
        /* module présent mais pas encore prêt */
      }
      if (typeof native.addListener === "function") {
        subscriptions.push(
          native.addListener("onStartedActivityCountChanged", (event) => {
            const count = typeof event?.count === "number" ? event.count : startedActivityCount;
            applyStartedActivityCount(count, "android_started_count");
          })
        );
      }
    } else {
      noteDriverAppStateForAttribution({
        prev: previousState,
        next: previousState,
        createdEpoch: false,
        resumeArmed,
        resumeEpoch: epoch,
        processForeground,
        source: "android_native_unavailable",
        startedActivityCount,
      });
    }
  }
}

/**
 * Cold start / SESSION_READY = epoch 0, processus déjà au premier plan.
 */
export function armDriverForegroundResumeAfterSessionReady(): void {
  resumeArmed = true;
  processForeground = true;
  previousState = "active";
  epoch = 0;
  lastResumeAtMs = 0;
  claimed.clear();
}

export function disarmDriverForegroundResumeAuthority(): void {
  resumeArmed = false;
  processForeground = false;
  epoch = 0;
  lastResumeAtMs = 0;
  claimed.clear();
}

export function getDriverResumeEpoch(): number {
  return epoch;
}

export function isDriverProcessForeground(): boolean {
  return processForeground;
}

export function getDriverStartedActivityCountForTests(): number {
  return startedActivityCount;
}

export function getDriverForegroundResumeListenerCountForTests(): number {
  return listeners.size;
}

export function wasDriverForegroundResumeRecent(windowMs = 2500): boolean {
  return lastResumeAtMs > 0 && Date.now() - lastResumeAtMs < windowMs;
}

export function subscribeDriverForegroundResume(listener: ResumeListener): () => void {
  ensureLifecycleSubscriptions();
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Tout changement réel de premier plan processus (pas AppState React). */
export function subscribeDriverProcessForeground(
  listener: ProcessForegroundListener
): () => void {
  ensureLifecycleSubscriptions();
  processForegroundListeners.add(listener);
  return () => {
    processForegroundListeners.delete(listener);
  };
}

/** @returns false si ce travail a déjà été pris pour cet epoch. */
export function tryClaimDriverResumeWork(
  kind: DriverResumeWorkKind,
  forEpoch: number
): boolean {
  if (forEpoch <= 0) return false;
  if (claimed.get(kind) === forEpoch) return false;
  claimed.set(kind, forEpoch);
  return true;
}

export function setDriverResumeAuthorityPlatformForTests(next: string | null): void {
  platformOverrideForTests = next;
}

export function resetDriverForegroundResumeAuthorityForTests(): void {
  for (const sub of subscriptions) {
    sub.remove();
  }
  subscriptions.length = 0;
  epoch = 0;
  lastResumeAtMs = 0;
  resumeArmed = true;
  processForeground = true;
  startedActivityCount = 1;
  previousState = AppState.currentState;
  claimed.clear();
  listeners.clear();
  processForegroundListeners.clear();
  platformOverrideForTests = null;
}

export function emitDriverForegroundAppStateForTests(next: AppStateStatus): void {
  ensureLifecycleSubscriptions();
  handleAppStateChange(next);
}

export function emitDriverWindowFocusForTests(hasFocus: boolean): void {
  ensureLifecycleSubscriptions();
  if (hasFocus) handleWindowFocus();
  else handleWindowBlur();
}

export function emitDriverStartedActivityCountForTests(count: number): void {
  ensureLifecycleSubscriptions();
  applyStartedActivityCount(count, "test_started_count");
}

export function emitDriverProcessForegroundForTests(next: boolean): void {
  ensureLifecycleSubscriptions();
  applyProcessForeground(next, "test_process");
}
