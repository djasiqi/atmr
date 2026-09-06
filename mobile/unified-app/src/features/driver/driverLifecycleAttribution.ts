import { AppState, type AppStateStatus } from "react-native";
import { emitDriverTelemetry } from "../../core/observability/driverTelemetry";
import { isDriverSessionNetworkReady } from "../../core/network/driverSessionNetworkGate";

type AttributionEvent = "app_state" | "focus" | "blur";

let started = false;
let lastKnownEpoch = 0;
let lastKnownArmed = false;
const subscriptions: { remove: () => void }[] = [];

function monotonicMs(): number {
  const perf = (globalThis as { performance?: { now?: () => number } }).performance;
  if (perf && typeof perf.now === "function") return Math.round(perf.now());
  return Date.now();
}

export function emitDriverLifecycleAttribution(input: {
  event: AttributionEvent;
  prevAppState?: AppStateStatus | string;
  nextAppState?: AppStateStatus | string;
  createdEpoch?: boolean;
  resumeArmed?: boolean;
  resumeEpoch?: number;
  processForeground?: boolean;
  source?: string;
  startedActivityCount?: number;
}): void {
  if (typeof input.resumeEpoch === "number") lastKnownEpoch = input.resumeEpoch;
  if (typeof input.resumeArmed === "boolean") lastKnownArmed = input.resumeArmed;
  const payload = {
    source: "driver.lifecycle.attribution",
    event: input.event,
    prev_app_state: input.prevAppState ?? AppState.currentState,
    next_app_state: input.nextAppState ?? AppState.currentState,
    app_state: AppState.currentState,
    resume_armed: lastKnownArmed,
    resume_epoch: lastKnownEpoch,
    process_foreground: input.processForeground ?? null,
    started_activity_count: input.startedActivityCount ?? null,
    authority_source: input.source ?? null,
    session_network_ready: isDriverSessionNetworkReady(),
    created_epoch: Boolean(input.createdEpoch),
    monotonic_ms: monotonicMs(),
    wall_ms: Date.now(),
  };
  emitDriverTelemetry("driver.lifecycle.attribution", payload);
  console.info("[LIFECYCLE-01C]", payload);
}

/** Appelé par l’autorité resume : n’altère pas la décision d’epoch. */
export function noteDriverAppStateForAttribution(input: {
  prev: AppStateStatus;
  next: AppStateStatus;
  createdEpoch: boolean;
  resumeArmed: boolean;
  resumeEpoch: number;
  processForeground?: boolean;
  source?: string;
  startedActivityCount?: number;
}): void {
  emitDriverLifecycleAttribution({
    event: "app_state",
    prevAppState: input.prev,
    nextAppState: input.next,
    createdEpoch: input.createdEpoch,
    resumeArmed: input.resumeArmed,
    resumeEpoch: input.resumeEpoch,
    processForeground: input.processForeground,
    source: input.source,
    startedActivityCount: input.startedActivityCount,
  });
}

export function startDriverLifecycleAttribution(): void {
  if (started) return;
  started = true;
  if (typeof AppState.addEventListener !== "function") return;

  for (const event of ["focus", "blur"] as const) {
    try {
      subscriptions.push(
        AppState.addEventListener(event, () => {
          emitDriverLifecycleAttribution({
            event,
            prevAppState: AppState.currentState,
            nextAppState: AppState.currentState,
            resumeEpoch: lastKnownEpoch,
            resumeArmed: lastKnownArmed,
          });
        })
      );
    } catch {
      /* focus/blur absents selon la plateforme */
    }
  }
}

export function resetDriverLifecycleAttributionForTests(): void {
  for (const sub of subscriptions) {
    sub.remove();
  }
  subscriptions.length = 0;
  started = false;
  lastKnownEpoch = 0;
  lastKnownArmed = false;
}
