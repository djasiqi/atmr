/**
 * P0-A — Contrôleur unique des opérations natives Expo Location (FGS).
 *
 * Invariants (par construction) :
 * - une seule opération native à la fois (START | STOP | RECOVER)
 * - START pendant STARTING → coalescé (même promesse / pas de 2e appel natif)
 * - START pendant STOPPING → pending, exécuté après STOP
 * - STOP pendant STARTING → pending, exécuté après START
 * - RECOVER pendant op → intention coalescée, exécutée une fois l'état résolu
 * - BLOCKED_FOREGROUND_REQUIRED → pas de retry agressif ; reprise uniquement
 *   après foreground *stable* (debounce), une tentative à la fois
 */

import { AppState, type AppStateStatus, type NativeEventSubscription } from "react-native";

export type NativeLifecyclePhase =
  | "STOPPED"
  | "STARTING"
  | "RUNNING"
  | "STOPPING"
  | "RECOVERING"
  | "BLOCKED_FOREGROUND_REQUIRED";

export const ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED =
  "ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED";

/** Fenêtre de stabilité FG avant reprise depuis BLOCKED (pas un simple flip). */
export const NATIVE_FOREGROUND_STABLE_MS = Number(
  process.env.EXPO_PUBLIC_NATIVE_FOREGROUND_STABLE_MS ?? "1500"
);

const BLOCKED_BACKOFF_INITIAL_MS = 2000;
const BLOCKED_BACKOFF_MAX_MS = 30_000;
const BLOCKED_MAX_ATTEMPTS_PER_WINDOW = 3;
const BLOCKED_ATTEMPT_WINDOW_MS = 60_000;

export type NativeStartRunResult = {
  ok: boolean;
  nativeStarted: boolean;
  /** true si startLocationUpdatesAsync a réellement été invoqué */
  invokedNativeStart?: boolean;
  errorCode?: string | null;
  errorName?: string | null;
  errorMessage?: string | null;
};

export type NativeStopRunResult = {
  ok: boolean;
  nativeStopped: boolean;
  errorCode?: string | null;
  errorName?: string | null;
  errorMessage?: string | null;
};

export type NativeStartOutcome =
  | "running"
  | "already_started"
  | "deferred_not_foreground"
  | "deferred_blocked"
  | "coalesced"
  | "failed"
  | "blocked_foreground"
  | "noop";

export type NativeStopOutcome = "stopped" | "noop" | "coalesced" | "failed";

export type NativeLifecycleSnapshot = {
  phase: NativeLifecyclePhase;
  start_in_flight: 0 | 1;
  stop_in_flight: 0 | 1;
  recover_pending: boolean;
  pending_recover_on_foreground: boolean;
  foreground_stable: boolean;
  blocked_backoff_until_ms: number;
  last_reason: string | null;
};

type StartKind = "start" | "recover";

type PendingStart = {
  kind: StartKind;
  reason: string;
  run: () => Promise<NativeStartRunResult>;
  resolve: (outcome: NativeStartControllerResult) => void;
  reject: (error: unknown) => void;
};

type PendingStop = {
  reason: string;
  run: () => Promise<NativeStopRunResult>;
  resolve: (outcome: NativeStopControllerResult) => void;
  reject: (error: unknown) => void;
};

export type NativeStartControllerResult = {
  outcome: NativeStartOutcome;
  phase: NativeLifecyclePhase;
  result?: NativeStartRunResult;
};

export type NativeStopControllerResult = {
  outcome: NativeStopOutcome;
  phase: NativeLifecyclePhase;
  result?: NativeStopRunResult;
};

type Listener = (snapshot: NativeLifecycleSnapshot) => void;

let phase: NativeLifecyclePhase = "STOPPED";
let lastReason: string | null = null;
let processing = false;

let inFlightStart: {
  kind: StartKind;
  reason: string;
  promise: Promise<NativeStartControllerResult>;
} | null = null;
let inFlightStop: {
  reason: string;
  promise: Promise<NativeStopControllerResult>;
} | null = null;

let pendingStart: PendingStart | null = null;
let pendingStop: PendingStop | null = null;
let recoverNeeded: PendingStart | null = null;

let pendingRecoverOnForeground = false;
let blockedBackoffUntilMs = 0;
let blockedBackoffMs = BLOCKED_BACKOFF_INITIAL_MS;
let blockedAttemptTimestamps: number[] = [];

let rawAppState: AppStateStatus = AppState.currentState;
let activeSinceMs: number | null = rawAppState === "active" ? Date.now() : null;
let stableTimer: ReturnType<typeof setTimeout> | null = null;
let appStateSubscription: NativeEventSubscription | null = null;
let appStateBridgeInstalled = false;

const listeners = new Set<Listener>();

function nowMs(): number {
  return Date.now();
}

function isForegroundErrorCode(code: string | null | undefined): boolean {
  if (!code) return false;
  return (
    code === ERR_FOREGROUND_SERVICE_START_NOT_ALLOWED ||
    code.includes("FOREGROUND_SERVICE_START_NOT_ALLOWED")
  );
}

function pruneBlockedAttempts(at: number): void {
  blockedAttemptTimestamps = blockedAttemptTimestamps.filter(
    (ts) => at - ts < BLOCKED_ATTEMPT_WINDOW_MS
  );
}

export function isForegroundStable(at: number = nowMs()): boolean {
  if (rawAppState !== "active" || activeSinceMs == null) return false;
  return at - activeSinceMs >= NATIVE_FOREGROUND_STABLE_MS;
}

function canAttemptWhileBlocked(at: number): boolean {
  if (phase !== "BLOCKED_FOREGROUND_REQUIRED") return true;
  if (at < blockedBackoffUntilMs) return false;
  pruneBlockedAttempts(at);
  if (blockedAttemptTimestamps.length >= BLOCKED_MAX_ATTEMPTS_PER_WINDOW) return false;
  return isForegroundStable(at);
}

export function getNativeLifecyclePhase(): NativeLifecyclePhase {
  return phase;
}

export function getNativeLifecycleSnapshot(): NativeLifecycleSnapshot {
  return {
    phase,
    start_in_flight: inFlightStart ? 1 : 0,
    stop_in_flight: inFlightStop ? 1 : 0,
    recover_pending: recoverNeeded != null,
    pending_recover_on_foreground: pendingRecoverOnForeground,
    foreground_stable: isForegroundStable(),
    blocked_backoff_until_ms: blockedBackoffUntilMs,
    last_reason: lastReason,
  };
}

/** Observabilité P0-A : mutuellement exclusifs par construction. */
export function getNativeLifecycleInFlight(): {
  start_in_flight: 0 | 1;
  stop_in_flight: 0 | 1;
} {
  return {
    start_in_flight: inFlightStart ? 1 : 0,
    stop_in_flight: inFlightStop ? 1 : 0,
  };
}

export function canAttemptNativeStartNow(at: number = nowMs()): boolean {
  if (processing || inFlightStart || inFlightStop) return false;
  if (phase === "STARTING" || phase === "STOPPING" || phase === "RECOVERING") return false;
  if (phase === "BLOCKED_FOREGROUND_REQUIRED") {
    return canAttemptWhileBlocked(at);
  }
  return true;
}

export function subscribeNativeLifecycle(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function emitSnapshot(): void {
  const snap = getNativeLifecycleSnapshot();
  for (const listener of listeners) {
    try {
      listener(snap);
    } catch {
      /* noop */
    }
  }
}

function setPhase(next: NativeLifecyclePhase, reason?: string): void {
  if (reason != null) lastReason = reason;
  if (phase === next) {
    emitSnapshot();
    return;
  }
  phase = next;
  emitSnapshot();
}

function clearStableTimer(): void {
  if (stableTimer != null) {
    clearTimeout(stableTimer);
    stableTimer = null;
  }
}

function scheduleStableForegroundCheck(): void {
  clearStableTimer();
  if (rawAppState !== "active" || activeSinceMs == null) return;
  if (phase !== "BLOCKED_FOREGROUND_REQUIRED") return;
  if (!pendingRecoverOnForeground && recoverNeeded == null && pendingStart == null) return;

  const at = nowMs();
  const stableIn = Math.max(0, NATIVE_FOREGROUND_STABLE_MS - (at - activeSinceMs));
  const backoffIn = Math.max(0, blockedBackoffUntilMs - at);
  const delay = Math.max(stableIn, backoffIn);

  stableTimer = setTimeout(() => {
    stableTimer = null;
    if (rawAppState !== "active") return;
    if (phase !== "BLOCKED_FOREGROUND_REQUIRED") return;
    if (!pendingRecoverOnForeground && recoverNeeded == null && pendingStart == null) return;
    if (!canAttemptWhileBlocked(nowMs())) {
      // Backoff ou stabilité pas encore atteints → replanifier
      scheduleStableForegroundCheck();
      return;
    }
    void drainQueue();
  }, delay);
}

export function notifyNativeLifecycleAppState(next: AppStateStatus): void {
  const prev = rawAppState;
  rawAppState = next;
  if (next === "active") {
    if (prev !== "active") {
      activeSinceMs = nowMs();
      scheduleStableForegroundCheck();
    }
  } else {
    activeSinceMs = null;
    clearStableTimer();
  }
  emitSnapshot();
}

/**
 * Installe le bridge AppState une seule fois (debounce FG stable).
 * Idempotent ; safe en tests si AppState.addEventListener absent.
 */
export function ensureNativeLifecycleAppStateBridge(): void {
  if (appStateBridgeInstalled) return;
  appStateBridgeInstalled = true;
  rawAppState = AppState.currentState;
  activeSinceMs = rawAppState === "active" ? nowMs() : null;
  if (typeof AppState.addEventListener !== "function") return;
  appStateSubscription = AppState.addEventListener("change", (next) => {
    notifyNativeLifecycleAppState(next);
  });
}

function markBlockedForeground(reason: string): void {
  pendingRecoverOnForeground = true;
  const at = nowMs();
  pruneBlockedAttempts(at);
  blockedAttemptTimestamps.push(at);
  blockedBackoffUntilMs = at + blockedBackoffMs;
  blockedBackoffMs = Math.min(BLOCKED_BACKOFF_MAX_MS, blockedBackoffMs * 2);
  setPhase("BLOCKED_FOREGROUND_REQUIRED", reason);
  scheduleStableForegroundCheck();
}

function clearBlockedState(): void {
  pendingRecoverOnForeground = false;
  blockedBackoffUntilMs = 0;
  blockedBackoffMs = BLOCKED_BACKOFF_INITIAL_MS;
  blockedAttemptTimestamps = [];
}

function resolveStartOutcomeFromResult(
  result: NativeStartRunResult,
  kind: StartKind
): NativeStartOutcome {
  if (isForegroundErrorCode(result.errorCode ?? null)) {
    return "blocked_foreground";
  }
  if (result.nativeStarted) {
    return result.invokedNativeStart === false ? "already_started" : "running";
  }
  if (!result.ok) return "failed";
  return kind === "recover" ? "noop" : "failed";
}

function extractErrorCode(error: unknown): string | null {
  if (error instanceof Error && "code" in error) {
    const raw = (error as Error & { code?: unknown }).code;
    if (typeof raw === "string" || typeof raw === "number") return String(raw);
  }
  return null;
}

async function executeStart(op: PendingStart): Promise<NativeStartControllerResult> {
  setPhase(op.kind === "recover" ? "RECOVERING" : "STARTING", op.reason);

  try {
    const result = await op.run();
    const outcome = resolveStartOutcomeFromResult(result, op.kind);

    if (outcome === "blocked_foreground") {
      markBlockedForeground(op.reason);
      return { outcome, phase, result };
    }
    if (result.nativeStarted) {
      clearBlockedState();
      setPhase("RUNNING", op.reason);
      return { outcome, phase: "RUNNING", result };
    }
    setPhase("STOPPED", op.reason);
    return { outcome, phase: "STOPPED", result };
  } catch (error) {
    const code = extractErrorCode(error);
    if (isForegroundErrorCode(code)) {
      markBlockedForeground(op.reason);
      return {
        outcome: "blocked_foreground",
        phase,
        result: {
          ok: false,
          nativeStarted: false,
          invokedNativeStart: true,
          errorCode: code,
          errorMessage: error instanceof Error ? error.message : String(error),
        },
      };
    }
    setPhase("STOPPED", op.reason);
    throw error;
  }
}

async function executeStop(op: PendingStop): Promise<NativeStopControllerResult> {
  setPhase("STOPPING", op.reason);
  try {
    const result = await op.run();
    clearBlockedState();
    setPhase("STOPPED", op.reason);
    return {
      outcome: result.ok || result.nativeStopped ? "stopped" : "noop",
      phase: "STOPPED",
      result,
    };
  } catch (error) {
    setPhase("STOPPED", op.reason);
    throw error;
  }
}

function takeNextStart(): PendingStart | null {
  if (pendingStart) {
    const op = pendingStart;
    pendingStart = null;
    return op;
  }
  if (recoverNeeded) {
    const op = recoverNeeded;
    recoverNeeded = null;
    return op;
  }
  return null;
}

async function drainQueue(): Promise<void> {
  if (processing) return;
  processing = true;
  try {
    while (true) {
      if (pendingStop) {
        const op = pendingStop;
        pendingStop = null;
        let settle!: (value: NativeStopControllerResult) => void;
        let fail!: (error: unknown) => void;
        const promise = new Promise<NativeStopControllerResult>((resolve, reject) => {
          settle = resolve;
          fail = reject;
        });
        inFlightStop = { reason: op.reason, promise };
        emitSnapshot();
        try {
          const result = await executeStop(op);
          settle(result);
          op.resolve(result);
        } catch (error) {
          fail(error);
          op.reject(error);
        } finally {
          inFlightStop = null;
          emitSnapshot();
        }
        continue;
      }

      const nextStart = takeNextStart();
      if (!nextStart) break;

      const at = nowMs();

      if (rawAppState !== "active") {
        pendingRecoverOnForeground = true;
        if (nextStart.kind === "recover") {
          recoverNeeded = nextStart;
        } else {
          pendingStart = nextStart;
        }
        nextStart.resolve({ outcome: "deferred_not_foreground", phase });
        break;
      }

      if (phase === "BLOCKED_FOREGROUND_REQUIRED" && !canAttemptWhileBlocked(at)) {
        pendingRecoverOnForeground = true;
        if (nextStart.kind === "recover") {
          recoverNeeded = nextStart;
        } else {
          pendingStart = nextStart;
        }
        nextStart.resolve({ outcome: "deferred_blocked", phase });
        scheduleStableForegroundCheck();
        break;
      }

      if (phase === "BLOCKED_FOREGROUND_REQUIRED") {
        pendingRecoverOnForeground = false;
      }

      let settle!: (value: NativeStartControllerResult) => void;
      let fail!: (error: unknown) => void;
      const promise = new Promise<NativeStartControllerResult>((resolve, reject) => {
        settle = resolve;
        fail = reject;
      });
      inFlightStart = { kind: nextStart.kind, reason: nextStart.reason, promise };
      emitSnapshot();
      try {
        const result = await executeStart(nextStart);
        settle(result);
        nextStart.resolve(result);
      } catch (error) {
        fail(error);
        nextStart.reject(error);
      } finally {
        inFlightStart = null;
        emitSnapshot();
      }
    }
  } finally {
    processing = false;
    emitSnapshot();
  }
}

function rememberStartIntention(entry: PendingStart): void {
  if (entry.kind === "recover") {
    if (recoverNeeded) {
      recoverNeeded.resolve({ outcome: "coalesced", phase });
    }
    recoverNeeded = entry;
    pendingRecoverOnForeground = true;
    return;
  }
  if (pendingStart) {
    pendingStart.resolve({ outcome: "coalesced", phase });
  }
  pendingStart = entry;
}

/**
 * Demande un START natif sérialisé.
 * `run` doit effectuer (ou no-op) l'appel Expo et retourner un résultat structuré.
 */
export function requestNativeStart(input: {
  reason: string;
  run: () => Promise<NativeStartRunResult>;
}): Promise<NativeStartControllerResult> {
  ensureNativeLifecycleAppStateBridge();

  // Second START pendant STARTING/RECOVERING → même promesse (pas de 2e natif)
  if (inFlightStart && (phase === "STARTING" || phase === "RECOVERING")) {
    return inFlightStart.promise;
  }

  if (phase === "BLOCKED_FOREGROUND_REQUIRED" && !canAttemptWhileBlocked(nowMs())) {
    rememberStartIntention({
      kind: "start",
      reason: input.reason,
      run: input.run,
      resolve: () => undefined,
      reject: () => undefined,
    });
    scheduleStableForegroundCheck();
    return Promise.resolve({ outcome: "deferred_blocked", phase });
  }

  return new Promise<NativeStartControllerResult>((resolve, reject) => {
    const entry: PendingStart = {
      kind: "start",
      reason: input.reason,
      run: input.run,
      resolve,
      reject,
    };

    if (inFlightStart || inFlightStop || processing || phase === "STOPPING") {
      rememberStartIntention(entry);
      void drainQueue();
      return;
    }

    pendingStart = entry;
    void drainQueue();
  });
}

/**
 * Demande un RECOVER (anti-zombie / fgs_recover / ensure) — coalescé, jamais parallèle.
 */
export function requestNativeRecover(input: {
  reason: string;
  run: () => Promise<NativeStartRunResult>;
}): Promise<NativeStartControllerResult> {
  ensureNativeLifecycleAppStateBridge();

  if (inFlightStart && (phase === "STARTING" || phase === "RECOVERING")) {
    return new Promise((resolve, reject) => {
      rememberStartIntention({
        kind: "recover",
        reason: input.reason,
        run: input.run,
        resolve,
        reject,
      });
      void drainQueue();
    });
  }

  if (phase === "BLOCKED_FOREGROUND_REQUIRED" && !canAttemptWhileBlocked(nowMs())) {
    rememberStartIntention({
      kind: "recover",
      reason: input.reason,
      run: input.run,
      resolve: () => undefined,
      reject: () => undefined,
    });
    scheduleStableForegroundCheck();
    return Promise.resolve({ outcome: "deferred_blocked", phase });
  }

  return new Promise<NativeStartControllerResult>((resolve, reject) => {
    const entry: PendingStart = {
      kind: "recover",
      reason: input.reason,
      run: input.run,
      resolve,
      reject,
    };

    if (inFlightStart || inFlightStop || processing || phase === "STOPPING") {
      rememberStartIntention(entry);
      void drainQueue();
      return;
    }

    recoverNeeded = entry;
    void drainQueue();
  });
}

/**
 * Demande un STOP natif sérialisé.
 */
export function requestNativeStop(input: {
  reason: string;
  run: () => Promise<NativeStopRunResult>;
}): Promise<NativeStopControllerResult> {
  ensureNativeLifecycleAppStateBridge();

  if (inFlightStop && phase === "STOPPING") {
    return inFlightStop.promise;
  }

  return new Promise<NativeStopControllerResult>((resolve, reject) => {
    const entry: PendingStop = {
      reason: input.reason,
      run: input.run,
      resolve,
      reject,
    };

    if (pendingStop) {
      pendingStop.resolve({ outcome: "coalesced", phase });
    }
    pendingStop = entry;
    void drainQueue();
  });
}

/** Test-only / logout : reset complet du contrôleur. */
export function __resetNativeTrackingLifecycleForTests(): void {
  clearStableTimer();
  if (appStateSubscription && typeof appStateSubscription.remove === "function") {
    appStateSubscription.remove();
  }
  appStateSubscription = null;
  appStateBridgeInstalled = false;
  phase = "STOPPED";
  lastReason = null;
  processing = false;
  inFlightStart = null;
  inFlightStop = null;
  pendingStart = null;
  pendingStop = null;
  recoverNeeded = null;
  pendingRecoverOnForeground = false;
  blockedBackoffUntilMs = 0;
  blockedBackoffMs = BLOCKED_BACKOFF_INITIAL_MS;
  blockedAttemptTimestamps = [];
  rawAppState = AppState.currentState;
  activeSinceMs = rawAppState === "active" ? nowMs() : null;
  listeners.clear();
}
