/**
 * P6 — FSM de recovery GPS persistante (event-driven, sans sleep long).
 *
 * Stages : HEALTHY → VERIFY_WATCH → RESTART_FGS → VERIFY_FGS →
 *          RECONNECT_TRANSPORT → VERIFY_ACK → REBUILD_RUNTIME → HEALTHY
 *          (ou DEGRADED_EXHAUSTED avec backoff 30 min si non résolu)
 *
 * Avance sur heartbeat / callback / foreground / wake via tickTrackingRecovery.
 * Feature flag `tracking_recovery_cascade_enabled` false → restartWatch seul.
 */
import AsyncStorage from "@react-native-async-storage/async-storage";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { realtimeManager } from "../../../core/realtime/realtimeManager";

export type RecoveryStage =
  | "HEALTHY"
  | "VERIFY_WATCH"
  | "RESTART_FGS"
  | "VERIFY_FGS"
  | "RECONNECT_TRANSPORT"
  | "VERIFY_ACK"
  | "REBUILD_RUNTIME"
  | "DEGRADED_EXHAUSTED";

/** Étapes legacy (compat runTrackingRecoveryCascade). */
export type RecoveryStep =
  | "restart_watch"
  | "restart_fgs"
  | "restart_socket"
  | "restart_engine";

const RECOVERY_STORAGE_KEY = "@driver:tracking_recovery_fsm_v1";
/** Fenêtre de vérification courte — pas de sleep 60s+. */
const VERIFY_WINDOW_MS = 5_000;
const MAX_ATTEMPTS_PER_GENERATION = 8;
/** Backoff après cascade non résolue — pas de nouvelle cascade avant. */
export const EXHAUSTED_BACKOFF_MS = 30 * 60 * 1000;

export type TrackingRecoveryPersistedState = {
  recoveryStage: RecoveryStage;
  recoveryGeneration: number;
  startedAt: number;
  nextCheckAt: number;
  attemptCount: number;
  lastEvidence: string | null;
};

export type RecoveryEvidence = {
  reason?: string;
  watchAlive?: boolean;
  fgsAlive?: boolean;
  transportOk?: boolean;
  ackRecent?: boolean;
  fixRecent?: boolean;
  /** Signal explicite optionnel pour démarrer / poursuivre une cascade. */
  triggerUnhealthy?: boolean;
};

export type RecoveryHandlers = {
  restartWatch: (reason: string) => Promise<void>;
  restartFgs: (reason: string) => Promise<void>;
  restartEngine: (reason: string) => Promise<void>;
  reconnectTransport?: (reason: string) => Promise<void>;
};

const inMemoryStorage = new Map<string, string>();
let memoryState: TrackingRecoveryPersistedState | null = null;

function healthyState(
  now: number,
  previousGeneration = 0
): TrackingRecoveryPersistedState {
  return {
    recoveryStage: "HEALTHY",
    recoveryGeneration: previousGeneration,
    startedAt: now,
    nextCheckAt: 0,
    attemptCount: 0,
    lastEvidence: null,
  };
}

function exhaustedState(
  now: number,
  state: TrackingRecoveryPersistedState,
  evidence: RecoveryEvidence
): TrackingRecoveryPersistedState {
  return {
    recoveryStage: "DEGRADED_EXHAUSTED",
    recoveryGeneration: state.recoveryGeneration,
    startedAt: state.startedAt,
    nextCheckAt: now + EXHAUSTED_BACKOFF_MS,
    attemptCount: state.attemptCount,
    lastEvidence: evidenceKey(evidence),
  };
}

async function readStorage(key: string): Promise<string | null> {
  const storage = AsyncStorage as unknown as {
    getItem?: (input: string) => Promise<string | null>;
  };
  if (typeof storage?.getItem === "function") {
    return storage.getItem(key);
  }
  return inMemoryStorage.get(key) ?? null;
}

async function writeStorage(key: string, value: string): Promise<void> {
  const storage = AsyncStorage as unknown as {
    setItem?: (k: string, v: string) => Promise<void>;
  };
  if (typeof storage?.setItem === "function") {
    await storage.setItem(key, value);
    return;
  }
  inMemoryStorage.set(key, value);
}

function parseState(raw: string | null): TrackingRecoveryPersistedState | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as Partial<TrackingRecoveryPersistedState>;
    if (!parsed || typeof parsed.recoveryStage !== "string") return null;
    const stages: RecoveryStage[] = [
      "HEALTHY",
      "VERIFY_WATCH",
      "RESTART_FGS",
      "VERIFY_FGS",
      "RECONNECT_TRANSPORT",
      "VERIFY_ACK",
      "REBUILD_RUNTIME",
      "DEGRADED_EXHAUSTED",
    ];
    if (!stages.includes(parsed.recoveryStage as RecoveryStage)) return null;
    return {
      recoveryStage: parsed.recoveryStage as RecoveryStage,
      recoveryGeneration:
        typeof parsed.recoveryGeneration === "number" ? parsed.recoveryGeneration : 0,
      startedAt: typeof parsed.startedAt === "number" ? parsed.startedAt : Date.now(),
      nextCheckAt: typeof parsed.nextCheckAt === "number" ? parsed.nextCheckAt : 0,
      attemptCount: typeof parsed.attemptCount === "number" ? parsed.attemptCount : 0,
      lastEvidence:
        typeof parsed.lastEvidence === "string" || parsed.lastEvidence === null
          ? (parsed.lastEvidence ?? null)
          : null,
    };
  } catch {
    return null;
  }
}

async function loadRecoveryState(now: number): Promise<TrackingRecoveryPersistedState> {
  if (memoryState) return memoryState;
  const parsed = parseState(await readStorage(RECOVERY_STORAGE_KEY));
  memoryState = parsed ?? healthyState(now);
  return memoryState;
}

async function persistRecoveryState(
  state: TrackingRecoveryPersistedState
): Promise<TrackingRecoveryPersistedState> {
  memoryState = state;
  await writeStorage(RECOVERY_STORAGE_KEY, JSON.stringify(state));
  return state;
}

function evidenceKey(evidence: RecoveryEvidence): string {
  return [
    evidence.reason ?? "",
    evidence.fixRecent === true ? "fix1" : evidence.fixRecent === false ? "fix0" : "fix?",
    evidence.watchAlive === true ? "w1" : evidence.watchAlive === false ? "w0" : "w?",
    evidence.fgsAlive === true ? "f1" : evidence.fgsAlive === false ? "f0" : "f?",
    evidence.transportOk === true ? "t1" : evidence.transportOk === false ? "t0" : "t?",
    evidence.ackRecent === true ? "a1" : evidence.ackRecent === false ? "a0" : "a?",
    evidence.triggerUnhealthy === true ? "tu1" : "tu0",
  ].join("|");
}

function looksHealthy(evidence: RecoveryEvidence): boolean {
  return evidence.fixRecent === true || (evidence.watchAlive === true && evidence.ackRecent === true);
}

/**
 * Unhealthy uniquement via signaux booléens explicites à false
 * (ou triggerUnhealthy). Une `reason` seule n'est PAS unhealthy.
 */
function looksUnhealthy(evidence: RecoveryEvidence): boolean {
  if (looksHealthy(evidence)) return false;
  return (
    evidence.fixRecent === false ||
    evidence.watchAlive === false ||
    evidence.fgsAlive === false ||
    evidence.transportOk === false ||
    evidence.ackRecent === false ||
    evidence.triggerUnhealthy === true
  );
}

async function startCascade(
  now: number,
  evidence: RecoveryEvidence,
  handlers: RecoveryHandlers,
  previousGeneration: number
): Promise<TrackingRecoveryPersistedState> {
  const reason = evidence.reason ?? "tracking_recovery_tick";
  await handlers.restartWatch(reason);
  const state = await persistRecoveryState({
    recoveryStage: "VERIFY_WATCH",
    recoveryGeneration: previousGeneration + 1,
    startedAt: now,
    nextCheckAt: now + VERIFY_WINDOW_MS,
    attemptCount: 1,
    lastEvidence: evidenceKey(evidence),
  });
  emitDriverTelemetry("tracking.recovery.tick", {
    source: "driver.tracking.recovery",
    stage: state.recoveryStage,
    reason,
    attempt_count: state.attemptCount,
  });
  return state;
}

/**
 * Tick event-driven de la FSM de recovery.
 * Ne fait jamais `await sleep(60s+)` — avance uniquement sur appel externe.
 */
export async function tickTrackingRecovery(
  now: number,
  evidence: RecoveryEvidence,
  handlers: RecoveryHandlers
): Promise<TrackingRecoveryPersistedState> {
  const reason = evidence.reason ?? "tracking_recovery_tick";

  // Flag off : comportement historique (restartWatch seul).
  if (!isFeatureEnabled("tracking_recovery_cascade_enabled")) {
    const prev = await loadRecoveryState(now);
    if (looksUnhealthy(evidence)) {
      await handlers.restartWatch(reason);
      emitDriverTelemetry("tracking.recovery.tick", {
        source: "driver.tracking.recovery",
        step: "restart_watch",
        reason,
        cascade_enabled: false,
      });
    }
    return persistRecoveryState(healthyState(now, prev.recoveryGeneration));
  }

  let state = await loadRecoveryState(now);

  if (looksHealthy(evidence) && state.recoveryStage !== "HEALTHY") {
    emitDriverTelemetry("tracking.recovery.recovered", {
      source: "driver.tracking.recovery",
      previous_stage: state.recoveryStage,
      reason,
      attempt_count: state.attemptCount,
      recovery_generation: state.recoveryGeneration,
    });
    return persistRecoveryState(healthyState(now, state.recoveryGeneration));
  }

  if (state.recoveryStage === "HEALTHY") {
    if (!looksUnhealthy(evidence)) {
      return state;
    }
    return startCascade(now, evidence, handlers, state.recoveryGeneration);
  }

  if (state.recoveryStage === "DEGRADED_EXHAUSTED") {
    // Cooldown : pas de nouvelle cascade avant nextCheckAt.
    if (now < state.nextCheckAt) {
      return state;
    }
    if (!looksUnhealthy(evidence)) {
      return persistRecoveryState(healthyState(now, state.recoveryGeneration));
    }
    return startCascade(now, evidence, handlers, state.recoveryGeneration);
  }

  // Fenêtre de verify non écoulée → no-op (pas de sleep).
  if (now < state.nextCheckAt) {
    return state;
  }

  if (state.attemptCount >= MAX_ATTEMPTS_PER_GENERATION) {
    emitDriverTelemetry("tracking.recovery.exhausted", {
      source: "driver.tracking.recovery",
      stage: state.recoveryStage,
      reason,
      attempt_count: state.attemptCount,
      recovery_generation: state.recoveryGeneration,
    });
    return persistRecoveryState(exhaustedState(now, state, evidence));
  }

  switch (state.recoveryStage) {
    case "VERIFY_WATCH": {
      if (evidence.watchAlive === true || evidence.fixRecent === true) {
        return persistRecoveryState(healthyState(now, state.recoveryGeneration));
      }
      await handlers.restartFgs(reason);
      return persistRecoveryState({
        ...state,
        recoveryStage: "VERIFY_FGS",
        nextCheckAt: now + VERIFY_WINDOW_MS,
        attemptCount: state.attemptCount + 1,
        lastEvidence: evidenceKey(evidence),
      });
    }
    case "RESTART_FGS":
    case "VERIFY_FGS": {
      if (evidence.fgsAlive === true || evidence.fixRecent === true) {
        return persistRecoveryState(healthyState(now, state.recoveryGeneration));
      }
      if (handlers.reconnectTransport) {
        await handlers.reconnectTransport(reason);
      } else {
        const snap = realtimeManager.getSnapshot();
        if (snap.activeContextId) {
          realtimeManager.connect(snap.activeContextId, { enableSocket: true });
        }
      }
      emitDriverTelemetry("tracking.recovery.step", {
        source: "driver.tracking.recovery",
        step: "reconnect_transport",
        reason,
      });
      return persistRecoveryState({
        ...state,
        recoveryStage: "VERIFY_ACK",
        nextCheckAt: now + VERIFY_WINDOW_MS,
        attemptCount: state.attemptCount + 1,
        lastEvidence: evidenceKey(evidence),
      });
    }
    case "RECONNECT_TRANSPORT":
    case "VERIFY_ACK": {
      if (evidence.ackRecent === true || evidence.transportOk === true) {
        return persistRecoveryState(healthyState(now, state.recoveryGeneration));
      }
      await handlers.restartEngine(reason);
      return persistRecoveryState({
        ...state,
        recoveryStage: "REBUILD_RUNTIME",
        nextCheckAt: now + VERIFY_WINDOW_MS,
        attemptCount: state.attemptCount + 1,
        lastEvidence: evidenceKey(evidence),
      });
    }
    case "REBUILD_RUNTIME": {
      if (looksHealthy(evidence)) {
        return persistRecoveryState(healthyState(now, state.recoveryGeneration));
      }
      // Cycle terminé sans preuve de santé → cooldown (pas de reset gen / pas de re-cascade immédiate).
      emitDriverTelemetry("tracking.recovery.rebuild_unresolved", {
        source: "driver.tracking.recovery",
        reason,
        attempt_count: state.attemptCount,
        recovery_generation: state.recoveryGeneration,
      });
      return persistRecoveryState(exhaustedState(now, state, evidence));
    }
    default:
      return persistRecoveryState(healthyState(now, state.recoveryGeneration));
  }
}

/**
 * Compat : entrée historique. Délègue au tick FSM (1 pas) sans sleep.
 */
export async function runTrackingRecoveryCascade(
  reason: string,
  handlers: RecoveryHandlers
): Promise<void> {
  await tickTrackingRecovery(
    Date.now(),
    {
      reason,
      fixRecent: false,
      watchAlive: false,
      fgsAlive: false,
      transportOk: false,
      ackRecent: false,
    },
    handlers
  );
}

/** Tests uniquement. */
export function __resetTrackingRecoveryForTests(): void {
  memoryState = null;
  inMemoryStorage.clear();
}
