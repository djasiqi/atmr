import * as Sentry from "@sentry/react-native";
import * as Location from "expo-location";
import { AppStateStatus } from "react-native";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import {
  probeTaskRegState,
  requestForegroundPermissionsWithCanaryProbe,
} from "./canaryD5NativeBoundaryProbe";

export const STALE_RESTART_THRESHOLD = 2;
export const STALE_RESTART_COOLDOWN_MS = 300_000;
export const WATCH_RESTART_MAX_PER_HOUR = 3;
export const ANTI_ZOMBIE_FIX_AGE_SEC = 60;

export type SelfHealBridgeSlice = {
  watchSubscription: Location.LocationSubscription | null;
  staleFallbackTimeouts: number;
  staleFallbackBlockedUntilMs: number;
  lastWatchAtMs: number | null;
  lastWatchedPosition: Location.LocationObject | null;
  lastWatchRestartAtMs: number;
  watchRestartTimestampsMs: number[];
  missionId: number | null;
};

export type SelfHealActions = {
  stopWatch: () => void;
  /** L2 uniquement — panne native positivement prouvée. L1 ne l'appelle pas. */
  stopBackground: (reason: string) => Promise<void>;
  ensureNativeForeground: () => Promise<void>;
  ensureLocationWatch: () => Promise<void>;
  triggerDeviceHealth: (reason: string) => void;
};

export type ForceRestartTrackingWatchOptions = {
  /**
   * D5 — L2 destructif (Unregister). Défaut false : recovery L1 non destructif.
   * N'activer que si panne native positivement démontrée.
   */
  allowDestructiveRestart?: boolean;
};

export function createSelfHealSlice(): Pick<
  SelfHealBridgeSlice,
  "lastWatchRestartAtMs" | "watchRestartTimestampsMs"
> {
  return {
    lastWatchRestartAtMs: 0,
    watchRestartTimestampsMs: [],
  };
}

function pruneRestartTimestamps(timestamps: number[], nowMs: number): number[] {
  const hourAgo = nowMs - 3_600_000;
  return timestamps.filter((t) => t >= hourAgo);
}

export function shouldForceRestartWatch(
  slice: SelfHealBridgeSlice,
  nowMs: number = Date.now()
): boolean {
  if (!isFeatureEnabled("tracking_self_heal_watch_restart_enabled")) {
    return false;
  }
  if (slice.staleFallbackTimeouts < STALE_RESTART_THRESHOLD) {
    return false;
  }
  if (nowMs - slice.lastWatchRestartAtMs < STALE_RESTART_COOLDOWN_MS) {
    return false;
  }
  const recent = pruneRestartTimestamps(slice.watchRestartTimestampsMs, nowMs);
  return recent.length < WATCH_RESTART_MAX_PER_HOUR;
}

export async function forceRestartTrackingWatch(
  reason: string,
  slice: SelfHealBridgeSlice,
  actions: SelfHealActions,
  appState: AppStateStatus,
  opts?: ForceRestartTrackingWatchOptions
): Promise<boolean> {
  if (!isFeatureEnabled("tracking_self_heal_watch_restart_enabled")) {
    return false;
  }
  const nowMs = Date.now();
  const recent = pruneRestartTimestamps(slice.watchRestartTimestampsMs, nowMs);
  if (recent.length >= WATCH_RESTART_MAX_PER_HOUR) {
    Sentry.captureMessage("tracking.watch.restart.exhausted", { level: "warning" });
    emitDriverTelemetry("tracking.watch.restart.exhausted", {
      source: "driver.tracking.self_heal",
      reason,
      mission_id: slice.missionId,
    });
    return false;
  }

  actions.stopWatch();
  // D5 : L1 par défaut — pas d'Unregister. L2 seulement si preuve native + flag.
  if (opts?.allowDestructiveRestart) {
    await actions.stopBackground("self_heal_restart").catch(() => undefined);
  }
  await probeTaskRegState("watch_restart", {
    caller: "forceRestartTrackingWatch",
    reason,
    missionId: slice.missionId,
    appState,
  });
  await requestForegroundPermissionsWithCanaryProbe({
    caller: "forceRestartTrackingWatch",
    reason,
    missionId: slice.missionId,
  });

  slice.staleFallbackTimeouts = 0;
  slice.staleFallbackBlockedUntilMs = 0;
  slice.lastWatchAtMs = null;
  slice.lastWatchedPosition = null;
  slice.lastWatchRestartAtMs = nowMs;
  slice.watchRestartTimestampsMs = [...recent, nowMs];

  await actions.ensureNativeForeground();
  await actions.ensureLocationWatch();

  actions.triggerDeviceHealth("stale_fallback_restart");
  emitDriverTelemetry("tracking.watch.restarted", {
    source: "driver.tracking.self_heal",
    reason,
    mission_id: slice.missionId,
    app_state: appState,
    destructive_restart: Boolean(opts?.allowDestructiveRestart),
    recovery_level: opts?.allowDestructiveRestart ? "L2" : "L1",
  });
  return true;
}

let lastAntiZombieTriggeredAtMs = 0;
const ANTI_ZOMBIE_COOLDOWN_MS = 60_000;

export function getFixAgeSeconds(
  lastFixProducedAtMs: number | null,
  nowMs: number = Date.now()
): number | null {
  if (lastFixProducedAtMs == null) {
    return null;
  }
  return Math.max(0, (nowMs - lastFixProducedAtMs) / 1000);
}

export function shouldTriggerAntiZombie(input: {
  isTrackingRunning: boolean;
  lastFixProducedAtMs: number | null;
  lastSentAt: string | null;
  /**
   * Horodatage de démarrage du runtime tracking. Permet de couvrir le cas
   * « tracking lancé mais aucun fix jamais produit » (zombie dès le départ),
   * où `lastFixProducedAtMs` reste `null`.
   */
  trackingStartedAtMs?: number | null;
  nowMs?: number;
}): boolean {
  if (!isFeatureEnabled("tracking_self_heal_watch_restart_enabled")) {
    return false;
  }
  if (!input.isTrackingRunning) {
    return false;
  }
  const nowMs = input.nowMs ?? Date.now();
  if (nowMs - lastAntiZombieTriggeredAtMs < ANTI_ZOMBIE_COOLDOWN_MS) {
    return false;
  }
  /**
   * Le signal de santé prioritaire est l'ENVOI réel (`lastSentAt`) : un fix
   * natif/local « frais » (`lastFixProducedAtMs`, `native_last_fix_age`) ne
   * suffit pas si plus rien n'est envoyé au backend — c'est précisément le FGS
   * zombie observé (service vivant, souscription morte, 0 envoi).
   *
   * D5 : `lastSentAt=null` ∧ `lastFix=null` = fraîcheur UNKNOWN, pas preuve
   * que la task Location est morte. Le fallback `startedAge` ne déclenche
   * plus l'anti-zombie destructif (Unregister). Un wake L1 éventuel peut
   * être ajouté séparément ; ici on refuse le trigger « zombie » sans preuve.
   */
  const sentAgeSec = input.lastSentAt
    ? (nowMs - Date.parse(input.lastSentAt)) / 1000
    : null;
  const fixAge = getFixAgeSeconds(input.lastFixProducedAtMs, nowMs);

  if (sentAgeSec !== null) {
    return sentAgeSec > ANTI_ZOMBIE_FIX_AGE_SEC;
  }
  if (fixAge !== null) {
    return fixAge > ANTI_ZOMBIE_FIX_AGE_SEC;
  }
  // UNKNOWN (startedAge ignoré pour Unregister / anti-zombie destructif)
  void input.trackingStartedAtMs;
  return false;
}

export function markAntiZombieTriggered(nowMs: number = Date.now()): void {
  lastAntiZombieTriggeredAtMs = nowMs;
}

export function resetAntiZombieForTests(): void {
  lastAntiZombieTriggeredAtMs = 0;
}

/**
 * Seuil (s) au-delà duquel un tracking « à froid » (manager arrêté) est
 * considéré gelé alors qu'une mission est active. Si aucune position n'a été
 * envoyée depuis ce délai, on re-arme le runtime.
 */
export const COLD_START_THRESHOLD_SEC = 120;
const COLD_START_COOLDOWN_MS = 60_000;
let lastColdStartTriggeredAtMs = 0;

/**
 * Catch-22 corrigé : `shouldTriggerAntiZombie` exige `isTrackingRunning=true`
 * et ne couvre donc pas le cas « le manager ne tourne pas du tout malgré une
 * mission active » (observé après login/logout, FGS tué par l'OS, JS timers
 * suspendus). Cette garde symétrique se déclenche **uniquement** quand le
 * tracking est arrêté (`isTrackingRunning=false`) pour relancer le runtime
 * complet via `startDriverTrackingBridge`.
 */
export function shouldTriggerColdStart(input: {
  hasActiveMission: boolean;
  isTrackingRunning: boolean;
  lastSentAt: string | null;
  nowMs?: number;
}): boolean {
  if (!isFeatureEnabled("tracking_self_heal_cold_start_enabled")) {
    return false;
  }
  if (!input.hasActiveMission) {
    return false;
  }
  // Cas « tracking en cours mais gelé » : couvert par l'anti-zombie.
  if (input.isTrackingRunning) {
    return false;
  }
  const nowMs = input.nowMs ?? Date.now();
  if (nowMs - lastColdStartTriggeredAtMs < COLD_START_COOLDOWN_MS) {
    return false;
  }
  // Jamais aucune position envoyée alors qu'une mission est active : runtime à froid.
  if (!input.lastSentAt) {
    return true;
  }
  const sentAt = Date.parse(input.lastSentAt);
  if (!Number.isFinite(sentAt)) {
    return true;
  }
  const sentAgeSec = (nowMs - sentAt) / 1000;
  return sentAgeSec > COLD_START_THRESHOLD_SEC;
}

export function markColdStartTriggered(nowMs: number = Date.now()): void {
  lastColdStartTriggeredAtMs = nowMs;
}

export function resetColdStartForTests(): void {
  lastColdStartTriggeredAtMs = 0;
}
