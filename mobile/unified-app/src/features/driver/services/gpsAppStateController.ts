/**
 * DRIVER-COLD P0 — contrôleur GPS vs AppState React.
 *
 * React AppState ≠ cycle processus. Sur Android, un flip active ↔ background
 * transitoire (activité système, overlay permission) ne doit jamais
 * démarrer ni arrêter le service GPS.
 *
 * Cadence / rate-limit / splash : hors scope.
 */

import { Platform, type AppStateStatus } from "react-native";
import type { DriverMissionSnapshot } from "../tracking/resolveMissionSnapshotReady";

export const GPS_RESUME_RECONCILE_REASONS = ["app_resume", "app_resume_pending"] as const;

export const GPS_OSCILLATION_WINDOW_MS = 2000;
export const GPS_OSCILLATION_MAX_TRANSITIONS = 8;
export const GPS_OSCILLATION_COOLDOWN_MS = 10_000;

export type GpsMissionStartHold = {
  blocked: boolean;
  reason: string | null;
};

export type GpsControllerTransition = "start" | "stop";

type HoldReader = () => GpsMissionStartHold;

let holdReader: HoldReader | null = null;
let lastDecisionKey: string | null = null;
let oscillationStamps: number[] = [];
let oscillationOpenUntilMs = 0;

export function shouldIgnoreAppStateForGps(platform: string = Platform.OS): boolean {
  return platform === "android";
}

export function isDirectResumeNativeStartReason(reason: string): boolean {
  return (
    reason === "app_resume" ||
    reason === "app_resume_pending"
  );
}

export function resolveGpsControllerForeground(input: {
  platform?: string;
  appState: AppStateStatus;
  processForeground: boolean;
}): boolean {
  if (shouldIgnoreAppStateForGps(input.platform ?? Platform.OS)) {
    return input.processForeground;
  }
  return input.appState === "active";
}

export function resolveGpsMissionStartHold(input: {
  snapshot: DriverMissionSnapshot;
  bridgeMissionId: number | null;
  nativeOwnerPresent: boolean;
  presenceWindow: boolean;
}): GpsMissionStartHold {
  if (input.snapshot.status === "pending") {
    return { blocked: true, reason: "mission_snapshot_pending" };
  }
  if (
    input.snapshot.status === "resolved_mission" &&
    input.bridgeMissionId !== input.snapshot.missionId
  ) {
    return { blocked: true, reason: "mission_snapshot_awaiting_start" };
  }
  if (!input.presenceWindow && !input.nativeOwnerPresent) {
    return { blocked: true, reason: "native_owner_absent" };
  }
  return { blocked: false, reason: null };
}

export function setGpsMissionStartHoldReader(reader: HoldReader | null): void {
  holdReader = reader;
}

export function readGpsMissionStartHold(): GpsMissionStartHold {
  if (!holdReader) {
    return { blocked: false, reason: null };
  }
  return holdReader();
}

export function buildGpsControllerDecisionKey(parts: (string | number | boolean | null | undefined)[]): string {
  return parts.map((part) => (part == null ? "" : String(part))).join("|");
}

/** Idempotence : même décision déjà appliquée → no-op. */
export function shouldApplyGpsControllerDecision(key: string): boolean {
  if (key === lastDecisionKey) return false;
  lastDecisionKey = key;
  return true;
}

export function invalidateGpsControllerDecision(): void {
  lastDecisionKey = null;
}

export function recordGpsControllerTransition(
  _kind: GpsControllerTransition,
  at: number = Date.now()
): "ok" | "tripped" {
  if (at < oscillationOpenUntilMs) return "tripped";
  oscillationStamps.push(at);
  oscillationStamps = oscillationStamps.filter((stamp) => at - stamp <= GPS_OSCILLATION_WINDOW_MS);
  if (oscillationStamps.length >= GPS_OSCILLATION_MAX_TRANSITIONS) {
    oscillationOpenUntilMs = at + GPS_OSCILLATION_COOLDOWN_MS;
    oscillationStamps = [];
    return "tripped";
  }
  return "ok";
}

export function isGpsOscillationOpen(at: number = Date.now()): boolean {
  return at < oscillationOpenUntilMs;
}

/** Home réel / snapshot devenu actionnable : on réarme le coupe-circuit. */
export function resetGpsOscillationOnTrustedSignal(): void {
  oscillationStamps = [];
  oscillationOpenUntilMs = 0;
}

export function resetGpsAppStateControllerForTests(): void {
  holdReader = null;
  lastDecisionKey = null;
  oscillationStamps = [];
  oscillationOpenUntilMs = 0;
  controllerChain = Promise.resolve();
}
