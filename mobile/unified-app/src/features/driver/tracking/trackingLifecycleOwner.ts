/**
 * Ownership unique du STOP natif (D5).
 *
 * Les callers (hook, manager, self-heal) demandent un STOP via
 * `requestTrackingStop` (bridge). Seul le chemin owner → tâche background
 * → API Expo Location matérialise l'Unregister, avec check de génération
 * immédiat pré-natif.
 */

export type TrackingDesiredState = "RUNNING" | "STOPPED" | "RECOVERING";

/** Autorité du STOP demandé. */
export type TrackingStopAuthority =
  /** Logout, leave driver, mission terminale, stop métier explicite. */
  | "explicit"
  /** Trou React / cache — ne doit pas Unregister immédiatement. */
  | "transient_loss"
  /** Recovery destructif uniquement si panne native positivement prouvée. */
  | "recovery_l2";

export type TrackingStopRequest = {
  reason: string;
  expectedGeneration: number;
  expectedMissionId?: number | null;
  authority: TrackingStopAuthority;
};

export type TrackingStopOutcome = "stopped" | "abandoned" | "deferred";

export type NativeStopAbortGuard = () => boolean;
