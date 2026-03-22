/**
 * P1 — Policy centrale de tracking chauffeur : une seule fonction de décision.
 * missionTrackingPolicy = sous-règle d’éligibilité mission (statuts) ; pas de duplication de la table.
 */

import { AppState } from "react-native";
import {
  shouldRunBackgroundTracking,
  type BgTrackingInputs,
  type LocationMode,
  type PermissionStatus,
} from "./backgroundTrackingGating";
import { resolveLocationModeFromState, resolvePresenceState } from "./locationPresenceFsm";
import { isMissionTrackingActiveStatus } from "./missionTrackingPolicy";
import type { MissionBarStatus } from "./missionState";

// ---------------------------------------------------------------------------
// Types (contrat figé P1)
// ---------------------------------------------------------------------------

export type TrackingMode =
  | "OFF"
  | "FOREGROUND_PRESENCE"
  | "BACKGROUND_PRESENCE"
  | "FOREGROUND_MISSION"
  | "BACKGROUND_MISSION";

export type DegradedReason =
  | "missing_foreground_permission"
  | "missing_background_permission_ios"
  | "android_fgs_deferred"
  | "driver_context_missing"
  | "network_offline"
  | "token_unavailable"
  | "mission_snapshot_missing"
  | "native_tracking_not_started"
  | null;

export type TransportPreference = "socket_then_http" | "http_only" | "deferred";

export type TrackingPolicy = {
  mode: TrackingMode;
  degraded: boolean;
  degradedReason: DegradedReason;

  shouldRunForegroundTracker: boolean;
  shouldRunNativeBackgroundTracking: boolean;
  shouldAllowBackgroundFlush: boolean;
  shouldEscalateMissionPriority: boolean;

  samplingIntervalMs: number | null;
  distanceIntervalM: number | null;
  flushIntervalMs: number | null;
  /** Cadences syncEngine distinctes du flush (queue) — alignées syncEngine historique. */
  missionHeartbeatIntervalMs: number | null;
  presenceHeartbeatIntervalMs: number | null;

  transportPreference: TransportPreference;
  /** Diagnostic court pour logs (stable par entrées identiques). */
  reason: string;
};

/** Snapshot stable des entrées de décision (aligné resolveTrackingPolicy + métadonnée reconcile). */
export type TrackingDecisionInputsSnapshot = ResolveTrackingPolicyInput & {
  reconcileReason: string;
  capturedAtMs: number;
};

export type ResolveTrackingPolicyInput = {
  platform: "ios" | "android" | "web";
  isAuthenticated: boolean;
  role: "driver" | "enterprise";
  appState: string;
  fgPermission: PermissionStatus;
  bgPermission: PermissionStatus;
  killSwitchEnabled: boolean;
  availabilityPresenceEnabled: boolean;
  hasActiveMission: boolean;
  missionBarStatus: MissionBarStatus | null;
  /** Mode persisté + FSM — aligné buildBgTrackingInputs. */
  persistedLocationMode: LocationMode;
  /** true si start natif Android reporté (FGS). */
  pendingAndroidFgsDeferred: boolean;
  /** Réseau joignable (false → policy dégradée + transport deferred). */
  networkReachable: boolean;
  /** driver_id présent en contexte flush natif. */
  driverTokenAvailable: boolean;
};

/** Cadences par mode — alignement locationTracker / syncEngine / locationTask. */
export function getTrackingCadenceForMode(mode: TrackingMode): {
  samplingIntervalMs: number | null;
  distanceIntervalM: number | null;
  flushIntervalMs: number | null;
} {
  const fastMs = parseInt(process.env.EXPO_PUBLIC_GPS_FAST_MS ?? "5000", 10) || 5000;
  const slowMs = parseInt(process.env.EXPO_PUBLIC_GPS_SLOW_MS ?? "60000", 10) || 60000;
  const flushMs = 15000;

  switch (mode) {
    case "OFF":
      return { samplingIntervalMs: null, distanceIntervalM: null, flushIntervalMs: null };
    case "FOREGROUND_MISSION":
      return { samplingIntervalMs: fastMs, distanceIntervalM: 10, flushIntervalMs: flushMs };
    case "BACKGROUND_MISSION":
      return { samplingIntervalMs: 15000, distanceIntervalM: 10, flushIntervalMs: flushMs };
    case "FOREGROUND_PRESENCE":
      return { samplingIntervalMs: slowMs, distanceIntervalM: 50, flushIntervalMs: flushMs };
    case "BACKGROUND_PRESENCE":
      return { samplingIntervalMs: 180000, distanceIntervalM: 50, flushIntervalMs: flushMs };
    default:
      return { samplingIntervalMs: null, distanceIntervalM: null, flushIntervalMs: null };
  }
}

function buildBgInputsForGating(
  input: ResolveTrackingPolicyInput,
  locationMode: LocationMode
): BgTrackingInputs {
  const missionStatusEnabledForTracking =
    input.hasActiveMission && input.missionBarStatus !== null
      ? isMissionTrackingActiveStatus(input.missionBarStatus)
      : false;
  return {
    isAuthenticated: input.isAuthenticated,
    role: input.role,
    platform: input.platform,
    hasActiveMission: input.hasActiveMission,
    missionStatusEnabledForTracking: input.hasActiveMission
      ? missionStatusEnabledForTracking
      : undefined,
    fgPermission: input.fgPermission,
    bgPermission: input.bgPermission,
    killSwitchEnabled: input.killSwitchEnabled,
    locationMode,
    availabilityPresenceEnabled: input.availabilityPresenceEnabled,
  };
}

/**
 * Décision unique : intention métier (mode), capacité réelle (degraded), stratégie d’exécution.
 * Ne démarre rien (séparation décision / exécution).
 */
export function resolveTrackingPolicy(input: ResolveTrackingPolicyInput): TrackingPolicy {
  const appActive = input.appState === "active";
  const requiresBgIos = input.platform === "ios";

  const missionStatusEnabledForTracking =
    input.hasActiveMission && input.missionBarStatus !== null
      ? isMissionTrackingActiveStatus(input.missionBarStatus)
      : false;

  const fsmState = resolvePresenceState({
    isAuthenticated: input.isAuthenticated,
    isDriver: input.role === "driver",
    hasFgPermission: input.fgPermission === "granted",
    hasBgPermission: requiresBgIos ? input.bgPermission === "granted" : true,
    appInBackground: !appActive,
    hasActiveMission: input.hasActiveMission,
    availabilityPresenceEnabled: input.availabilityPresenceEnabled,
  });
  const fsmMode = resolveLocationModeFromState(fsmState);

  const eligibleMission =
    input.hasActiveMission &&
    input.missionBarStatus !== null &&
    isMissionTrackingActiveStatus(input.missionBarStatus);

  const locationMode: LocationMode =
    eligibleMission || fsmMode === "mission_live" ? "mission_live" : input.persistedLocationMode;

  const bgInputs = buildBgInputsForGating(input, locationMode);
  const nativeWanted = shouldRunBackgroundTracking(bgInputs);

  let mode: TrackingMode = "OFF";
  if (!input.isAuthenticated || input.role !== "driver" || input.killSwitchEnabled) {
    mode = "OFF";
  } else if (locationMode === "passive_last_known") {
    mode = "OFF";
  } else if (eligibleMission && locationMode === "mission_live") {
    mode = appActive ? "FOREGROUND_MISSION" : "BACKGROUND_MISSION";
  } else if (locationMode === "availability_presence" && input.availabilityPresenceEnabled) {
    mode = appActive ? "FOREGROUND_PRESENCE" : "BACKGROUND_PRESENCE";
  } else {
    mode = "OFF";
  }

  const cadence = getTrackingCadenceForMode(mode);

  const fgOk = input.fgPermission === "granted";
  const bgOk = requiresBgIos ? input.bgPermission === "granted" : true;

  let degradedReason: DegradedReason = null;
  if (input.role === "driver" && !input.isAuthenticated) {
    degradedReason = "driver_context_missing";
  } else if (input.hasActiveMission && input.missionBarStatus === null) {
    degradedReason = "mission_snapshot_missing";
  } else if (mode !== "OFF" && !fgOk) {
    degradedReason = "missing_foreground_permission";
  } else if (mode !== "OFF" && requiresBgIos && !bgOk) {
    degradedReason = "missing_background_permission_ios";
  } else if (!input.networkReachable && mode !== "OFF") {
    degradedReason = "network_offline";
  } else if (input.role === "driver" && input.isAuthenticated && !input.driverTokenAvailable) {
    degradedReason = "token_unavailable";
  } else if (
    input.platform === "android" &&
    input.pendingAndroidFgsDeferred &&
    (mode === "BACKGROUND_MISSION" || mode === "BACKGROUND_PRESENCE")
  ) {
    degradedReason = "android_fgs_deferred";
  }
  const degraded = degradedReason !== null;

  const shouldRunForegroundTracker =
    input.role === "driver" &&
    input.isAuthenticated &&
    fgOk &&
    appActive &&
    mode !== "OFF";

  const shouldRunNativeBackgroundTracking = nativeWanted;

  const shouldEscalateMissionPriority = missionStatusEnabledForTracking;

  const shouldAllowBackgroundFlush =
    input.role === "driver" &&
    input.isAuthenticated &&
    fgOk &&
    bgOk &&
    !input.killSwitchEnabled &&
    (shouldEscalateMissionPriority ||
      (locationMode === "availability_presence" && input.availabilityPresenceEnabled));

  let transportPreference: TransportPreference = "socket_then_http";
  if (!input.networkReachable) {
    transportPreference = "deferred";
  } else if (mode === "BACKGROUND_PRESENCE" || degradedReason === "android_fgs_deferred") {
    transportPreference = "http_only";
  }

  const missionHeartbeatIntervalMs =
    shouldEscalateMissionPriority &&
    (mode === "FOREGROUND_MISSION" || mode === "BACKGROUND_MISSION")
      ? 60000
      : null;
  const presenceHeartbeatIntervalMs =
    mode === "FOREGROUND_PRESENCE" || mode === "BACKGROUND_PRESENCE" ? 180000 : null;

  const reasonParts = [
    `mode=${mode}`,
    `lm=${locationMode}`,
    `fsm=${fsmMode}`,
    `native=${shouldRunNativeBackgroundTracking ? "y" : "n"}`,
    `esc=${shouldEscalateMissionPriority ? "y" : "n"}`,
  ];
  const reason = reasonParts.join("|");

  return {
    mode,
    degraded,
    degradedReason,
    shouldRunForegroundTracker,
    shouldRunNativeBackgroundTracking,
    shouldAllowBackgroundFlush,
    shouldEscalateMissionPriority,
    samplingIntervalMs: cadence.samplingIntervalMs,
    distanceIntervalM: cadence.distanceIntervalM,
    flushIntervalMs: cadence.flushIntervalMs,
    missionHeartbeatIntervalMs,
    presenceHeartbeatIntervalMs,
    transportPreference,
    reason,
  };
}

/** Comparaison structurelle pour skip reconcile (tous les champs sauf raison libre — reason incluse). */
export function isTrackingPolicyStructurallyEqual(
  a: TrackingPolicy | null,
  b: TrackingPolicy
): boolean {
  if (!a) return false;
  return (
    a.mode === b.mode &&
    a.degraded === b.degraded &&
    a.degradedReason === b.degradedReason &&
    a.shouldRunForegroundTracker === b.shouldRunForegroundTracker &&
    a.shouldRunNativeBackgroundTracking === b.shouldRunNativeBackgroundTracking &&
    a.shouldAllowBackgroundFlush === b.shouldAllowBackgroundFlush &&
    a.shouldEscalateMissionPriority === b.shouldEscalateMissionPriority &&
    a.samplingIntervalMs === b.samplingIntervalMs &&
    a.distanceIntervalM === b.distanceIntervalM &&
    a.flushIntervalMs === b.flushIntervalMs &&
    a.missionHeartbeatIntervalMs === b.missionHeartbeatIntervalMs &&
    a.presenceHeartbeatIntervalMs === b.presenceHeartbeatIntervalMs &&
    a.transportPreference === b.transportPreference &&
    a.reason === b.reason
  );
}

// ---------------------------------------------------------------------------
// P1.0 — Shadow : approximation legacy (lecture seule des mêmes primitives)
// ---------------------------------------------------------------------------

export type LegacyTrackingShadowSnapshot = {
  legacyMode: TrackingMode;
  legacyBgShouldRun: boolean;
  legacyForegroundTrackerAssumed: boolean;
  legacyFlushIntervalMs: number;
};

export function computeLegacyTrackingShadowSnapshot(args: {
  bgInputs: BgTrackingInputs;
  appState: string;
  currentLocationMode: LocationMode;
}): LegacyTrackingShadowSnapshot {
  const { bgInputs, appState, currentLocationMode } = args;
  const active = appState === "active";
  const legacyBgShouldRun = shouldRunBackgroundTracking(bgInputs);

  let legacyMode: TrackingMode = "OFF";
  if (!bgInputs.isAuthenticated || bgInputs.role !== "driver") {
    legacyMode = "OFF";
  } else if (currentLocationMode === "passive_last_known") {
    legacyMode = "OFF";
  } else if (
    currentLocationMode === "mission_live" &&
    bgInputs.hasActiveMission &&
    bgInputs.missionStatusEnabledForTracking !== false
  ) {
    legacyMode = active ? "FOREGROUND_MISSION" : "BACKGROUND_MISSION";
  } else if (currentLocationMode === "availability_presence") {
    legacyMode = active ? "FOREGROUND_PRESENCE" : "BACKGROUND_PRESENCE";
  }

  const legacyForegroundTrackerAssumed =
    bgInputs.role === "driver" && bgInputs.isAuthenticated && active;

  return {
    legacyMode,
    legacyBgShouldRun,
    legacyForegroundTrackerAssumed,
    legacyFlushIntervalMs: 15000,
  };
}

export function computeTrackingPolicyShadowDiff(args: {
  resolved: TrackingPolicy;
  legacy: LegacyTrackingShadowSnapshot;
}): {
  diffDetected: boolean;
  diffReason: string | null;
} {
  const { resolved, legacy } = args;
  const reasons: string[] = [];
  if (resolved.mode !== legacy.legacyMode) {
    reasons.push(`mode:${legacy.legacyMode}->${resolved.mode}`);
  }
  if (resolved.shouldRunNativeBackgroundTracking !== legacy.legacyBgShouldRun) {
    reasons.push(`native_bg:${legacy.legacyBgShouldRun}->${resolved.shouldRunNativeBackgroundTracking}`);
  }
  if (resolved.shouldRunForegroundTracker !== legacy.legacyForegroundTrackerAssumed) {
    reasons.push(
      `fg_tracker:${legacy.legacyForegroundTrackerAssumed}->${resolved.shouldRunForegroundTracker}`
    );
  }
  if (
    (resolved.flushIntervalMs ?? 0) !== legacy.legacyFlushIntervalMs &&
    resolved.flushIntervalMs !== null
  ) {
    reasons.push(
      `flush:${legacy.legacyFlushIntervalMs}->${resolved.flushIntervalMs ?? "null"}`
    );
  }
  const diffDetected = reasons.length > 0;
  return {
    diffDetected,
    diffReason: diffDetected ? reasons.join(";") : null,
  };
}

/** Construit l’input policy depuis l’état courant (sync helpers). */
export function buildResolveTrackingPolicyInput(params: {
  bgInputs: BgTrackingInputs;
  pendingAndroidFgsDeferred: boolean;
  networkReachable: boolean;
  driverTokenAvailable: boolean;
  missionBarStatus: MissionBarStatus | null;
  persistedLocationMode: LocationMode;
}): ResolveTrackingPolicyInput {
  const { bgInputs } = params;
  return {
    platform:
      bgInputs.platform === "ios" || bgInputs.platform === "android" ? bgInputs.platform : "web",
    isAuthenticated: bgInputs.isAuthenticated,
    role: bgInputs.role,
    appState: AppState.currentState,
    fgPermission: bgInputs.fgPermission,
    bgPermission: bgInputs.bgPermission,
    killSwitchEnabled: bgInputs.killSwitchEnabled,
    availabilityPresenceEnabled: !!bgInputs.availabilityPresenceEnabled,
    hasActiveMission: bgInputs.hasActiveMission,
    missionBarStatus: params.missionBarStatus,
    persistedLocationMode: params.persistedLocationMode,
    pendingAndroidFgsDeferred: params.pendingAndroidFgsDeferred,
    networkReachable: params.networkReachable,
    driverTokenAvailable: params.driverTokenAvailable,
  };
}
