import * as Sentry from "@sentry/react-native";
import { AppState, AppStateStatus, Platform } from "react-native";
import * as Location from "expo-location";
import { sendDriverLocation } from "../api/driverHttp";
import { DriverMissionStatus, type DriverMission, type DriverLocationAckStatus } from "../types";
import { isTrackingActiveStatus } from "../domain/status";
import { resolveMissionTrackingMode } from "../domain/resolveMissionTrackingMode";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { evaluateConnectivityPolicy } from "../../../core/network/connectivityPolicy";
import { getNetworkSnapshot } from "../../../core/network/networkState";
import { realtimeManager } from "../../../core/realtime/realtimeManager";
import { TrackingManager } from "../../../core/tracking/trackingManager";
import { resolveTrackingCadence, TrackingNetworkProfile } from "../../../core/tracking/cadenceResolver";
import { driverTrackingQueue, DriverTrackingMode } from "./driverTrackingQueue";
import { createCaptureId } from "./captureId";
import { resolveBridgeAckFields } from "./bridgeAckSemantics";
import { hideMissionBarAndroid } from "../missionBarAndroid";
import { stopMissionLiveActivity } from "../missionBarIOS";
import {
  ensureNativeTrackingWhileForeground,
  initializeBackgroundLocationTask,
  resumePendingNativeTrackingIfNeeded,
  setBackgroundTrackingMissionContext,
  stopBackgroundLocationTask,
} from "./backgroundLocationTask";
import { canUseBackgroundLocation } from "./backgroundRuntimeCompat";
import { formatTrackingSendError } from "./driverTrackingSendErrorFormat";
import {
  isLiveTrackingDisclosureAccepted,
  isPresenceDisclosureAccepted,
} from "./liveTrackingDisclosureSession";
import { emitBatteryBaselineIfTracing } from "../../../core/observability/gpsFidelityTrace";
import {
  canAttemptTrackingOperation,
  recordTrackingCircuitFailure,
  recordTrackingCircuitSuccess,
} from "../tracking/trackingCircuitBreaker";
import {
  forceRestartTrackingWatch,
  shouldForceRestartWatch,
  shouldTriggerAntiZombie,
  markAntiZombieTriggered,
  type SelfHealBridgeSlice,
  ANTI_ZOMBIE_FIX_AGE_SEC,
} from "../tracking/trackingSelfHeal";
import type {
  TrackingDesiredState,
  TrackingStopOutcome,
  TrackingStopRequest,
} from "../tracking/trackingLifecycleOwner";
export type {
  TrackingDesiredState,
  TrackingStopAuthority,
  TrackingStopOutcome,
  TrackingStopRequest,
} from "../tracking/trackingLifecycleOwner";
import {
  resolveTrackingFsmState,
  type TrackingFsmState,
} from "../tracking/TrackingStateMachine";
import {
  getDriverAvailabilityActive,
  setDriverAvailabilityActive,
} from "./driverAvailabilityBridge";
import { getTrackingPermissionsReady } from "./trackingPermissionsReady";
import {
  resolvePresenceGpsAccuracy,
  resolveTrackingEligibility,
  type TrackingEligibilityResult,
} from "../tracking/trackingEligibility";
import { tickTrackingRecovery } from "../tracking/TrackingRecoveryOrchestrator";
import {
  captureActiveRuntime,
  ensureTrackingAuthTerminalSubscription,
  isRuntimeActive,
  clearActiveRuntimeIfGeneration,
  registerTrackingPhysicalStop,
  startOrJoinTrackingRuntime,
  toNativeTrackingOwner,
  updateMissionContext,
  type TrackingMissionContext,
  type TrackingRuntimeIdentity,
} from "./trackingRuntimeRegistry";
import { getTrackingAuthAvailability } from "../../../core/auth/sessionAuthDecision";
import { computeFixAgeMs, WATCH_STALE_MS } from "./driverTrackingFixAge";
import {
  readTrackingContextLease,
  setTrackingContextLeaseDriverActive,
  setTrackingContextLeaseSwitching,
} from "./trackingContextLease";

export { computeFixAgeMs, WATCH_STALE_MS } from "./driverTrackingFixAge";

const FOREGROUND_INTERVAL_MS = Number(process.env.EXPO_PUBLIC_DRIVER_GPS_FOREGROUND_INTERVAL_MS ?? "8000");
const AGGRESSIVE_FOREGROUND_INTERVAL_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_GPS_AGGRESSIVE_FOREGROUND_INTERVAL_MS ?? "4000"
);
const BACKGROUND_INTERVAL_MS = Number(process.env.EXPO_PUBLIC_DRIVER_GPS_BACKGROUND_INTERVAL_MS ?? "20000");
const MAX_BACKOFF_MS = 60_000;
/** Dernière exception du tick tracking (pour logs __DEV__ / télémétrie, sans PII). */
let lastTrackingTickFailure: unknown = null;

const WATCH_DISTANCE_METERS = Number(process.env.EXPO_PUBLIC_DRIVER_GPS_WATCH_DISTANCE_METERS ?? "10");
const AGGRESSIVE_WATCH_DISTANCE_METERS = Number(
  process.env.EXPO_PUBLIC_DRIVER_GPS_AGGRESSIVE_WATCH_DISTANCE_METERS ?? "5"
);
const POSITION_TIMEOUT_MS = Number(process.env.EXPO_PUBLIC_DRIVER_POSITION_TIMEOUT_MS ?? "7000");
const STALE_FALLBACK_COOLDOWN_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_STALE_FALLBACK_COOLDOWN_MS ?? "20000"
);
const STALE_FALLBACK_BREAKER_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_STALE_FALLBACK_BREAKER_MS ?? "60000"
);
let permissionRequestInFlight: Promise<boolean> | null = null;
let nativeTrackingAppStateSubscribed = false;
let stopDriverTrackingInProgress: Promise<void> | null = null;
/** Génération de cycle de vie : un ancien stop ne peut pas muter une génération plus récente. */
let lifecycleGeneration = 0;
/** Intention lifecycle owner (D5) — STOP natif seulement si STOPPED. */
let desiredTrackingState: TrackingDesiredState = "STOPPED";
/** Watchdog : pas de callback GPS pendant mission EN_ROUTE. */
let noLocationCallbackTimer: ReturnType<typeof setTimeout> | null = null;
const NO_LOCATION_CALLBACK_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_NO_LOCATION_CALLBACK_MS ?? String(3 * 60_000)
);

function resolveForegroundIntervalMs(): number {
  if (isFeatureEnabled("driver_capture_aggressive_enabled")) {
    return AGGRESSIVE_FOREGROUND_INTERVAL_MS;
  }
  return FOREGROUND_INTERVAL_MS;
}

function resolveWatchDistanceMeters(): number {
  if (isFeatureEnabled("driver_capture_aggressive_enabled")) {
    return AGGRESSIVE_WATCH_DISTANCE_METERS;
  }
  return WATCH_DISTANCE_METERS;
}

function resolveBridgeDriverId(): number {
  const availability = getTrackingAuthAvailability();
  if (availability.kind === "SESSION_AVAILABLE") {
    return availability.driverId;
  }
  return 0;
}

function ensureNativeTrackingAppStateListener(): void {
  if (nativeTrackingAppStateSubscribed || Platform.OS === "web") return;
  nativeTrackingAppStateSubscribed = true;
  initializeBackgroundLocationTask();
  ensureTrackingAuthTerminalSubscription();
  registerTrackingPhysicalStop(async (request) => {
    await stopDriverTrackingBridge({
      expectedTrackingGenerationId: request.expectedTrackingGenerationId,
      reason: request.reason,
      skipRegistryStop: true,
    });
  });
  AppState.addEventListener("change", (next) => {
    // Toute transition active ↔ background/inactive : recalcul d’éligibilité
    // puis start / update mode / stop (une seule runtime GPS).
    ensureManagerState(next);
    notifyTrackingBridgeListeners();

    if (next === "active") {
      const runtime = captureActiveRuntime();
      if (!runtime) {
        void resumePendingNativeTrackingIfNeeded();
        return;
      }
      void resumePendingNativeTrackingIfNeeded();
      if (
        !isRuntimeActive(runtime.identity) ||
        !isFeatureEnabled("tracking_background_enabled")
      ) {
        return;
      }
      const missionSnapshot = runtime.missionContext;
      if (missionSnapshot.missionId != null) {
        void ensureNativeTrackingWhileForeground(
          missionSnapshot.missionId,
          missionSnapshot.missionStatus,
          {},
          "app_resume"
        );
      } else if (
        state.driverAvailable &&
        state.presenceWindowOpen &&
        isPresenceDisclosureAccepted()
      ) {
        void ensureNativeTrackingWhileForeground(
          null,
          null,
          { presenceWindow: true },
          "app_resume_presence"
        );
      }
    }
  });
}

type MissionSchedulingSnapshot = Pick<DriverMission, "scheduled_time" | "time_confirmed" | "scheduling">;

type TrackingBridgeState = {
  missionId: number | null;
  missionStatus: DriverMissionStatus | null;
  missionScheduling: MissionSchedulingSnapshot | null;
  /** Toggle disponibilité chauffeur (signal séparé de la fenêtre horaire). */
  driverAvailable: boolean;
  /**
   * Fenêtre horaire 07h–19h ouverte (présence arrière-plan uniquement).
   * Ne décide pas seule du start/stop GPS — passer par resolveTrackingEligibility.
   */
  presenceWindowOpen: boolean;
  /** Accuracy du watch GPS courant (pour redémarrer High ↔ Balanced). */
  watchAccuracy: number | null;
  lastSentAt: string | null;
  lastCapturedAt: string | null;
  lastEnqueuedAt: string | null;
  lastTransportAttemptAt: string | null;
  lastIngestedAt: string | null;
  lastPersistedAt: string | null;
  lastAckAt: string | null;
  lastAckIsQueued: boolean;
  lastAckStatus: DriverLocationAckStatus | null;
  lastAckError: string | null;
  bridgeLastAttemptAt: string | null;
  currentAttemptSeq: number;
  lastAckAttemptSeq: number | null;
  currentAttemptEventId: string | null;
  lastAckEventId: string | null;
  lastBackendAckLatencyMs: number | null;
  queueDepth: number;
  flushPathUsed: "http_fallback" | "socket_batch" | null;
  permission: "unknown" | "granted" | "denied";
  watchSubscription: Location.LocationSubscription | null;
  lastWatchAtMs: number | null;
  lastWatchedPosition: Location.LocationObject | null;
  networkProfile: TrackingNetworkProfile;
  profileSinceMs: number;
  staleFallbackBlockedUntilMs: number;
  staleFallbackTimeouts: number;
  lastStaleFallbackAttemptMs: number | null;
  lastHttpFallbackTrackingEventId: string | null;
  fallbackTrackingSeq: number;
  lastWatchRestartAtMs: number;
  watchRestartTimestampsMs: number[];
  fsmState: TrackingFsmState;
  lastFixProducedAtMs: number | null;
  trackingStartedAtMs: number | null;
};

export type DriverTrackingPosition = {
  lat: number;
  lng: number;
};

export type DriverTrackingBridgeSnapshot = {
  missionId: number | null;
  missionStatus: DriverMissionStatus | null;
  appState: AppStateStatus;
  isRunning: boolean;
  permission: "unknown" | "granted" | "denied";
  lastSentAt: string | null;
  lastCapturedAt: string | null;
  lastEnqueuedAt: string | null;
  lastTransportAttemptAt: string | null;
  lastIngestedAt: string | null;
  lastPersistedAt: string | null;
  lastAckAt: string | null;
  lastAckIsQueued: boolean;
  lastAckStatus: DriverLocationAckStatus | null;
  lastAckError: string | null;
  currentAttemptSeq: number;
  lastAckAttemptSeq: number | null;
  currentAttemptEventId: string | null;
  lastAckEventId: string | null;
  queueDepth: number;
  flushPathUsed: "http_fallback" | "socket_batch" | null;
  networkProfile: TrackingNetworkProfile;
  lastWatchAt: string | null;
  /** Epoch ms du dernier callback watch (≠ Location.timestamp). */
  lastWatchAtMs: number | null;
  /** Epoch ms = Location.timestamp du dernier fix produit. */
  lastFixProducedAtMs: number | null;
  lastPosition: DriverTrackingPosition | null;
  lastAttemptAt: string | null;
  consecutiveFailures: number;
  backoffUntilMs: number;
  fsmState: TrackingFsmState;
};

type DriverTrackingBridgeListener = (snapshot: DriverTrackingBridgeSnapshot) => void;

const state: TrackingBridgeState = {
  missionId: null,
  missionStatus: null,
  missionScheduling: null,
  driverAvailable: false,
  presenceWindowOpen: false,
  watchAccuracy: null,
  lastSentAt: null,
  lastCapturedAt: null,
  lastEnqueuedAt: null,
  lastTransportAttemptAt: null,
  lastIngestedAt: null,
  lastPersistedAt: null,
  lastAckAt: null,
  lastAckIsQueued: false,
  lastAckStatus: null,
  lastAckError: null,
  bridgeLastAttemptAt: null,
  currentAttemptSeq: 0,
  lastAckAttemptSeq: null,
  currentAttemptEventId: null,
  lastAckEventId: null,
  lastBackendAckLatencyMs: null,
  queueDepth: 0,
  flushPathUsed: null,
  permission: "unknown",
  watchSubscription: null,
  lastWatchAtMs: null,
  lastWatchedPosition: null,
  networkProfile: "normal",
  profileSinceMs: Date.now(),
  staleFallbackBlockedUntilMs: 0,
  staleFallbackTimeouts: 0,
  lastStaleFallbackAttemptMs: null,
  lastHttpFallbackTrackingEventId: null,
  fallbackTrackingSeq: 0,
  lastWatchRestartAtMs: 0,
  watchRestartTimestampsMs: [],
  fsmState: "IDLE",
  lastFixProducedAtMs: null,
  trackingStartedAtMs: null,
};

const trackingBridgeListeners = new Set<DriverTrackingBridgeListener>();

function readDriverLastKnownPosition(): DriverTrackingPosition | null {
  if (!state.lastWatchedPosition || state.lastWatchAtMs == null) return null;
  if (computeFixAgeMs(state.lastWatchedPosition, state.lastWatchAtMs) > WATCH_STALE_MS) {
    return null;
  }
  const { latitude, longitude } = state.lastWatchedPosition.coords;
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  return { lat: latitude, lng: longitude };
}

function buildTrackingBridgeSnapshot(): DriverTrackingBridgeSnapshot {
  const managerSnapshot = trackingManager.getSnapshot();
  return {
    missionId: state.missionId,
    missionStatus: state.missionStatus,
    appState: managerSnapshot.appState,
    isRunning: managerSnapshot.isRunning,
    permission: state.permission,
    lastSentAt: state.lastSentAt,
    lastCapturedAt: state.lastCapturedAt,
    lastEnqueuedAt: state.lastEnqueuedAt,
    lastTransportAttemptAt: state.lastTransportAttemptAt,
    lastIngestedAt: state.lastIngestedAt,
    lastPersistedAt: state.lastPersistedAt,
    lastAckAt: state.lastAckAt,
    lastAckIsQueued: state.lastAckIsQueued,
    lastAckStatus: state.lastAckStatus,
    lastAckError: state.lastAckError,
    currentAttemptSeq: state.currentAttemptSeq,
    lastAckAttemptSeq: state.lastAckAttemptSeq,
    currentAttemptEventId: state.currentAttemptEventId,
    lastAckEventId: state.lastAckEventId,
    queueDepth: state.queueDepth,
    flushPathUsed: state.flushPathUsed,
    networkProfile: state.networkProfile,
    lastWatchAt: state.lastWatchAtMs ? new Date(state.lastWatchAtMs).toISOString() : null,
    lastWatchAtMs: state.lastWatchAtMs,
    /** Location.timestamp du dernier vrai fix (autorité GNSS pour observabilité). */
    lastFixProducedAtMs: state.lastFixProducedAtMs,
    lastPosition: readDriverLastKnownPosition(),
    lastAttemptAt: state.bridgeLastAttemptAt ?? managerSnapshot.lastAttemptAt,
    consecutiveFailures: managerSnapshot.consecutiveFailures,
    backoffUntilMs: managerSnapshot.backoffUntilMs,
    fsmState: state.fsmState,
  };
}

function notifyTrackingBridgeListeners() {
  const snapshot = buildTrackingBridgeSnapshot();
  trackingBridgeListeners.forEach((listener) => {
    listener(snapshot);
  });
}

function beginBridgeAttempt(eventId: string): number {
  const attemptSeq = state.currentAttemptSeq + 1;
  state.currentAttemptSeq = attemptSeq;
  state.currentAttemptEventId = eventId;
  state.bridgeLastAttemptAt = new Date().toISOString();
  state.lastTransportAttemptAt = state.bridgeLastAttemptAt;
  state.lastAckError = null;
  return attemptSeq;
}

function applyBridgeAckStatus(
  ackStatus: DriverLocationAckStatus | null,
  ackAt: string | number | null,
  attemptSeq: number,
  eventId: string
): boolean {
  if (ackStatus == null) return false;
  if (attemptSeq !== state.currentAttemptSeq) return false;
  if (eventId !== state.currentAttemptEventId) return false;
  const ackAtIso =
    typeof ackAt === "number"
      ? new Date(ackAt).toISOString()
      : ackAt ?? new Date().toISOString();
  const fields = resolveBridgeAckFields(ackStatus, ackAtIso);
  state.lastAckAttemptSeq = attemptSeq;
  state.lastAckEventId = eventId;
  state.lastAckStatus = fields.lastAckStatus;
  state.lastAckError = fields.lastAckError;
  state.lastAckIsQueued = fields.lastAckIsQueued;
  if (fields.lastAckAt != null) {
    state.lastAckAt = fields.lastAckAt;
  }
  return true;
}

function getSelfHealSlice(): SelfHealBridgeSlice {
  return {
    watchSubscription: state.watchSubscription,
    staleFallbackTimeouts: state.staleFallbackTimeouts,
    staleFallbackBlockedUntilMs: state.staleFallbackBlockedUntilMs,
    lastWatchAtMs: state.lastWatchAtMs,
    lastWatchedPosition: state.lastWatchedPosition,
    lastWatchRestartAtMs: state.lastWatchRestartAtMs,
    watchRestartTimestampsMs: state.watchRestartTimestampsMs,
    missionId: state.missionId,
  };
}

function getSelfHealActions() {
  return {
    stopWatch: () => stopLocationWatch(),
    stopBackground: async (reason: string) => {
      await requestTrackingStop({
        reason,
        expectedGeneration: lifecycleGeneration,
        expectedMissionId: state.missionId,
        authority: "recovery_l2",
      });
    },
    ensureNativeForeground: async () => {
      if (state.missionId != null && isFeatureEnabled("tracking_background_enabled")) {
        await ensureNativeTrackingWhileForeground(
          state.missionId,
          state.missionStatus,
          {},
          "self_heal_restart"
        );
      }
    },
    ensureLocationWatch: () => ensureLocationWatch(),
    triggerDeviceHealth: (reason: string) => {
      try {
        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const heartbeat = require("./deviceHealthHeartbeat") as typeof import("./deviceHealthHeartbeat");
        if (typeof heartbeat.triggerDeviceHealthNow === "function") {
          void heartbeat.triggerDeviceHealthNow(reason).catch(() => undefined);
        }
      } catch {
        /* noop */
      }
    },
  };
}

/** Remote kick backend — redémarrage watch/FGS. */
export async function forceRestartTrackingWatchFromBridge(
  reason: string,
  appState: AppStateStatus = trackingManager.getSnapshot().appState
): Promise<boolean> {
  return forceRestartTrackingWatch(reason, getSelfHealSlice(), getSelfHealActions(), appState);
}

/**
 * Redémarrage DUR du runtime tracking : teardown complet (FGS natif + watch +
 * engine) puis reconstruction via `startDriverTrackingBridge`. Nécessaire pour
 * un FGS zombie Samsung dont le service est vivant mais la souscription GPS
 * native est morte : relancer seulement le watch JS (`forceRestartTrackingWatch`)
 * ne recrée pas le task natif. La mission courante est capturée depuis l'état
 * du bridge ; à froid (état vide après login/logout), on retombe sur `fallback`
 * (résolu depuis le cache des missions par l'appelant).
 */
export async function hardRestartDriverTrackingBridge(
  fallback: {
    missionId: number;
    status: DriverMissionStatus;
    scheduling?: MissionSchedulingSnapshot | null;
  } | null,
  reason: string
): Promise<boolean> {
  let missionId = state.missionId;
  let status = state.missionStatus;
  let scheduling = state.missionScheduling;
  if (missionId == null && fallback) {
    missionId = fallback.missionId;
    status = fallback.status;
    scheduling = fallback.scheduling ?? null;
  }
  await stopDriverTrackingBridge();
  if (missionId != null && status != null && isTrackingActiveStatus(status)) {
    startDriverTrackingBridge(missionId, status, scheduling);
    emitDriverTelemetry("tracking.hard_restart", {
      source: "driver.tracking.bridge",
      mission_id: missionId,
      mission_status: status,
      reason,
    });
    return true;
  }
  return false;
}

async function handleAntiZombieIfNeeded(appState: AppStateStatus): Promise<void> {
  const managerSnapshot = trackingManager.getSnapshot();
  if (
    !shouldTriggerAntiZombie({
      isTrackingRunning: managerSnapshot.isRunning,
      lastFixProducedAtMs: state.lastFixProducedAtMs,
      lastSentAt: state.lastSentAt,
      trackingStartedAtMs: state.trackingStartedAtMs,
    })
  ) {
    return;
  }
  markAntiZombieTriggered();
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const pipelineObs = require("./trackingPipelineObservability") as typeof import("./trackingPipelineObservability");
    pipelineObs.recordPipelineRecoveryReason("anti_zombie_fix_stale");
  } catch {
    /* instrumentation-only — ne doit jamais impacter la recovery */
  }
  emitDriverTelemetry("tracking.anti_zombie.triggered", {
    source: "driver.tracking.bridge",
    mission_id: state.missionId,
    app_state: appState,
    last_fix_age_sec: state.lastFixProducedAtMs
      ? (Date.now() - state.lastFixProducedAtMs) / 1000
      : null,
    threshold_sec: ANTI_ZOMBIE_FIX_AGE_SEC,
    fsm_state: state.fsmState,
  });
  getSelfHealActions().triggerDeviceHealth("anti_zombie_fix_stale");
  // P6 : tick FSM event-driven (pas de sleep long). Flag off → restartWatch seul.
  await tickTrackingRecovery(
    Date.now(),
    {
      reason: "anti_zombie_fix_stale",
      fixRecent: false,
      watchAlive: false,
    },
    {
      restartWatch: async (r) => {
        await forceRestartTrackingWatchFromBridge(r, appState);
      },
      restartFgs: async (r) => {
        if (state.missionId != null && isFeatureEnabled("tracking_background_enabled")) {
          const runtime = captureActiveRuntime();
          await ensureNativeTrackingWhileForeground(
            state.missionId,
            state.missionStatus,
            {
              nativeOwner: runtime ? toNativeTrackingOwner(runtime) : undefined,
            },
            r
          );
        }
      },
      restartEngine: async () => {
        const capturedMissionId = state.missionId;
        const capturedStatus = state.missionStatus;
        const capturedScheduling = state.missionScheduling;
        const gen = lifecycleGeneration;
        await stopDriverTrackingBridge();
        if (gen !== lifecycleGeneration) return;
        if (capturedMissionId != null && capturedStatus != null) {
          startDriverTrackingBridge(capturedMissionId, capturedStatus, capturedScheduling);
        }
      },
      reconnectTransport: async () => {
        const snap = realtimeManager.getSnapshot();
        if (snap.activeContextId) {
          realtimeManager.connect(snap.activeContextId, { enableSocket: true });
        }
      },
    }
  );
}

function resolvePayloadLocationMode(mode: DriverTrackingMode): DriverTrackingMode {
  if (state.missionId === null && mode === "mission_live") {
    return "availability_presence";
  }
  return mode;
}

function refreshFsmState(appState: AppStateStatus, fixStale: boolean) {
  if (!isFeatureEnabled("tracking_state_machine_enabled")) {
    return;
  }
  const eligibility = resolveBridgeEligibility(appState);
  state.fsmState = resolveTrackingFsmState({
    hasMission: hasActiveMission(),
    presenceEligible:
      eligibility.foregroundPresenceEligible ||
      eligibility.backgroundPresenceEligible,
    blocked: eligibility.blocked,
    enService: getDriverAvailabilityActive() === true,
    appForeground: appState === "active",
    missionLive: resolveTrackingMode(appState) === "mission_live",
    fixStale,
    circuitOpen: !canAttemptTrackingOperation(Date.now(), true),
    missionTerminal: state.missionStatus != null && !isTrackingActiveStatus(state.missionStatus),
  });
}

function hasActiveMission(): boolean {
  return state.missionId !== null && isTrackingActiveStatus(state.missionStatus);
}

function resolveBridgeEligibility(
  appState: AppStateStatus = trackingManager.getSnapshot().appState
): TrackingEligibilityResult {
  // SoT en-service : driverAvailabilityBridge (hydraté depuis Driver.is_available)
  const driverAvailable = getDriverAvailabilityActive();
  const knownAvailable = driverAvailable === true;
  if (state.driverAvailable !== knownAvailable) {
    state.driverAvailable = knownAvailable;
  }
  return resolveTrackingEligibility({
    driverAvailable,
    presenceWindowOpen: true,
    appForeground: appState === "active",
    presenceDisclosureAccepted: isPresenceDisclosureAccepted(),
    permissionsReady: getTrackingPermissionsReady(),
    hasActiveMission: hasActiveMission(),
  });
}

function isEligible(
  appState: AppStateStatus = trackingManager.getSnapshot().appState
) {
  return resolveBridgeEligibility(appState).trackingEligible;
}

function resolveExpoLocationAccuracy(appState: AppStateStatus): number {
  const tier = resolvePresenceGpsAccuracy({
    hasActiveMission: hasActiveMission(),
    appForeground: appState === "active",
  });
  return tier === "high" ? Location.Accuracy.High : Location.Accuracy.Balanced;
}

async function ensurePermission(appState: AppStateStatus) {
  if (state.permission === "denied") return false;
  if (permissionRequestInFlight) {
    return permissionRequestInFlight;
  }
  permissionRequestInFlight = (async () => {
    const foreground = await Location.requestForegroundPermissionsAsync();
    state.permission = foreground.granted ? "granted" : "denied";
    notifyTrackingBridgeListeners();
    if (!foreground.granted) {
      emitDriverTelemetry("tracking.permission.denied", {
        source: "driver.tracking.bridge",
        mission_id: state.missionId,
        app_state: appState,
      });
      return false;
    }
    const missionLive =
      state.missionId != null && isTrackingActiveStatus(state.missionStatus);
    if (
      missionLive &&
      isLiveTrackingDisclosureAccepted() &&
      isFeatureEnabled("tracking_background_enabled") &&
      canUseBackgroundLocation()
    ) {
      await Location.requestBackgroundPermissionsAsync().catch(() => undefined);
    }
    return true;
  })();
  try {
    return await permissionRequestInFlight;
  } finally {
    permissionRequestInFlight = null;
  }
}

function resetPermissionState() {
  permissionRequestInFlight = null;
  state.permission = "unknown";
  notifyTrackingBridgeListeners();
}

function getCadenceForTick(appState: AppStateStatus, mode: DriverTrackingMode) {
  if (!isFeatureEnabled("tracking_adaptive_cadence_enabled")) {
    return {
      networkProfile: "normal" as TrackingNetworkProfile,
      foregroundIntervalMs: resolveForegroundIntervalMs(),
      backgroundIntervalMs: BACKGROUND_INTERVAL_MS,
      ackStaleMs: 75_000,
    };
  }
  const managerSnapshot = trackingManager.getSnapshot();
  const connectivityMode = isFeatureEnabled("driver_network_2g_cadence_enabled")
    ? evaluateConnectivityPolicy(getNetworkSnapshot()).mode
    : null;
  const cadence = resolveTrackingCadence({
    mode,
    appState: appState === "active" ? "active" : "background",
    queueDepth: state.queueDepth,
    socketReady: state.flushPathUsed === "socket_batch",
    consecutiveFailures: managerSnapshot.consecutiveFailures,
    previousProfile: state.networkProfile,
    profileSinceMs: state.profileSinceMs,
    nowMs: Date.now(),
    networkModeHint: connectivityMode,
  });
  if (cadence.networkProfile !== state.networkProfile) {
    state.networkProfile = cadence.networkProfile;
    state.profileSinceMs = Date.now();
  }
  trackingManager.setIntervals({
    foregroundIntervalMs: cadence.foregroundIntervalMs,
    backgroundIntervalMs: cadence.backgroundIntervalMs,
  });
  return cadence;
}

async function getCurrentPositionWithTimeout(
  appState: AppStateStatus
): Promise<Location.LocationObject | null> {
  // Mission / présence foreground → High ; présence background → Balanced.
  const positionAccuracy = resolveExpoLocationAccuracy(appState);
  if (!isFeatureEnabled("tracking_safe_stale_fallback_enabled")) {
    return Location.getCurrentPositionAsync({
      accuracy: positionAccuracy,
      mayShowUserSettingsDialog: true,
    }).catch(() => null);
  }
  const now = Date.now();
  if (
    state.lastStaleFallbackAttemptMs !== null &&
    now - state.lastStaleFallbackAttemptMs < STALE_FALLBACK_COOLDOWN_MS
  ) {
    return null;
  }
  if (now < state.staleFallbackBlockedUntilMs) {
    return null;
  }
  state.lastStaleFallbackAttemptMs = now;
  const timeoutPromise = new Promise<null>((resolve) => {
    setTimeout(() => resolve(null), POSITION_TIMEOUT_MS);
  });
  const currentPosition = await Promise.race([
    Location.getCurrentPositionAsync({
      accuracy: positionAccuracy,
      mayShowUserSettingsDialog: true,
    }).catch(() => null),
    timeoutPromise,
  ]);
  if (currentPosition) {
    state.staleFallbackTimeouts = 0;
    recordTrackingCircuitSuccess();
    state.lastFixProducedAtMs = Date.now();
    return currentPosition;
  }
  state.staleFallbackTimeouts += 1;
  recordTrackingCircuitFailure();
  state.staleFallbackBlockedUntilMs = Date.now() + STALE_FALLBACK_BREAKER_MS;
  emitDriverTelemetry("tracking.stale_fallback.timeout", {
    source: "driver.tracking.bridge",
    mission_id: state.missionId,
    app_state: appState,
    stale_fallback_timeout_total: state.staleFallbackTimeouts,
    stale_fallback_blocked_until_ms: state.staleFallbackBlockedUntilMs,
  });
  /* Heartbeat forcé : on signale au backend que le device n'a plus de fix
   * (cas Samsung One UI : FGS éteint silencieusement, getCurrentPositionAsync
   * timeout). Lazy-required pour éviter les cycles d'import (bridge -> heartbeat
   * -> bridge.snapshot). Errors silencieuses, observabilité uniquement. */
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const heartbeat = require("./deviceHealthHeartbeat") as typeof import("./deviceHealthHeartbeat");
    if (typeof heartbeat.triggerDeviceHealthNow === "function") {
      void heartbeat.triggerDeviceHealthNow("stale_fallback_timeout").catch(() => undefined);
    }
  } catch {
    /* noop */
  }
  if (shouldForceRestartWatch(getSelfHealSlice())) {
    void forceRestartTrackingWatch(
      "stale_fallback_breaker",
      getSelfHealSlice(),
      getSelfHealActions(),
      appState
    );
  }
  return Location.getLastKnownPositionAsync({
    maxAge: WATCH_STALE_MS,
    requiredAccuracy: 100,
  }).catch(() => null);
}

async function flushPoint(appState: AppStateStatus) {
  if (!isEligible()) return;
  if (!canAttemptTrackingOperation(Date.now(), true)) {
    return;
  }
  const runtimeAtStart = captureActiveRuntime();
  const identitySnapshot: TrackingRuntimeIdentity | null =
    runtimeAtStart?.identity ?? null;
  const missionSnapshot: TrackingMissionContext | null =
    runtimeAtStart?.missionContext ?? null;
  const capturedMissionId =
    missionSnapshot?.missionId ?? state.missionId;

  void emitBatteryBaselineIfTracing("driver.tracking.bridge");
  const granted = await ensurePermission(appState);
  if (!granted) return;
  if (identitySnapshot && !isRuntimeActive(identitySnapshot)) return;
  const mode = resolveTrackingMode(appState);
  const payloadMode = resolvePayloadLocationMode(mode);
  refreshFsmState(appState, false);
  const position = await resolvePositionFromWatchOrFallback(appState, mode);
  if (!position) {
    refreshFsmState(appState, true);
    emitDriverTelemetry("tracking.send.skipped", {
      source: "driver.tracking.bridge",
      mission_id: capturedMissionId,
      reason: "no_position_fix",
      app_state: appState,
    });
    return;
  }
  if (identitySnapshot && !isRuntimeActive(identitySnapshot)) return;
  const fixTs =
    typeof position.timestamp === "number" && Number.isFinite(position.timestamp)
      ? position.timestamp
      : Date.now();
  const nowIso = new Date(fixTs).toISOString();
  state.lastFixProducedAtMs = fixTs;
  state.lastCapturedAt = nowIso;
  recordTrackingCircuitSuccess();
  const cadence = getCadenceForTick(appState, mode);
  const enqueuedItem = await driverTrackingQueue.enqueue({
    missionId: capturedMissionId,
    appState,
    locationMode: payloadMode,
    captureId: createCaptureId(),
    trackingGenerationId: identitySnapshot?.trackingGenerationId ?? null,
    missionContextVersion: missionSnapshot?.missionContextVersion ?? null,
    payload: {
      latitude: position.coords.latitude,
      longitude: position.coords.longitude,
      accuracy: position.coords.accuracy ?? undefined,
      heading: position.coords.heading ?? undefined,
      speed: position.coords.speed ?? undefined,
      missionId: capturedMissionId,
      isBackground: appState !== "active",
      timestamp: nowIso,
      locationMode: payloadMode,
      trackingGenerationId: identitySnapshot?.trackingGenerationId,
      missionContextVersion: missionSnapshot?.missionContextVersion,
      trackingIdentityId: identitySnapshot?.trackingIdentityId,
    },
  });
  if (!enqueuedItem) {
    // Session ledger non-READY (register en cours / échec) — drop observé côté queue.
    return;
  }
  state.lastEnqueuedAt = new Date().toISOString();
  const attemptSeq = beginBridgeAttempt(enqueuedItem.id);
  const flushResult = await driverTrackingQueue.flush({
    ackStaleMs: cadence.ackStaleMs,
    networkProfile: cadence.networkProfile,
    forceHttpFallback: appState !== "active",
  });
  // Mutations runtime / ACK UI seulement si génération encore active.
  if (identitySnapshot && !isRuntimeActive(identitySnapshot)) {
    return;
  }
  state.queueDepth = flushResult.queueDepth;
  state.flushPathUsed = flushResult.flushPathUsed;
  if (flushResult.ingestedEventIds.includes(enqueuedItem.id)) {
    state.lastIngestedAt = new Date().toISOString();
  }
  if (flushResult.persistedEventIds.includes(enqueuedItem.id)) {
    state.lastPersistedAt = new Date().toISOString();
  }
  const ackMatchesCurrentPoint =
    flushResult.lastBackendAckRequestEventId === enqueuedItem.id;
  if (
    ackMatchesCurrentPoint &&
    flushResult.lastBackendAckStatus != null &&
    flushResult.lastBackendAckAt != null
  ) {
    applyBridgeAckStatus(
      flushResult.lastBackendAckStatus,
      flushResult.lastBackendAckAt,
      attemptSeq,
      enqueuedItem.id
    );
    if (state.lastAckAt) {
      state.lastBackendAckLatencyMs =
        flushResult.lastBackendAckAt - Date.parse(nowIso);
    }
  }

  const lastAckMs = state.lastAckAt ? Date.parse(state.lastAckAt) : null;
  // Un ACK `queued` (HTTP 202 + Kafka async) est valide : ne pas relancer un HTTP bridge.
  const ackIsStale =
    lastAckMs !== null &&
    !state.lastAckIsQueued &&
    Date.now() - lastAckMs > cadence.ackStaleMs;
  if (ackIsStale) {
    emitDriverTelemetry("tracking.send.backoff", {
      source: "driver.tracking.bridge",
      mission_id: state.missionId,
      reason: "ack_stale",
      stale_ms: Date.now() - lastAckMs!,
      ack_stale_ms: cadence.ackStaleMs,
      network_profile_active: cadence.networkProfile,
      app_state: appState,
    });
  }

  const queueHandledTransport =
    flushResult.sent > 0 ||
    flushResult.backendAcked > 0 ||
    flushResult.flushPathUsed === "http_fallback" ||
    flushResult.lastBackendAckStatus === "queued";

  const shouldFallback =
    isFeatureEnabled("tracking_http_fallback_enabled") &&
    !queueHandledTransport &&
    (ackIsStale ||
      (flushResult.sent === 0 &&
        flushResult.backendAcked === 0 &&
        flushResult.socketEmitted === 0 &&
        flushResult.dropped === 0));
  const secondBucket = Math.floor((position.timestamp ?? Date.now()) / 1000);
  state.fallbackTrackingSeq = (state.fallbackTrackingSeq + 1) % 1000;
  const fallbackTrackingEventId = `bridge_fb_${state.missionId ?? "presence"}_${secondBucket}_${state.fallbackTrackingSeq}`;
  const shouldSkipDuplicateFallback =
    state.lastHttpFallbackTrackingEventId === fallbackTrackingEventId;
  if (shouldFallback && !shouldSkipDuplicateFallback) {
    const fallbackAttemptSeq = beginBridgeAttempt(fallbackTrackingEventId);
    try {
      const response = await sendDriverLocation({
        latitude: position.coords.latitude,
        longitude: position.coords.longitude,
        accuracy: position.coords.accuracy ?? undefined,
        heading: position.coords.heading ?? undefined,
        speed: position.coords.speed ?? undefined,
        missionId: state.missionId,
        isBackground: appState !== "active",
        timestamp: nowIso,
        locationMode: payloadMode,
        trackingEventId: fallbackTrackingEventId,
        captureId: enqueuedItem.captureId ?? enqueuedItem.payload.captureId ?? null,
      });
      if (
        response.tracking_event_id != null &&
        response.tracking_event_id !== fallbackTrackingEventId
      ) {
        state.lastAckStatus = response.ack_status;
        state.lastAckError = "ack_event_id_mismatch";
      } else {
        applyBridgeAckStatus(
          response.ack_status,
          nowIso,
          fallbackAttemptSeq,
          fallbackTrackingEventId
        );
      }
    } catch (error) {
      state.lastAckError = formatTrackingSendError(error).error_message ?? "network_error";
      state.lastAckStatus = null;
      throw error;
    }
    state.lastHttpFallbackTrackingEventId = fallbackTrackingEventId;
    state.flushPathUsed = "http_fallback";
  }
  // lastSentAt seulement si le point COURANT a réellement été émis ou confirmé.
  const currentId = enqueuedItem.id;
  const currentHandled =
    flushResult.socketEmittedEventIds.includes(currentId) ||
    flushResult.ingestedEventIds.includes(currentId) ||
    flushResult.persistedEventIds.includes(currentId) ||
    (shouldFallback &&
      !shouldSkipDuplicateFallback &&
      state.lastHttpFallbackTrackingEventId === fallbackTrackingEventId &&
      state.lastAckError == null);
  if (isEligible() && currentHandled) {
    state.lastSentAt = new Date().toISOString();
  }
  emitDriverTelemetry("tracking.bridge.health", {
    source: "driver.tracking.bridge",
    mission_id: state.missionId,
    queue_depth: state.queueDepth,
    oldest_item_age_ms: flushResult.oldestItemAgeMs,
    network_profile_active: cadence.networkProfile,
    backend_ack_latency_ms: state.lastBackendAckLatencyMs,
    tracking_flush_batch_size: flushResult.sent,
    socket_emit_without_backend_ack_total: flushResult.socketEmitted,
  });
  notifyTrackingBridgeListeners();
}

async function resolvePositionFromWatchOrFallback(
  appState: AppStateStatus,
  _mode: DriverTrackingMode
) {
  const hasWatch =
    state.lastWatchedPosition !== null && state.lastWatchAtMs !== null;
  if (hasWatch && state.lastWatchedPosition) {
    const fixAgeMs = computeFixAgeMs(
      state.lastWatchedPosition,
      state.lastWatchAtMs
    );
    // Même seuil pour tous les modes (plus d'exemption availability_presence).
    if (fixAgeMs < WATCH_STALE_MS) {
      return state.lastWatchedPosition;
    }
  }
  return getCurrentPositionWithTimeout(appState);
}

function buildActiveMissionSnapshot(): DriverMission | null {
  if (state.missionId == null || !state.missionStatus) return null;
  return {
    id: state.missionId,
    status: state.missionStatus,
    scheduled_time: state.missionScheduling?.scheduled_time ?? null,
    time_confirmed: state.missionScheduling?.time_confirmed ?? null,
    scheduling: state.missionScheduling?.scheduling ?? null,
  };
}

function resolveTrackingMode(appState: AppStateStatus): DriverTrackingMode {
  const eligibility = resolveBridgeEligibility(appState);
  if (
    !eligibility.missionEligible &&
    (eligibility.foregroundPresenceEligible ||
      eligibility.backgroundPresenceEligible)
  ) {
    return "availability_presence";
  }
  const mission = buildActiveMissionSnapshot();
  if (mission) {
    const missionMode = resolveMissionTrackingMode(mission);
    if (missionMode) return missionMode;
  }
  if (appState !== "active" && isFeatureEnabled("tracking_background_enabled")) {
    return "availability_presence";
  }
  return "mission_live";
}

async function ensureLocationWatch(appStateOverride?: AppStateStatus) {
  if (Platform.OS === "web") {
    /* Sur web, expo-location brise le cleanup (LocationEventEmitter.removeSubscription n'existe pas). */
    return;
  }
  const appState = appStateOverride ?? trackingManager.getSnapshot().appState;
  if (!isEligible(appState) || state.watchSubscription) return;
  const granted = await ensurePermission(appState);
  if (!granted) return;
  const accuracy = resolveExpoLocationAccuracy(appState);
  try {
    state.watchSubscription = await Location.watchPositionAsync(
      {
        accuracy,
        distanceInterval: resolveWatchDistanceMeters(),
        timeInterval: appState === "active" ? resolveForegroundIntervalMs() : BACKGROUND_INTERVAL_MS,
        mayShowUserSettingsDialog: true,
      },
      (position) => {
        state.lastWatchedPosition = position;
        state.lastWatchAtMs = Date.now();
        if (
          typeof position.timestamp === "number" &&
          Number.isFinite(position.timestamp)
        ) {
          state.lastFixProducedAtMs = position.timestamp;
          state.lastCapturedAt = new Date(position.timestamp).toISOString();
        }
        notifyTrackingBridgeListeners();
      }
    );
    state.watchAccuracy = accuracy;
    emitDriverTelemetry("tracking.watch.started", {
      source: "driver.tracking.bridge",
      mission_id: state.missionId,
      app_state: appState,
      distance_m: resolveWatchDistanceMeters(),
      accuracy_tier:
        resolvePresenceGpsAccuracy({
          hasActiveMission: hasActiveMission(),
          appForeground: appState === "active",
        }),
    });
  } catch (error) {
    emitDriverTelemetry("tracking.watch.unavailable", {
      source: "driver.tracking.bridge",
      mission_id: state.missionId,
      reason: error instanceof Error ? error.message : "watch_failed",
    });
  }
}

function stopLocationWatch() {
  if (state.watchSubscription) {
    const sub = state.watchSubscription;
    state.watchSubscription = null;
    try {
      if (typeof sub.remove === "function") {
        sub.remove();
      }
    } catch {
      /* ex. web ou impl incomplete : ignorer le cleanup d'ecoute */
    }
  }
  state.watchAccuracy = null;
  state.lastWatchAtMs = null;
  state.lastWatchedPosition = null;
  notifyTrackingBridgeListeners();
}

export async function flushDriverTrackingQueueNow() {
  const managerSnapshot = trackingManager.getSnapshot();
  const mode = resolveTrackingMode(managerSnapshot.appState);
  const cadence = getCadenceForTick(managerSnapshot.appState, mode);
  const result = await driverTrackingQueue.flush({
    ackStaleMs: cadence.ackStaleMs,
    networkProfile: cadence.networkProfile,
    forceHttpFallback: managerSnapshot.appState !== "active",
  });
  state.queueDepth = result.queueDepth;
  if (
    result.lastBackendAckRequestEventId != null &&
    result.lastBackendAckStatus != null &&
    result.lastBackendAckAt != null
  ) {
    const attemptSeq = beginBridgeAttempt(result.lastBackendAckRequestEventId);
    applyBridgeAckStatus(
      result.lastBackendAckStatus,
      result.lastBackendAckAt,
      attemptSeq,
      result.lastBackendAckRequestEventId
    );
  }
  if (result.flushPathUsed) {
    state.flushPathUsed = result.flushPathUsed;
  }
  notifyTrackingBridgeListeners();
}

export async function getDriverTrackingQueueSnapshot() {
  return driverTrackingQueue.getSnapshot();
}

export async function syncBridgeQueueDepthFromPersistence() {
  const snapshot = await driverTrackingQueue.getSnapshot();
  state.queueDepth = snapshot.queueDepth;
  notifyTrackingBridgeListeners();
}

async function sendLegacyPoint(appState: AppStateStatus, nowIso: string) {
  const granted = await ensurePermission(appState);
  if (!granted) return;
  const position = await getCurrentPositionWithTimeout(appState);
  if (!position) return;
  state.lastWatchedPosition = position;
  state.lastWatchAtMs = Date.now();
  if (state.missionId === null && !isEligible(appState)) return;
  const legacyEventId = `bridge_legacy_${state.missionId ?? "presence"}_${Date.now()}`;
  const attemptSeq = beginBridgeAttempt(legacyEventId);
  try {
    const response = await sendDriverLocation({
      latitude: position.coords.latitude,
      longitude: position.coords.longitude,
      accuracy: position.coords.accuracy ?? undefined,
      heading: position.coords.heading ?? undefined,
      speed: position.coords.speed ?? undefined,
      missionId: state.missionId,
      isBackground: appState !== "active",
      timestamp: nowIso,
      locationMode: resolveTrackingMode(appState),
      trackingEventId: legacyEventId,
      captureId: createCaptureId(),
    });
    if (
      response.tracking_event_id != null &&
      response.tracking_event_id !== legacyEventId
    ) {
      state.lastAckStatus = response.ack_status;
      state.lastAckError = "ack_event_id_mismatch";
      throw new Error("ack_event_id_mismatch");
    }
    const applied = applyBridgeAckStatus(
      response.ack_status,
      nowIso,
      attemptSeq,
      legacyEventId
    );
    if (!applied || state.lastAckError) {
      throw new Error(state.lastAckError ?? `ack_${response.ack_status}`);
    }
  } catch (error) {
    if (!state.lastAckError) {
      state.lastAckError =
        formatTrackingSendError(error).error_message ?? "network_error";
    }
    throw error;
  }
}

const trackingManager = new TrackingManager({
  foregroundIntervalMs: resolveForegroundIntervalMs(),
  backgroundIntervalMs: BACKGROUND_INTERVAL_MS,
  maxBackoffMs: MAX_BACKOFF_MS,
  onTick: async ({ appState }) => {
    if (!isEligible()) return "skipped";
    try {
      await handleAntiZombieIfNeeded(appState);
      if (isFeatureEnabled("tracking_persistent_runtime_enabled")) {
        await flushPoint(appState);
      } else {
        const nowIso = new Date().toISOString();
        await sendLegacyPoint(appState, nowIso);
        state.lastSentAt = nowIso;
      }
      lastTrackingTickFailure = null;
      return "success";
    } catch (error) {
      lastTrackingTickFailure = error;
      return "failed";
    }
  },
  onFailure: ({ appState, consecutiveFailures, backoffMs }) => {
    const errMeta = formatTrackingSendError(lastTrackingTickFailure);
    const persistentQueue = isFeatureEnabled("tracking_persistent_runtime_enabled");
    emitDriverTelemetry("tracking.send.failure", {
      source: "driver.tracking.bridge",
      mission_id: state.missionId,
      retry_count: consecutiveFailures,
      app_state: appState,
      error_message: errMeta.error_message,
      error_class: errMeta.error_class,
      http_status: errMeta.http_status,
      api_error_code: errMeta.api_error_code,
      transport_code: errMeta.transport_code,
      tracking_tick_path: persistentQueue ? "persistent_queue" : "legacy_http",
      flush_path_last: state.flushPathUsed,
      network_profile: state.networkProfile,
      presence_window_active: state.presenceWindowOpen,
      driver_available: state.driverAvailable,
      driver_socket_ready: realtimeManager.isDriverSocketReady(),
    });
    emitDriverTelemetry("tracking.send.backoff", {
      source: "driver.tracking.bridge",
      mission_id: state.missionId,
      retry_count: consecutiveFailures,
      app_state: appState,
      backoff_ms: backoffMs,
      error_class: errMeta.error_class,
    });
    if (__DEV__) {
      console.warn("[driver_tracking_bridge_send_failed]", {
        failures: consecutiveFailures,
        backoffMs,
        ...errMeta,
        tracking_tick_path: persistentQueue ? "persistent_queue" : "legacy_http",
        flush_path_last: state.flushPathUsed,
        network_profile: state.networkProfile,
        driver_socket_ready: realtimeManager.isDriverSocketReady(),
      });
    }
    /** Breadcrumbs légers sur rafales d’échecs (évite le bruit à chaque tick). */
    if (consecutiveFailures === 3 || consecutiveFailures === 10 || consecutiveFailures === 25) {
      Sentry.addBreadcrumb({
        category: "driver.tracking",
        type: "default",
        level: "warning",
        message: "tracking_send_failure_burst",
        data: {
          failures: consecutiveFailures,
          backoff_ms: backoffMs,
          error_class: errMeta.error_class,
          http_status: errMeta.http_status,
          api_error_code: errMeta.api_error_code,
          transport_code: errMeta.transport_code,
          tracking_tick_path: persistentQueue ? "persistent_queue" : "legacy_http",
          mission_id: state.missionId,
          app_state: appState,
        },
      });
    }
  },
  onRecovered: ({ appState, previousFailures }) => {
    emitDriverTelemetry("tracking.send.recovered", {
      source: "driver.tracking.bridge",
      mission_id: state.missionId,
      retry_count: previousFailures,
      app_state: appState,
    });
  },
});

async function stopMissionTrackingBridge(expectedGeneration?: number): Promise<void> {
  if (expectedGeneration != null && expectedGeneration !== lifecycleGeneration) {
    return;
  }
  await flushDriverTrackingQueueNow();
  if (expectedGeneration != null && expectedGeneration !== lifecycleGeneration) {
    return;
  }
  await syncBridgeQueueDepthFromPersistence();
  if (expectedGeneration != null && expectedGeneration !== lifecycleGeneration) {
    return;
  }
  state.missionId = null;
  state.missionStatus = null;
  state.missionScheduling = null;
  state.lastSentAt = null;
  state.lastCapturedAt = null;
  state.lastEnqueuedAt = null;
  state.lastTransportAttemptAt = null;
  state.lastIngestedAt = null;
  state.lastPersistedAt = null;
  state.lastAckAt = null;
  state.lastAckIsQueued = false;
  state.lastAckStatus = null;
  state.lastAckError = null;
  state.lastAckAttemptSeq = null;
  state.lastAckEventId = null;
  state.currentAttemptEventId = null;
  notifyTrackingBridgeListeners();
}

async function ensurePresenceTrackingState(): Promise<void> {
  await syncBridgeQueueDepthFromPersistence();
  ensureManagerState();
  notifyTrackingBridgeListeners();
}

async function stopTrackingRuntime(): Promise<void> {
  await setBackgroundTrackingMissionContext(null, null);
  await requestTrackingStop({
    reason: "tracking_bridge_stopped",
    expectedGeneration: lifecycleGeneration,
    expectedMissionId: state.missionId,
    authority: "explicit",
  });
  stopLocationWatch();
  trackingManager.stop();
  state.trackingStartedAtMs = null;
  notifyTrackingBridgeListeners();
}

/**
 * D5 — seule entrée bridge pour un STOP natif intentionnel.
 * Check génération + desiredState ; abort immédiat pré-Unregister.
 */
export async function requestTrackingStop(
  req: TrackingStopRequest
): Promise<TrackingStopOutcome> {
  emitDriverTelemetry("tracking.lifecycle.stop.requested", {
    source: "driver.tracking.bridge",
    reason: req.reason,
    authority: req.authority,
    expected_generation: req.expectedGeneration,
    actual_generation: lifecycleGeneration,
    expected_mission_id: req.expectedMissionId ?? null,
    mission_id: state.missionId,
    desired_state: desiredTrackingState,
  });

  if (req.authority === "transient_loss") {
    emitDriverTelemetry("tracking.lifecycle.stop.deferred", {
      source: "driver.tracking.bridge",
      reason: req.reason,
      authority: req.authority,
      expected_generation: req.expectedGeneration,
      mission_id: state.missionId,
    });
    return "deferred";
  }

  if (req.authority === "recovery_l2") {
    // L2 destructif : réservé aux preuves natives positives ; le self-heal
    // par défaut est L1 (pas d'appel stopBackground). Si on arrive ici,
    // on exige quand même le garde génération.
  }

  desiredTrackingState = "STOPPED";

  if (req.expectedGeneration !== lifecycleGeneration) {
    emitDriverTelemetry("tracking.lifecycle.stop.abandoned", {
      source: "driver.tracking.bridge",
      reason: req.reason,
      abandon_reason: "generation_mismatch",
      expected_generation: req.expectedGeneration,
      actual_generation: lifecycleGeneration,
      authority: req.authority,
    });
    return "abandoned";
  }

  if (
    req.expectedMissionId != null &&
    state.missionId != null &&
    req.expectedMissionId !== state.missionId
  ) {
    emitDriverTelemetry("tracking.lifecycle.stop.abandoned", {
      source: "driver.tracking.bridge",
      reason: req.reason,
      abandon_reason: "mission_mismatch",
      expected_mission_id: req.expectedMissionId,
      mission_id: state.missionId,
      authority: req.authority,
    });
    return "abandoned";
  }

  const expectedGen = req.expectedGeneration;
  const { nativeStopped } = await stopBackgroundLocationTask(req.reason, {
    shouldAbortNativeStop: () => {
      if (expectedGen !== lifecycleGeneration) return true;
      if (desiredTrackingState !== "STOPPED") return true;
      return false;
    },
  });

  if (!nativeStopped) {
    if (expectedGen !== lifecycleGeneration || desiredTrackingState !== "STOPPED") {
      emitDriverTelemetry("tracking.lifecycle.stop.abandoned", {
        source: "driver.tracking.bridge",
        reason: req.reason,
        abandon_reason: "pre_native_or_controller",
        expected_generation: expectedGen,
        actual_generation: lifecycleGeneration,
        desired_state: desiredTrackingState,
        authority: req.authority,
      });
      return "abandoned";
    }
    // Task déjà arrêtée / noop — traité comme stopped côté intention.
  }

  return "stopped";
}

/** Test-only : génération lifecycle courante. */
export function __getLifecycleGenerationForTests(): number {
  return lifecycleGeneration;
}

/** Génération lifecycle courante (callers ownership / recovery). */
export function getTrackingLifecycleGeneration(): number {
  return lifecycleGeneration;
}

/** Test-only : desired state. */
export function __getDesiredTrackingStateForTests(): TrackingDesiredState {
  return desiredTrackingState;
}

/** Canary D5-C4 — snapshot fraîcheur (QA panel uniquement). */
export type CanaryFreshnessSnapshot = {
  lastSentAt: string | null;
  lastFixProducedAtMs: number | null;
  trackingStartedAtMs: number | null;
};

export function __canaryD5SnapshotFreshness(): CanaryFreshnessSnapshot {
  return {
    lastSentAt: state.lastSentAt,
    lastFixProducedAtMs: state.lastFixProducedAtMs,
    trackingStartedAtMs: state.trackingStartedAtMs,
  };
}

/** Force fraîcheur UNKNOWN + startedAge élevé (simule T4 Prod126). */
export function __canaryD5ApplyUnknownFreshness(
  fakeStartedAgeSec = 120
): CanaryFreshnessSnapshot {
  const prev = __canaryD5SnapshotFreshness();
  state.lastSentAt = null;
  state.lastFixProducedAtMs = null;
  state.trackingStartedAtMs = Date.now() - Math.max(60, fakeStartedAgeSec) * 1000;
  return prev;
}

export function __canaryD5RestoreFreshness(prev: CanaryFreshnessSnapshot): void {
  state.lastSentAt = prev.lastSentAt;
  state.lastFixProducedAtMs = prev.lastFixProducedAtMs;
  state.trackingStartedAtMs = prev.trackingStartedAtMs;
}

/** Probe anti-zombie sur l'état courant (sans side-effect). */
export function __canaryD5WouldTriggerAntiZombie(): boolean {
  return shouldTriggerAntiZombie({
    isTrackingRunning: trackingManager.getSnapshot().isRunning,
    lastFixProducedAtMs: state.lastFixProducedAtMs,
    lastSentAt: state.lastSentAt,
    trackingStartedAtMs: state.trackingStartedAtMs,
  });
}

/**
 * Arrêt local sans flush réseau — obligatoire après switch vers COMPANY.
 * Conserve SQLite ; n'appelle jamais flushDriverTrackingQueueNow.
 */
export async function hardStopDriverContextRuntime(reason = "context_left_driver"): Promise<void> {
  clearNoLocationCallbackWatchdog();
  void hideMissionBarAndroid();
  const missionIdForBar = state.missionId;
  if (missionIdForBar != null && isFeatureEnabled("driver_mission_bar_enabled")) {
    void stopMissionLiveActivity(missionIdForBar);
  }

  const expectedGenId = captureActiveRuntime()?.identity.trackingGenerationId;
  if (expectedGenId) {
    clearActiveRuntimeIfGeneration(expectedGenId);
  }

  // Reset mission state SANS flush réseau
  state.missionId = null;
  state.missionStatus = null;
  state.missionScheduling = null;
  state.lastSentAt = null;
  state.lastCapturedAt = null;
  state.lastEnqueuedAt = null;
  state.lastTransportAttemptAt = null;
  state.lastIngestedAt = null;
  state.lastPersistedAt = null;
  state.lastAckAt = null;
  state.lastAckIsQueued = false;
  state.lastAckStatus = null;
  state.lastAckError = null;
  state.lastAckAttemptSeq = null;
  state.lastAckEventId = null;
  state.currentAttemptEventId = null;
  state.lastBackendAckLatencyMs = null;
  state.networkProfile = "normal";
  state.profileSinceMs = Date.now();
  state.staleFallbackBlockedUntilMs = 0;
  state.staleFallbackTimeouts = 0;
  state.lastStaleFallbackAttemptMs = null;
  state.lastHttpFallbackTrackingEventId = null;
  resetPermissionState();

  // Await obligatoire : taskContext effacé avant resolve
  await setBackgroundTrackingMissionContext(null, null);
  await requestTrackingStop({
    reason,
    expectedGeneration: lifecycleGeneration,
    expectedMissionId: null,
    authority: "explicit",
  });
  stopLocationWatch();
  trackingManager.stop();
  state.trackingStartedAtMs = null;
  await syncBridgeQueueDepthFromPersistence();
  notifyTrackingBridgeListeners();

  emitDriverTelemetry("tracking.context.hard_stop", {
    source: "driver.tracking.bridge",
    reason,
  });
}

function ensureManagerState(appStateOverride?: AppStateStatus) {
  const appState = appStateOverride ?? trackingManager.getSnapshot().appState;
  const eligibility = resolveBridgeEligibility(appState);

  if (!eligibility.trackingEligible) {
    void syncBridgeQueueDepthFromPersistence();
    // D5 : plus de stopBackgroundLocationTask direct (bypass B2).
    void requestTrackingStop({
      reason: "ineligible_tracking_state",
      expectedGeneration: lifecycleGeneration,
      expectedMissionId: state.missionId,
      authority: "explicit",
    });
    trackingManager.stop();
    state.trackingStartedAtMs = null;
    stopLocationWatch();
    void setBackgroundTrackingMissionContext(null, null);
    return;
  }

  ensureNativeTrackingAppStateListener();

  if (eligibility.missionEligible && state.missionId != null) {
    const runtime = captureActiveRuntime();
    const nativeOwner = runtime ? toNativeTrackingOwner(runtime) : undefined;
    void setBackgroundTrackingMissionContext(
      state.missionId,
      state.missionStatus,
      "mission",
      state.missionScheduling,
      nativeOwner
    );
    if (isFeatureEnabled("tracking_background_enabled")) {
      void ensureNativeTrackingWhileForeground(
        state.missionId,
        state.missionStatus,
        { scheduling: state.missionScheduling, nativeOwner },
        "ensure_manager_state"
      );
    }
  } else if (eligibility.backgroundPresenceEligible) {
    const runtime = captureActiveRuntime();
    const nativeOwner = runtime ? toNativeTrackingOwner(runtime) : undefined;
    void setBackgroundTrackingMissionContext(null, null, "presence_window", null, nativeOwner);
    if (isFeatureEnabled("tracking_background_enabled")) {
      void ensureNativeTrackingWhileForeground(
        null,
        null,
        { presenceWindow: true, nativeOwner },
        "ensure_manager_presence"
      );
    }
  } else if (eligibility.foregroundPresenceEligible) {
    // Présence FG : watch + FGS si background tracking activé (contrat HOME/lock).
    if (isFeatureEnabled("tracking_background_enabled")) {
      const runtime = captureActiveRuntime();
      const nativeOwner = runtime ? toNativeTrackingOwner(runtime) : undefined;
      void setBackgroundTrackingMissionContext(null, null, "presence_window", null, nativeOwner);
      void ensureNativeTrackingWhileForeground(
        null,
        null,
        { presenceWindow: true, nativeOwner },
        "ensure_manager_presence_fg"
      );
    }
  } else if (eligibility.blocked) {
    // BLOCKED : en service sans permissionsReady — ne pas STOP comme hors service.
    emitDriverTelemetry("tracking.presence.blocked", {
      source: "driver.tracking.bridge",
      reason: "permissions_not_ready",
      driver_available: getDriverAvailabilityActive(),
    });
  }

  const desiredAccuracy = resolveExpoLocationAccuracy(appState);
  if (
    state.watchSubscription &&
    state.watchAccuracy != null &&
    state.watchAccuracy !== desiredAccuracy
  ) {
    stopLocationWatch();
  }
  void ensureLocationWatch(appState);
  void syncBridgeQueueDepthFromPersistence();
  const mode = resolveTrackingMode(appState);
  const snapshot = trackingManager.getSnapshot();
  if (!snapshot.isRunning) {
    state.trackingStartedAtMs = Date.now();
    trackingManager.start(mode);
    return;
  }
  trackingManager.updateMode(mode);
}

function clearNoLocationCallbackWatchdog(): void {
  if (noLocationCallbackTimer) {
    clearTimeout(noLocationCallbackTimer);
    noLocationCallbackTimer = null;
  }
}

function armNoLocationCallbackWatchdog(missionId: number, generation: number): void {
  clearNoLocationCallbackWatchdog();
  noLocationCallbackTimer = setTimeout(() => {
    noLocationCallbackTimer = null;
    if (generation !== lifecycleGeneration) return;
    if (state.missionId !== missionId) return;
    if (state.lastFixProducedAtMs != null) return;
    emitDriverTelemetry("tracking.no_location_callback", {
      source: "driver.tracking.bridge",
      mission_id: missionId,
      lifecycle_generation: generation,
    });
    void forceRestartTrackingWatchFromBridge("no_location_callback", AppState.currentState).catch(
      () => undefined
    );
  }, NO_LOCATION_CALLBACK_MS);
}

export function startDriverTrackingBridge(
  missionId: number,
  status: DriverMissionStatus,
  scheduling?: MissionSchedulingSnapshot | null
) {
  // Incrémente la génération : tout stop en cours avec une génération plus ancienne
  // ne doit plus effacer missionId / arrêter le runtime.
  lifecycleGeneration += 1;
  const operationGeneration = lifecycleGeneration;
  desiredTrackingState = "RUNNING";
  const driverId = resolveBridgeDriverId();
  const previousMissionId = state.missionId;
  const isMissionSwitch =
    previousMissionId != null && previousMissionId !== missionId;

  // Mise à jour mémoire immédiate (évite race avec ticks concurrents).
  state.missionId = missionId;
  state.missionStatus = status;
  state.missionScheduling = scheduling ?? null;
  state.networkProfile = "normal";
  state.profileSinceMs = Date.now();
  state.staleFallbackBlockedUntilMs = 0;
  state.staleFallbackTimeouts = 0;
  state.lastStaleFallbackAttemptMs = null;
  state.lastHttpFallbackTrackingEventId = null;
  resetPermissionState();
  void hideMissionBarAndroid();
  ensureNativeTrackingAppStateListener();

  void (async () => {
    try {
      // Switch A→B : lease switching avant mutation contexte/owner.
      if (isMissionSwitch) {
        const currentLease = await readTrackingContextLease();
        await setTrackingContextLeaseSwitching({
          fromDriver: true,
          previousDriverActive:
            currentLease?.state === "driver_active" ? currentLease : null,
        });
      }

      const runtime = await startOrJoinTrackingRuntime({
        driverId,
        missionId,
        missionStatus: status,
      });
      if (operationGeneration !== lifecycleGeneration) return;

      const owner = toNativeTrackingOwner(runtime);
      // Toujours écrire contexte + owner ensemble (jamais context sans owner pendant switch).
      await setBackgroundTrackingMissionContext(
        missionId,
        status,
        "mission",
        scheduling,
        owner
      );
      await setTrackingContextLeaseDriverActive({
        contextId: `driver:${driverId}`,
        driverId,
        sessionGenerationId: runtime.identity.sessionGenerationId,
        trackingGenerationId: runtime.identity.trackingGenerationId,
        trackingIdentityId: runtime.identity.trackingIdentityId,
        missionId: runtime.missionContext.missionId,
        missionContextVersion: runtime.missionContext.missionContextVersion,
      });

      if (
        operationGeneration === lifecycleGeneration &&
        isFeatureEnabled("tracking_background_enabled")
      ) {
        await ensureNativeTrackingWhileForeground(
          missionId,
          status,
          { scheduling: scheduling ?? null, nativeOwner: owner },
          isMissionSwitch ? "mission_switch" : "mission_started"
        );
      }
    } catch {
      /* best-effort : ensureManagerState reprendra */
    }
  })();

  ensureManagerState();
  armNoLocationCallbackWatchdog(missionId, operationGeneration);
  notifyTrackingBridgeListeners();
}

export function updateDriverTrackingBridgeStatus(status: DriverMissionStatus) {
  state.missionStatus = status;
  const runtime = updateMissionContext(state.missionId, status);
  if (runtime) {
    const owner = toNativeTrackingOwner(runtime);
    void setTrackingContextLeaseDriverActive({
      contextId: `driver:${runtime.identity.driverId}`,
      driverId: runtime.identity.driverId,
      sessionGenerationId: runtime.identity.sessionGenerationId,
      trackingGenerationId: runtime.identity.trackingGenerationId,
      trackingIdentityId: runtime.identity.trackingIdentityId,
      missionId: runtime.missionContext.missionId,
      missionContextVersion: runtime.missionContext.missionContextVersion,
    });
    if (state.missionId != null) {
      void setBackgroundTrackingMissionContext(
        state.missionId,
        status,
        "mission",
        state.missionScheduling,
        owner
      );
    }
  }
  void hideMissionBarAndroid();
  if (!isEligible()) {
    void stopDriverTrackingBridge();
    return;
  }
  ensureManagerState();
  notifyTrackingBridgeListeners();
}

export async function stopDriverTrackingBridge(opts?: {
  expectedTrackingGenerationId?: string;
  reason?:
    | "explicit_logout"
    | "account_revoked"
    | "identity_changed"
    | "manual_stop"
    | "forced_recovery"
    | "runtime_replaced"
    | "context_left_driver";
  skipRegistryStop?: boolean;
}): Promise<void> {
  if (opts?.reason === "context_left_driver") {
    await hardStopDriverContextRuntime("context_left_driver");
    return;
  }

  const stopGeneration = lifecycleGeneration;
  const expectedGenId =
    opts?.expectedTrackingGenerationId ??
    captureActiveRuntime()?.identity.trackingGenerationId;

  if (
    opts?.expectedTrackingGenerationId &&
    captureActiveRuntime()?.identity.trackingGenerationId &&
    captureActiveRuntime()!.identity.trackingGenerationId !==
      opts.expectedTrackingGenerationId
  ) {
    return;
  }

  if (stopDriverTrackingInProgress) {
    return stopDriverTrackingInProgress;
  }

  stopDriverTrackingInProgress = (async () => {
    clearNoLocationCallbackWatchdog();
    void hideMissionBarAndroid();
    const missionIdForBar = state.missionId;
    if (missionIdForBar != null && isFeatureEnabled("driver_mission_bar_enabled")) {
      void stopMissionLiveActivity(missionIdForBar);
    }
    // Si un start plus récent a déjà pris le relais, abandonner sans muter l'état.
    if (stopGeneration !== lifecycleGeneration) {
      return;
    }
    if (expectedGenId && !opts?.skipRegistryStop) {
      clearActiveRuntimeIfGeneration(expectedGenId);
    }
    state.lastBackendAckLatencyMs = null;
    state.networkProfile = "normal";
    state.profileSinceMs = Date.now();
    state.staleFallbackBlockedUntilMs = 0;
    state.staleFallbackTimeouts = 0;
    state.lastStaleFallbackAttemptMs = null;
    state.lastHttpFallbackTrackingEventId = null;
    resetPermissionState();

    await stopMissionTrackingBridge(stopGeneration);

    if (stopGeneration !== lifecycleGeneration) {
      return;
    }

    if (resolveBridgeEligibility().trackingEligible) {
      await ensurePresenceTrackingState();
    } else {
      await stopTrackingRuntime();
    }
  })();

  try {
    await stopDriverTrackingInProgress;
  } finally {
    stopDriverTrackingInProgress = null;
  }
}

export type DriverPresenceContext = {
  available: boolean | null;
  windowOpen: boolean;
};

/**
 * Met à jour les signaux présence (disponibilité), puis
 * réconcilie via le resolver central. SoT = driverAvailabilityBridge.
 * `available=null` = UNKNOWN : ne pas forcer hors service.
 */
export function setDriverTrackingPresenceContext(ctx: DriverPresenceContext) {
  const available = ctx.available;
  setDriverAvailabilityActive(available);
  const knownAvailable = available === true;
  const windowOpen = true;
  if (
    state.driverAvailable === knownAvailable &&
    state.presenceWindowOpen === windowOpen
  ) {
    return;
  }
  state.driverAvailable = knownAvailable;
  state.presenceWindowOpen = windowOpen;
  emitDriverTelemetry("tracking.presence_context.updated", {
    source: "driver.tracking.bridge",
    driver_available: knownAvailable,
    availability_pending: available == null,
    presence_window_open: windowOpen,
    mission_id: state.missionId,
  });
  ensureManagerState();
  notifyTrackingBridgeListeners();
}

/**
 * @deprecated Préférer `setDriverTrackingPresenceContext`.
 * Conserve un mapping approximatif pour d’éventuels call sites hérités.
 */
export function setDriverTrackingPresenceWindow(active: boolean) {
  setDriverTrackingPresenceContext({
    available: active,
    windowOpen: active,
  });
}

export function getDriverTrackingPresenceWindowActive(): boolean {
  return state.presenceWindowOpen;
}

export function getDriverTrackingPresenceContext(): DriverPresenceContext {
  return {
    available: getDriverAvailabilityActive(),
    windowOpen: state.presenceWindowOpen,
  };
}

/** Re-applique le pipeline natif après acceptation disclosure présence. */
export function refreshDriverTrackingBridgeState(): void {
  void import("./hydrateTrackingPermissionsReady").then((m) =>
    m.hydrateTrackingPermissionsReady().finally(() => {
      ensureManagerState();
      notifyTrackingBridgeListeners();
    })
  );
}

export function getDriverTrackingBridgeSnapshot() {
  return buildTrackingBridgeSnapshot();
}

/** JZ-R1 — âge session tracking (ms epoch) pour durable_ack jamais observé. */
export function getDriverTrackingStartedAtMs(): number | null {
  return state.trackingStartedAtMs;
}

export function getDriverLastKnownPosition(): DriverTrackingPosition | null {
  return readDriverLastKnownPosition();
}

export function subscribeDriverTrackingBridge(
  listener: DriverTrackingBridgeListener
): () => void {
  trackingBridgeListeners.add(listener);
  listener(buildTrackingBridgeSnapshot());
  return () => {
    trackingBridgeListeners.delete(listener);
  };
}

export function disposeDriverTrackingBridge() {
  void stopDriverTrackingBridge();
  trackingManager.dispose();
}
