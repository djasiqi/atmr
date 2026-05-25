import * as Sentry from "@sentry/react-native";
import { AppStateStatus, Platform } from "react-native";
import * as Location from "expo-location";
import { sendDriverLocation } from "../api/driverHttp";
import { DriverMissionStatus } from "../types";
import { isTrackingActiveStatus } from "../domain/status";
import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";
import { evaluateConnectivityPolicy } from "../../../core/network/connectivityPolicy";
import { getNetworkSnapshot } from "../../../core/network/networkState";
import { realtimeManager } from "../../../core/realtime/realtimeManager";
import { TrackingManager } from "../../../core/tracking/trackingManager";
import { resolveTrackingCadence, TrackingNetworkProfile } from "../../../core/tracking/cadenceResolver";
import { driverTrackingQueue, DriverTrackingMode } from "./driverTrackingQueue";
import { hideMissionBarAndroid, showMissionBarAndroid } from "../missionBarAndroid";
import {
  startMissionLiveActivity,
  stopMissionLiveActivity,
  updateMissionLiveActivity,
} from "../missionBarIOS";
import {
  setBackgroundTrackingMissionContext,
  startBackgroundLocationTaskIfEligible,
  stopBackgroundLocationTask,
} from "./backgroundLocationTask";
import { canUseBackgroundLocation } from "./backgroundRuntimeCompat";
import { formatTrackingSendError } from "./driverTrackingSendErrorFormat";

const FOREGROUND_INTERVAL_MS = Number(process.env.EXPO_PUBLIC_DRIVER_GPS_FOREGROUND_INTERVAL_MS ?? "8000");
const BACKGROUND_INTERVAL_MS = Number(process.env.EXPO_PUBLIC_DRIVER_GPS_BACKGROUND_INTERVAL_MS ?? "20000");
const MAX_BACKOFF_MS = 60_000;
/** Dernière exception du tick tracking (pour logs __DEV__ / télémétrie, sans PII). */
let lastTrackingTickFailure: unknown = null;

const WATCH_STALE_MS = 25_000;
const WATCH_DISTANCE_METERS = 10;
const POSITION_TIMEOUT_MS = Number(process.env.EXPO_PUBLIC_DRIVER_POSITION_TIMEOUT_MS ?? "7000");
const STALE_FALLBACK_COOLDOWN_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_STALE_FALLBACK_COOLDOWN_MS ?? "20000"
);
const STALE_FALLBACK_BREAKER_MS = Number(
  process.env.EXPO_PUBLIC_DRIVER_STALE_FALLBACK_BREAKER_MS ?? "60000"
);
let permissionRequestInFlight: Promise<boolean> | null = null;

type TrackingBridgeState = {
  missionId: number | null;
  missionStatus: DriverMissionStatus | null;
  /**
   * True quand on track « par fenêtre horaire » (07h–19h) sans mission active.
   * Permet d'émettre des points GPS de présence pour les opérations même quand
   * aucune mission n'est en cours.
   */
  presenceWindowActive: boolean;
  lastSentAt: string | null;
  lastAckAt: string | null;
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
  lastAckAt: string | null;
  queueDepth: number;
  flushPathUsed: "http_fallback" | "socket_batch" | null;
  networkProfile: TrackingNetworkProfile;
  lastWatchAt: string | null;
  lastPosition: DriverTrackingPosition | null;
  lastAttemptAt: string | null;
  consecutiveFailures: number;
  backoffUntilMs: number;
};

type DriverTrackingBridgeListener = (snapshot: DriverTrackingBridgeSnapshot) => void;

const state: TrackingBridgeState = {
  missionId: null,
  missionStatus: null,
  presenceWindowActive: false,
  lastSentAt: null,
  lastAckAt: null,
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
};

const trackingBridgeListeners = new Set<DriverTrackingBridgeListener>();

function readDriverLastKnownPosition(): DriverTrackingPosition | null {
  if (!state.lastWatchedPosition || state.lastWatchAtMs == null) return null;
  if (Date.now() - state.lastWatchAtMs > WATCH_STALE_MS) return null;
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
    lastAckAt: state.lastAckAt,
    queueDepth: state.queueDepth,
    flushPathUsed: state.flushPathUsed,
    networkProfile: state.networkProfile,
    lastWatchAt: state.lastWatchAtMs ? new Date(state.lastWatchAtMs).toISOString() : null,
    lastPosition: readDriverLastKnownPosition(),
    lastAttemptAt: managerSnapshot.lastAttemptAt,
    consecutiveFailures: managerSnapshot.consecutiveFailures,
    backoffUntilMs: managerSnapshot.backoffUntilMs,
  };
}

function notifyTrackingBridgeListeners() {
  const snapshot = buildTrackingBridgeSnapshot();
  trackingBridgeListeners.forEach((listener) => {
    listener(snapshot);
  });
}

function hasActiveMission(): boolean {
  return state.missionId !== null && isTrackingActiveStatus(state.missionStatus);
}

function isEligible() {
  return hasActiveMission() || state.presenceWindowActive;
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
    if (isFeatureEnabled("tracking_background_enabled") && canUseBackgroundLocation()) {
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
      foregroundIntervalMs: FOREGROUND_INTERVAL_MS,
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
  if (!isFeatureEnabled("tracking_safe_stale_fallback_enabled")) {
    return Location.getCurrentPositionAsync({
      accuracy: Location.Accuracy.Balanced,
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
      accuracy: Location.Accuracy.Balanced,
      mayShowUserSettingsDialog: true,
    }).catch(() => null),
    timeoutPromise,
  ]);
  if (currentPosition) {
    state.staleFallbackTimeouts = 0;
    return currentPosition;
  }
  state.staleFallbackTimeouts += 1;
  state.staleFallbackBlockedUntilMs = Date.now() + STALE_FALLBACK_BREAKER_MS;
  emitDriverTelemetry("tracking.stale_fallback.timeout", {
    source: "driver.tracking.bridge",
    mission_id: state.missionId,
    app_state: appState,
    stale_fallback_timeout_total: state.staleFallbackTimeouts,
    stale_fallback_blocked_until_ms: state.staleFallbackBlockedUntilMs,
  });
  return Location.getLastKnownPositionAsync({
    maxAge: WATCH_STALE_MS,
    requiredAccuracy: 100,
  }).catch(() => null);
}

async function flushPoint(appState: AppStateStatus) {
  if (!isEligible()) return;
  const granted = await ensurePermission(appState);
  if (!granted) return;
  const position = await resolvePositionFromWatchOrFallback(appState);
  if (!position) return;
  const nowIso = new Date().toISOString();
  const mode = resolveTrackingMode(appState);
  const cadence = getCadenceForTick(appState, mode);
  await driverTrackingQueue.enqueue({
    missionId: state.missionId,
    appState,
    locationMode: mode,
    payload: {
      latitude: position.coords.latitude,
      longitude: position.coords.longitude,
      accuracy: position.coords.accuracy ?? undefined,
      heading: position.coords.heading ?? undefined,
      speed: position.coords.speed ?? undefined,
      missionId: state.missionId,
      isBackground: appState !== "active",
      timestamp: nowIso,
      locationMode: mode,
    },
  });
  const flushResult = await driverTrackingQueue.flush({
    ackStaleMs: cadence.ackStaleMs,
    networkProfile: cadence.networkProfile,
    forceHttpFallback: appState !== "active",
  });
  state.queueDepth = flushResult.queueDepth;
  state.flushPathUsed = flushResult.flushPathUsed;
  if (flushResult.backendAcked > 0 && flushResult.lastBackendAckAt) {
    state.lastAckAt = new Date(flushResult.lastBackendAckAt).toISOString();
    state.lastBackendAckLatencyMs =
      flushResult.lastBackendAckAt - Date.parse(nowIso);
  }

  const lastAckMs = state.lastAckAt ? Date.parse(state.lastAckAt) : null;
  const ackIsStale = lastAckMs !== null && Date.now() - lastAckMs > cadence.ackStaleMs;
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

  const shouldFallback =
    isFeatureEnabled("tracking_http_fallback_enabled") &&
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
    await sendDriverLocation({
      latitude: position.coords.latitude,
      longitude: position.coords.longitude,
      accuracy: position.coords.accuracy ?? undefined,
      heading: position.coords.heading ?? undefined,
      speed: position.coords.speed ?? undefined,
      missionId: state.missionId,
      isBackground: appState !== "active",
      timestamp: nowIso,
      locationMode: mode,
      trackingEventId: fallbackTrackingEventId,
    });
    state.lastAckAt = nowIso;
    state.lastHttpFallbackTrackingEventId = fallbackTrackingEventId;
    state.flushPathUsed = "http_fallback";
  }
  state.lastSentAt = nowIso;
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

async function resolvePositionFromWatchOrFallback(appState: AppStateStatus) {
  const hasFreshWatch =
    state.lastWatchedPosition !== null &&
    state.lastWatchAtMs !== null &&
    Date.now() - state.lastWatchAtMs < WATCH_STALE_MS;
  if (hasFreshWatch) return state.lastWatchedPosition;
  return getCurrentPositionWithTimeout(appState);
}

function resolveTrackingMode(appState: AppStateStatus): DriverTrackingMode {
  /* Pas de mission ET fenêtre 07h–19h ouverte = présence pure. */
  if (!hasActiveMission() && state.presenceWindowActive) {
    return "availability_presence";
  }
  if (state.missionStatus === "ASSIGNED" && isFeatureEnabled("tracking_presence_mode_enabled")) {
    return "availability_presence";
  }
  if (appState !== "active" && isFeatureEnabled("tracking_background_enabled")) {
    return "availability_presence";
  }
  return "mission_live";
}

async function ensureLocationWatch() {
  if (Platform.OS === "web") {
    /* Sur web, expo-location brise le cleanup (LocationEventEmitter.removeSubscription n'existe pas). */
    return;
  }
  if (!isEligible() || state.watchSubscription) return;
  const appState = trackingManager.getSnapshot().appState;
  const granted = await ensurePermission(appState);
  if (!granted) return;
  try {
    state.watchSubscription = await Location.watchPositionAsync(
      {
        accuracy: Location.Accuracy.Balanced,
        distanceInterval: WATCH_DISTANCE_METERS,
        timeInterval: appState === "active" ? FOREGROUND_INTERVAL_MS : BACKGROUND_INTERVAL_MS,
        mayShowUserSettingsDialog: true,
      },
      (position) => {
        state.lastWatchedPosition = position;
        state.lastWatchAtMs = Date.now();
        notifyTrackingBridgeListeners();
      }
    );
    emitDriverTelemetry("tracking.watch.started", {
      source: "driver.tracking.bridge",
      mission_id: state.missionId,
      app_state: appState,
      distance_m: WATCH_DISTANCE_METERS,
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
  if (result.backendAcked > 0 && result.lastBackendAckAt) {
    state.lastAckAt = new Date(result.lastBackendAckAt).toISOString();
  }
  if (result.flushPathUsed) {
    state.flushPathUsed = result.flushPathUsed;
  }
  notifyTrackingBridgeListeners();
}

export async function getDriverTrackingQueueSnapshot() {
  return driverTrackingQueue.getSnapshot();
}

async function updateQueueDepth() {
  const snapshot = await driverTrackingQueue.getSnapshot();
  state.queueDepth = snapshot.queueDepth;
  notifyTrackingBridgeListeners();
}

async function sendLegacyPoint(appState: AppStateStatus, nowIso: string) {
  const granted = await ensurePermission(appState);
  if (!granted) return;
  const position = await getCurrentPositionWithTimeout(appState);
  if (!position) return;
  if (state.missionId === null && !state.presenceWindowActive) return;
  await sendDriverLocation({
    latitude: position.coords.latitude,
    longitude: position.coords.longitude,
    accuracy: position.coords.accuracy ?? undefined,
    heading: position.coords.heading ?? undefined,
    speed: position.coords.speed ?? undefined,
    missionId: state.missionId,
    isBackground: appState !== "active",
    timestamp: nowIso,
    locationMode: resolveTrackingMode(appState),
  });
  state.lastAckAt = nowIso;
}

const trackingManager = new TrackingManager({
  foregroundIntervalMs: FOREGROUND_INTERVAL_MS,
  backgroundIntervalMs: BACKGROUND_INTERVAL_MS,
  maxBackoffMs: MAX_BACKOFF_MS,
  onTick: async ({ appState }) => {
    if (!isEligible()) return "skipped";
    try {
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
      presence_window_active: state.presenceWindowActive,
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

function ensureManagerState() {
  if (!isEligible()) {
    void stopBackgroundLocationTask("ineligible_tracking_state");
    trackingManager.stop();
    stopLocationWatch();
    void setBackgroundTrackingMissionContext(null, null);
    return;
  }
  if (state.missionId != null) {
    void setBackgroundTrackingMissionContext(state.missionId, state.missionStatus, "mission");
    if (isFeatureEnabled("tracking_background_enabled")) {
      void startBackgroundLocationTaskIfEligible(state.missionId, state.missionStatus);
    }
  } else if (state.presenceWindowActive) {
    void setBackgroundTrackingMissionContext(null, null, "presence_window");
    if (isFeatureEnabled("tracking_background_enabled")) {
      void startBackgroundLocationTaskIfEligible(null, null, { presenceWindow: true });
    }
  }
  void ensureLocationWatch();
  void updateQueueDepth();
  const mode = resolveTrackingMode(trackingManager.getSnapshot().appState);
  const snapshot = trackingManager.getSnapshot();
  if (!snapshot.isRunning) {
    trackingManager.start(mode);
    return;
  }
  trackingManager.updateMode(mode);
}

export function startDriverTrackingBridge(missionId: number, status: DriverMissionStatus) {
  state.missionId = missionId;
  state.missionStatus = status;
  state.networkProfile = "normal";
  state.profileSinceMs = Date.now();
  state.staleFallbackBlockedUntilMs = 0;
  state.staleFallbackTimeouts = 0;
  state.lastStaleFallbackAttemptMs = null;
  state.lastHttpFallbackTrackingEventId = null;
  resetPermissionState();
  void showMissionBarAndroid(missionId, status);
  void startMissionLiveActivity({ missionId, status });
  void setBackgroundTrackingMissionContext(missionId, status);
  if (isFeatureEnabled("tracking_background_enabled")) {
    void startBackgroundLocationTaskIfEligible(missionId, status);
  }
  ensureManagerState();
  notifyTrackingBridgeListeners();
}

export function updateDriverTrackingBridgeStatus(status: DriverMissionStatus) {
  state.missionStatus = status;
  if (state.missionId != null) {
    void showMissionBarAndroid(state.missionId, status);
    void updateMissionLiveActivity({ missionId: state.missionId, status });
  }
  if (!isEligible()) {
    stopDriverTrackingBridge();
    return;
  }
  ensureManagerState();
  notifyTrackingBridgeListeners();
}

export function stopDriverTrackingBridge() {
  void hideMissionBarAndroid();
  if (state.missionId != null) {
    void stopMissionLiveActivity(state.missionId);
  }
  state.missionId = null;
  state.missionStatus = null;
  state.lastBackendAckLatencyMs = null;
  state.networkProfile = "normal";
  state.profileSinceMs = Date.now();
  state.staleFallbackBlockedUntilMs = 0;
  state.staleFallbackTimeouts = 0;
  state.lastStaleFallbackAttemptMs = null;
  state.lastHttpFallbackTrackingEventId = null;
  resetPermissionState();
  if (state.presenceWindowActive) {
    /* Si la window 07h–19h est ouverte on garde le tracking en mode présence
     * même après la fin d'une mission. Sinon on coupe complètement. */
    ensureManagerState();
    notifyTrackingBridgeListeners();
    return;
  }
  void setBackgroundTrackingMissionContext(null, null);
  void stopBackgroundLocationTask("tracking_bridge_stopped");
  stopLocationWatch();
  trackingManager.stop();
  notifyTrackingBridgeListeners();
}

/**
 * Active ou désactive le mode présence par fenêtre horaire (07h–19h).
 * Quand actif sans mission, l'app continue d'envoyer des points GPS de
 * présence avec `missionId = null` et `locationMode = availability_presence`.
 * Quand inactif, le tracking ne tourne que si une mission est éligible.
 */
export function setDriverTrackingPresenceWindow(active: boolean) {
  if (state.presenceWindowActive === active) return;
  state.presenceWindowActive = active;
  emitDriverTelemetry("tracking.presence_window.toggled", {
    source: "driver.tracking.bridge",
    presence_window_active: active,
    mission_id: state.missionId,
  });
  if (!active && state.missionId === null) {
    /* Window vient de se fermer et pas de mission active : on coupe tout. */
    void setBackgroundTrackingMissionContext(null, null);
    void stopBackgroundLocationTask("presence_window_closed");
    stopLocationWatch();
    trackingManager.stop();
    notifyTrackingBridgeListeners();
    return;
  }
  ensureManagerState();
  notifyTrackingBridgeListeners();
}

export function getDriverTrackingPresenceWindowActive(): boolean {
  return state.presenceWindowActive;
}

export function getDriverTrackingBridgeSnapshot() {
  return buildTrackingBridgeSnapshot();
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
  stopDriverTrackingBridge();
  trackingManager.dispose();
}
