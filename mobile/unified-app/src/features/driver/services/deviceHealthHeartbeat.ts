/**
 * Heartbeat de santé tracking côté driver.
 *
 * Émet périodiquement (60s) un POST /drivers/me/device-status pour indiquer au
 * backend que l'app est vivante même quand le FGS est tué silencieusement
 * (notamment Samsung One UI / Doze qui coupe le BG GPS sans notifier l'app).
 *
 * Le heartbeat est purement observabilité : aucune erreur n'est propagée, les
 * échecs réseau sont silencieusement comptés via télémétrie.
 *
 * Imports internes (`./backgroundLocationTask`, `./driverTrackingBridge`,
 * `./batteryOptimization`) sont lazy-required pour éviter les cycles
 * d'import — `backgroundLocationTask.ts` et `driverTrackingBridge.ts`
 * appellent `triggerDeviceHealthNow` depuis ce module.
 */
import * as Battery from "expo-battery";
import * as Location from "expo-location";
import { AppState, AppStateStatus, Platform } from "react-native";

import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { getTrackingRuntimeSnapshot } from "./trackingRuntime";

export type DevicePermissionStatus = "granted" | "denied" | "undetermined";

/** Diagnostic Lot 1 — autorisation de précision iOS (inférée, lecture seule). */
export type IosAccuracyAuthorization = "full" | "reduced" | "unknown";

/** Diagnostic Lot 1 — statut Background App Refresh iOS. */
export type IosBackgroundRefreshStatus =
  | "available"
  | "denied"
  | "restricted"
  | "unknown";

export type TrackingHealthState =
  | "starting"
  | "healthy"
  | "capture_failed"
  | "queue_blocked"
  | "auth_blocked"
  | "offline"
  | "stopped";

export type DeviceHealthPayload = {
  kind: "tracking_health";
  manufacturer?: string | null;
  model?: string | null;
  platform?: string | null;
  fgs_running: boolean;
  /** @deprecated Préférer tracking_state — conservé pour compat backend. */
  tracking_active?: boolean;
  /** État calculé (pas un alias de fgs_running). */
  tracking_state?: TrackingHealthState;
  fg_permission: DevicePermissionStatus;
  bg_permission: DevicePermissionStatus;
  location_permission?: DevicePermissionStatus | "always" | "when_in_use" | "denied" | "undetermined";
  notifications_enabled?: boolean;
  gps_provider_enabled: boolean;
  battery_optimized: boolean;
  battery_level: number | null;
  is_charging: boolean | null;
  last_fix_age_seconds: number | null;
  fix_success_rate_last_5min: number | null;
  constraint_reason: string | null;
  app_state?: AppStateStatus | string | null;
  native_start_phase?: string | null;
  native_start_error?: string | null;
  native_task_defined?: boolean | null;
  native_started_before?: boolean | null;
  native_started_after?: boolean | null;
  // --- Diagnostic Lot 1 (observabilité uniquement, aucun changement de comportement) ---
  /** Version applicative (native) — ex. "1.42.0". */
  app_version?: string | null;
  /** Version OS — ex. "17.4" (iOS) / "14" (Android). */
  os_version?: string | null;
  /** Âge (s) du dernier callback du task natif background-location — distinct du watch JS. */
  native_last_fix_age_seconds?: number | null;
  /** Le task natif de localisation tourne réellement (hasStartedLocationUpdatesAsync). */
  native_task_running?: boolean | null;
  /** iOS : autorisation de précision inférée (full/reduced) — null hors iOS. */
  ios_accuracy_authorization?: IosAccuracyAuthorization | null;
  /** iOS : mode économie d'énergie actif — null hors iOS. */
  ios_low_power_mode?: boolean | null;
  /** iOS : statut Background App Refresh — null hors iOS. */
  ios_background_refresh_status?: IosBackgroundRefreshStatus | null;
  /** Métriques session (sans JWT). */
  tracking_session_id?: string | null;
  sequence?: number | null;
  queue_depth?: number | null;
};

export type DeviceHealthRequestPayload = DeviceHealthPayload & {
  trigger_reason?: string;
};

export type StartDeviceHealthHeartbeatOptions = {
  /** Override de l'intervalle (par défaut 60 000 ms). */
  intervalMs?: number;
};

const DEFAULT_HEARTBEAT_INTERVAL_MS = 120_000;
const MISSION_HEARTBEAT_INTERVAL_MS = 60_000;
const FIX_STALE_THRESHOLD_SECONDS = 300;
const DEVICE_HEALTH_ENDPOINT = "/driver/me/device-health";
const LEGACY_DEVICE_STATUS_ENDPOINT = "/driver/me/device-status";

let activeStop: (() => void) | null = null;
let lastSentAtMs = 0;
let lastSentConstraintReason: string | null = null;

function toPermissionStatus(input: { status?: string } | null | undefined): DevicePermissionStatus {
  const status = input?.status;
  if (status === "granted") return "granted";
  if (status === "denied") return "denied";
  return "undetermined";
}

async function readNotificationsEnabled(): Promise<boolean | undefined> {
  if (Platform.OS === "web") return undefined;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const Notifications = require("expo-notifications") as {
      getPermissionsAsync?: () => Promise<{ granted?: boolean; status?: string }>;
    };
    if (typeof Notifications.getPermissionsAsync !== "function") return undefined;
    const perm = await Notifications.getPermissionsAsync();
    return Boolean(perm?.granted || perm?.status === "granted");
  } catch {
    return undefined;
  }
}

function readDeviceIdentity(): { manufacturer: string | null; model: string | null } {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const Device = require("expo-device") as {
      manufacturer?: string | null;
      modelName?: string | null;
    };
    return {
      manufacturer: Device?.manufacturer ? String(Device.manufacturer) : null,
      model: Device?.modelName ? String(Device.modelName) : null,
    };
  } catch {
    return { manufacturer: null, model: null };
  }
}

function readAppOsVersion(): { appVersion: string | null; osVersion: string | null } {
  let appVersion: string | null = null;
  let osVersion: string | null = null;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const Application = require("expo-application") as {
      nativeApplicationVersion?: string | null;
    };
    appVersion = Application?.nativeApplicationVersion
      ? String(Application.nativeApplicationVersion)
      : null;
  } catch {
    appVersion = null;
  }
  if (!appVersion) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      const Constants = require("expo-constants").default as {
        expoConfig?: { version?: string | null } | null;
      };
      appVersion = Constants?.expoConfig?.version
        ? String(Constants.expoConfig.version)
        : null;
    } catch {
      /* noop */
    }
  }
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const Device = require("expo-device") as { osVersion?: string | null };
    osVersion = Device?.osVersion ? String(Device.osVersion) : null;
  } catch {
    osVersion = null;
  }
  return { appVersion, osVersion };
}

async function readIosLowPowerMode(): Promise<boolean | null> {
  if (Platform.OS !== "ios") return null;
  try {
    if (typeof Battery.isLowPowerModeEnabledAsync !== "function") return null;
    return await Battery.isLowPowerModeEnabledAsync();
  } catch {
    return null;
  }
}

async function readIosBackgroundRefreshStatus(): Promise<IosBackgroundRefreshStatus | null> {
  if (Platform.OS !== "ios") return null;
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const BackgroundFetch = require("expo-background-fetch") as {
      getStatusAsync?: () => Promise<number | null>;
      BackgroundFetchStatus?: { Restricted?: number; Denied?: number; Available?: number };
    };
    if (typeof BackgroundFetch.getStatusAsync !== "function") return "unknown";
    const status = await BackgroundFetch.getStatusAsync();
    const enums = BackgroundFetch.BackgroundFetchStatus ?? {};
    if (status === enums.Available) return "available";
    if (status === enums.Denied) return "denied";
    if (status === enums.Restricted) return "restricted";
    return "unknown";
  } catch {
    return "unknown";
  }
}

/** Inférence lecture seule de la précision iOS via la dernière position connue.
 *  Reduced accuracy iOS => accuracy horizontale très large (≈ km). Pas de getter
 *  natif exposé par expo-location ; heuristique conservatrice à seuils. */
async function readIosAccuracyAuthorization(): Promise<IosAccuracyAuthorization | null> {
  if (Platform.OS !== "ios") return null;
  try {
    if (typeof Location.getLastKnownPositionAsync !== "function") return "unknown";
    const pos = await Location.getLastKnownPositionAsync();
    const acc = pos?.coords?.accuracy;
    if (typeof acc !== "number" || !Number.isFinite(acc)) return "unknown";
    if (acc <= 200) return "full";
    if (acc >= 1000) return "reduced";
    return "unknown";
  } catch {
    return "unknown";
  }
}

function computeNativeLastFixAgeSeconds(): number | null {
  const ts = getTrackingRuntimeSnapshot().lastTaskInvokedAt;
  if (typeof ts !== "number" || !Number.isFinite(ts)) return null;
  const ageMs = Date.now() - ts;
  if (ageMs < 0) return 0;
  return Math.round(ageMs / 1000);
}

async function readForegroundPermission(): Promise<DevicePermissionStatus> {
  if (typeof Location.getForegroundPermissionsAsync !== "function") return "undetermined";
  try {
    const fg = await Location.getForegroundPermissionsAsync();
    return toPermissionStatus(fg);
  } catch {
    return "undetermined";
  }
}

async function readBackgroundPermission(): Promise<DevicePermissionStatus> {
  if (typeof Location.getBackgroundPermissionsAsync !== "function") return "undetermined";
  try {
    const bg = await Location.getBackgroundPermissionsAsync();
    return toPermissionStatus(bg);
  } catch {
    return "undetermined";
  }
}

async function readGpsProviderEnabled(): Promise<boolean> {
  if (typeof Location.hasServicesEnabledAsync !== "function") return false;
  try {
    return await Location.hasServicesEnabledAsync();
  } catch {
    return false;
  }
}

async function readFgsRunning(): Promise<boolean> {
  try {
    /* Lazy require pour éviter le cycle deviceHealthHeartbeat <-> backgroundLocationTask. */
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./backgroundLocationTask") as typeof import("./backgroundLocationTask");
    if (typeof mod.getNativeTaskLifecycleStatus !== "function") return false;
    const lifecycle = await mod.getNativeTaskLifecycleStatus();
    return Boolean(lifecycle?.taskStarted);
  } catch {
    return false;
  }
}

async function readBatteryOptimized(): Promise<boolean> {
  if (Platform.OS !== "android") return false;
  try {
    /* Lazy require — le helper batteryOptimization peut ne pas encore être mergé
     * (chantier UX bannière en parallèle) ; on retombe alors sur false. */
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./batteryOptimization") as typeof import("./batteryOptimization");
    if (typeof mod.checkBatteryOptimizationStatus !== "function") return false;
    const status = await mod.checkBatteryOptimizationStatus();
    /* `isIgnoring=true` => app exemptée (allowlist Doze) ; donc PAS optimisée.
     * `isIgnoring=null` (indéterminable) => par sécurité on ne signale pas la
     * contrainte (false) pour éviter du faux-positif sur OEM exotiques. */
    return status?.isIgnoring === false;
  } catch {
    return false;
  }
}

async function readBatteryLevel(): Promise<number | null> {
  try {
    if (typeof Battery.getBatteryLevelAsync !== "function") return null;
    const level = await Battery.getBatteryLevelAsync();
    return typeof level === "number" && Number.isFinite(level) ? level : null;
  } catch {
    return null;
  }
}

async function readBatteryCharging(): Promise<boolean | null> {
  try {
    if (typeof Battery.getBatteryStateAsync !== "function") return null;
    const state = await Battery.getBatteryStateAsync();
    /* expo-battery BatteryState : UNKNOWN=0, UNPLUGGED=1, CHARGING=2, FULL=3. */
    if (state === 2 || state === 3) return true;
    if (state === 1) return false;
    return null;
  } catch {
    return null;
  }
}

type BridgeSnapshotLike = {
  missionId: number | null;
  lastWatchAt: string | null;
  queueDepth?: number | null;
};

function readBridgeSnapshot(): BridgeSnapshotLike | null {
  try {
    /* Lazy require — driverTrackingBridge importe triggerDeviceHealthNow. */
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./driverTrackingBridge") as typeof import("./driverTrackingBridge");
    if (typeof mod.getDriverTrackingBridgeSnapshot !== "function") return null;
    const snapshot = mod.getDriverTrackingBridgeSnapshot();
    return {
      missionId: snapshot.missionId ?? null,
      lastWatchAt: snapshot.lastWatchAt ?? null,
      queueDepth: (snapshot as { queueDepth?: number | null }).queueDepth ?? null,
    };
  } catch {
    return null;
  }
}

function readPresenceWindowActive(): boolean {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("./driverTrackingBridge") as typeof import("./driverTrackingBridge");
    if (typeof mod.getDriverTrackingPresenceWindowActive !== "function") return false;
    return Boolean(mod.getDriverTrackingPresenceWindowActive());
  } catch {
    return false;
  }
}

function computeLastFixAgeSeconds(snapshot: BridgeSnapshotLike | null): number | null {
  if (!snapshot?.lastWatchAt) return null;
  const ts = Date.parse(snapshot.lastWatchAt);
  if (!Number.isFinite(ts)) return null;
  const ageMs = Date.now() - ts;
  if (ageMs < 0) return 0;
  return Math.round(ageMs / 1000);
}

function expectFgsRunning(snapshot: BridgeSnapshotLike | null): boolean {
  /* On n'attend le FGS que quand on track activement : mission en cours ou
   * fenêtre de présence ouverte. Sinon l'absence de FGS est normale. */
  if (snapshot && snapshot.missionId !== null) return true;
  return readPresenceWindowActive();
}

function resolveConstraintReason(input: {
  fgPermission: DevicePermissionStatus;
  bgPermission: DevicePermissionStatus;
  gpsProviderEnabled: boolean;
  batteryOptimized: boolean;
  fgsRunning: boolean;
  fgsExpected: boolean;
  lastFixAgeSeconds: number | null;
}): string | null {
  if (input.fgPermission === "denied") return "permission_fg_denied";
  if (input.bgPermission === "denied") return "permission_bg_denied";
  if (!input.gpsProviderEnabled) return "gps_provider_disabled";
  if (Platform.OS === "android" && input.batteryOptimized) return "battery_optimized";
  if (input.fgsExpected && !input.fgsRunning) return "fgs_not_running";
  if (
    input.lastFixAgeSeconds !== null &&
    input.lastFixAgeSeconds > FIX_STALE_THRESHOLD_SECONDS
  ) {
    return "fix_stale";
  }
  return null;
}

export async function collectDeviceHealth(): Promise<DeviceHealthPayload> {
  const [
    fgPermission,
    bgPermission,
    gpsProviderEnabled,
    fgsRunning,
    batteryOptimized,
    batteryLevel,
    isCharging,
    notificationsEnabled,
    iosLowPowerMode,
    iosBackgroundRefreshStatus,
    iosAccuracyAuthorization,
  ] = await Promise.all([
    readForegroundPermission(),
    readBackgroundPermission(),
    readGpsProviderEnabled(),
    readFgsRunning(),
    readBatteryOptimized(),
    readBatteryLevel(),
    readBatteryCharging(),
    readNotificationsEnabled(),
    readIosLowPowerMode(),
    readIosBackgroundRefreshStatus(),
    readIosAccuracyAuthorization(),
  ]);

  const snapshot = readBridgeSnapshot();
  const lastFixAgeSeconds = computeLastFixAgeSeconds(snapshot);
  const nativeLastFixAgeSeconds = computeNativeLastFixAgeSeconds();
  const fgsExpected = expectFgsRunning(snapshot);
  const identity = readDeviceIdentity();
  const { appVersion, osVersion } = readAppOsVersion();

  const trackingState = resolveTrackingHealthState({
    fgsRunning,
    fgsExpected,
    lastFixAgeSeconds,
    nativeLastFixAgeSeconds,
    gpsProviderEnabled,
    missionId: snapshot?.missionId ?? null,
    queueDepth: snapshot?.queueDepth ?? null,
  });
  const trackingActive =
    trackingState === "healthy" || trackingState === "starting";

  const constraintReason = resolveConstraintReason({
    fgPermission,
    bgPermission,
    gpsProviderEnabled,
    batteryOptimized,
    fgsRunning,
    fgsExpected,
    lastFixAgeSeconds,
  });

  // Fix NULL pendant mission attendue = capture_failed (pas un tracking « sain »).
  const effectiveConstraint =
    constraintReason
    ?? (trackingState === "capture_failed" ? "no_location_fix" : null);

  const locationPermission =
    bgPermission === "granted"
      ? "always"
      : fgPermission === "granted"
        ? "when_in_use"
        : fgPermission;

  const trackingRuntime = getTrackingRuntimeSnapshot();
  const nativeDiag = trackingRuntime.nativeStartDiagnostics;
  const nativeStartError =
    nativeDiag.native_start_error ?? trackingRuntime.lastNativeStartError ?? null;

  return {
    kind: "tracking_health",
    manufacturer: identity.manufacturer,
    model: identity.model,
    platform: Platform.OS,
    fgs_running: fgsRunning,
    tracking_active: trackingActive,
    tracking_state: trackingState,
    fg_permission: fgPermission,
    bg_permission: bgPermission,
    location_permission: locationPermission,
    notifications_enabled: notificationsEnabled,
    gps_provider_enabled: gpsProviderEnabled,
    battery_optimized: batteryOptimized,
    battery_level: batteryLevel,
    is_charging: isCharging,
    last_fix_age_seconds: lastFixAgeSeconds,
    fix_success_rate_last_5min: null,
    constraint_reason: effectiveConstraint,
    app_state: AppState.currentState,
    native_start_phase: nativeDiag.native_start_phase,
    native_start_error: nativeStartError,
    native_task_defined: nativeDiag.native_task_defined,
    native_started_before: nativeDiag.native_started_before,
    native_started_after: nativeDiag.native_started_after,
    app_version: appVersion,
    os_version: osVersion,
    native_last_fix_age_seconds: nativeLastFixAgeSeconds,
    native_task_running: fgsRunning,
    ios_accuracy_authorization: iosAccuracyAuthorization,
    ios_low_power_mode: iosLowPowerMode,
    ios_background_refresh_status: iosBackgroundRefreshStatus,
    queue_depth: snapshot?.queueDepth ?? null,
  };
}

function resolveTrackingHealthState(input: {
  fgsRunning: boolean;
  fgsExpected: boolean;
  lastFixAgeSeconds: number | null;
  nativeLastFixAgeSeconds: number | null;
  gpsProviderEnabled: boolean;
  missionId: number | null;
  queueDepth: number | null;
}): TrackingHealthState {
  if (!input.fgsExpected && input.missionId == null) {
    return "stopped";
  }
  if (!input.gpsProviderEnabled) {
    return "offline";
  }
  if (input.fgsExpected && !input.fgsRunning) {
    return "starting";
  }
  // FGS vivant mais aucun fix (NULL) pendant une mission attendue = capture_failed.
  if (
    input.missionId != null
    && input.fgsRunning
    && input.lastFixAgeSeconds == null
    && input.nativeLastFixAgeSeconds == null
  ) {
    return "capture_failed";
  }
  if (
    input.lastFixAgeSeconds != null
    && input.lastFixAgeSeconds > FIX_STALE_THRESHOLD_SECONDS
  ) {
    return "capture_failed";
  }
  if ((input.queueDepth ?? 0) > 50) {
    return "queue_blocked";
  }
  if (input.fgsRunning || (input.lastFixAgeSeconds != null && input.lastFixAgeSeconds < 120)) {
    return "healthy";
  }
  return input.fgsExpected ? "starting" : "stopped";
}

type ApiClientLike = {
  post: (url: string, body?: unknown) => Promise<unknown>;
};

function getApiClient(): ApiClientLike | null {
  try {
    /* Lazy require — évite de tirer expo-secure-store / Constants quand
     * deviceHealthHeartbeat est lazy-required par backgroundLocationTask
     * dans des contextes (tests, web) où apiClient n'est pas hydraté. */
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    const mod = require("../../../core/api/client") as { apiClient?: ApiClientLike };
    return mod.apiClient ?? null;
  } catch {
    return null;
  }
}

export async function sendDeviceHealth(payload: DeviceHealthRequestPayload): Promise<void> {
  const now = Date.now();
  const triggerReason = payload.trigger_reason ?? null;
  const criticalSignal =
    Boolean(payload.constraint_reason) ||
    (triggerReason !== null &&
      triggerReason !== "health_monitor_ok" &&
      !triggerReason.startsWith("health_monitor_ok"));
  const constraintChanged = payload.constraint_reason !== lastSentConstraintReason;
  const throttleMs =
    payload.tracking_state === "healthy" || payload.tracking_state === "starting" || payload.tracking_state === "capture_failed"
      ? MISSION_HEARTBEAT_INTERVAL_MS
      : DEFAULT_HEARTBEAT_INTERVAL_MS;
  // Mission active (fix NULL / capture_failed inclus) → throttle 60s.
  const intervalElapsed = now - lastSentAtMs >= throttleMs;

  if (!(intervalElapsed || constraintChanged || (criticalSignal && lastSentAtMs === 0))) {
    emitDriverTelemetry("tracking.device_health.send_skipped", {
      source: "driver.device_health",
      reason: "throttled",
      trigger_reason: triggerReason,
      age_ms: now - lastSentAtMs,
    });
    return;
  }

  const client = getApiClient();
  if (!client || typeof client.post !== "function") {
    emitDriverTelemetry("tracking.device_health.send_failed", {
      source: "driver.device_health",
      error: "api_client_unavailable",
      http_status: null,
      trigger_reason: payload.trigger_reason ?? null,
    });
    return;
  }
  try {
    try {
      await client.post(DEVICE_HEALTH_ENDPOINT, payload);
    } catch (error) {
      const httpStatus =
        (error as { response?: { status?: number } })?.response?.status ?? null;
      if (httpStatus === 404) {
        await client.post(LEGACY_DEVICE_STATUS_ENDPOINT, payload);
      } else {
        throw error;
      }
    }
    emitDriverTelemetry("tracking.device_health.sent", {
      source: "driver.device_health",
      kind: payload.kind,
      constraint_reason: payload.constraint_reason ?? null,
      fgs_running: payload.fgs_running,
      battery_optimized: payload.battery_optimized,
      trigger_reason: payload.trigger_reason ?? null,
    });
    lastSentAtMs = now;
    lastSentConstraintReason = payload.constraint_reason ?? null;
  } catch (error) {
    const httpStatus =
      (error as { response?: { status?: number } })?.response?.status ?? null;
    const message = error instanceof Error ? error.message : String(error);
    emitDriverTelemetry("tracking.device_health.send_failed", {
      source: "driver.device_health",
      error: message,
      http_status: httpStatus,
      trigger_reason: payload.trigger_reason ?? null,
    });
    if (typeof console !== "undefined" && typeof console.warn === "function") {
      console.warn("[device_health_send_failed]", { message, status: httpStatus });
    }
  }
}

async function tickHeartbeat(triggerReason?: string): Promise<void> {
  try {
    const base = await collectDeviceHealth();
    const payload: DeviceHealthRequestPayload = triggerReason
      ? { ...base, trigger_reason: triggerReason }
      : base;
    await sendDeviceHealth(payload);
  } catch {
    /* Defensive : collectDeviceHealth ne devrait pas throw, mais on ne veut
     * surtout pas laisser un tick faire crasher le ticker. */
  }
}

export function startDeviceHealthHeartbeat(
  opts: StartDeviceHealthHeartbeatOptions = {}
): () => void {
  if (Platform.OS === "web") {
    return () => undefined;
  }
  if (activeStop) {
    /* Idempotent : un seul ticker actif, on retourne le stop existant. */
    return activeStop;
  }
  const intervalMs = opts.intervalMs ?? DEFAULT_HEARTBEAT_INTERVAL_MS;
  let stopped = false;

  const safeTick = () => {
    if (stopped) return;
    void tickHeartbeat();
  };

  /* Premier tick immédiat (asynchrone) pour signaler la session au backend. */
  safeTick();

  const timer = setInterval(safeTick, intervalMs);

  let appStateSubscription: { remove?: () => void } | null = null;
  try {
    if (typeof AppState?.addEventListener === "function") {
      appStateSubscription = AppState.addEventListener("change", (next: AppStateStatus) => {
        if (next === "active") {
          safeTick();
        }
      });
    }
  } catch {
    appStateSubscription = null;
  }

  const stop = () => {
    if (stopped) return;
    stopped = true;
    clearInterval(timer);
    try {
      appStateSubscription?.remove?.();
    } catch {
      /* noop */
    }
    if (activeStop === stop) {
      activeStop = null;
    }
  };

  activeStop = stop;
  return stop;
}

export async function triggerDeviceHealthNow(reason: string): Promise<void> {
  if (Platform.OS === "web") return;
  await tickHeartbeat(reason);
}

/** Test-only : reset du singleton stop courant. */
export function __resetDeviceHealthHeartbeatForTests(): void {
  if (activeStop) {
    try {
      activeStop();
    } catch {
      /* noop */
    }
  }
  activeStop = null;
  lastSentAtMs = 0;
  lastSentConstraintReason = null;
}
