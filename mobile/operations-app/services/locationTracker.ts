// services/locationTracker.ts
// P1 — Exécuteur (start/stop natif, foreground) ; le mode métier est résolu par trackingPolicy.ts +
// trackingReconcile.ts (ne pas réinférer la criticité ici — lire getLastResolvedTrackingPolicy côté sync).
// Tracker GPS adaptatif avec fréquence variable selon mouvement
// Plan 2G/3G Phase 4 : enqueue dans locationQueue au lieu d'envoi direct (offline-safe)
// Background orchestrator : mission-active only, start/stop idempotent

import { AppState, Platform } from "react-native";
import AsyncStorage from "@react-native-async-storage/async-storage";
import * as Location from "expo-location";
import { getLogger } from "@/utils/logger";
import { assertDriverSurface } from "@/services/authSurface";
import { getDistanceInMeters } from "./location";
import { enqueueLocation } from "./locationQueue";
import {
  shouldRunBackgroundTracking,
  getFirstStopCondition,
  deriveStopContract,
  type BgTrackingInputs,
  type LocationMode,
  type PermissionStatus,
} from "./backgroundTrackingGating";
import {
  getMissionNotificationContent,
  dismissMissionNotification,
} from "./missionBarAndroid";
import { MissionStateManager, type MissionBarStatus } from "./missionState";
import { resolveLocationModeFromState, resolvePresenceState } from "./locationPresenceFsm";
import { resolveMissionContext } from "./locationMissionContext";
import {
  isMissionTrackingActiveStatus,
  isMissionTrackingEligibleNow,
} from "./missionTrackingPolicy";
import { getLastResolvedTrackingPolicy } from "./trackingRuntime";

// Constante locale pour éviter de charger locationTask en __DEV__ (boucle reload expo/expo#25325)
const LOCATION_TASK_NAME = "background-location-task";

const log = getLogger("Tracker");
const trackLog = getLogger("TRACK");

// ---------------------------------------------------------------------------
// Background orchestrator — mission-active only
// ---------------------------------------------------------------------------

const BG_DIAG_KEY = "@atmr:bg_tracking_diag";
const KILL_SWITCH_KEY = "driver_background_tracking_enabled";
const LOCATION_MODE_KEY = "@atmr:location_mode";
const AVAILABILITY_PRESENCE_ENABLED_KEY = "@atmr:availability_presence_enabled";

export type BgStartReason = "mission_active" | "reconciliation" | "notification_refresh";
export type BgStopReason =
  | "mission_ended"
  | "logout"
  | "permission_denied"
  | "permission_revoked"
  | "kill_switch"
  | "role_changed"
  | "reconciliation"
  | "notification_refresh";

/** Mutex partagé : une seule opération start/stop à la fois. Le stop a priorité. */
let bgOperationInProgress = false;
let lastStartReason: string | null = null;
let lastStopReason: string | null = null;
let lastReconciliationTrigger: string | null = null;
let lastStateChangeTs = 0;
let bgTrackingStartedCache = false;
/** P0 : défaut mémoire aligné sur présence (pas mission tant qu'aucune mission éligible). */
let currentLocationMode: LocationMode = "availability_presence";

/** Dernières permissions connues (mis à jour par buildBgTrackingInputs et initLocationModeRuntime). */
let lastPermissionSnapshot: { fg: PermissionStatus; bg: PermissionStatus } | null = null;

let lastLoggedPermissionKey = "";
let lastLoggedMissionKey = "";
let lastMissionEscalationEnabled: boolean | null = null;

export type PendingBackgroundTrackingState = {
  active: boolean;
  reason?: string;
  missionSnapshot?: { id: number | null; status: MissionBarStatus | null };
  deferredAt?: number;
};

let pendingBackgroundTrackingStart: PendingBackgroundTrackingState = { active: false };

export function getLastPermissionSnapshot(): {
  fg: PermissionStatus;
  bg: PermissionStatus;
} | null {
  return lastPermissionSnapshot;
}

export function getPendingBackgroundTrackingStart(): PendingBackgroundTrackingState {
  return { ...pendingBackgroundTrackingStart };
}

function setPendingBackgroundTrackingStart(
  reason: string,
  missionSnapshot?: { id: number | null; status: MissionBarStatus | null }
): void {
  pendingBackgroundTrackingStart = {
    active: true,
    reason,
    missionSnapshot,
    deferredAt: Date.now(),
  };
}

export function clearPendingBackgroundTrackingStart(clearReason: string): void {
  if (!pendingBackgroundTrackingStart.active) return;
  pendingBackgroundTrackingStart = { active: false };
  trackLog.info("PENDING_BACKGROUND_TRACKING_CLEARED", { clearReason });
}

/** P1 — Natif : policy résolue si disponible ; __DEV__ : assert vs gating legacy. */
function nativeBackgroundShouldRunFromPolicyOrLegacy(inputs: BgTrackingInputs): boolean {
  const policy = getLastResolvedTrackingPolicy();
  if (policy !== null) {
    const legacy = shouldRunBackgroundTracking(inputs);
    if (__DEV__ && legacy !== policy.shouldRunNativeBackgroundTracking) {
      trackLog.warn("TRACKING_NATIVE_POLICY_SHADOW_MISMATCH", {
        legacyShouldRun: legacy,
        policyShouldRun: policy.shouldRunNativeBackgroundTracking,
      });
    }
    return policy.shouldRunNativeBackgroundTracking;
  }
  return shouldRunBackgroundTracking(inputs);
}

/** Snapshot diagnostic lisible pour QA (@atmr:bg_tracking_diag). */
async function persistDiagnostic(): Promise<void> {
  try {
    const snapshot = {
      current_runtime_state: bgTrackingStartedCache ? "started" : "stopped",
      last_start_reason: lastStartReason,
      last_stop_reason: lastStopReason,
      last_reconciliation_trigger: lastReconciliationTrigger,
      last_state_change_ts: lastStateChangeTs,
    };
    await AsyncStorage.setItem(BG_DIAG_KEY, JSON.stringify(snapshot));
  } catch (e) {
    log.warn("bg diagnostic persist failed", { error: e });
  }
}

/** Kill switch : true = arrêt prioritaire. Valeur par défaut si absente = false (autoriser). */
async function isKillSwitchEnabled(): Promise<boolean> {
  try {
    const v = await AsyncStorage.getItem(KILL_SWITCH_KEY);
    if (v == null || v === "") return false;
    const lower = String(v).toLowerCase().trim();
    if (lower === "true" || lower === "1") return false;
    if (lower === "false" || lower === "0") return true;
    return false;
  } catch {
    return false;
  }
}

/**
 * Définit le kill switch (priorité absolue).
 * @param enabled true = arrêt immédiat du tracking, false = autoriser (si autres conditions OK)
 */
export async function setKillSwitchEnabled(enabled: boolean): Promise<void> {
  try {
    await AsyncStorage.setItem(KILL_SWITCH_KEY, enabled ? "false" : "true");
  } catch (e) {
    log.warn("set kill switch failed", { error: e });
  }
}

/** Lit le diagnostic persistant pour QA/debug. */
export async function getPersistedDiagnostic(): Promise<{
  current_runtime_state?: string;
  last_start_reason?: string;
  last_stop_reason?: string;
  last_reconciliation_trigger?: string;
  last_state_change_ts?: number;
} | null> {
  try {
    const raw = await AsyncStorage.getItem(BG_DIAG_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

/** Règle D : kill switch priorité absolue. Stop a priorité sur start. */
export async function ensureBackgroundTrackingStopped(
  reason: BgStopReason
): Promise<void> {
  if (Platform.OS === "web") return;
  if (__DEV__) return; // Workaround expo/expo#25325 : pas de task en __DEV__
  if (bgOperationInProgress) return;
  clearPendingBackgroundTrackingStart(`stop:${reason}`);
  const stopRequestedTs = Date.now();
  bgOperationInProgress = true;
  try {
    const started = await Location.hasStartedLocationUpdatesAsync(LOCATION_TASK_NAME);
    if (!started) {
      log.info("bg_tracking_stop_skip", { reason, already_stopped: true });
      return;
    }
    await Location.stopLocationUpdatesAsync(LOCATION_TASK_NAME);
    const stopEffectiveTs = Date.now();
    const slaMs = stopEffectiveTs - stopRequestedTs;
    bgTrackingStartedCache = false;
    lastStopReason = reason;
    lastStateChangeTs = stopEffectiveTs;
    await persistDiagnostic();
    log.info("bg_tracking_stopped", {
      reason,
      stop_requested_ts: stopRequestedTs,
      stop_effective_ts: stopEffectiveTs,
      sla_ms: slaMs,
    });
  } catch (e: any) {
    log.warn("bg_tracking_stop_error", { reason, error: e?.message });
  } finally {
    bgOperationInProgress = false;
  }
}

export async function ensureBackgroundTrackingStarted(
  reason: BgStartReason,
  inputs: BgTrackingInputs
): Promise<void> {
  if (Platform.OS === "web") return;
  if (__DEV__) return; // Workaround expo/expo#25325 : pas de task en __DEV__
  if (bgOperationInProgress) return;

  const ms = MissionStateManager.getState();
  const missionSnapshot = {
    id: ms.activeMission?.id ?? null,
    status: ms.activeMission ? ms.currentStatus : null,
  };

  trackLog.info("BACKGROUND_TRACKING_START_REQUESTED", {
    reason,
    appState: AppState.currentState,
    platform: Platform.OS,
    missionSnapshot,
  });

  if (!nativeBackgroundShouldRunFromPolicyOrLegacy(inputs)) {
    log.debug("bg_tracking_start_skip", { reason, inputs_deny: true });
    clearPendingBackgroundTrackingStart("should_not_run_inputs_deny");
    return;
  }
  // Android 10+ : un foreground service de localisation ne peut pas être démarré
  // tant que l'app n'est pas en "active". Sinon Expo rejette avec
  // "Foreground service cannot be started when the application is in the background"
  // (cf. logcat LocationTaskConsumer + [Tracker] bg_tracking_start_error).
  if (Platform.OS === "android" && AppState.currentState !== "active") {
    setPendingBackgroundTrackingStart("android_fgs_requires_foreground", missionSnapshot);
    log.debug("bg_tracking_start_deferred", {
      reason,
      app_state: AppState.currentState,
      hint: "android_fgs_requires_foreground",
    });
    trackLog.info("BACKGROUND_TRACKING_START_DEFERRED", {
      reason,
      appState: AppState.currentState,
      missionSnapshot,
      hint: "android_fgs_requires_foreground",
    });
    return;
  }
  bgOperationInProgress = true;
  try {
    const started = await Location.hasStartedLocationUpdatesAsync(LOCATION_TASK_NAME);
    if (started) {
      log.info("bg_tracking_start_skip", { reason, already_started: true });
      clearPendingBackgroundTrackingStart("already_started_native");
      trackLog.info("BACKGROUND_TRACKING_STARTED", {
        reason,
        locationMode: inputs.locationMode,
        already_started: true,
      });
      return;
    }
    const isPresenceMode = inputs.locationMode === "availability_presence";
    const opts: Location.LocationTaskOptions = {
      accuracy: Location.Accuracy.Balanced,
      timeInterval: isPresenceMode ? 180000 : 15000,
      distanceInterval: isPresenceMode ? 50 : 10,
    };
    if (Platform.OS === "android") {
      const state = MissionStateManager.getState();
      const { title } = getMissionNotificationContent(state);
      opts.foregroundService = {
        notificationTitle: title,
        notificationBody: "Suivi de localisation en cours",
        notificationColor: "#0A7F59",
      };
    }
    await Location.startLocationUpdatesAsync(LOCATION_TASK_NAME, opts);
    bgTrackingStartedCache = true;
    lastStartReason = reason;
    await dismissMissionNotification();
    lastStateChangeTs = Date.now();
    await persistDiagnostic();
    clearPendingBackgroundTrackingStart("native_started_ok");
    log.info("bg_tracking_started", { reason });
    trackLog.info("BACKGROUND_TRACKING_STARTED", {
      reason,
      locationMode: inputs.locationMode,
      already_started: false,
    });
  } catch (e: any) {
    log.warn("bg_tracking_start_error", { reason, error: e?.message });
  } finally {
    bgOperationInProgress = false;
  }
}

/** Redémarre le tracking pour mettre à jour la notification (mission status changé). Retourne true si rafraîchi. */
export async function refreshBackgroundTrackingNotification(
  inputs: BgTrackingInputs
): Promise<boolean> {
  if (Platform.OS === "web") return false;
  if (__DEV__) return false; // Workaround expo/expo#25325 : pas de task en __DEV__
  // Ne pas stopper le task en cours si on ne peut pas le redémarrer (même contrainte FGS).
  if (Platform.OS === "android" && AppState.currentState !== "active") {
    return false;
  }
  try {
    const started = await Location.hasStartedLocationUpdatesAsync(LOCATION_TASK_NAME);
    if (!started) return false;
    if (!nativeBackgroundShouldRunFromPolicyOrLegacy(inputs)) return false;
  } catch {
    return false;
  }
  await ensureBackgroundTrackingStopped("notification_refresh");
  await ensureBackgroundTrackingStarted("notification_refresh", inputs);
  return true;
}

export async function reconcileBackgroundTrackingState(
  trigger: string,
  inputs: BgTrackingInputs
): Promise<void> {
  if (Platform.OS === "web") return;
  if (__DEV__) return; // Workaround expo/expo#25325 : pas de task en __DEV__
  lastReconciliationTrigger = trigger;
  const shouldRun = nativeBackgroundShouldRunFromPolicyOrLegacy(inputs);
  let started = false;
  try {
    started = await Location.hasStartedLocationUpdatesAsync(LOCATION_TASK_NAME);
  } catch {
    return;
  }
  if (shouldRun && !started) {
    await ensureBackgroundTrackingStarted("reconciliation", inputs);
  } else if (!shouldRun && started) {
    const stopReason = deriveStopReasonFromInputs(inputs);
    // Détecte permission_revoked quand l'utilisateur revient des Settings après avoir retiré la permission
    if (stopReason === "permission_revoked") {
      log.info("permission_revoked detected, stopping background tracking", { trigger });
    }
    await ensureBackgroundTrackingStopped(stopReason);
  }
  await persistDiagnostic();
}

/** Dérive une raison de stop explicite depuis les inputs (contrat d'arrêt). */
function deriveStopReasonFromInputs(inputs: BgTrackingInputs): BgStopReason {
  const contract = deriveStopContract(inputs);
  const cond = getFirstStopCondition(contract);
  if (cond === "kill_switch") return "kill_switch";
  if (cond === "permission_revoked") return "permission_revoked";
  if (cond === "logout") return "logout";
  if (cond === "mission_ended") return "mission_ended";
  if (cond === "role_non_driver") return "role_changed";
  return "reconciliation";
}

export function getTrackingRuntimeState(): {
  started: boolean;
  lastReason?: string;
} {
  return {
    started: bgTrackingStartedCache,
    lastReason: lastStartReason ?? lastStopReason ?? undefined,
  };
}

export function getLastStartReason(): string | null {
  return lastStartReason;
}

export function getLastStopReason(): string | null {
  return lastStopReason;
}

/** Helper pour construire BgTrackingInputs depuis les permissions expo-location. */
export async function getPermissionStatuses(): Promise<{
  fg: PermissionStatus;
  bg: PermissionStatus;
}> {
  try {
    const [fg, bg] = await Promise.all([
      Location.getForegroundPermissionsAsync(),
      Location.getBackgroundPermissionsAsync(),
    ]);
    const toStatus = (s: { status: string }): PermissionStatus =>
      s?.status === "granted" ? "granted" : s?.status === "denied" ? "denied" : "undetermined";
    return { fg: toStatus(fg), bg: toStatus(bg) };
  } catch {
    return { fg: "undetermined", bg: "undetermined" };
  }
}

/**
 * Hydrate le mode mémoire et le snapshot permissions au démarrage session driver,
 * avant syncEngine.start(), pour éviter un défaut mission_live incohérent.
 */
export async function initLocationModeRuntime(): Promise<void> {
  await MissionStateManager.ensureHydrated({ skipNetwork: true });
  try {
    const perms = await getPermissionStatuses();
    lastPermissionSnapshot = { fg: perms.fg, bg: perms.bg };
    const permKey = `${perms.fg}:${perms.bg}`;
    if (permKey !== lastLoggedPermissionKey) {
      lastLoggedPermissionKey = permKey;
      trackLog.info("PERMISSION_SNAPSHOT", {
        fg: perms.fg,
        bg: perms.bg,
        platform: Platform.OS,
        source: "init_runtime",
      });
    }
  } catch {
    // snapshot inchangé
  }

  const hasMission = MissionStateManager.isActive();
  const eligible = isMissionTrackingEligibleNow();
  const persisted = await getPersistedLocationMode(hasMission);
  const nextMode: LocationMode = eligible ? "mission_live" : persisted;
  const prevMode = currentLocationMode;
  currentLocationMode = nextMode;
  if (prevMode !== nextMode) {
    trackLog.info("TRACKING_MODE_CHANGED", {
      from: prevMode,
      to: nextMode,
      source: "init_runtime",
      hasActiveMission: hasMission,
      missionStatus: hasMission ? MissionStateManager.getState().currentStatus : null,
    });
  }
}

/** Construit les inputs pour le gating (appelé par le layout / mission listener). */
export async function buildBgTrackingInputs(params: {
  isAuthenticated: boolean;
  role: "driver" | "enterprise";
  hasActiveMission: boolean;
}): Promise<BgTrackingInputs> {
  const { fg, bg } = await getPermissionStatuses();
  lastPermissionSnapshot = { fg, bg };
  const permKey = `${fg}:${bg}`;
  if (permKey !== lastLoggedPermissionKey) {
    lastLoggedPermissionKey = permKey;
    trackLog.info("PERMISSION_SNAPSHOT", {
      fg,
      bg,
      platform: Platform.OS,
      source: "build_inputs",
    });
  }

  const killSwitchEnabled = await isKillSwitchEnabled();
  const availabilityPresenceEnabled = await isAvailabilityPresenceEnabled();
  const persistedMode = await getPersistedLocationMode(params.hasActiveMission);
  const fsmState = resolvePresenceState({
    isAuthenticated: params.isAuthenticated,
    isDriver: params.role === "driver",
    hasFgPermission: fg === "granted",
    hasBgPermission: bg === "granted",
    appInBackground: false,
    hasActiveMission: params.hasActiveMission,
    availabilityPresenceEnabled,
  });
  const fsmMode = resolveLocationModeFromState(fsmState);
  const missionState = MissionStateManager.getState();
  const missionStatusEnabledForTracking =
    params.hasActiveMission &&
    isMissionTrackingActiveStatus(missionState.currentStatus);

  const missionKey = `${missionState.activeMission?.id ?? "none"}:${missionState.currentStatus}:${missionStatusEnabledForTracking}`;
  if (missionKey !== lastLoggedMissionKey) {
    lastLoggedMissionKey = missionKey;
    trackLog.info("MISSION_SNAPSHOT", {
      missionId: missionState.activeMission?.id ?? null,
      currentStatus: missionState.currentStatus,
      hasActiveMission: params.hasActiveMission,
      missionStatusEnabledForTracking,
    });
  }

  const prevEsc = lastMissionEscalationEnabled;
  const enabled = !!missionStatusEnabledForTracking;
  if (prevEsc !== null && prevEsc !== enabled) {
    if (enabled) {
      trackLog.info("MISSION_TRACKING_ESCALATED", {
        missionId: missionState.activeMission?.id ?? null,
        currentStatus: missionState.currentStatus,
        enabled: true,
      });
    } else {
      trackLog.info("MISSION_TRACKING_DEESCALATED", {
        missionId: missionState.activeMission?.id ?? null,
        currentStatus: missionState.currentStatus,
        enabled: false,
      });
    }
  }
  lastMissionEscalationEnabled = enabled;

  const eligible = isMissionTrackingEligibleNow();
  const prevMode = currentLocationMode;
  const locationMode: LocationMode =
    eligible || fsmMode === "mission_live" ? "mission_live" : persistedMode;
  currentLocationMode = locationMode;
  if (prevMode !== locationMode) {
    trackLog.info("TRACKING_MODE_CHANGED", {
      from: prevMode,
      to: locationMode,
      source: "build_inputs",
      hasActiveMission: params.hasActiveMission,
      missionStatus: missionState.activeMission ? missionState.currentStatus : null,
    });
  }
  return {
    ...params,
    platform: Platform.OS === "ios" || Platform.OS === "android" ? Platform.OS : "web",
    missionStatusEnabledForTracking: params.hasActiveMission ? missionStatusEnabledForTracking : undefined,
    fgPermission: fg,
    bgPermission: bg,
    killSwitchEnabled,
    locationMode,
    availabilityPresenceEnabled,
  };
}

export async function getPersistedLocationMode(
  hasActiveMission = false
): Promise<LocationMode> {
  try {
    const raw = (await AsyncStorage.getItem(LOCATION_MODE_KEY))?.trim();
    if (
      raw === "mission_live" ||
      raw === "availability_presence" ||
      raw === "passive_last_known"
    ) {
      return raw;
    }
  } catch {
    // ignore
  }
  return hasActiveMission ? "mission_live" : "availability_presence";
}

export async function setPersistedLocationMode(mode: LocationMode): Promise<void> {
  currentLocationMode = mode;
  await AsyncStorage.setItem(LOCATION_MODE_KEY, mode);
}

export function getCurrentLocationMode(): LocationMode {
  return currentLocationMode;
}

export async function isAvailabilityPresenceEnabled(): Promise<boolean> {
  try {
    const raw = await AsyncStorage.getItem(AVAILABILITY_PRESENCE_ENABLED_KEY);
    if (!raw) {
      return process.env.EXPO_PUBLIC_LOCATION_PRESENCE_HEARTBEAT_ENABLED !== "false";
    }
    return raw === "true";
  } catch {
    return process.env.EXPO_PUBLIC_LOCATION_PRESENCE_HEARTBEAT_ENABLED !== "false";
  }
}

/**
 * Demande la permission background si foreground accordée et mission active.
 * Déclenché par mission_started (action métier explicite).
 */
export async function requestBackgroundPermissionIfNeeded(): Promise<PermissionStatus> {
  try {
    const fg = await Location.getForegroundPermissionsAsync();
    if (fg.status !== "granted") return "undetermined";
    const bg = await Location.getBackgroundPermissionsAsync();
    if (bg.status === "granted") return "granted";
    const { status } = await Location.requestBackgroundPermissionsAsync();
    return status === "granted" ? "granted" : status === "denied" ? "denied" : "undetermined";
  } catch {
    return "undetermined";
  }
}

/**
 * Tracker GPS adaptatif qui ajuste la fréquence selon la vitesse.
 * 
 * - En mouvement (> 3.6 km/h) : 5s
 * - Immobile : 30s
 * - Batterie < 20% : 60s (mode économie)
 */
export type GpsStatus = "active" | "disabled" | "unavailable" | "unknown";

export class AdaptiveLocationTracker {
  private locationSub: Location.LocationSubscription | null = null;
  private updateInterval: number = 5000; // 5s par défaut
  private lastPosition: Location.LocationObject | null = null;
  private lastSpeed: number = 0;
  private lastSentAt: number = 0;
  private isTracking: boolean = false;
  private batteryCheckInterval: ReturnType<typeof setInterval> | null = null;
  private trackingTimeout: ReturnType<typeof setTimeout> | null = null;

  private _gpsStatus: GpsStatus = "unknown";
  private gpsStatusListeners: Set<(status: GpsStatus) => void> = new Set();
  private positionListeners: Set<(location: Location.LocationObject) => void> = new Set();

  /** Plan 2G/3G : Expose position pour UI (mission, dashboard). */
  subscribeToPosition(listener: (location: Location.LocationObject) => void): () => void {
    this.positionListeners.add(listener);
    if (this.lastPosition) listener(this.lastPosition);
    return () => this.positionListeners.delete(listener);
  }

  /** Plan 2G/3G Phase 6 : Dernière position pour heartbeat syncEngine. */
  getLastPosition(): Location.LocationObject | null {
    return this.lastPosition;
  }

  onGpsStatusChange(listener: (status: GpsStatus) => void): () => void {
    this.gpsStatusListeners.add(listener);
    listener(this._gpsStatus);
    return () => this.gpsStatusListeners.delete(listener);
  }

  get gpsStatus(): GpsStatus {
    return this._gpsStatus;
  }

  private setGpsStatus(status: GpsStatus): void {
    if (status === this._gpsStatus) return;
    this._gpsStatus = status;
    this.gpsStatusListeners.forEach((fn) => fn(status));
  }

  // Seuils de vitesse (m/s)
  private readonly SPEED_THRESHOLD_MOVING = 1.0; // 3.6 km/h
  // Config via env: EXPO_PUBLIC_GPS_FAST_MS=5000, EXPO_PUBLIC_GPS_SLOW_MS=30000
  private readonly INTERVAL_MOVING_MS =
    parseInt(process.env.EXPO_PUBLIC_GPS_FAST_MS ?? "5000", 10) || 5000;
  private readonly INTERVAL_STATIONARY_MS =
    parseInt(process.env.EXPO_PUBLIC_GPS_SLOW_MS ?? "60000", 10) || 60000;
  private readonly INTERVAL_BATTERY_LOW_MS = 60000; // 60s si batterie < 20%

  /**
   * Démarrer le tracking adaptatif.
   */
  async startTracking(): Promise<void> {
    if (!assertDriverSurface("AdaptiveLocationTracker.startTracking")) {
      return;
    }
    if (this.isTracking) {
      log.debug("tracking already active");
      return;
    }

    // Demander permissions
    const { status } = await Location.requestForegroundPermissionsAsync();
    if (status !== "granted") {
      throw new Error("Permission de localisation refusée");
    }

    this.isTracking = true;
    this.lastSentAt = Date.now();

    // Démarrer vérification batterie (toutes les 60s)
    this.startBatteryMonitoring();

    // Démarrer le tracking
    this.scheduleNextUpdate();
    log.success("adaptive tracking started");
  }

  /**
   * Arrêter le tracking.
   */
  stopTracking(): void {
    if (!this.isTracking) {
      return;
    }

    this.isTracking = false;

    if (this.locationSub) {
      try {
        this.locationSub.remove();
      } catch (e) {
        log.warn("stop subscription error", { error: e });
      }
      this.locationSub = null;
    }

    if (this.trackingTimeout) {
      clearTimeout(this.trackingTimeout);
      this.trackingTimeout = null;
    }

    if (this.batteryCheckInterval) {
      clearInterval(this.batteryCheckInterval);
      this.batteryCheckInterval = null;
    }

    log.success("tracking stopped");
  }

  /**
   * Programmer la prochaine mise à jour.
   */
  private scheduleNextUpdate(): void {
    if (!this.isTracking) {
      return;
    }

    const mode = getCurrentLocationMode();
    if (mode === "mission_live") {
      this.updateInterval = this.INTERVAL_MOVING_MS; // v2.1: 5s fixé
    } else if (mode === "availability_presence") {
      this.updateInterval = this.INTERVAL_STATIONARY_MS; // v2.1: 60s fixé
    } else {
      this.updateInterval = this.INTERVAL_STATIONARY_MS;
    }

    // Vérifier batterie et ajuster si nécessaire
    this.checkBatteryLevel().then((batteryLevel) => {
      if (batteryLevel !== null && batteryLevel < 0.2) {
        // < 20% : mission_live plafonné à 15s, présence reste légère.
        if (mode === "mission_live") {
          this.updateInterval = 15000;
        } else if (mode === "availability_presence") {
          this.updateInterval = 180000;
        } else {
          this.updateInterval = this.INTERVAL_BATTERY_LOW_MS;
        }
        log.info("low battery interval reduced", {
          batteryLevel: (batteryLevel * 100).toFixed(0),
          intervalMs: this.updateInterval,
        });
      }
    });

    // Programmer prochaine mise à jour
    this.trackingTimeout = setTimeout(() => {
      this.updateLocation();
    }, this.updateInterval);
  }

  /**
   * Mettre à jour la position.
   */
  private async updateLocation(): Promise<void> {
    if (!this.isTracking) {
      return;
    }

    try {
      const enabled = await Location.hasServicesEnabledAsync();
      if (!enabled) {
        this.setGpsStatus("disabled");
        log.warn("location services disabled, retrying later");
        this.scheduleNextUpdate();
        return;
      }

      let location: Location.LocationObject | null = null;

      try {
        location = await Location.getCurrentPositionAsync({
          accuracy: Location.Accuracy.Balanced,
          timeInterval: 10000,
        });
      } catch {
        log.warn("getCurrentPositionAsync failed, trying last known position");
        location = await Location.getLastKnownPositionAsync();
      }

      if (!location) {
        this.setGpsStatus("unavailable");
        log.warn("no location available, retrying later");
        this.scheduleNextUpdate();
        return;
      }

      this.setGpsStatus("active");

      // Calculer vitesse
      const speed = this.calculateSpeedFromLocation(location);
      this.lastSpeed = speed;

      // Vérifier si on doit envoyer (distance ou temps)
      const shouldSend = this.shouldSendLocation(location);

      if (shouldSend) {
        await this.sendLocation(location);
        this.lastSentAt = Date.now();
      }

      this.lastPosition = location;
      this.positionListeners.forEach((fn) => {
        try {
          fn(location);
        } catch (e) {
          log.warn("position listener error", { error: e });
        }
      });

      // Programmer prochaine mise à jour
      this.scheduleNextUpdate();
    } catch (error) {
      log.error("position update failed", { error });
      // Réessayer après un délai
      this.trackingTimeout = setTimeout(() => {
        this.updateLocation();
      }, this.updateInterval);
    }
  }

  /**
   * Calculer vitesse moyenne depuis dernière position.
   */
  private calculateSpeed(): number {
    if (!this.lastPosition) {
      return 0;
    }

    // Utiliser vitesse du GPS si disponible
    if (this.lastPosition.coords.speed !== null && this.lastPosition.coords.speed > 0) {
      return this.lastPosition.coords.speed; // m/s
    }

    // Sinon, calculer depuis distance/temps
    return this.lastSpeed; // Garder dernière vitesse calculée
  }

  /**
   * Calculer vitesse depuis une nouvelle position.
   */
  private calculateSpeedFromLocation(location: Location.LocationObject): number {
    if (!this.lastPosition) {
      return 0;
    }

    // Utiliser vitesse GPS si disponible
    if (location.coords.speed !== null && location.coords.speed > 0) {
      return location.coords.speed; // m/s
    }

    // Calculer depuis distance/temps
    const distance = getDistanceInMeters(
      this.lastPosition.coords.latitude,
      this.lastPosition.coords.longitude,
      location.coords.latitude,
      location.coords.longitude
    );

    const timeDelta = ((location.timestamp || Date.now()) - (this.lastPosition.timestamp || Date.now())) / 1000; // secondes

    if (timeDelta <= 0) {
      return 0;
    }

    return distance / timeDelta; // m/s
  }

  /**
   * Déterminer si on doit envoyer la position.
   */
  private shouldSendLocation(location: Location.LocationObject): boolean {
    const now = Date.now();
    const timeSinceLastSend = now - this.lastSentAt;

    const currentMode = getCurrentLocationMode();
    const heartbeatMs = currentMode === "mission_live" ? 15000 : 60000;
    if (timeSinceLastSend >= heartbeatMs) {
      return true;
    }

    if (!this.lastPosition) {
      return true; // Première position
    }

    // Vérifier distance
    const distance = getDistanceInMeters(
      this.lastPosition.coords.latitude,
      this.lastPosition.coords.longitude,
      location.coords.latitude,
      location.coords.longitude
    );

    // Envoyer si déplacement > 10m
    return distance >= 10;
  }

  /**
   * Plan 2G/3G : Enqueue dans locationQueue (flush par syncEngine).
   * Plus d'envoi direct — offline-safe.
   */
  private async sendLocation(location: Location.LocationObject): Promise<void> {
    const { latitude, longitude, speed, heading, accuracy } = location.coords;
    const driverIdStr = await AsyncStorage.getItem("driver_id");
    const driver_id = driverIdStr ? parseInt(driverIdStr, 10) : 0;
    if (!driver_id || !Number.isFinite(driver_id)) {
      log.warn("no driver_id, skip enqueue");
      return;
    }

    try {
      const { missionId, mode } = resolveMissionContext();
      await enqueueLocation({
        latitude: Number(latitude),
        longitude: Number(longitude),
        speed: speed ? Number(speed) : 0,
        heading: heading ? Number(heading) : 0,
        accuracy: accuracy ? Number(accuracy) : 0,
        timestamp: location.timestamp || Date.now(),
        driver_id,
        location_mode: mode,
        mission_id: missionId,
        recorded_at: new Date(location.timestamp || Date.now()).toISOString(),
        sent_at: new Date().toISOString(),
        is_background: false,
      });
      log.success("position enqueued", {
        speedKmh: (this.lastSpeed * 3.6).toFixed(1),
        intervalMs: this.updateInterval,
      });
    } catch (error) {
      log.error("enqueue position failed", { error });
    }
  }

  /**
   * Démarrer monitoring batterie.
   */
  private startBatteryMonitoring(): void {
    // Vérifier batterie toutes les 60s
    this.batteryCheckInterval = setInterval(() => {
      this.checkBatteryLevel().then((batteryLevel) => {
        if (batteryLevel !== null) {
          log.info("battery level", { level: (batteryLevel * 100).toFixed(0) });
        }
      });
    }, 60000);
  }

  /**
   * Vérifier niveau batterie (si API disponible).
   */
  private async checkBatteryLevel(): Promise<number | null> {
    try {
      // Expo Battery API (si disponible)
      const Battery = require("expo-battery");
      if (Battery && Battery.getBatteryLevelAsync) {
        const batteryLevel = await Battery.getBatteryLevelAsync();
        return batteryLevel; // 0.0 à 1.0
      }
    } catch (error) {
      // API batterie non disponible (normal sur certaines plateformes)
    }
    return null;
  }

  /**
   * Obtenir statistiques du tracker.
   */
  getStats(): {
    isTracking: boolean;
    updateInterval: number;
    lastSpeed: number;
    lastSentAt: number;
  } {
    return {
      isTracking: this.isTracking,
      updateInterval: this.updateInterval,
      lastSpeed: this.lastSpeed,
      lastSentAt: this.lastSentAt,
    };
  }
}

// Instance globale (singleton)
let globalTracker: AdaptiveLocationTracker | null = null;

/**
 * Obtenir l'instance globale du tracker.
 */
export function getAdaptiveLocationTracker(): AdaptiveLocationTracker {
  if (!globalTracker) {
    globalTracker = new AdaptiveLocationTracker();
  }
  return globalTracker;
}

/**
 * Démarrer le tracking adaptatif (helper).
 */
export async function startAdaptiveLocationTracking(): Promise<void> {
  if (!assertDriverSurface("startAdaptiveLocationTracking")) {
    return;
  }
  try {
    await MissionStateManager.syncActiveMissionFromServerIfMissing();
  } catch {
    // best-effort : le tracking reste utilisable en availability_presence
  }
  const tracker = getAdaptiveLocationTracker();
  await tracker.startTracking();
}

/**
 * Arrêter le tracking adaptatif (helper).
 */
export function stopAdaptiveLocationTracking(): void {
  const tracker = getAdaptiveLocationTracker();
  tracker.stopTracking();
}

