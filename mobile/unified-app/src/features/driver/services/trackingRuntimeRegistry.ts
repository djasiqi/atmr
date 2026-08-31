/**
 * Registre runtime GPS (Phase 1C) — identité de génération séparée du contexte mission.
 */
import type { DriverMissionStatus } from "../types";
import { getSessionGenerationId } from "../../../core/auth/authCredentialStore";
import {
  getTrackingAuthAvailability,
  subscribeToTrackingAuthTerminalEvents,
  TRACKING_AUTH_EFFECT_POLICY,
  type TrackingAuthTerminalEvent,
} from "../../../core/auth/sessionAuthDecision";

export type TrackingRuntimeIdentity = {
  trackingGenerationId: string;
  sessionGenerationId: number;
  trackingIdentityId: string;
  driverId: number;
  startedAt: number;
};

export type TrackingMissionContext = {
  missionContextVersion: number;
  missionId: number | null;
  missionStatus: DriverMissionStatus | null;
};

export type ActiveTrackingRuntime = {
  identity: TrackingRuntimeIdentity;
  missionContext: TrackingMissionContext;
  status: "starting" | "running" | "stopping";
};

export type StopTrackingRequest = {
  expectedTrackingGenerationId: string;
  reason:
    | "explicit_logout"
    | "account_revoked"
    | "identity_changed"
    | "manual_stop"
    | "forced_recovery"
    | "runtime_replaced";
  quarantinePolicy: "none" | "identity_partition";
};

export type StopTrackingResult =
  | { status: "stopped"; generationId: string }
  | { status: "ignored_stale_stop"; activeGenerationId: string | null }
  | { status: "already_stopped" };

export type NativeTrackingOwner = {
  trackingGenerationId: string;
  sessionGenerationId: number;
  trackingIdentityId: string;
  missionContextVersion: number;
  /** Mission portée par le propriétaire natif (null = présence / hors mission). */
  missionId: number | null;
  driverId: number;
};

let activeRuntime: ActiveTrackingRuntime | null = null;
let startTrackingInFlight: Promise<ActiveTrackingRuntime> | null = null;
let terminalUnsub: (() => void) | null = null;
/** Callback bridge pour arrêt physique (évite dépendance circulaire). */
let physicalStopCallback:
  | ((request: StopTrackingRequest) => Promise<void>)
  | null = null;

function newTrackingGenerationId(): string {
  return `trk-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 12)}`;
}

export function captureActiveRuntime(): ActiveTrackingRuntime | null {
  return activeRuntime;
}

export function isRuntimeActive(identity: TrackingRuntimeIdentity): boolean {
  const current = activeRuntime?.identity;
  return Boolean(
    current &&
      current.trackingGenerationId === identity.trackingGenerationId &&
      current.sessionGenerationId === identity.sessionGenerationId &&
      current.trackingIdentityId === identity.trackingIdentityId
  );
}

export function commitRuntimeMutationIfActive(
  identity: TrackingRuntimeIdentity,
  mutation: () => void
): boolean {
  if (!isRuntimeActive(identity)) return false;
  mutation();
  return isRuntimeActive(identity);
}

export async function runIfRuntimeActive<T>(
  identity: TrackingRuntimeIdentity,
  operation: () => Promise<T>
): Promise<{ status: "completed"; value: T } | { status: "ignored_stale_runtime" }> {
  if (!isRuntimeActive(identity)) {
    return { status: "ignored_stale_runtime" };
  }
  const value = await operation();
  if (!isRuntimeActive(identity)) {
    return { status: "ignored_stale_runtime" };
  }
  return { status: "completed", value };
}

export function resolveTrackingIdentityId(driverId: number, companyId?: string | number | null): string {
  return `driver:${driverId}:company:${companyId ?? "unknown"}`;
}

/**
 * Démarre (ou rejoint) un runtime. Changement de mission = même génération, missionContextVersion++.
 */
export async function startOrJoinTrackingRuntime(params: {
  driverId: number;
  companyId?: string | number | null;
  missionId: number | null;
  missionStatus: DriverMissionStatus | null;
  forceNewGeneration?: boolean;
}): Promise<ActiveTrackingRuntime> {
  const availability = getTrackingAuthAvailability();
  const trackingIdentityId = resolveTrackingIdentityId(params.driverId, params.companyId);
  const sessionGenerationId =
    availability.kind === "SESSION_AVAILABLE"
      ? availability.sessionGenerationId
      : getSessionGenerationId();

  if (startTrackingInFlight) {
    const joined = await startTrackingInFlight;
    if (
      joined.identity.trackingIdentityId === trackingIdentityId &&
      !params.forceNewGeneration
    ) {
      return updateMissionContext(params.missionId, params.missionStatus) ?? joined;
    }
  }

  const run = (async () => {
    const current = activeRuntime;
    const sameIdentity =
      current &&
      current.identity.trackingIdentityId === trackingIdentityId &&
      current.identity.sessionGenerationId === sessionGenerationId &&
      !params.forceNewGeneration;

    if (sameIdentity && current) {
      current.status = "running";
      current.missionContext = {
        missionContextVersion: current.missionContext.missionContextVersion + 1,
        missionId: params.missionId,
        missionStatus: params.missionStatus,
      };
      return current;
    }

    const identity: TrackingRuntimeIdentity = {
      trackingGenerationId: newTrackingGenerationId(),
      sessionGenerationId,
      trackingIdentityId,
      driverId: params.driverId,
      startedAt: Date.now(),
    };
    activeRuntime = {
      identity,
      missionContext: {
        missionContextVersion: 1,
        missionId: params.missionId,
        missionStatus: params.missionStatus,
      },
      status: "running",
    };
    return activeRuntime;
  })();

  startTrackingInFlight = run;
  try {
    return await run;
  } finally {
    if (startTrackingInFlight === run) {
      startTrackingInFlight = null;
    }
  }
}

export function updateMissionContext(
  missionId: number | null,
  missionStatus: DriverMissionStatus | null
): ActiveTrackingRuntime | null {
  if (!activeRuntime) return null;
  activeRuntime.missionContext = {
    missionContextVersion: activeRuntime.missionContext.missionContextVersion + 1,
    missionId,
    missionStatus,
  };
  return activeRuntime;
}

/**
 * Efface le registre si la génération correspond (sans arrêt physique).
 * Utilisé par le bridge qui orchestre déjà stopMissionTrackingBridge.
 */
export function clearActiveRuntimeIfGeneration(
  expectedTrackingGenerationId: string
): boolean {
  if (!activeRuntime) return false;
  if (activeRuntime.identity.trackingGenerationId !== expectedTrackingGenerationId) {
    return false;
  }
  activeRuntime = null;
  return true;
}

export async function stopTrackingRuntime(
  request: StopTrackingRequest,
  opts?: { invokePhysicalStop?: boolean }
): Promise<StopTrackingResult> {
  if (!activeRuntime) {
    return { status: "already_stopped" };
  }
  if (activeRuntime.identity.trackingGenerationId !== request.expectedTrackingGenerationId) {
    return {
      status: "ignored_stale_stop",
      activeGenerationId: activeRuntime.identity.trackingGenerationId,
    };
  }
  activeRuntime.status = "stopping";
  const generationId = request.expectedTrackingGenerationId;
  activeRuntime = null;
  if (opts?.invokePhysicalStop !== false && physicalStopCallback) {
    await physicalStopCallback(request).catch(() => undefined);
  }
  return { status: "stopped", generationId };
}

export function registerTrackingPhysicalStop(
  handler: (request: StopTrackingRequest) => Promise<void>
): void {
  physicalStopCallback = handler;
}

export function toNativeTrackingOwner(
  runtime: ActiveTrackingRuntime
): NativeTrackingOwner {
  return {
    trackingGenerationId: runtime.identity.trackingGenerationId,
    sessionGenerationId: runtime.identity.sessionGenerationId,
    trackingIdentityId: runtime.identity.trackingIdentityId,
    missionContextVersion: runtime.missionContext.missionContextVersion,
    missionId: runtime.missionContext.missionId,
    driverId: runtime.identity.driverId,
  };
}

export function isNativeOwnerCurrent(
  owner: NativeTrackingOwner | null | undefined
): boolean {
  if (!owner || !activeRuntime) return false;
  return (
    owner.trackingGenerationId === activeRuntime.identity.trackingGenerationId &&
    owner.trackingIdentityId === activeRuntime.identity.trackingIdentityId &&
    owner.sessionGenerationId === activeRuntime.identity.sessionGenerationId &&
    owner.driverId === activeRuntime.identity.driverId &&
    owner.missionId === activeRuntime.missionContext.missionId &&
    owner.missionContextVersion === activeRuntime.missionContext.missionContextVersion
  );
}

/**
 * Validation headless durable — n'utilise PAS activeRuntime (null après process death iOS).
 */
export function validateNativeOwnerForHeadless(params: {
  owner: NativeTrackingOwner | null | undefined;
  lease: {
    state: string;
    driverId?: number;
    sessionGenerationId?: number;
    trackingGenerationId?: string;
    trackingIdentityId?: string;
    contextId?: string;
    missionId?: number | null;
    missionContextVersion?: number;
  } | null;
  authUsable: boolean;
}): { ok: true } | { ok: false; reason: string } {
  const { owner, lease, authUsable } = params;
  if (!lease || lease.state !== "driver_active") {
    return { ok: false, reason: "lease_not_driver_active" };
  }
  if (!owner) {
    return { ok: false, reason: "missing_native_owner" };
  }
  if (!authUsable) {
    return { ok: false, reason: "auth_not_usable" };
  }
  if (owner.driverId !== lease.driverId) {
    return { ok: false, reason: "driver_id_mismatch" };
  }
  if (owner.sessionGenerationId !== lease.sessionGenerationId) {
    return { ok: false, reason: "session_generation_mismatch" };
  }
  if (owner.trackingGenerationId !== lease.trackingGenerationId) {
    return { ok: false, reason: "tracking_generation_mismatch" };
  }
  if (owner.trackingIdentityId !== lease.trackingIdentityId) {
    return { ok: false, reason: "tracking_identity_mismatch" };
  }
  const expectedContext = `driver:${owner.driverId}`;
  if (lease.contextId !== expectedContext) {
    return { ok: false, reason: "context_id_mismatch" };
  }
  const leaseMissionId =
    lease.missionId === undefined ? null : lease.missionId;
  if (owner.missionId !== leaseMissionId) {
    return { ok: false, reason: "mission_id_mismatch" };
  }
  if (
    typeof lease.missionContextVersion !== "number" ||
    owner.missionContextVersion !== lease.missionContextVersion
  ) {
    return { ok: false, reason: "mission_context_version_mismatch" };
  }
  return { ok: true };
}

export type NativeOwnerResolveSource =
  | "task_context_persisted"
  | "active_runtime"
  | "lease_reconstructed";

export function buildNativeOwnerFromDriverActiveLease(lease: {
  driverId: number;
  sessionGenerationId: number;
  trackingGenerationId: string;
  trackingIdentityId: string;
  missionId: number | null;
  missionContextVersion: number;
}): NativeTrackingOwner {
  return {
    driverId: lease.driverId,
    sessionGenerationId: lease.sessionGenerationId,
    trackingGenerationId: lease.trackingGenerationId,
    trackingIdentityId: lease.trackingIdentityId,
    missionId: lease.missionId,
    missionContextVersion: lease.missionContextVersion,
  };
}

/**
 * Résout l'ownership headless sans dépendre d'un objet JS volatile seul.
 * Ordre : owner persisté → runtime mémoire → reconstruction depuis lease durable.
 */
export function resolveNativeOwnerForHeadlessCapture(params: {
  persistedOwner: NativeTrackingOwner | null | undefined;
  lease: {
    state: string;
    contextId?: string;
    driverId?: number;
    sessionGenerationId?: number;
    trackingGenerationId?: string;
    trackingIdentityId?: string;
    missionId?: number | null;
    missionContextVersion?: number;
  } | null;
  authUsable: boolean;
}): { ok: true; owner: NativeTrackingOwner; source: NativeOwnerResolveSource } | { ok: false; reason: string } {
  const { persistedOwner, lease, authUsable } = params;
  if (!lease || lease.state !== "driver_active") {
    return { ok: false, reason: "lease_not_driver_active" };
  }
  if (!authUsable) {
    return { ok: false, reason: "auth_not_usable" };
  }

  const runtimeOwner = activeRuntime ? toNativeTrackingOwner(activeRuntime) : null;

  // Owner persisté présent : valider ou rejeter — pas de reconstruction (anti-bypass stale).
  if (persistedOwner) {
    const persistedCheck = validateNativeOwnerForHeadless({
      owner: persistedOwner,
      lease,
      authUsable,
    });
    if (persistedCheck.ok) {
      return { ok: true, owner: persistedOwner, source: "task_context_persisted" };
    }
    return { ok: false, reason: persistedCheck.reason };
  }

  if (runtimeOwner) {
    const runtimeCheck = validateNativeOwnerForHeadless({
      owner: runtimeOwner,
      lease,
      authUsable,
    });
    if (runtimeCheck.ok) {
      return { ok: true, owner: runtimeOwner, source: "active_runtime" };
    }
  }

  if (
    typeof lease.driverId === "number" &&
    typeof lease.sessionGenerationId === "number" &&
    typeof lease.trackingGenerationId === "string" &&
    typeof lease.trackingIdentityId === "string" &&
    typeof lease.missionContextVersion === "number"
  ) {
    const reconstructed = buildNativeOwnerFromDriverActiveLease({
      driverId: lease.driverId,
      sessionGenerationId: lease.sessionGenerationId,
      trackingGenerationId: lease.trackingGenerationId,
      trackingIdentityId: lease.trackingIdentityId,
      missionId: lease.missionId === undefined ? null : lease.missionId,
      missionContextVersion: lease.missionContextVersion,
    });
    const reconstructedCheck = validateNativeOwnerForHeadless({
      owner: reconstructed,
      lease,
      authUsable,
    });
    if (reconstructedCheck.ok) {
      if (
        runtimeOwner &&
        runtimeOwner.trackingGenerationId !== reconstructed.trackingGenerationId
      ) {
        return { ok: false, reason: "missing_native_owner" };
      }
      return { ok: true, owner: reconstructed, source: "lease_reconstructed" };
    }
  }

  return { ok: false, reason: "missing_native_owner" };
}

/**
 * Flush durable : génération de capture non requise ; identité + non-quarantaine oui.
 */
export function canFlushDurableEvent(params: {
  trackingIdentityId: string;
  partitionQuarantined: boolean;
}): boolean {
  if (params.partitionQuarantined) return false;
  const availability = getTrackingAuthAvailability();
  if (availability.kind === "SESSION_AVAILABLE") {
    return availability.trackingIdentityId === params.trackingIdentityId;
  }
  // Reauth temporaire : autoriser si même identité que le runtime ou snapshot précédent.
  if (availability.kind === "AUTH_TEMPORARILY_UNAVAILABLE") {
    return (
      activeRuntime?.identity.trackingIdentityId === params.trackingIdentityId ||
      false
    );
  }
  return false;
}

function onTerminalEvent(event: TrackingAuthTerminalEvent): void {
  const runtime = activeRuntime;
  if (!runtime) return;

  if (event.kind === "EXPLICIT_LOGOUT") {
    const policy = TRACKING_AUTH_EFFECT_POLICY.explicit_logout;
    if (policy.stop) {
      void stopTrackingRuntime({
        expectedTrackingGenerationId: runtime.identity.trackingGenerationId,
        reason: "explicit_logout",
        quarantinePolicy: policy.quarantine ? "identity_partition" : "none",
      });
    }
    return;
  }
  if (event.kind === "ACCOUNT_REVOKED") {
    const policy = TRACKING_AUTH_EFFECT_POLICY.account_revoked;
    if (policy.stop) {
      void stopTrackingRuntime({
        expectedTrackingGenerationId: runtime.identity.trackingGenerationId,
        reason: "account_revoked",
        quarantinePolicy: policy.quarantine ? "identity_partition" : "none",
      });
    }
    return;
  }
  if (event.kind === "IDENTITY_CHANGED") {
    void stopTrackingRuntime({
      expectedTrackingGenerationId: runtime.identity.trackingGenerationId,
      reason: "identity_changed",
      quarantinePolicy: "identity_partition",
    });
  }
}

export function ensureTrackingAuthTerminalSubscription(): void {
  if (terminalUnsub) return;
  terminalUnsub = subscribeToTrackingAuthTerminalEvents(onTerminalEvent);
}

/** Tests uniquement. */
export function __resetTrackingRuntimeRegistryForTests(): void {
  activeRuntime = null;
  startTrackingInFlight = null;
  physicalStopCallback = null;
  if (terminalUnsub) {
    terminalUnsub();
    terminalUnsub = null;
  }
}
