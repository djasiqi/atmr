/**
 * Coordinateur de récupération auth (F2 / PR2) :
 * cold start offline, recovery REST, file PendingRevocation, logout crash-safe,
 * révocation terminale avec preuve durable.
 */
import type { AuthContext, BootstrapResponse } from "../contracts/auth";
import {
  appendPendingRevocation,
  bumpSessionGeneration,
  clearLocalAuthCredentialsLocked,
  deletePendingRevocationIfOperationMatches,
  getSessionGenerationId,
  isCurrentSessionGeneration,
  persistTerminalRevocationEvidenceLocked,
  readInstallationId,
  readPendingRevocations,
  readRecoveryCredential,
  readRefreshToken,
  readSessionEnvelope,
  writeSessionEnvelope,
  type PendingRevocation,
  type SessionEnvelope,
  type SessionGenerationId,
} from "./authCredentialStore";
import {
  claimNextSessionGenerationIfCurrent,
  withCredentialStoreLock,
  withSessionCredentialMutation,
} from "./sessionCredentialMutex";
import { newLifecycleOperationId } from "./sessionLifecycle";
import { appendSessionJournalEvent } from "../observability/sessionJournal";
import {
  refreshAuthTokenNow,
  revokeSessionPending,
  sessionResumeRequest,
  setAuthToken,
} from "../api/client";
import { getNetworkSnapshot } from "../network/networkState";

const TERMINAL_ERROR_CODES = new Set(["session_revoked", "refresh_replay_detected", "account_disabled"]);
const KEEP_LOCAL_ERROR_CODES = new Set([
  "refresh_store_unavailable",
  "session_validation_unavailable",
]);
const ROTATION_RECOVERY_CODE = "rotation_recovery_required";

export type AuthErrorClass = "terminal" | "keep_local" | "rotation_recovery" | "unknown";

export function classifyAuthErrorCode(code: string | null | undefined): AuthErrorClass {
  if (!code) return "unknown";
  if (TERMINAL_ERROR_CODES.has(code)) return "terminal";
  if (KEEP_LOCAL_ERROR_CODES.has(code)) return "keep_local";
  if (code === ROTATION_RECOVERY_CODE) return "rotation_recovery";
  return "unknown";
}

export type RecoveryOutcome = "recovered" | "terminal" | "keep_local" | "no_action";

export async function attemptRestRecovery(reason: string): Promise<RecoveryOutcome> {
  void appendSessionJournalEvent("auth.recovery.attempt", { reason });
  const refreshed = await refreshAuthTokenNow();
  if (refreshed) {
    void appendSessionJournalEvent("auth.recovery.success", { reason, via: "refresh" });
    return "recovered";
  }

  const { getLastRefreshErrorCode } = await import("../api/client");
  const refreshErrorClass = classifyAuthErrorCode(getLastRefreshErrorCode());
  if (refreshErrorClass === "terminal") {
    void appendSessionJournalEvent("auth.recovery.terminal", {
      reason,
      code: getLastRefreshErrorCode(),
    });
    return "terminal";
  }
  if (refreshErrorClass === "keep_local") {
    void appendSessionJournalEvent("auth.recovery.keep_local", {
      reason,
      code: getLastRefreshErrorCode(),
    });
    return "keep_local";
  }

  const resumeOutcome = await sessionResumeRequest();
  if (resumeOutcome.ok) {
    void appendSessionJournalEvent("auth.recovery.success", { reason, via: "session_resume" });
    return "recovered";
  }
  const resumeErrorClass = classifyAuthErrorCode(resumeOutcome.code);
  if (resumeErrorClass === "terminal") {
    void appendSessionJournalEvent("auth.recovery.terminal", { reason, code: resumeOutcome.code });
    return "terminal";
  }
  void appendSessionJournalEvent("auth.recovery.failed", { reason, code: resumeOutcome.code });
  return resumeErrorClass === "keep_local" || resumeOutcome.retryable ? "keep_local" : "no_action";
}

export type ColdStartRestoreResult =
  | {
      kind: "restored";
      activeContext: AuthContext | null;
      bootstrap: BootstrapResponse | null;
    }
  | { kind: "anonymous" }
  | { kind: "storage_locked" }
  | { kind: "revoked" }
  | { kind: "incoherent" }
  | {
      kind: "interrupted_logout";
      pending: PendingRevocation;
      envelopeSessionId: string | null;
    };

/**
 * Restauration offline.
 * PendingRevocation ≠ preuve revoked ; permanently_invalidated → revoked.
 */
export async function restoreOfflineSessionSnapshot(): Promise<ColdStartRestoreResult> {
  const refresh = await readRefreshToken();
  if (refresh.status === "temporarily_unavailable") {
    return { kind: "storage_locked" };
  }
  if (refresh.status === "permanently_invalidated") {
    return { kind: "revoked" };
  }

  const recovery = await readRecoveryCredential();
  if (recovery.status === "temporarily_unavailable") {
    return { kind: "storage_locked" };
  }
  if (recovery.status === "permanently_invalidated") {
    return { kind: "revoked" };
  }

  const pendingResult = await readPendingRevocations();
  if (pendingResult.status === "temporarily_unavailable") {
    return { kind: "storage_locked" };
  }
  const pendingList = pendingResult.status === "found" ? pendingResult.value : [];

  const envelope = await readSessionEnvelope();
  if (envelope.status === "temporarily_unavailable") {
    return { kind: "storage_locked" };
  }
  if (envelope.status === "permanently_invalidated") {
    return { kind: "revoked" };
  }

  const matchingExplicit = pendingList.find(
    (p) =>
      p.origin === "explicit_logout" &&
      envelope.status === "found" &&
      p.session_id === envelope.value.session_id
  );
  if (matchingExplicit) {
    return {
      kind: "interrupted_logout",
      pending: matchingExplicit,
      envelopeSessionId: envelope.status === "found" ? envelope.value.session_id : null,
    };
  }

  if (envelope.status !== "found") {
    return { kind: "anonymous" };
  }

  if (recovery.status !== "found") {
    return { kind: "incoherent" };
  }

  const installation = await readInstallationId();
  if (installation.status === "temporarily_unavailable") {
    return { kind: "storage_locked" };
  }
  if (installation.status !== "found") {
    return { kind: "incoherent" };
  }
  if (installation.value !== envelope.value.device_installation_id) {
    return { kind: "incoherent" };
  }

  return {
    kind: "restored",
    activeContext: envelope.value.cached_active_context ?? null,
    bootstrap: envelope.value.cached_bootstrap ?? null,
  };
}

export async function persistOfflineSnapshot(
  bootstrap: BootstrapResponse,
  activeContext: AuthContext | null
): Promise<void> {
  const existing = await readSessionEnvelope();
  if (existing.status !== "found") return;
  const next: SessionEnvelope = {
    ...existing.value,
    active_context_id: activeContext?.context_id ?? existing.value.active_context_id,
    cached_active_context: activeContext,
    cached_bootstrap: bootstrap,
    last_authenticated_at: new Date().toISOString(),
  };
  await writeSessionEnvelope(next);
}

export type ExplicitLogoutInFlight = {
  sourceGeneration: SessionGenerationId;
  sourceSessionId: string;
  lifecycleOperationId: string;
  promise: Promise<ExplicitLogoutResult>;
};

export type ExplicitLogoutResult = {
  status: "completed" | "stale_superseded" | "already_anonymous";
  logoutGeneration: SessionGenerationId | null;
  lifecycleOperationId: string | null;
};

let explicitLogoutInFlight: ExplicitLogoutInFlight | null = null;

export type PerformExplicitLogoutParams = {
  sourceGeneration: SessionGenerationId;
  sourceSessionId: string;
  lifecycleOperationId: string;
  trackingIdentity: {
    user_id: string;
    driver_id: string;
    company_id: string;
  } | null;
  quarantineRequired: boolean;
  /** Appelé sync juste après claim réussi (deny auto-bootstrap + logging_out). */
  onLogoutClaimed?: (logoutGeneration: SessionGenerationId) => void;
  /** Quarantaine GPS — hors mutex, avant réseau. */
  runQuarantine: (args: {
    identity: {
      userId: string;
      driverId: string;
      companyId: string;
    };
    lifecycleOperationId: string;
  }) => Promise<void>;
  /** Compensation si logout devenu stale après quarantine. */
  clearQuarantineIfOperationMatches: (lifecycleOperationId: string) => Promise<void>;
  /**
   * Commit UI synchrone sous garde de génération (après purge credentials).
   * Retourne true si appliqué.
   */
  commitSessionStateIfCurrent: (logoutGeneration: SessionGenerationId) => boolean;
};

/**
 * Logout explicite single-flight par (session_id, sourceGeneration).
 * Ordre : claim → pending+local_cleanup → quarantine → réseau → purge+commit UI.
 */
export async function performExplicitLogout(
  params: PerformExplicitLogoutParams
): Promise<ExplicitLogoutResult> {
  const current = explicitLogoutInFlight;
  if (
    current &&
    current.sourceSessionId === params.sourceSessionId &&
    current.sourceGeneration === params.sourceGeneration
  ) {
    return current.promise;
  }

  const promise = runExplicitLogout(params).finally(() => {
    if (explicitLogoutInFlight?.promise === promise) {
      explicitLogoutInFlight = null;
    }
  });

  explicitLogoutInFlight = {
    sourceGeneration: params.sourceGeneration,
    sourceSessionId: params.sourceSessionId,
    lifecycleOperationId: params.lifecycleOperationId,
    promise,
  };

  return promise;
}

async function runExplicitLogout(
  params: PerformExplicitLogoutParams
): Promise<ExplicitLogoutResult> {
  const claim = await claimNextSessionGenerationIfCurrent(params.sourceGeneration);
  if (claim.status === "stale") {
    return {
      status: "stale_superseded",
      logoutGeneration: null,
      lifecycleOperationId: params.lifecycleOperationId,
    };
  }
  const logoutGeneration = claim.generation;
  params.onLogoutClaimed?.(logoutGeneration);

  const envelope = await readSessionEnvelope();
  const revocationSecret =
    envelope.status === "found" ? envelope.value.revocation_secret ?? null : null;
  const deviceInstallationId =
    envelope.status === "found"
      ? envelope.value.device_installation_id
      : params.sourceSessionId;

  let pending: PendingRevocation | null = null;
  if (revocationSecret) {
    pending = {
      operation_id: params.lifecycleOperationId,
      session_id: params.sourceSessionId,
      device_installation_id: deviceInstallationId,
      revocation_secret: revocationSecret,
      created_at: new Date().toISOString(),
      origin: "explicit_logout",
      local_cleanup: {
        tracking_identity: params.trackingIdentity,
        quarantine_required: params.quarantineRequired,
      },
    };
    await withCredentialStoreLock(async () => {
      await appendPendingRevocation(pending!);
    });
    void appendSessionJournalEvent("auth.revocation.pending_written", {
      session_id: pending.session_id,
      origin: pending.origin,
      before_network: true,
      before_quarantine: true,
    });
  }

  if (
    params.quarantineRequired &&
    params.trackingIdentity
  ) {
    await params
      .runQuarantine({
        identity: {
          userId: params.trackingIdentity.user_id,
          driverId: params.trackingIdentity.driver_id,
          companyId: params.trackingIdentity.company_id,
        },
        lifecycleOperationId: params.lifecycleOperationId,
      })
      .catch(() => undefined);
  }

  if (!isCurrentSessionGeneration(logoutGeneration)) {
    await params
      .clearQuarantineIfOperationMatches(params.lifecycleOperationId)
      .catch(() => undefined);
    // La révocation réseau de A continue même si B a supersédé
    if (pending) {
      void flushSinglePendingRevocation(pending);
    }
    return {
      status: "stale_superseded",
      logoutGeneration,
      lifecycleOperationId: params.lifecycleOperationId,
    };
  }

  let ack = false;
  if (pending) {
    try {
      ack = await revokeSessionPending(
        pending.session_id,
        pending.revocation_secret,
        pending.operation_id
      );
    } catch {
      ack = false;
    }
  } else {
    try {
      const { logoutSession } = await import("../api/client");
      await logoutSession({ skipLocalPurge: true });
      ack = true;
    } catch {
      ack = false;
    }
  }

  const mutation = await withSessionCredentialMutation(logoutGeneration, async () => {
    setAuthToken(null);
    await clearLocalAuthCredentialsLocked();
    if (ack && pending) {
      await deletePendingRevocationIfOperationMatches(pending.operation_id);
      void appendSessionJournalEvent("auth.revocation.pending_acked", {
        session_id: pending.session_id,
      });
    } else if (pending) {
      void appendSessionJournalEvent("auth.revocation.pending_retained", {
        session_id: pending.session_id,
        network_connected: getNetworkSnapshot().connected,
      });
    }
    const committed = params.commitSessionStateIfCurrent(logoutGeneration);
    return committed;
  });

  if (mutation.status === "stale") {
    if (pending) {
      void flushSinglePendingRevocation(pending);
    }
    return {
      status: "stale_superseded",
      logoutGeneration,
      lifecycleOperationId: params.lifecycleOperationId,
    };
  }

  return {
    status: "completed",
    logoutGeneration,
    lifecycleOperationId: params.lifecycleOperationId,
  };
}

/**
 * Compat : logout sans callbacks UI (tests / chemins legacy).
 * Préférer performExplicitLogout depuis SessionProvider.
 */
export async function performLogout(): Promise<void> {
  const sourceGeneration = getSessionGenerationId();
  const envelope = await readSessionEnvelope();
  const sourceSessionId =
    envelope.status === "found" ? envelope.value.session_id : `anon-${sourceGeneration}`;
  await performExplicitLogout({
    sourceGeneration,
    sourceSessionId,
    lifecycleOperationId: newLifecycleOperationId(),
    trackingIdentity: null,
    quarantineRequired: false,
    runQuarantine: async () => undefined,
    clearQuarantineIfOperationMatches: async () => undefined,
    commitSessionStateIfCurrent: () => true,
  });
}

async function flushSinglePendingRevocation(pending: PendingRevocation): Promise<boolean> {
  try {
    const ok = await revokeSessionPending(
      pending.session_id,
      pending.revocation_secret,
      pending.operation_id
    );
    if (ok) {
      await withCredentialStoreLock(async () => {
        await deletePendingRevocationIfOperationMatches(pending.operation_id);
      });
      void appendSessionJournalEvent("auth.revocation.pending_flushed", {
        session_id: pending.session_id,
        origin: pending.origin,
      });
    }
    return ok;
  } catch {
    return false;
  }
}

/** Flush de toute la file pending (cold start / retour réseau) — idempotent. */
export async function flushPendingRevocationTombstone(): Promise<boolean> {
  const result = await readPendingRevocations();
  if (result.status === "temporarily_unavailable") return false;
  if (result.status !== "found" || result.value.length === 0) return true;
  let allOk = true;
  for (const pending of result.value) {
    const ok = await flushSinglePendingRevocation(pending);
    if (!ok) allOk = false;
  }
  return allOk;
}

/**
 * Termine un logout interrompu (crash après pending, avant/après quarantine).
 * Résultat UI : anonymous — jamais revoked.
 */
export async function finishInterruptedExplicitLogout(
  pending: PendingRevocation,
  opts: {
    runQuarantine: PerformExplicitLogoutParams["runQuarantine"];
  }
): Promise<void> {
  const cleanup = pending.local_cleanup;
  if (cleanup?.quarantine_required && cleanup.tracking_identity) {
    await opts
      .runQuarantine({
        identity: {
          userId: cleanup.tracking_identity.user_id,
          driverId: cleanup.tracking_identity.driver_id,
          companyId: cleanup.tracking_identity.company_id,
        },
        lifecycleOperationId: pending.operation_id,
      })
      .catch(() => undefined);
  }

  await withCredentialStoreLock(async () => {
    setAuthToken(null);
    await clearLocalAuthCredentialsLocked();
  });

  void flushSinglePendingRevocation(pending);
  void appendSessionJournalEvent("auth.logout.interrupted_finished", {
    session_id: pending.session_id,
    operation_id: pending.operation_id,
  });
}

/**
 * Révocation terminale : compare-and-bump atomique + preuve invalidate* avant cleanup.
 */
export async function applyTerminalRevocationIfCurrent(
  sourceGeneration: SessionGenerationId,
  reason: string,
  commitSessionStateIfCurrent: (terminalGeneration: SessionGenerationId) => boolean
): Promise<"applied" | "stale"> {
  return withCredentialStoreLock(async () => {
    if (!isCurrentSessionGeneration(sourceGeneration)) {
      return "stale";
    }
    const terminalGeneration = bumpSessionGeneration();
    await persistTerminalRevocationEvidenceLocked(reason);
    setAuthToken(null);
    const committed = commitSessionStateIfCurrent(terminalGeneration);
    void appendSessionJournalEvent("auth.terminal_revocation.applied", {
      reason,
      committed,
      generation: terminalGeneration,
    });
    return "applied";
  });
}

/** Enqueue révocation orpheline (login stale) — store lock, pas de garde de génération. */
export async function enqueueOrphanedLoginRevocation(args: {
  sessionId: string;
  deviceInstallationId: string;
  revocationSecret: string;
  operationId?: string;
}): Promise<PendingRevocation> {
  const pending: PendingRevocation = {
    operation_id: args.operationId ?? newLifecycleOperationId(),
    session_id: args.sessionId,
    device_installation_id: args.deviceInstallationId,
    revocation_secret: args.revocationSecret,
    created_at: new Date().toISOString(),
    origin: "orphaned_login_cleanup",
  };
  await withCredentialStoreLock(async () => {
    await appendPendingRevocation(pending);
  });
  void appendSessionJournalEvent("auth.revocation.orphan_enqueued", {
    session_id: pending.session_id,
  });
  return pending;
}

/** Flush background d'un orphan — ne mute pas la session courante. */
export function flushOrphanedLoginRevocationInBackground(
  pending: PendingRevocation
): void {
  void flushSinglePendingRevocation(pending);
}

/**
 * Indique s'il reste des pending (flush).
 * Ne bloque PAS un nouveau login : pending ≠ preuve terminale.
 */
export async function hasPendingRevocationTombstone(): Promise<boolean> {
  const t = await readPendingRevocations();
  return t.status === "found" && t.value.length > 0;
}

/** Réservé aux tests. */
export function __resetExplicitLogoutInFlightForTests(): void {
  explicitLogoutInFlight = null;
}
