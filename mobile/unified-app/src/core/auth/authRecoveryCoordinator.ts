/**
 * Coordinateur de récupération auth (F2) : cold start offline, recovery REST,
 * logout crash-safe (tombstone avant réseau, purge avant ACK).
 */
import type { AuthContext, BootstrapResponse } from "../contracts/auth";
import {
  bumpAuthEpoch,
  deleteRecoveryCredential,
  deleteRefreshToken,
  deleteRevocationTombstone,
  deleteSessionEnvelope,
  readInstallationId,
  readRecoveryCredential,
  readRevocationTombstone,
  readSessionEnvelope,
  writeRevocationTombstone,
  writeSessionEnvelope,
  type RevocationTombstone,
  type SessionEnvelope,
} from "./authCredentialStore";
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
  | { kind: "incoherent" };

/**
 * Restauration offline : envelope + recovery + installation_id match + pas de tombstone.
 */
export async function restoreOfflineSessionSnapshot(): Promise<ColdStartRestoreResult> {
  const tombstone = await readRevocationTombstone();
  if (tombstone.status === "found") {
    return { kind: "revoked" };
  }
  if (tombstone.status === "temporarily_unavailable") {
    return { kind: "storage_locked" };
  }

  const envelope = await readSessionEnvelope();
  if (envelope.status === "temporarily_unavailable") {
    return { kind: "storage_locked" };
  }
  if (envelope.status === "permanently_invalidated") {
    return { kind: "revoked" };
  }
  if (envelope.status !== "found") {
    return { kind: "anonymous" };
  }

  const recovery = await readRecoveryCredential();
  if (recovery.status === "temporarily_unavailable") {
    return { kind: "storage_locked" };
  }
  if (recovery.status === "permanently_invalidated") {
    return { kind: "revoked" };
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
  // Enveloppe petite : ne pas stocker de grandes listes missions — bootstrap minimal ok
  const next: SessionEnvelope = {
    ...existing.value,
    active_context_id: activeContext?.context_id ?? existing.value.active_context_id,
    cached_active_context: activeContext,
    cached_bootstrap: bootstrap,
    last_authenticated_at: new Date().toISOString(),
  };
  await writeSessionEnvelope(next);
}

function newOperationId(): string {
  return `op-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

/**
 * Logout crash-safe :
 * 1 bumpAuthEpoch
 * 2 lire session_id + revocation_secret
 * 3 tombstone AVANT réseau
 * 4 tenter révocation
 * 5 purger credentials locaux
 * 6 ACK → supprimer tombstone ; sinon conserver
 * 7 anonymous
 */
export async function performLogout(): Promise<void> {
  bumpAuthEpoch();
  const envelope = await readSessionEnvelope();
  let tombstone: RevocationTombstone | null = null;

  if (
    envelope.status === "found" &&
    envelope.value.revocation_secret &&
    envelope.value.session_id
  ) {
    tombstone = {
      operation: "revoke_session",
      operation_id: newOperationId(),
      session_id: envelope.value.session_id,
      device_installation_id: envelope.value.device_installation_id,
      revocation_secret: envelope.value.revocation_secret,
      created_at: new Date().toISOString(),
    };
    await writeRevocationTombstone(tombstone);
    void appendSessionJournalEvent("auth.revocation.tombstone_written", {
      session_id: tombstone.session_id,
      before_network: true,
    });
  }

  let ack = false;
  if (tombstone) {
    try {
      ack = await revokeSessionPending(
        tombstone.session_id,
        tombstone.revocation_secret,
        tombstone.operation_id
      );
    } catch {
      ack = false;
    }
  } else {
    // Legacy : tenter logout HTTP avec access/refresh si présents
    try {
      const { logoutSession } = await import("../api/client");
      await logoutSession({ skipLocalPurge: true });
      ack = true;
    } catch {
      ack = false;
    }
  }

  // Purge locale AVANT suppression du tombstone
  setAuthToken(null);
  await deleteRefreshToken();
  await deleteRecoveryCredential();
  await deleteSessionEnvelope();
  try {
    await (
      await import("@react-native-async-storage/async-storage")
    ).default.removeItem("@atmr/auth/pending_refresh_operation");
  } catch {
    /* ignore */
  }

  if (ack && tombstone) {
    await deleteRevocationTombstone();
    void appendSessionJournalEvent("auth.revocation.tombstone_acked", {
      session_id: tombstone.session_id,
    });
  } else if (tombstone) {
    void appendSessionJournalEvent("auth.revocation.tombstone_retained", {
      session_id: tombstone.session_id,
      network_connected: getNetworkSnapshot().connected,
    });
  }
}

/** Flush tombstone pending (cold start / retour réseau) — idempotent. */
export async function flushPendingRevocationTombstone(): Promise<boolean> {
  const tombstone = await readRevocationTombstone();
  if (tombstone.status === "temporarily_unavailable") return false;
  if (tombstone.status !== "found") return true;
  const ok = await revokeSessionPending(
    tombstone.value.session_id,
    tombstone.value.revocation_secret,
    tombstone.value.operation_id
  );
  if (ok) {
    await deleteRevocationTombstone();
    void appendSessionJournalEvent("auth.revocation.tombstone_flushed", {
      session_id: tombstone.value.session_id,
    });
  }
  return ok;
}

/** Bloque un nouveau login si un tombstone est en attente. */
export async function hasPendingRevocationTombstone(): Promise<boolean> {
  const t = await readRevocationTombstone();
  return t.status === "found";
}
