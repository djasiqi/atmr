/**
 * Coordinateur de récupération auth (PR C) : classification des codes d'erreur
 * durable-session, tentative de récupération REST (refresh puis session-resume),
 * et purge/flush de la révocation en attente (logout hors-ligne).
 */
import type { AuthContext, BootstrapResponse } from "../contracts/auth";
import {
  deleteRecoveryCredential,
  deleteRevocationTombstone,
  deleteSessionEnvelope,
  readRevocationTombstone,
  readSessionEnvelope,
  writeRevocationTombstone,
  writeSessionEnvelope,
  type SessionEnvelope,
} from "./authCredentialStore";
import { appendSessionJournalEvent } from "../observability/sessionJournal";
import {
  getLastRefreshErrorCode,
  logoutSession,
  refreshAuthTokenNow,
  revokeSessionPending,
  sessionResumeRequest,
} from "../api/client";
import { getNetworkSnapshot } from "../network/networkState";

/** Codes terminaux : la session durable est morte, il faut une reconnexion explicite. */
const TERMINAL_ERROR_CODES = new Set(["session_revoked", "refresh_replay_detected"]);
/** Codes transitoires : le serveur est temporairement indisponible, on garde l'état local. */
const KEEP_LOCAL_ERROR_CODES = new Set(["refresh_store_unavailable", "session_validation_unavailable"]);
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

/**
 * Tente de récupérer une session locale valide : refresh classique d'abord,
 * puis reprise dédiée `/auth/session-resume` (recovery credential) si le refresh
 * échoue avec un code suggérant une rotation/reprise (ou est inconnu).
 * Ne purge jamais l'état local elle-même — c'est à l'appelant de décider (cf. sessionProvider).
 */
export async function attemptRestRecovery(reason: string): Promise<RecoveryOutcome> {
  void appendSessionJournalEvent("auth.recovery.attempt", { reason });
  const refreshed = await refreshAuthTokenNow();
  if (refreshed) {
    void appendSessionJournalEvent("auth.recovery.success", { reason, via: "refresh" });
    return "recovered";
  }

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

  // rotation_recovery_required ou code inconnu : tenter la reprise dédiée par recovery credential.
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
  | { kind: "restored"; activeContext: AuthContext | null; bootstrap: BootstrapResponse | null }
  | { kind: "anonymous" }
  | { kind: "storage_locked" };

/**
 * Lit installation_id + enveloppe de session + recovery credential pour restaurer
 * immédiatement l'UI en mode `authenticated_offline`, avant tout appel réseau (cf. PR C, point 4).
 */
export async function restoreOfflineSessionSnapshot(): Promise<ColdStartRestoreResult> {
  const envelope = await readSessionEnvelope();
  if (envelope.status === "temporarily_unavailable") {
    return { kind: "storage_locked" };
  }
  if (envelope.status !== "found") {
    return { kind: "anonymous" };
  }
  return {
    kind: "restored",
    activeContext: envelope.value.cached_active_context ?? null,
    bootstrap: envelope.value.cached_bootstrap ?? null,
  };
}

/** Persiste un instantané hors-ligne (contexte actif + bootstrap) réutilisable au prochain cold start. */
export async function persistOfflineSnapshot(
  bootstrap: BootstrapResponse,
  activeContext: AuthContext | null
): Promise<void> {
  const existing = await readSessionEnvelope();
  if (existing.status !== "found") return; // pas de session durable (legacy) : rien à mettre à jour
  const next: SessionEnvelope = {
    ...existing.value,
    active_context_id: activeContext?.context_id ?? existing.value.active_context_id,
    cached_active_context: activeContext,
    cached_bootstrap: bootstrap,
    last_authenticated_at: new Date().toISOString(),
  };
  await writeSessionEnvelope(next);
}

/**
 * Logout hors-ligne (point 8) : purge locale immédiate + tombstone de révocation
 * à flusher dès que le réseau revient, avant tout nouveau login.
 */
export async function performLogout(): Promise<void> {
  const envelope = await readSessionEnvelope();
  try {
    await logoutSession();
  } catch {
    // Best-effort : logoutSession purge déjà l'access/refresh local dans son `finally`.
  }
  const offline = !getNetworkSnapshot().connected;
  if (
    offline &&
    envelope.status === "found" &&
    envelope.value.revocation_secret &&
    envelope.value.session_id
  ) {
    await writeRevocationTombstone({
      operation: "revoke_session",
      session_id: envelope.value.session_id,
      device_installation_id: envelope.value.device_installation_id,
      revocation_secret: envelope.value.revocation_secret,
      created_at: new Date().toISOString(),
    });
    void appendSessionJournalEvent("auth.revocation.tombstone_written", {
      session_id: envelope.value.session_id,
    });
  }
  await deleteRecoveryCredential();
  await deleteSessionEnvelope();
}

/** À appeler au retour réseau (cold start / bootstrap) avant tout nouveau login. */
export async function flushPendingRevocationTombstone(): Promise<void> {
  const tombstone = await readRevocationTombstone();
  if (tombstone.status !== "found") return;
  const ok = await revokeSessionPending(
    tombstone.value.session_id,
    tombstone.value.revocation_secret
  );
  if (ok) {
    await deleteRevocationTombstone();
    void appendSessionJournalEvent("auth.revocation.tombstone_flushed", {
      session_id: tombstone.value.session_id,
    });
  }
}
