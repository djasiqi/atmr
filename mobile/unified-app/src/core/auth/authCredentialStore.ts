/**
 * Stockage strict des credentials auth natifs (PR C0).
 * Aucun fallback mémoire / AsyncStorage / localStorage sur native.
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import type { AuthContext, BootstrapResponse } from "../contracts/auth";

/**
 * Clés SecureStore : uniquement [A-Za-z0-9._-] (regex expo-secure-store /^[\w.-]+$/).
 * Les anciennes clés `@atmr/auth/...` (avec `@` et `/`) étaient rejetées systématiquement
 * → DEVICE_ID_UNAVAILABLE / STORAGE_UNAVAILABLE sur login mobile.
 */
const RECOVERY_KEY = "atmr.auth.recovery_credential";
const RECOVERY_TOMBSTONE_KEY = "atmr.auth.recovery_credential_tombstone";
const REFRESH_KEY = "atmr.auth.refresh_token";
const REFRESH_TOMBSTONE_KEY = "atmr.auth.refresh_token_tombstone";
const INSTALLATION_KEY = "atmr.auth.installation_id";
const TOMBSTONE_KEY = "atmr.auth.revocation_tombstone";
const PENDING_REVOCATIONS_KEY = "atmr.auth.pending_revocations";
const ENVELOPE_KEY = "atmr.auth.session_envelope";

/** Exporté pour tests de non-régression (format de clé SecureStore). */
export const AUTH_SECURE_STORE_KEYS = [
  RECOVERY_KEY,
  RECOVERY_TOMBSTONE_KEY,
  REFRESH_KEY,
  REFRESH_TOMBSTONE_KEY,
  INSTALLATION_KEY,
  TOMBSTONE_KEY,
  PENDING_REVOCATIONS_KEY,
  ENVELOPE_KEY,
] as const;

export type SecureCredentialReadResult =
  | { status: "found"; value: string }
  | { status: "missing" }
  | { status: "temporarily_unavailable"; cause: string }
  | { status: "permanently_invalidated"; cause: string };

/** Variante typée pour les valeurs JSON (enveloppe de session, tombstone de révocation). */
export type TypedCredentialReadResult<T> =
  | { status: "found"; value: T }
  | { status: "missing" }
  | { status: "temporarily_unavailable"; cause: string }
  | { status: "permanently_invalidated"; cause: string };

export type SecureCredentialWriteResult =
  | { status: "ok" }
  | { status: "temporarily_unavailable"; cause: string }
  | { status: "failed"; cause: string };

/** Options iOS uniquement — keychainAccessible n'a pas d'effet Android et peut perturber. */
const NATIVE_OPTIONS: SecureStore.SecureStoreOptions =
  Platform.OS === "ios"
    ? {
        keychainAccessible: SecureStore.AFTER_FIRST_UNLOCK,
        requireAuthentication: false,
      }
    : {
        requireAuthentication: false,
      };

function assertSecureStoreKey(key: string): void {
  if (!/^[\w.-]+$/.test(key)) {
    throw new Error(
      `Invalid SecureStore key "${key}". Only alphanumeric, ".", "-" and "_" are allowed.`
    );
  }
}

function isNative(): boolean {
  return Platform.OS === "ios" || Platform.OS === "android";
}

const webMemory = new Map<string, string>();

async function nativeGet(key: string): Promise<SecureCredentialReadResult> {
  assertSecureStoreKey(key);
  if (!isNative()) {
    // Web / tests : stockage mémoire isolé (pas de credentials prod).
    const mem = webMemory.get(key);
    if (typeof mem === "string" && mem.length > 0) return { status: "found", value: mem };
    return { status: "missing" };
  }
  try {
    const value = await SecureStore.getItemAsync(key, NATIVE_OPTIONS);
    if (value == null || value === "") return { status: "missing" };
    return { status: "found", value };
  } catch (err) {
    const cause = err instanceof Error ? err.message : String(err);
    return { status: "temporarily_unavailable", cause };
  }
}

async function nativeSet(key: string, value: string): Promise<SecureCredentialWriteResult> {
  assertSecureStoreKey(key);
  if (!isNative()) {
    webMemory.set(key, value);
    return { status: "ok" };
  }
  try {
    await SecureStore.setItemAsync(key, value, NATIVE_OPTIONS);
    const readBack = await SecureStore.getItemAsync(key, NATIVE_OPTIONS);
    if (readBack !== value) {
      return { status: "failed", cause: "read_back_mismatch" };
    }
    return { status: "ok" };
  } catch (err) {
    const cause = err instanceof Error ? err.message : String(err);
    return { status: "temporarily_unavailable", cause };
  }
}

async function nativeDelete(key: string): Promise<SecureCredentialWriteResult> {
  assertSecureStoreKey(key);
  if (!isNative()) {
    webMemory.delete(key);
    return { status: "ok" };
  }
  try {
    await SecureStore.deleteItemAsync(key, NATIVE_OPTIONS);
    return { status: "ok" };
  } catch (err) {
    const cause = err instanceof Error ? err.message : String(err);
    return { status: "temporarily_unavailable", cause };
  }
}

/**
 * Lecture d'un credential révocable : distingue "jamais écrit" (missing) de
 * "explicitement invalidé" (permanently_invalidated, ex. session_revoked / refresh_replay_detected).
 * Le marqueur d'invalidation est purgé au prochain write réussi.
 */
async function readWithInvalidationMarker(
  key: string,
  tombstoneKey: string
): Promise<SecureCredentialReadResult> {
  const raw = await nativeGet(key);
  if (raw.status !== "missing") return raw;
  const marker = await nativeGet(tombstoneKey);
  if (marker.status === "found") {
    return { status: "permanently_invalidated", cause: marker.value };
  }
  return raw;
}

async function writeAndClearInvalidationMarker(
  key: string,
  tombstoneKey: string,
  value: string
): Promise<SecureCredentialWriteResult> {
  const written = await nativeSet(key, value);
  if (written.status === "ok") {
    await nativeDelete(tombstoneKey);
  }
  return written;
}

async function deleteAndClearInvalidationMarker(
  key: string,
  tombstoneKey: string
): Promise<SecureCredentialWriteResult> {
  const deleted = await nativeDelete(key);
  await nativeDelete(tombstoneKey);
  return deleted;
}

/** Invalide définitivement un credential (session_revoked, refresh_replay_detected, …). */
async function invalidateCredential(
  key: string,
  tombstoneKey: string,
  reason: string
): Promise<SecureCredentialWriteResult> {
  await nativeDelete(key);
  return nativeSet(tombstoneKey, reason);
}

export async function readRecoveryCredential(): Promise<SecureCredentialReadResult> {
  return readWithInvalidationMarker(RECOVERY_KEY, RECOVERY_TOMBSTONE_KEY);
}

export async function writeRecoveryCredential(value: string): Promise<SecureCredentialWriteResult> {
  return writeAndClearInvalidationMarker(RECOVERY_KEY, RECOVERY_TOMBSTONE_KEY, value);
}

export async function deleteRecoveryCredential(): Promise<SecureCredentialWriteResult> {
  return deleteAndClearInvalidationMarker(RECOVERY_KEY, RECOVERY_TOMBSTONE_KEY);
}

export async function invalidateRecoveryCredential(reason: string): Promise<SecureCredentialWriteResult> {
  return invalidateCredential(RECOVERY_KEY, RECOVERY_TOMBSTONE_KEY, reason);
}

export async function readRefreshToken(): Promise<SecureCredentialReadResult> {
  return readWithInvalidationMarker(REFRESH_KEY, REFRESH_TOMBSTONE_KEY);
}

export async function writeRefreshToken(value: string): Promise<SecureCredentialWriteResult> {
  return writeAndClearInvalidationMarker(REFRESH_KEY, REFRESH_TOMBSTONE_KEY, value);
}

export async function deleteRefreshToken(): Promise<SecureCredentialWriteResult> {
  return deleteAndClearInvalidationMarker(REFRESH_KEY, REFRESH_TOMBSTONE_KEY);
}

export async function invalidateRefreshToken(reason: string): Promise<SecureCredentialWriteResult> {
  return invalidateCredential(REFRESH_KEY, REFRESH_TOMBSTONE_KEY, reason);
}

export async function readInstallationId(): Promise<SecureCredentialReadResult> {
  return nativeGet(INSTALLATION_KEY);
}

/** Génère + persiste + relit l'ID d'installation : n'ignore jamais un échec d'écriture. */
export async function createAndPersistInstallationId(): Promise<SecureCredentialReadResult> {
  const existing = await readInstallationId();
  if (existing.status === "found") return existing;
  if (existing.status === "temporarily_unavailable") return existing;

  const generated = `atmr-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
  const written = await nativeSet(INSTALLATION_KEY, generated);
  if (written.status !== "ok") {
    return {
      status: "temporarily_unavailable",
      cause: written.cause || "device_identity_storage_unavailable",
    };
  }
  const readBack = await readInstallationId();
  if (readBack.status !== "found" || readBack.value !== generated) {
    return {
      status: "temporarily_unavailable",
      cause: "device_identity_storage_unavailable",
    };
  }
  return readBack;
}

/** @deprecated Utiliser PendingRevocation — conservé pour migration du singleton historique. */
export type RevocationTombstone = {
  operation: "revoke_session";
  operation_id: string;
  session_id: string;
  device_installation_id: string;
  revocation_secret: string;
  created_at: string;
};

export type PendingRevocationOrigin = "explicit_logout" | "orphaned_login_cleanup";

export type PendingRevocationLocalCleanup = {
  tracking_identity: {
    user_id: string;
    driver_id: string;
    company_id: string;
  } | null;
  quarantine_required: boolean;
};

/**
 * Intention durable de révocation réseau (≠ preuve terminale revoked).
 * Une PendingRevocation appartient à une session historique ; son écriture
 * ne dépend pas du fait que sa génération soit encore courante.
 */
export type PendingRevocation = {
  operation_id: string;
  session_id: string;
  device_installation_id: string;
  revocation_secret: string;
  created_at: string;
  origin: PendingRevocationOrigin;
  local_cleanup?: PendingRevocationLocalCleanup;
};

function tombstoneToPending(t: RevocationTombstone): PendingRevocation {
  return {
    operation_id: t.operation_id,
    session_id: t.session_id,
    device_installation_id: t.device_installation_id,
    revocation_secret: t.revocation_secret,
    created_at: t.created_at,
    origin: "explicit_logout",
  };
}

async function migrateLegacyTombstoneIntoList(
  list: PendingRevocation[]
): Promise<PendingRevocation[]> {
  const legacy = await nativeGet(TOMBSTONE_KEY);
  if (legacy.status !== "found") return list;
  try {
    const parsed = JSON.parse(legacy.value) as RevocationTombstone;
    if (
      parsed &&
      typeof parsed.operation_id === "string" &&
      typeof parsed.session_id === "string" &&
      typeof parsed.revocation_secret === "string"
    ) {
      const exists = list.some((p) => p.operation_id === parsed.operation_id);
      if (!exists) {
        list = [...list, tombstoneToPending(parsed)];
      }
    }
  } catch {
    /* ignore parse error — on purge le singleton */
  }
  await nativeDelete(TOMBSTONE_KEY);
  return list;
}

async function readPendingRevocationsRaw(): Promise<PendingRevocation[]> {
  const raw = await nativeGet(PENDING_REVOCATIONS_KEY);
  let list: PendingRevocation[] = [];
  if (raw.status === "found") {
    try {
      const parsed = JSON.parse(raw.value) as unknown;
      if (Array.isArray(parsed)) {
        list = parsed.filter(
          (item): item is PendingRevocation =>
            !!item &&
            typeof item === "object" &&
            typeof (item as PendingRevocation).operation_id === "string" &&
            typeof (item as PendingRevocation).session_id === "string" &&
            typeof (item as PendingRevocation).revocation_secret === "string"
        );
      }
    } catch {
      list = [];
    }
  }
  const legacyBefore = await nativeGet(TOMBSTONE_KEY);
  const migrated = await migrateLegacyTombstoneIntoList(list);
  if (legacyBefore.status === "found" || migrated.length !== list.length) {
    await writePendingRevocationsRaw(migrated);
  }
  return migrated;
}

async function writePendingRevocationsRaw(
  list: PendingRevocation[]
): Promise<SecureCredentialWriteResult> {
  return nativeSet(PENDING_REVOCATIONS_KEY, JSON.stringify(list));
}

/** Lecture de la file (hors verrou — préférer via withCredentialStoreLock pour muter). */
export async function readPendingRevocations(): Promise<
  TypedCredentialReadResult<PendingRevocation[]>
> {
  try {
    const list = await readPendingRevocationsRaw();
    return { status: "found", value: list };
  } catch (err) {
    const cause = err instanceof Error ? err.message : String(err);
    return { status: "temporarily_unavailable", cause };
  }
}

/** Append sans écrasement — à appeler sous withCredentialStoreLock. */
export async function appendPendingRevocation(
  entry: PendingRevocation
): Promise<SecureCredentialWriteResult> {
  const list = await readPendingRevocationsRaw();
  if (list.some((p) => p.operation_id === entry.operation_id)) {
    return { status: "ok" };
  }
  list.push(entry);
  const written = await writePendingRevocationsRaw(list);
  if (written.status === "ok") {
    await nativeDelete(TOMBSTONE_KEY);
  }
  return written;
}

/** Suppression par operation_id — sous withCredentialStoreLock. */
export async function deletePendingRevocationIfOperationMatches(
  operationId: string
): Promise<SecureCredentialWriteResult> {
  const list = await readPendingRevocationsRaw();
  const next = list.filter((p) => p.operation_id !== operationId);
  if (next.length === list.length) {
    return { status: "ok" };
  }
  return writePendingRevocationsRaw(next);
}

export async function replacePendingRevocations(
  list: PendingRevocation[]
): Promise<SecureCredentialWriteResult> {
  return writePendingRevocationsRaw(list);
}

/** Compat lecture : premier pending ou singleton migré. */
export async function readRevocationTombstone(): Promise<TypedCredentialReadResult<RevocationTombstone>> {
  const listResult = await readPendingRevocations();
  if (listResult.status === "temporarily_unavailable") return listResult;
  if (listResult.status === "permanently_invalidated") return listResult;
  const first = listResult.status === "found" ? listResult.value[0] : undefined;
  if (!first) return { status: "missing" };
  return {
    status: "found",
    value: {
      operation: "revoke_session",
      operation_id: first.operation_id,
      session_id: first.session_id,
      device_installation_id: first.device_installation_id,
      revocation_secret: first.revocation_secret,
      created_at: first.created_at,
    },
  };
}

/** Compat écriture : append en file (n'écrase plus les autres). */
export async function writeRevocationTombstone(
  tombstone: RevocationTombstone
): Promise<SecureCredentialWriteResult> {
  return appendPendingRevocation(tombstoneToPending(tombstone));
}

/** Compat : supprime toute la file (préférer deletePendingRevocationIfOperationMatches). */
export async function deleteRevocationTombstone(): Promise<SecureCredentialWriteResult> {
  await nativeDelete(TOMBSTONE_KEY);
  return writePendingRevocationsRaw([]);
}

export type SessionEnvelope = {
  schema_version: number;
  session_id: string;
  device_installation_id: string;
  user_public_id: string;
  driver_id: number | null;
  role: string;
  active_context_id: string | null;
  refresh_generation: number;
  last_authenticated_at: string;
  /** Secret local (jamais transmis sauf via revoke-pending) pour la révocation hors-ligne. */
  revocation_secret?: string | null;
  /**
   * Extension locale (non contractuelle côté backend) : permet de réhydrater
   * immédiatement l'UI en mode authenticated_offline au cold start, avant tout appel réseau.
   */
  cached_active_context?: AuthContext | null;
  cached_bootstrap?: BootstrapResponse | null;
};

export async function readSessionEnvelope(): Promise<TypedCredentialReadResult<SessionEnvelope>> {
  const raw = await nativeGet(ENVELOPE_KEY);
  if (raw.status !== "found") return raw;
  try {
    return { status: "found", value: JSON.parse(raw.value) as SessionEnvelope };
  } catch {
    return { status: "permanently_invalidated", cause: "envelope_parse_error" };
  }
}

export async function writeSessionEnvelope(
  envelope: SessionEnvelope
): Promise<SecureCredentialWriteResult> {
  return nativeSet(ENVELOPE_KEY, JSON.stringify(envelope));
}

export async function deleteSessionEnvelope(): Promise<SecureCredentialWriteResult> {
  return nativeDelete(ENVELOPE_KEY);
}

/** Génération de session monotone (alias runtime de authEpoch) — process-local. */
export type SessionGenerationId = number;

let authEpoch = 0;

export function bumpAuthEpoch(): SessionGenerationId {
  authEpoch += 1;
  return authEpoch;
}

export function getAuthEpoch(): SessionGenerationId {
  return authEpoch;
}

export function isCurrentAuthEpoch(epoch: SessionGenerationId): boolean {
  return epoch === authEpoch;
}

/** Alias PR2 — même runtime que authEpoch. */
export const bumpSessionGeneration = bumpAuthEpoch;
export const getSessionGenerationId = getAuthEpoch;
export const isCurrentSessionGeneration = isCurrentAuthEpoch;

async function clearPendingRefreshOperationMarker(): Promise<void> {
  try {
    // eslint-disable-next-line @typescript-eslint/no-require-imports -- évite import() dynamique sous Jest
    const AsyncStorage = require("@react-native-async-storage/async-storage")
      .default as { removeItem: (key: string) => Promise<void> };
    await AsyncStorage.removeItem("@atmr/auth/pending_refresh_operation");
  } catch {
    /* ignore */
  }
}

/** Purge credentials locaux sans bump — appelant sous verrou / mutation de session. */
export async function clearLocalAuthCredentialsLocked(): Promise<void> {
  await deleteRefreshToken();
  await deleteRecoveryCredential();
  await deleteSessionEnvelope();
  await clearPendingRefreshOperationMarker();
}

/**
 * Preuve terminale durable puis purge enveloppe / access.
 * À appeler sous withSessionCredentialMutation après claim terminal.
 */
export async function persistTerminalRevocationEvidenceLocked(
  reason: string
): Promise<void> {
  await invalidateRefreshToken(reason);
  await invalidateRecoveryCredential(reason);
  await deleteSessionEnvelope();
  await clearPendingRefreshOperationMarker();
}

/** Réservé aux tests. */
export function __resetSessionGenerationForTests(): void {
  authEpoch = 0;
}
