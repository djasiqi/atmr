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
const ENVELOPE_KEY = "atmr.auth.session_envelope";

/** Exporté pour tests de non-régression (format de clé SecureStore). */
export const AUTH_SECURE_STORE_KEYS = [
  RECOVERY_KEY,
  RECOVERY_TOMBSTONE_KEY,
  REFRESH_KEY,
  REFRESH_TOMBSTONE_KEY,
  INSTALLATION_KEY,
  TOMBSTONE_KEY,
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
    void nativeDelete(tombstoneKey);
  }
  return written;
}

async function deleteAndClearInvalidationMarker(
  key: string,
  tombstoneKey: string
): Promise<SecureCredentialWriteResult> {
  const deleted = await nativeDelete(key);
  void nativeDelete(tombstoneKey);
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

export type RevocationTombstone = {
  operation: "revoke_session";
  operation_id: string;
  session_id: string;
  device_installation_id: string;
  revocation_secret: string;
  created_at: string;
};

export async function readRevocationTombstone(): Promise<TypedCredentialReadResult<RevocationTombstone>> {
  const raw = await nativeGet(TOMBSTONE_KEY);
  if (raw.status !== "found") return raw;
  try {
    return { status: "found", value: JSON.parse(raw.value) as RevocationTombstone };
  } catch {
    return { status: "permanently_invalidated", cause: "tombstone_parse_error" };
  }
}

export async function writeRevocationTombstone(
  tombstone: RevocationTombstone
): Promise<SecureCredentialWriteResult> {
  return nativeSet(TOMBSTONE_KEY, JSON.stringify(tombstone));
}

export async function deleteRevocationTombstone(): Promise<SecureCredentialWriteResult> {
  return nativeDelete(TOMBSTONE_KEY);
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

/** Epoch auth monotone — partagé avec client / sessionProvider / realtime. */
let authEpoch = 0;

export function bumpAuthEpoch(): number {
  authEpoch += 1;
  return authEpoch;
}

export function getAuthEpoch(): number {
  return authEpoch;
}

export function isCurrentAuthEpoch(epoch: number): boolean {
  return epoch === authEpoch;
}
