/**
 * Stockage strict des credentials auth natifs (PR C0).
 * Aucun fallback mémoire / AsyncStorage / localStorage sur native.
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";

const RECOVERY_KEY = "@atmr/auth/recovery_credential";
const REFRESH_KEY = "@atmr/auth/refresh_token";
const INSTALLATION_KEY = "@atmr/auth/installation_id";
const TOMBSTONE_KEY = "@atmr/auth/revocation_tombstone";
const ENVELOPE_KEY = "@atmr/auth/session_envelope";

export type SecureCredentialReadResult =
  | { status: "found"; value: string }
  | { status: "missing" }
  | { status: "temporarily_unavailable"; cause: string }
  | { status: "permanently_invalidated"; cause: string };

export type SecureCredentialWriteResult =
  | { status: "ok" }
  | { status: "temporarily_unavailable"; cause: string }
  | { status: "failed"; cause: string };

const NATIVE_OPTIONS: SecureStore.SecureStoreOptions = {
  keychainAccessible: SecureStore.AFTER_FIRST_UNLOCK,
  requireAuthentication: false,
};

function isNative(): boolean {
  return Platform.OS === "ios" || Platform.OS === "android";
}

async function nativeGet(key: string): Promise<SecureCredentialReadResult> {
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

const webMemory = new Map<string, string>();

export async function readRecoveryCredential(): Promise<SecureCredentialReadResult> {
  return nativeGet(RECOVERY_KEY);
}

export async function writeRecoveryCredential(value: string): Promise<SecureCredentialWriteResult> {
  return nativeSet(RECOVERY_KEY, value);
}

export async function deleteRecoveryCredential(): Promise<SecureCredentialWriteResult> {
  return nativeDelete(RECOVERY_KEY);
}

export async function readRefreshToken(): Promise<SecureCredentialReadResult> {
  return nativeGet(REFRESH_KEY);
}

export async function writeRefreshToken(value: string): Promise<SecureCredentialWriteResult> {
  return nativeSet(REFRESH_KEY, value);
}

export async function deleteRefreshToken(): Promise<SecureCredentialWriteResult> {
  return nativeDelete(REFRESH_KEY);
}

export async function readInstallationId(): Promise<SecureCredentialReadResult> {
  return nativeGet(INSTALLATION_KEY);
}

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
  return { status: "found", value: generated };
}

export type RevocationTombstone = {
  operation: "revoke_session";
  session_id: string;
  device_installation_id: string;
  revocation_secret: string;
  created_at: string;
};

export async function readRevocationTombstone(): Promise<
  | { status: "found"; value: RevocationTombstone }
  | { status: "missing" }
  | { status: "temporarily_unavailable"; cause: string }
> {
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
};

export async function readSessionEnvelope(): Promise<
  | { status: "found"; value: SessionEnvelope }
  | { status: "missing" }
  | { status: "temporarily_unavailable"; cause: string }
> {
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
