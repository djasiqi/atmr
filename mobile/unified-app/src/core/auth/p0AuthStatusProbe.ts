/**
 * Instrumentation DEBUG locale P0 — statuts SecureStore uniquement.
 * Aucune valeur de credential / token / secret n'est jamais loggée ni écrite.
 * Activée uniquement si EXPO_PUBLIC_P0_VALIDATE=1.
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import {
  deleteRefreshToken,
  readInstallationId,
  readRecoveryCredential,
  readRefreshToken,
  readSessionEnvelope,
  type SecureCredentialReadResult,
} from "./authCredentialStore";

const TAG = "ATMR_P0_PROBE";
const ENABLED = process.env.EXPO_PUBLIC_P0_VALIDATE === "1";

const NATIVE_OPTIONS: SecureStore.SecureStoreOptions =
  Platform.OS === "ios"
    ? {
        keychainAccessible: SecureStore.AFTER_FIRST_UNLOCK,
        requireAuthentication: false,
      }
    : {
        requireAuthentication: false,
      };

export type P0StatusLabel =
  | "FOUND"
  | "MISSING"
  | "TEMPORARILY_UNAVAILABLE"
  | "PERMANENTLY_INVALIDATED";

function toLabel(result: SecureCredentialReadResult | { status: string }): P0StatusLabel {
  switch (result.status) {
    case "found":
      return "FOUND";
    case "missing":
      return "MISSING";
    case "temporarily_unavailable":
      return "TEMPORARILY_UNAVAILABLE";
    case "permanently_invalidated":
      return "PERMANENTLY_INVALIDATED";
    default:
      return "MISSING";
  }
}

function emit(line: string): void {
  if (!ENABLED) return;
  // eslint-disable-next-line no-console
  console.log(`${TAG} ${line}`);
}

export function isP0ValidateEnabled(): boolean {
  return ENABLED;
}

export type P0CredentialSnapshot = {
  installation: P0StatusLabel;
  refresh: P0StatusLabel;
  envelope: P0StatusLabel;
  recovery: P0StatusLabel;
};

export async function probeCredentialStatuses(
  phase: string,
  extras?: Record<string, string | number | boolean | null | undefined>
): Promise<P0CredentialSnapshot> {
  const [installation, refresh, envelope, recovery] = await Promise.all([
    readInstallationId(),
    readRefreshToken(),
    readSessionEnvelope(),
    readRecoveryCredential(),
  ]);
  const snap: P0CredentialSnapshot = {
    installation: toLabel(installation),
    refresh: toLabel(refresh),
    envelope: toLabel(envelope),
    recovery: toLabel(recovery),
  };
  const extraParts = extras
    ? Object.entries(extras)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => `${k}=${v === null ? "null" : String(v)}`)
        .join(" ")
    : "";
  emit(
    `phase=${phase} installation=${snap.installation} refresh=${snap.refresh} envelope=${snap.envelope} recovery=${snap.recovery}${extraParts ? ` ${extraParts}` : ""}`
  );
  return snap;
}

export function probeEvent(
  event: string,
  extras?: Record<string, string | number | boolean | null | undefined>
): void {
  const extraParts = extras
    ? Object.entries(extras)
        .filter(([, v]) => v !== undefined)
        .map(([k, v]) => `${k}=${v === null ? "null" : String(v)}`)
        .join(" ")
    : "";
  emit(`event=${event}${extraParts ? ` ${extraParts}` : ""}`);
}

/**
 * État contrôlé INCOHERENT_LOCAL : refresh MISSING, siblings conservés.
 * Ne touche jamais aux valeurs — delete strict refresh uniquement.
 */
export async function p0ForceDeleteStrictRefreshOnly(): Promise<P0CredentialSnapshot> {
  if (!ENABLED) {
    throw new Error("P0 validate disabled");
  }
  probeEvent("credential_op", { op: "delete_strict_refresh_only", when: "before" });
  await deleteRefreshToken();
  // Tombstone éventuel côté store : on force aussi une absence pure via SecureStore.
  try {
    await SecureStore.deleteItemAsync("atmr.auth.refresh_token", NATIVE_OPTIONS);
  } catch {
    /* ignore */
  }
  return probeCredentialStatuses("after_delete_strict_refresh");
}
