/**
 * Identifiants opaques via CSPRNG.
 *
 * Ordre :
 * 1. globalThis.crypto.randomUUID (Web Crypto / Hermes récent)
 * 2. globalThis.expo.uuidv4 (natif expo-modules-core → UUID.randomUUID / SecureRandom)
 * 3. sinon → secure_random_unavailable (fail-closed)
 *
 * Pas de Math.random, Date.now, ni identifiant matériel.
 */

/** UUID v4 (RFC 4122) — crypto.randomUUID et java.util.UUID.randomUUID. */
const UUID_V4_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

type ExpoGlobal = {
  uuidv4?: () => unknown;
};

function readExpoUuidV4(): string | null {
  const expoObj = (globalThis as typeof globalThis & { expo?: ExpoGlobal }).expo;
  if (!expoObj || typeof expoObj.uuidv4 !== "function") {
    return null;
  }
  const raw = expoObj.uuidv4();
  if (typeof raw !== "string") {
    return null;
  }
  const trimmed = raw.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function readCryptoRandomUuid(): string | null {
  const cryptoObj = globalThis.crypto;
  if (!cryptoObj || typeof cryptoObj.randomUUID !== "function") {
    return null;
  }
  const raw = cryptoObj.randomUUID();
  if (typeof raw !== "string") {
    return null;
  }
  const trimmed = raw.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function assertUuidV4OrThrow(value: string): string {
  if (!UUID_V4_RE.test(value)) {
    throw new Error("secure_random_unavailable");
  }
  return value;
}

export function createSecureRandomId(prefix: string): string {
  const fromCrypto = readCryptoRandomUuid();
  if (fromCrypto != null) {
    return `${prefix}${assertUuidV4OrThrow(fromCrypto)}`;
  }

  const fromExpo = readExpoUuidV4();
  if (fromExpo != null) {
    return `${prefix}${assertUuidV4OrThrow(fromExpo)}`;
  }

  throw new Error("secure_random_unavailable");
}
