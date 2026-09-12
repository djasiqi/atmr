/**
 * Identifiants opaques via CSPRNG (Web Crypto / Hermes).
 * Pas de fallback Math.random : un secret ou un binding device
 * ne doit pas se dégrader silencieusement.
 */

export function createSecureRandomId(prefix: string): string {
  const cryptoObj = globalThis.crypto;
  if (!cryptoObj || typeof cryptoObj.randomUUID !== "function") {
    throw new Error("secure_random_unavailable");
  }
  return `${prefix}${cryptoObj.randomUUID()}`;
}
