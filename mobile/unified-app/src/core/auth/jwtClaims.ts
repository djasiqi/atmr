/**
 * Décodage local (non vérifié) du payload d'un JWT — uniquement pour hydrater
 * l'enveloppe de session locale (PR C). Ne jamais utiliser pour une décision de sécurité :
 * la signature n'est pas vérifiée côté client.
 */

function base64UrlDecode(segment: string): string {
  const normalized = segment.replace(/-/g, "+").replace(/_/g, "/");
  const padded = normalized + "=".repeat((4 - (normalized.length % 4)) % 4);
  if (typeof globalThis.atob === "function") {
    return globalThis.atob(padded);
  }
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const { Buffer } = require("buffer") as {
    Buffer: { from(data: string, encoding: string): { toString(encoding: string): string } };
  };
  return Buffer.from(padded, "base64").toString("binary");
}

export function decodeJwtClaims(token: string): Record<string, unknown> | null {
  try {
    const parts = token.split(".");
    if (parts.length < 2) return null;
    const json = base64UrlDecode(parts[1]);
    const parsed = JSON.parse(json);
    return parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}
