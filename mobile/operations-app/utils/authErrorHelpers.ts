/**
 * P0.2 — Helpers pour classifier les erreurs auth/réseau.
 * Permet de distinguer : offline/timeout (ne pas logout) vs 401/403 (session invalide).
 */

/** Erreur réseau : pas de réponse, timeout, DNS, etc. */
export function isNetworkError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { response?: unknown; code?: string; message?: string };
  if (!e.response) return true; // Pas de réponse HTTP = réseau
  if (e.code === "ECONNABORTED") return true; // Timeout
  if (e.code === "ERR_NETWORK") return true;
  if (e.code === "ENOTFOUND" || e.code === "ECONNREFUSED") return true;
  const msg = (e.message || "").toLowerCase();
  if (msg.includes("network") || msg.includes("timeout")) return true;
  return false;
}

/** Erreur HTTP 401 ou 403 (session invalide / compte désactivé). */
export function isHttpAuthError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const status = (err as { response?: { status?: number } })?.response?.status;
  return status === 401 || status === 403;
}

/** Extrait le status HTTP si présent. */
export function getHttpStatus(err: unknown): number | null {
  if (!err || typeof err !== "object") return null;
  const status = (err as { response?: { status?: number } })?.response?.status;
  return status ?? null;
}
