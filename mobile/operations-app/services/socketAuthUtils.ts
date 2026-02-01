/**
 * P2.1.1 — Utilitaires auth pour connect_error (extraction status 401/403).
 * Module isolé sans dépendances lourdes pour tests unitaires.
 */

/** Extrait le status HTTP 401/403 depuis connect_error (serveur/adapter variable). */
export function extractAuthStatus(err: unknown): 401 | 403 | null {
  if (!err || typeof err !== "object") return null;
  const e = err as Record<string, unknown>;
  const data = e?.data as Record<string, unknown> | undefined;
  const status = data?.status ?? data?.code;
  if (status === 401 || status === 403) return status as 401 | 403;
  const msg = String(e?.message ?? e?.description ?? "").toLowerCase();
  if (msg.includes("401") || msg.includes("unauthorized")) return 401;
  if (msg.includes("403") || msg.includes("forbidden")) return 403;
  return null;
}
