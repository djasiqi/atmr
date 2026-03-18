/**
 * P2.1.1 — Utilitaires auth pour connect_error (extraction status 401/403).
 * Module isolé sans dépendances lourdes pour tests unitaires.
 */
import type {
  AuthFailureReason,
  AuthFailureSeverity,
} from "@/services/authGuards";

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

function extractAuthReason(err: unknown): AuthFailureReason {
  if (!err || typeof err !== "object") return "unknown_socket_auth_error";
  const e = err as Record<string, unknown>;
  const data = (e?.data as Record<string, unknown> | undefined) ?? {};
  const response = (e?.response as Record<string, unknown> | undefined) ?? {};
  const responseData =
    (response?.data as Record<string, unknown> | undefined) ?? {};
  const reason =
    data?.reason ?? data?.error_reason ?? responseData?.reason ?? responseData?.error;
  const normalized = String(reason ?? "").toLowerCase();
  const msg = String(e?.message ?? e?.description ?? "").toLowerCase();

  if (normalized.includes("refresh_invalid")) return "refresh_invalid";
  if (normalized.includes("refresh_expired")) return "refresh_expired";
  if (normalized.includes("session_revoked")) return "session_revoked";
  if (normalized.includes("account_disabled")) return "account_disabled";
  if (normalized.includes("tenant_access_revoked")) return "tenant_access_revoked";
  if (msg.includes("offline")) return "network_error";
  if (msg.includes("network") || msg.includes("transport")) return "socket_transport_error";
  return "unknown_socket_auth_error";
}

export function getSocketAuthFailureDecision(err: unknown): {
  status: 401 | 403 | null;
  reason: AuthFailureReason;
  severity: AuthFailureSeverity;
  shouldLogout: boolean;
} {
  const status = extractAuthStatus(err);
  const reason = extractAuthReason(err);
  const hardReasons: AuthFailureReason[] = [
    "refresh_invalid",
    "refresh_expired",
    "session_revoked",
    "account_disabled",
    "tenant_access_revoked",
  ];
  const shouldLogout = hardReasons.includes(reason);
  return {
    status,
    reason,
    severity: shouldLogout ? "AUTH_HARD_FAILURE" : "AUTH_SOFT_FAILURE",
    shouldLogout,
  };
}
