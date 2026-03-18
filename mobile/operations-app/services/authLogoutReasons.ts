/**
 * AuthLogoutReason — source unique de vérité pour les causes de logout.
 *
 * Tous les chemins de logout doivent passer par cet enum.
 * Évite les strings magiques et permet corrélation logs / métriques.
 */

export type AuthFailureSeverity =
  | "AUTH_HARD_FAILURE"
  | "AUTH_SOFT_FAILURE"
  | "AUTH_MANUAL";

export type AuthTriggerSource =
  | "api_interceptor"
  | "foreground_resume"
  | "proactive_refresh"
  | "socket_reconnect"
  | "bootstrap"
  | "manual_action";

/** Reasons driver — pour invokeForceLogoutDriver */
export type DriverLogoutReason =
  | "manual_logout"
  | "security_revocation"
  | "session_revoked"
  | "refresh_invalid"
  | "refresh_expired"
  | "account_disabled"
  | "tenant_access_revoked"
  // Legacy raisons encore présentes dans le code (compat migration).
  | "refresh_rejected_401"
  | "refresh_rejected_403"
  | "profile_401_403"
  | "profile_auth_invalid"
  | "boot_autologin_failed"
  | "login_profile_failed";

/** Reasons enterprise — pour invokeForceLogoutEnterprise */
export type EnterpriseLogoutReason =
  | "manual_logout"
  | "security_revocation"
  | "session_revoked"
  | "refresh_invalid"
  | "refresh_expired"
  | "account_disabled"
  | "tenant_access_revoked"
  // Legacy raisons encore présentes dans le code (compat migration).
  | "refresh_rejected_401"
  | "refresh_rejected_403";

/** Union centralisée — tous les reasons possibles */
export type AuthLogoutReason = DriverLogoutReason | EnterpriseLogoutReason;

export type ForceLogoutMetadata = {
  reason: AuthLogoutReason;
  severity: AuthFailureSeverity;
  source: "driver" | "enterprise";
  trigger_source: AuthTriggerSource;
  role?: string;
  tenant_id?: string | number | null;
  session_id?: string | null;
  device_id?: string | null;
};

export function normalizeLogoutReason(reason: string): AuthLogoutReason {
  switch (reason) {
    case "refresh_rejected_401":
      return "refresh_invalid";
    case "refresh_rejected_403":
      return "session_revoked";
    default:
      return reason as AuthLogoutReason;
  }
}

export function getLogoutSeverity(reason: AuthLogoutReason): AuthFailureSeverity {
  const normalized = normalizeLogoutReason(reason);
  if (normalized === "manual_logout") return "AUTH_MANUAL";
  return "AUTH_HARD_FAILURE";
}

/** Reasons qui déclenchent la bannière "Session expirée" sur login. */
export const SESSION_EXPIRED_REASONS: readonly AuthLogoutReason[] = [
  "refresh_invalid",
  "refresh_expired",
  "session_revoked",
  "refresh_rejected_401",
  "refresh_rejected_403",
  "profile_auth_invalid",
] as const;

/** Reasons qui déclenchent la bannière "Compte désactivé" sur login. */
export const ACCOUNT_DISABLED_REASONS: readonly AuthLogoutReason[] = [
  "account_disabled",
] as const;

export function isSessionExpiredReason(reason: string): boolean {
  return (SESSION_EXPIRED_REASONS as readonly string[]).includes(reason);
}

export function isAccountDisabledReason(reason: string): boolean {
  return (ACCOUNT_DISABLED_REASONS as readonly string[]).includes(reason);
}
