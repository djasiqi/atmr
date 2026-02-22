/**
 * AuthLogoutReason — source unique de vérité pour les causes de logout.
 *
 * Tous les chemins de logout doivent passer par cet enum.
 * Évite les strings magiques et permet corrélation logs / métriques.
 */

/** Reasons driver — pour invokeForceLogoutDriver */
export type DriverLogoutReason =
  | "manual_logout"
  | "refresh_rejected_401"
  | "refresh_rejected_403"
  | "account_disabled"
  | "profile_401_403"
  | "profile_auth_invalid"
  | "boot_autologin_failed"
  | "login_profile_failed";

/** Reasons enterprise — pour invokeForceLogoutEnterprise */
export type EnterpriseLogoutReason =
  | "manual_logout"
  | "refresh_rejected_401"
  | "refresh_rejected_403"
  | "account_disabled";

/** Union centralisée — tous les reasons possibles */
export type AuthLogoutReason = DriverLogoutReason | EnterpriseLogoutReason;

/** Reasons qui déclenchent la bannière "Session expirée" sur login. */
export const SESSION_EXPIRED_REASONS: readonly AuthLogoutReason[] = [
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
