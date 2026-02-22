/**
 * P0.3.A — Cooldown + backoff sur refresh proactif.
 * Évite boucles refresh, spam logs, battery drain quand refresh échoue (401/403/5xx/network).
 */

import { getLogger } from "@/utils/logger";

const log = getLogger("RefreshCD");
const INITIAL_BACKOFF_MS = 30 * 1000; // 30s
const MAX_BACKOFF_MS = 15 * 60 * 1000; // 15min

type CooldownState = { nextAllowedAt: number; failureCount: number };

const driverState: CooldownState = { nextAllowedAt: 0, failureCount: 0 };
const enterpriseState: CooldownState = { nextAllowedAt: 0, failureCount: 0 };

function getBackoffMs(failureCount: number): number {
  return Math.min(
    INITIAL_BACKOFF_MS * Math.pow(2, Math.min(failureCount, 5)),
    MAX_BACKOFF_MS
  );
}

/** Vérifie si on est en cooldown (doit attendre avant prochaine tentative). */
export function isDriverProactiveRefreshInCooldown(): boolean {
  return Date.now() < driverState.nextAllowedAt;
}

export function isEnterpriseProactiveRefreshInCooldown(): boolean {
  return Date.now() < enterpriseState.nextAllowedAt;
}

/** Retourne le délai restant avant prochaine tentative autorisée (ms). */
export function getDriverProactiveRefreshCooldownRemaining(): number {
  const remaining = driverState.nextAllowedAt - Date.now();
  return Math.max(0, remaining);
}

export function getEnterpriseProactiveRefreshCooldownRemaining(): number {
  const remaining = enterpriseState.nextAllowedAt - Date.now();
  return Math.max(0, remaining);
}

/** Enregistre un échec et retourne le backoff à appliquer (ms). */
export function recordDriverProactiveRefreshFailure(): number {
  driverState.failureCount++;
  const backoff = getBackoffMs(driverState.failureCount);
  driverState.nextAllowedAt = Date.now() + backoff;
  log.info("driver cooldown", { backoffSec: backoff / 1000, failureCount: driverState.failureCount });
  return backoff;
}

export function recordEnterpriseProactiveRefreshFailure(): number {
  enterpriseState.failureCount++;
  const backoff = getBackoffMs(enterpriseState.failureCount);
  enterpriseState.nextAllowedAt = Date.now() + backoff;
  log.info("enterprise cooldown", { backoffSec: backoff / 1000, failureCount: enterpriseState.failureCount });
  return backoff;
}

/** Reset cooldown après un refresh réussi. */
export function resetDriverProactiveRefreshCooldown(): void {
  driverState.nextAllowedAt = 0;
  driverState.failureCount = 0;
}

export function resetEnterpriseProactiveRefreshCooldown(): void {
  enterpriseState.nextAllowedAt = 0;
  enterpriseState.failureCount = 0;
}
