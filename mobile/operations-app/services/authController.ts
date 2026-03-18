/**
 * P0 — Contrôleur central d'invalidation de session
 *
 * Une seule porte de sortie quand on nettoie une session : forceLogoutDriver / forceLogoutEnterprise.
 * Les intercepteurs (api.ts, enterpriseAuth.ts) NE doivent JAMAIS appeler clearAll/clearAuth directement.
 * Ils invoquent ce contrôleur, qui délègue au callback enregistré par useAuth.
 *
 * P0.1 — Pending drain (cold start safe) + dedupe (anti double-logout).
 */

import Constants from "expo-constants";
import { getLogger } from "@/utils/logger";
import { getNetworkStateSnapshot } from "@/services/networkState";

const log = getLogger("AuthCtrl");
import {
  getLogoutSeverity,
  normalizeLogoutReason,
  type ForceLogoutMetadata,
  type DriverLogoutReason,
  type EnterpriseLogoutReason,
} from "@/services/authLogoutReasons";
import { logAuthEvent } from "@/services/authLogging";

export type {
  DriverLogoutReason,
  EnterpriseLogoutReason,
  ForceLogoutMetadata,
};

type ForceLogoutDriverCallback = (
  reason: DriverLogoutReason,
  metadata: ForceLogoutMetadata
) => void | Promise<void>;
type ForceLogoutEnterpriseCallback = (
  reason: EnterpriseLogoutReason,
  metadata: ForceLogoutMetadata
) => void | Promise<void>;

let forceLogoutDriverCallback: ForceLogoutDriverCallback | null = null;
let forceLogoutEnterpriseCallback: ForceLogoutEnterpriseCallback | null = null;

/** P0.1 — Pending logout si callback absent (cold start). */
type PendingLogout = {
  route: "driver" | "enterprise";
  reason: string;
  ts: number;
  metadata: ForceLogoutMetadata;
};
let pendingDriver: PendingLogout | null = null;
let pendingEnterprise: PendingLogout | null = null;

/** P0.1 — Dedupe : fenêtre courte pour éviter double exécution. */
const DEDUPE_MS = 2000;
let lastDriverLogout: { reason: string; ts: number } | null = null;
let lastEnterpriseLogout: { reason: string; ts: number } | null = null;

/**
 * Enregistre le callback de force-logout driver (appelé par useAuth au montage).
 * P0.1 — Drain pending immédiatement si un invoke a eu lieu avant le register.
 */
export function registerForceLogoutDriver(cb: ForceLogoutDriverCallback): () => void {
  forceLogoutDriverCallback = cb;
  if (pendingDriver) {
    const { reason, metadata } = pendingDriver;
    pendingDriver = null;
    log.info("drain pending driver logout", { reason });
    void Promise.resolve(cb(reason as DriverLogoutReason, metadata)).catch((e: unknown) => {
      log.error("drain pending driver error", { error: e });
    });
  }
  return () => {
    forceLogoutDriverCallback = null;
  };
}

/**
 * Enregistre le callback de force-logout enterprise (appelé par useAuth au montage).
 * P0.1 — Drain pending immédiatement si un invoke a eu lieu avant le register.
 */
export function registerForceLogoutEnterprise(cb: ForceLogoutEnterpriseCallback): () => void {
  forceLogoutEnterpriseCallback = cb;
  if (pendingEnterprise) {
    const { reason, metadata } = pendingEnterprise;
    pendingEnterprise = null;
    log.info("drain pending enterprise logout", { reason });
    void Promise.resolve(cb(reason as EnterpriseLogoutReason, metadata)).catch((e: unknown) => {
      log.error("drain pending enterprise error", { error: e });
    });
  }
  return () => {
    forceLogoutEnterpriseCallback = null;
  };
}

/**
 * Log structuré pour diagnostic prod (reason, route, ts, app_version, network_state).
 * Toujours émis (prod + staging) pour prouver la cause réelle.
 */
function logLogoutTransition(
  route: "driver" | "enterprise",
  metadata: ForceLogoutMetadata,
  extra?: Record<string, unknown>
): void {
  const normalizedReason = normalizeLogoutReason(metadata.reason);
  const network = getNetworkStateSnapshot();
  logAuthEvent("LOGOUT_TRANSITION", {
    route,
    reason: normalizedReason,
    severity: metadata.severity,
    trigger_source: metadata.trigger_source,
    ...(metadata.role ? { role: metadata.role } : {}),
    ...(metadata.tenant_id ? { tenant_id: metadata.tenant_id } : {}),
    ...(metadata.session_id ? { session_id: metadata.session_id } : {}),
    ...(metadata.device_id ? { device_id: metadata.device_id } : {}),
    app_version: Constants.expoConfig?.version ?? "?",
    ...(network ? { network_connected: network.isConnected } : {}),
    ...extra,
  });
}

/**
 * P0.1 — Dedupe : évite double exécution sur (route, reason) dans une fenêtre courte.
 */
function shouldDedupeDriver(reason: string): boolean {
  const now = Date.now();
  if (lastDriverLogout && now - lastDriverLogout.ts < DEDUPE_MS && lastDriverLogout.reason === reason) {
    return true;
  }
  lastDriverLogout = { reason, ts: now };
  return false;
}

function shouldDedupeEnterprise(reason: string): boolean {
  const now = Date.now();
  if (lastEnterpriseLogout && now - lastEnterpriseLogout.ts < DEDUPE_MS && lastEnterpriseLogout.reason === reason) {
    return true;
  }
  lastEnterpriseLogout = { reason, ts: now };
  return false;
}

/**
 * Invalide la session driver de manière centralisée.
 * Appelé par api.ts (intercepteurs) ou useAuth (logout manuel, refreshProfile 401/403).
 * P0.1 — Pending si callback absent ; dedupe si invoqué en rafale.
 */
export async function invokeForceLogoutDriver(
  metadata: ForceLogoutMetadata & { reason: DriverLogoutReason }
): Promise<void> {
  if (!metadata.trigger_source || !metadata.severity || !metadata.source) {
    throw new Error("forceLogout driver metadata incomplete");
  }
  const reason = normalizeLogoutReason(metadata.reason) as DriverLogoutReason;
  const finalMetadata: ForceLogoutMetadata = {
    ...metadata,
    source: "driver",
    reason,
    severity: metadata.severity || getLogoutSeverity(reason),
  };
  logLogoutTransition("driver", {
    ...finalMetadata,
  });

  if (shouldDedupeDriver(reason)) {
    log.info("dedupe driver logout", { reason, windowMs: DEDUPE_MS });
    return;
  }

  if (forceLogoutDriverCallback) {
    await forceLogoutDriverCallback(reason, finalMetadata);
  } else {
    pendingDriver = {
      route: "driver",
      reason,
      ts: Date.now(),
      metadata: {
        ...finalMetadata,
      },
    };
    log.warn("force logout driver callback not registered", { reason });
  }
}

/**
 * Invalide la session enterprise de manière centralisée.
 * Appelé par enterpriseAuth.ts (intercepteurs) ou useAuth (logoutEnterprise).
 * P0.1 — Pending si callback absent ; dedupe si invoqué en rafale.
 */
export async function invokeForceLogoutEnterprise(
  metadata: ForceLogoutMetadata & {
    reason: EnterpriseLogoutReason;
  }
): Promise<void> {
  if (!metadata.trigger_source || !metadata.severity || !metadata.source) {
    throw new Error("forceLogout enterprise metadata incomplete");
  }
  const reason = normalizeLogoutReason(metadata.reason) as EnterpriseLogoutReason;
  const finalMetadata: ForceLogoutMetadata = {
    ...metadata,
    source: "enterprise",
    reason,
    severity: metadata.severity || getLogoutSeverity(reason),
  };
  logLogoutTransition("enterprise", {
    ...finalMetadata,
  });

  if (shouldDedupeEnterprise(reason)) {
    log.info("dedupe enterprise logout", { reason, windowMs: DEDUPE_MS });
    return;
  }

  if (forceLogoutEnterpriseCallback) {
    await forceLogoutEnterpriseCallback(reason, finalMetadata);
  } else {
    pendingEnterprise = {
      route: "enterprise",
      reason,
      ts: Date.now(),
      metadata: {
        ...finalMetadata,
      },
    };
    log.warn("force logout enterprise callback not registered", { reason });
  }
}
