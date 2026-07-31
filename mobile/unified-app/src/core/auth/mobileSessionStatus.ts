/** Machine d'état de session mobile (PR C / PR2) — orthogonale au `status` legacy de sessionProvider. */
export type MobileSessionStatus =
  | "initializing"
  | "anonymous"
  | "logging_out"
  /** @deprecated Preférer auth_recovering — conservé pour compat lectures. */
  | "restoring"
  | "authenticated_online"
  | "authenticated_offline"
  | "auth_recovering"
  | "storage_locked"
  | "revoked";

export type OfflineCapabilities = {
  canReadCachedProfile: boolean;
  canReadCachedMissions: boolean;
  canCaptureGps: boolean;
  canPersistGpsQueue: boolean;
  /** Toute mutation qui exige une confirmation serveur immédiate (paiement, changement de statut, …). */
  canPerformOnlineMutation: boolean;
};

const OFFLINE_FIRST_CAPABILITIES: OfflineCapabilities = {
  canReadCachedProfile: true,
  canReadCachedMissions: true,
  canCaptureGps: true,
  canPersistGpsQueue: true,
  canPerformOnlineMutation: false,
};

const ONLINE_CAPABILITIES: OfflineCapabilities = {
  ...OFFLINE_FIRST_CAPABILITIES,
  canPerformOnlineMutation: true,
};

const NO_CAPABILITIES: OfflineCapabilities = {
  canReadCachedProfile: false,
  canReadCachedMissions: false,
  canCaptureGps: false,
  canPersistGpsQueue: false,
  canPerformOnlineMutation: false,
};

/**
 * Capacités locales dérivées du statut de session — pilotent l'UI hors-ligne :
 * un chauffeur authentifié hors-ligne garde son profil/missions en cache et peut
 * continuer à capturer/mettre en file le GPS, mais ne peut pas muter côté serveur.
 */
export function resolveOfflineCapabilities(status: MobileSessionStatus): OfflineCapabilities {
  switch (status) {
    case "authenticated_online":
      return ONLINE_CAPABILITIES;
    case "authenticated_offline":
    case "auth_recovering":
      return OFFLINE_FIRST_CAPABILITIES;
    case "initializing":
    case "restoring":
    case "logging_out":
    case "anonymous":
    case "storage_locked":
    case "revoked":
    default:
      return NO_CAPABILITIES;
  }
}
