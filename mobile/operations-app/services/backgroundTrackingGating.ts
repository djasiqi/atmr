/**
 * Background tracking gating — Phase 1: mission active uniquement.
 *
 * Vérité métier purement dérivée. Aucune lecture de l'état natif.
 * Aucun effet de bord. Consommé par l'orchestrateur (locationTracker).
 */

export type PermissionStatus = "granted" | "denied" | "undetermined";

export interface BgTrackingInputs {
  isAuthenticated: boolean;
  role: "driver" | "enterprise";
  hasActiveMission: boolean;
  fgPermission: PermissionStatus;
  bgPermission: PermissionStatus;
  killSwitchEnabled: boolean;
}

/**
 * Contrat de démarrage — conditions nécessaires pour que le background puisse démarrer.
 * Toutes les conditions doivent être vraies pour autoriser le start.
 */
export interface StartContract {
  /** Utilisateur authentifié (chauffeur) */
  isAuthenticated: boolean;
  /** Rôle = chauffeur */
  roleIsDriver: boolean;
  /** Mission active (ASSIGNED | EN_ROUTE | IN_PROGRESS) */
  hasActiveMission: boolean;
  /** Permission foreground accordée */
  fgPermissionGranted: boolean;
  /** Permission background accordée */
  bgPermissionGranted: boolean;
  /** Kill switch non activé (driver_background_tracking_enabled != false) */
  killSwitchAllowed: boolean;
  /** Tracking non déjà démarré (vérifié côté orchestrateur) */
  notAlreadyStarted: boolean;
}

/**
 * Déduit le contrat de démarrage depuis les inputs.
 * notAlreadyStarted doit être fourni par l'orchestrateur.
 */
export function deriveStartContract(
  inputs: BgTrackingInputs,
  notAlreadyStarted: boolean
): StartContract {
  return {
    isAuthenticated: inputs.isAuthenticated,
    roleIsDriver: inputs.role === "driver",
    hasActiveMission: inputs.hasActiveMission,
    fgPermissionGranted: inputs.fgPermission === "granted",
    bgPermissionGranted: inputs.bgPermission === "granted",
    killSwitchAllowed: !inputs.killSwitchEnabled,
    notAlreadyStarted,
  };
}

/**
 * Vérifie si toutes les conditions du contrat de démarrage sont remplies.
 */
export function satisfiesStartContract(contract: StartContract): boolean {
  return (
    contract.isAuthenticated &&
    contract.roleIsDriver &&
    contract.hasActiveMission &&
    contract.fgPermissionGranted &&
    contract.bgPermissionGranted &&
    contract.killSwitchAllowed &&
    contract.notAlreadyStarted
  );
}

/**
 * Contrat d'arrêt — conditions de coupure immédiate.
 * Si au moins une condition est vraie → stop immédiat.
 */
export type StopCondition =
  | "mission_ended"
  | "driver_off_duty"
  | "logout"
  | "permission_revoked"
  | "role_non_driver"
  | "kill_switch"
  | "reconciliation";

export interface StopContract {
  /** Mission terminée (COMPLETED | CANCELED | RETURN_COMPLETED) */
  missionEnded: boolean;
  /** Chauffeur hors service / déconnecté */
  logout: boolean;
  /** Permission foreground ou background retirée */
  permissionRevoked: boolean;
  /** Rôle non chauffeur */
  roleNonDriver: boolean;
  /** Kill switch activé */
  killSwitch: boolean;
}

/**
 * Déduit le contrat d'arrêt depuis les inputs.
 */
export function deriveStopContract(inputs: BgTrackingInputs): StopContract {
  const fgOk = inputs.fgPermission === "granted";
  const bgOk = inputs.bgPermission === "granted";
  return {
    missionEnded: !inputs.hasActiveMission,
    logout: !inputs.isAuthenticated,
    permissionRevoked: !fgOk || !bgOk,
    roleNonDriver: inputs.role !== "driver",
    killSwitch: inputs.killSwitchEnabled,
  };
}

/**
 * Retourne la première condition d'arrêt qui s'applique (priorité absolue kill switch).
 */
export function getFirstStopCondition(contract: StopContract): StopCondition | null {
  if (contract.killSwitch) return "kill_switch";
  if (contract.permissionRevoked) return "permission_revoked";
  if (contract.logout) return "logout";
  if (contract.missionEnded) return "mission_ended";
  if (contract.roleNonDriver) return "role_non_driver";
  return null;
}

/**
 * Vérifie si au moins une condition d'arrêt est remplie.
 */
export function satisfiesStopContract(contract: StopContract): boolean {
  return getFirstStopCondition(contract) !== null;
}

/**
 * Détermine si le tracking background peut démarrer.
 * Règle D : kill switch priorité absolue.
 */
export function shouldRunBackgroundTracking(inputs: BgTrackingInputs): boolean {
  if (inputs.killSwitchEnabled) return false;
  if (inputs.role !== "driver") return false;
  if (!inputs.isAuthenticated) return false;
  if (!inputs.hasActiveMission) return false;
  if (inputs.fgPermission !== "granted") return false;
  if (inputs.bgPermission !== "granted") return false;
  return true;
}
