import { resolveLocationModeFromState, resolvePresenceState } from "./locationPresenceFsm";

/**
 * Background tracking gating — Phase 1: mission active uniquement.
 *
 * P1 : sous-règle d’exécution (shouldRunBackgroundTracking) consommée aussi par
 * resolveTrackingPolicy() ; la source unique de modes métier reste trackingPolicy.ts.
 *
 * Vérité métier purement dérivée. Aucune lecture de l'état natif.
 * Aucun effet de bord. Consommé par l'orchestrateur (locationTracker).
 */

export type PermissionStatus = "granted" | "denied" | "undetermined";
export type LocationMode =
  | "mission_live"
  | "availability_presence"
  | "passive_last_known";

export interface BgTrackingInputs {
  isAuthenticated: boolean;
  role: "driver" | "enterprise";
  platform?: "ios" | "android" | "web";
  hasActiveMission: boolean;
  /** En mission_live : true uniquement si EN_ROUTE ou IN_PROGRESS (pas ASSIGNED). Évite la notif "Suivi en cours" avant que le chauffeur ait appuyé sur En route. */
  missionStatusEnabledForTracking?: boolean;
  fgPermission: PermissionStatus;
  bgPermission: PermissionStatus;
  killSwitchEnabled: boolean;
  locationMode: LocationMode;
  availabilityPresenceEnabled?: boolean;
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
  const requiresBgPermission = inputs.platform === "ios";
  return {
    isAuthenticated: inputs.isAuthenticated,
    roleIsDriver: inputs.role === "driver",
    hasActiveMission: inputs.hasActiveMission,
    fgPermissionGranted: inputs.fgPermission === "granted",
    // Android: foreground service location can continue with foreground permission.
    // iOS: requires background permission for true background updates.
    bgPermissionGranted: requiresBgPermission
      ? inputs.bgPermission === "granted"
      : true,
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
  const requiresBgPermission = inputs.platform === "ios";
  const bgOk = requiresBgPermission ? inputs.bgPermission === "granted" : true;
  return {
    missionEnded:
      inputs.locationMode === "mission_live" && !inputs.hasActiveMission,
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
  const requiresBgPermission = inputs.platform === "ios";
  if (inputs.killSwitchEnabled) return false;
  if (inputs.role !== "driver") return false;
  if (!inputs.isAuthenticated) return false;
  const fsmState = resolvePresenceState({
    isAuthenticated: inputs.isAuthenticated,
    isDriver: inputs.role === "driver",
    hasFgPermission: inputs.fgPermission === "granted",
    hasBgPermission: requiresBgPermission ? inputs.bgPermission === "granted" : true,
    appInBackground: true,
    hasActiveMission: inputs.hasActiveMission,
    availabilityPresenceEnabled: !!inputs.availabilityPresenceEnabled,
  });
  if (resolveLocationModeFromState(fsmState) === "passive_last_known") return false;
  if (inputs.locationMode === "passive_last_known") return false;
  if (
    inputs.locationMode === "availability_presence" &&
    !inputs.availabilityPresenceEnabled
  ) {
    return false;
  }
  if (inputs.locationMode === "mission_live" && !inputs.hasActiveMission) return false;
  // En mission_live : aligné sur missionTrackingPolicy (ASSIGNED | EN_ROUTE | IN_PROGRESS)
  if (
    inputs.locationMode === "mission_live" &&
    inputs.missionStatusEnabledForTracking === false
  ) {
    return false;
  }
  if (inputs.fgPermission !== "granted") return false;
  if (requiresBgPermission && inputs.bgPermission !== "granted") return false;
  return true;
}
