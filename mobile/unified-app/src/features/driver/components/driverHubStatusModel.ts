/**
 * DRIVER-COLD-03 — une ligne de statut, pas de réserve verticale.
 * NORMAL = GPS / Localisation… ; PROBLÈME = la même ligne (alerte).
 */
export const DRIVER_DASHBOARD_STATUS_LINE_HEIGHT = 12;

/** @deprecated Plus de réserve 48 px — alias de la ligne secondaire. */
export const DRIVER_DASHBOARD_STATUS_AREA_HEIGHT = DRIVER_DASHBOARD_STATUS_LINE_HEIGHT;

export type DriverStatusTone = "warn" | "error";

export type DriverStatusActionKind =
  | "disclosure"
  | "settings"
  | "battery"
  | "oem"
  | "tracking"
  | null;

export type DriverStatusIssue = {
  id: string;
  title: string;
  message: string;
  tone: DriverStatusTone;
  actionKind: DriverStatusActionKind;
  actionLabel?: string;
};

export type DriverStatusAreaView =
  | { mode: "empty"; count: 0 }
  | { mode: "single"; count: 1; issue: DriverStatusIssue }
  | { mode: "summary"; count: number; label: string };

export type DriverStatusIssueFlags = {
  hideTrackingPrepDuplicates: boolean;
  trackingNeedsAttention: boolean;
  pushDisclosure: boolean;
  pushPending: boolean;
  pushFailed: boolean;
  pushDenied: boolean;
  offline: boolean;
  socketDegraded: boolean;
  gpsDisabled: boolean;
  batteryOptimization: boolean;
  oemRequired: boolean;
  oemManufacturer?: string;
  sessionError: boolean;
};

export function collectDriverStatusIssues(flags: DriverStatusIssueFlags): DriverStatusIssue[] {
  const hidePrep = flags.hideTrackingPrepDuplicates;
  const issues: DriverStatusIssue[] = [];

  if (flags.trackingNeedsAttention) {
    issues.push({
      id: "tracking",
      title: "Suivi à vérifier",
      message: "Permissions ou GPS à confirmer pour le suivi.",
      tone: "warn",
      actionKind: "tracking",
      actionLabel: "Vérifier",
    });
  }

  if (!hidePrep && flags.pushDisclosure) {
    issues.push({
      id: "push_disclosure",
      title: "Notifications",
      message: "Confirmez l'utilisation des notifications pour recevoir vos missions.",
      tone: "warn",
      actionKind: "disclosure",
      actionLabel: "Continuer",
    });
  }
  if (flags.pushPending) {
    issues.push({
      id: "push_pending",
      title: "Enregistrement en attente",
      message: "La synchronisation des notifications reprendra automatiquement.",
      tone: "warn",
      actionKind: null,
    });
  }
  if (flags.pushFailed) {
    issues.push({
      id: "push_failed",
      title: "Notifications indisponibles",
      message: "Impossible d'enregistrer les notifications. Réessayez après connexion.",
      tone: "error",
      actionKind: "settings",
      actionLabel: "Ouvrir les réglages",
    });
  }
  if (!hidePrep && flags.pushDenied && !flags.pushDisclosure) {
    issues.push({
      id: "push_denied",
      title: "Notifications désactivées",
      message: "Activez les notifications pour recevoir vos missions.",
      tone: "error",
      actionKind: "settings",
      actionLabel: "Ouvrir les réglages",
    });
  }
  if (flags.offline) {
    issues.push({
      id: "offline",
      title: "Mode hors ligne",
      message: "Connexion indisponible. Les actions sont mises en file et rejouées.",
      tone: "warn",
      actionKind: null,
    });
  }
  if (flags.socketDegraded) {
    issues.push({
      id: "socket",
      title: "Temps réel instable",
      message: "Sync dégradée, léger retard possible.",
      tone: "warn",
      actionKind: null,
    });
  }
  if (!hidePrep && flags.gpsDisabled) {
    issues.push({
      id: "gps",
      title: "GPS désactivé",
      message: "Activez la localisation pour maintenir le suivi mission.",
      tone: "error",
      actionKind: null,
    });
  }
  if (!hidePrep && flags.batteryOptimization) {
    issues.push({
      id: "battery",
      title: "Optimisation batterie active",
      message: "Vos positions GPS peuvent ne pas être transmises en arrière-plan.",
      tone: "warn",
      actionKind: "battery",
      actionLabel: "Corriger",
    });
  }
  if (!hidePrep && flags.oemRequired) {
    issues.push({
      id: "oem",
      title: "Réglages fabricant requis",
      message: `Sur ${flags.oemManufacturer || "votre appareil"}, ouvrez aussi Auto-start / apps protégées.`,
      tone: "warn",
      actionKind: "oem",
      actionLabel: "Réglages fabricant",
    });
  }
  if (flags.sessionError) {
    issues.push({
      id: "session",
      title: "Session indisponible",
      message: "La session a expiré ou le bootstrap a échoué. Reconnectez-vous.",
      tone: "error",
      actionKind: null,
    });
  }

  return issues;
}

export function resolveDriverStatusAreaView(issues: DriverStatusIssue[]): DriverStatusAreaView {
  if (issues.length === 0) return { mode: "empty", count: 0 };
  if (issues.length === 1) {
    return { mode: "single", count: 1, issue: issues[0]! };
  }
  return {
    mode: "summary",
    count: issues.length,
    label: `${issues.length} éléments à vérifier`,
  };
}
