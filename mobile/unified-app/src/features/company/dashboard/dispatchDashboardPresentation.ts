import type { ComponentProps } from "react";
import type { Ionicons } from "@expo/vector-icons";
import type { CompanyRealtimeStatus } from "../realtime/companyRealtimeState";
import type { CompanyDispatchMission } from "../api/contracts";
import { isAdvancedMetricAvailable, type AdvancedMetricKey } from "./dataAvailability";

type IonName = ComponentProps<typeof Ionicons>["name"];

export type CompanyDispatchMode = "manual" | "semi_auto" | "fully_auto" | "unknown";

const ROUTE_RIDES = "/(app)/(company)/rides" as const;
const ROUTE_DISPATCH = "/(app)/(company)/dispatch" as const;

/**
 * Repli quand le dashboard ne reçoit pas `hasDispatchScreen` (ex. tests).
 * En prod, `dashboard.tsx` passe le feature flag `company_dispatch_screen_enabled`.
 */
export const HAS_DISPATCH_SCREEN = false;

export type KpiBlockDisplay =
  | { kind: "value"; line1: string; line2?: string }
  | { kind: "unavailable" }
  | { kind: "hidden" };

export type DashboardKpiKey =
  | "assign_pending"
  | "assign_in_progress"
  | "delayed"
  | "opportunities"
  | "drivers_available"
  | "proposals"
  | "assign_without_proposal"
  | "auto_assigned"
  | "exceptions"
  | "realtime_health";

export type DashboardKpiDef = { key: DashboardKpiKey; label: string; icon: IonName };

export type BaseDashboardConfig = {
  mode: CompanyDispatchMode;
  operationalQuestion: string;
  contextTitle: string;
  contextMessageDefault: string;
  kpiRow: [DashboardKpiDef, DashboardKpiDef, DashboardKpiDef, DashboardKpiDef];
  baseAlertOrder: string[];
  primaryCta: { label: string; path: string; params?: Record<string, string> };
};

const KPI_MANUAL: BaseDashboardConfig["kpiRow"] = [
  { key: "assign_pending", label: "À assigner", icon: "time-outline" },
  { key: "assign_in_progress", label: "En cours", icon: "play-outline" },
  { key: "delayed", label: "Retards", icon: "warning-outline" },
  { key: "drivers_available", label: "Chauffeurs dispos", icon: "navigate-circle-outline" },
] as const;

const KPI_SEMI: BaseDashboardConfig["kpiRow"] = [
  { key: "proposals", label: "Propositions", icon: "bulb-outline" },
  { key: "assign_without_proposal", label: "Sans proposition", icon: "help-circle-outline" },
  { key: "delayed", label: "Retards", icon: "time-outline" },
  { key: "opportunities", label: "Optimisations", icon: "git-compare-outline" },
] as const;

const KPI_AUTO: BaseDashboardConfig["kpiRow"] = [
  { key: "auto_assigned", label: "Auto-assignées", icon: "flash-outline" },
  { key: "exceptions", label: "Exceptions", icon: "alert-circle-outline" },
  { key: "delayed", label: "Retards", icon: "time-outline" },
  { key: "realtime_health", label: "Santé temps réel", icon: "pulse-outline" },
] as const;

const KPI_UNKNOWN: BaseDashboardConfig["kpiRow"] = KPI_MANUAL;

/**
 * Cadrage stable (mode seul) — le message et le CTA visibles sont ajustés par
 * `buildDashboardPresentation` selon le runtime.
 */
export function getDashboardModeConfig(mode: CompanyDispatchMode): BaseDashboardConfig {
  switch (mode) {
    case "manual":
      return {
        mode: "manual",
        operationalQuestion: "Que dois-je faire maintenant ?",
        contextTitle: "Pilotage manuel",
        contextMessageDefault: "Vous gardez la main sur les assignations.",
        kpiRow: KPI_MANUAL,
        baseAlertOrder: [
          "pending_past",
          "delayed",
          "network",
          "stale",
        ],
        primaryCta: { label: "Assigner les courses", path: ROUTE_RIDES },
      };
    case "semi_auto":
      return {
        mode: "semi_auto",
        operationalQuestion: "Que dois-je valider maintenant ?",
        contextTitle: "Pilotage semi-automatique",
        contextMessageDefault: "LIRIE propose, vous validez.",
        kpiRow: KPI_SEMI,
        baseAlertOrder: ["delayed", "stale", "network", "pending_past"],
        primaryCta: { label: "Voir les propositions", path: ROUTE_DISPATCH },
      };
    case "fully_auto":
      return {
        mode: "fully_auto",
        operationalQuestion: "Que dois-je surveiller ou corriger ?",
        contextTitle: "Dispatch automatique",
        contextMessageDefault:
          "LIRIE assigne automatiquement. Surveillez les exceptions.",
        kpiRow: KPI_AUTO,
        baseAlertOrder: [
          "engine",
          "stale",
          "delayed",
          "pending_past",
          "network",
        ],
        primaryCta: { label: "Voir les exceptions", path: ROUTE_RIDES, params: { filter: "exceptions" } },
      };
    default:
      return {
        mode: "unknown",
        operationalQuestion: "État d’exploitation",
        contextTitle: "Mode dispatch inconnu",
        contextMessageDefault: "Vérifiez la configuration côté serveur.",
        kpiRow: KPI_UNKNOWN,
        baseAlertOrder: ["stale", "network", "delayed", "pending_past"],
        primaryCta: { label: "Voir les courses", path: ROUTE_RIDES },
      };
  }
}

function mapKpiToAdvancedKey(k: DashboardKpiKey): AdvancedMetricKey | null {
  if (k === "proposals") return "propositions";
  if (k === "assign_without_proposal") return "unassignedWithoutProposition";
  if (k === "auto_assigned") return "autoAssigned";
  if (k === "exceptions") return "exceptions";
  return null;
}

function isAdvancedKpiKey(k: DashboardKpiKey): boolean {
  return mapKpiToAdvancedKey(k) != null;
}

function advancedMetricValue(
  key: DashboardKpiKey,
  metrics: DashboardRuntimeMetrics
): KpiBlockDisplay {
  const mk = mapKpiToAdvancedKey(key);
  if (!mk) return { kind: "unavailable" };
  if (!isAdvancedMetricAvailable(mk)) return { kind: "unavailable" };
  const n = metrics.advancedCounts?.[mk];
  if (n == null) return { kind: "unavailable" };
  return { kind: "value", line1: String(n) };
}

function resolveKpiValue(
  key: DashboardKpiKey,
  metrics: DashboardRuntimeMetrics,
  mode: CompanyDispatchMode
): KpiBlockDisplay {
  if (isAdvancedKpiKey(key)) {
    return advancedMetricValue(key, metrics);
  }
  if (key === "assign_pending")
    return { kind: "value", line1: String(metrics.missionsPending) };
  if (key === "assign_in_progress")
    return { kind: "value", line1: String(metrics.missionsInProgress) };
  if (key === "drivers_available")
    return { kind: "value", line1: String(metrics.driversAvailable) };
  if (key === "delayed") {
    const fromClient = mode === "manual" || mode === "unknown";
    if (!fromClient && !metrics.delayedBookingsMetricsAvailable) {
      return { kind: "unavailable" };
    }
    if (typeof metrics.delayedBookings === "number")
      return { kind: "value", line1: String(metrics.delayedBookings) };
    return { kind: "unavailable" };
  }
  if (key === "opportunities") {
    if (mode === "semi_auto" && !metrics.opportunitiesMetricsAvailable) {
      return { kind: "unavailable" };
    }
    if (typeof metrics.opportunities === "number")
      return { kind: "value", line1: String(metrics.opportunities) };
    return { kind: "unavailable" };
  }
  if (key === "realtime_health")
    return {
      kind: "value",
      line1: metrics.realtimeHealthyData ? "OK" : "À vérifier",
    };
  return { kind: "unavailable" };
}

export type DashboardRuntimeMetrics = {
  missions: CompanyDispatchMission[];
  missionsPending: number;
  missionsInProgress: number;
  /** Fiables en manuel / `unknown` (calcul local) ; sinon lié à `delayed_bookings_metrics_available` côté API. */
  delayedBookings: number;
  /** `false` si l’API n’expose pas le compteur retards (hors mode manuel où le calcul local suffit). */
  delayedBookingsMetricsAvailable: boolean;
  opportunities: number;
  /** Métier semi-auto : opportunités seulement si l’API renvoie le tableau. */
  opportunitiesMetricsAvailable: boolean;
  /**
   * Quand `isAdvancedMetricAvailable` est vrai, renseigner la valeur ; sinon l’UI reste en « non dispo ».
   */
  advancedCounts?: Partial<Record<AdvancedMetricKey, number | null | undefined>>;
  driversAvailable: number;
  driversEnMission: number;
  driversOffline: number;
  onlineDrivers: number;
  totalDrivers: number;
  isPotentiallyStale: boolean;
  hasPendingOverdue: boolean;
  isLikelyNetworkError: boolean;
  errMsg: string;
  dataHealthLabel: "Temps réel" | "Repli";
  realtimeHealthyData: boolean;
  /** Erreur HTTP 401/403 : une seule alerte explicite, sans « données obsolètes » ni doublon réseau. */
  isAuthFailure: boolean;
};

export type CompanyOptimizerRuntime = {
  optimizerEnabled: boolean;
  optimizerState: "idle" | "running" | "degraded" | "failed";
};

export type CompanyDispatchState = "idle" | "running" | "degraded" | "failed" | "unknown";

export type DashboardBuildInput = {
  config: BaseDashboardConfig;
  dispatchState: CompanyDispatchState;
  optimizer: CompanyOptimizerRuntime;
  socketStatus: CompanyRealtimeStatus;
  connected: boolean;
  metrics: DashboardRuntimeMetrics;
  hasDispatchScreen?: boolean;
};

export type AlertLine = { id: string; severity: "error" | "warning"; text: string };

export type DashboardViewModel = {
  operationalQuestion: string;
  contextTitle: string;
  contextMessage: string;
  showAutomationCaution: boolean;
  kpi: { def: DashboardKpiDef; display: KpiBlockDisplay }[];
  primaryCta: { label: string; path: string; params?: Record<string, string> };
  technicalLines: { label: string; value: string }[];
  alertLines: AlertLine[];
  dataHealthLabel: "Temps réel" | "Repli";
  optimizerLine: string;
};

/** Automatisation fully_auto considérée saine = TR + moteur + optim alignés. */
function automationCaution(
  mode: CompanyDispatchMode,
  input: {
    socketStatus: CompanyRealtimeStatus;
    connected: boolean;
    optimizer: CompanyOptimizerRuntime;
    dispatchState: CompanyDispatchState;
    isPotentiallyStale: boolean;
  }
): boolean {
  if (mode !== "fully_auto") return false;
  if (!input.connected) return true;
  if (
    input.socketStatus === "reconnecting" ||
    input.socketStatus === "connecting" ||
    input.socketStatus === "failed" ||
    input.socketStatus === "idle"
  )
    return true;
  if (input.isPotentiallyStale) return true;
  if (!input.optimizer.optimizerEnabled) return true;
  if (input.optimizer.optimizerState === "failed" || input.dispatchState === "failed") return true;
  if (input.dispatchState === "degraded" || input.dispatchState === "unknown")
    return true;
  return false;
}

function anyFluxConcern(input: { connected: boolean; isPotentiallyStale: boolean; socketStatus: CompanyRealtimeStatus }): boolean {
  if (!input.connected) return true;
  if (input.isPotentiallyStale) return true;
  if (
    input.socketStatus === "reconnecting" ||
    input.socketStatus === "connecting" ||
    input.socketStatus === "failed" ||
    input.socketStatus === "idle"
  )
    return true;
  return false;
}

function resolvePrimaryCta(
  base: BaseDashboardConfig,
  hasDispatch: boolean
): { label: string; path: string; params?: Record<string, string> } {
  if (base.mode === "semi_auto") {
    if (hasDispatch) return { label: "Voir les propositions", path: ROUTE_DISPATCH };
    return { label: "Voir les courses à traiter", path: ROUTE_RIDES };
  }
  if (base.mode === "fully_auto") {
    return { label: "Voir les exceptions", path: ROUTE_RIDES, params: { filter: "exceptions" } };
  }
  const c = { label: base.primaryCta.label, path: base.primaryCta.path };
  return base.primaryCta.params
    ? { ...c, params: base.primaryCta.params }
    : c;
}

/**
 * Ajuste la config stable selon moteur, TR et disponibilité des compteurs.
 */
export function buildDashboardPresentation(
  build: DashboardBuildInput
): DashboardViewModel {
  const { config, dispatchState, optimizer, socketStatus, connected, metrics } = build;
  const hasScreen = build.hasDispatchScreen ?? HAS_DISPATCH_SCREEN;
  const autoCaution = automationCaution(config.mode, {
    socketStatus,
    connected,
    optimizer,
    dispatchState,
    isPotentiallyStale: metrics.isPotentiallyStale,
  });

  const flux = anyFluxConcern({ connected, isPotentiallyStale: metrics.isPotentiallyStale, socketStatus });

  let contextMessage = config.contextMessageDefault;
  let showAutomationCaution = false;
  if (config.mode === "fully_auto") {
    if (autoCaution) {
      contextMessage = "Mode automatique configuré, mais automatisation à vérifier.";
      showAutomationCaution = true;
    }
  } else if (config.mode === "manual" || config.mode === "semi_auto" || config.mode === "unknown") {
    if (flux) {
      contextMessage = "Les données en temps réel ou le moteur nécessitent un contrôle.";
    }
  }

  const kpi = config.kpiRow.map((def) => ({
    def,
    display: resolveKpiValue(def.key, metrics, config.mode),
  }));
  const primaryCta = resolvePrimaryCta(config, hasScreen);

  const engineWarning: AlertLine[] =
    config.mode === "fully_auto" && autoCaution
      ? [
          {
            id: "engine",
            severity: "warning",
            text: "Moteur dispatch ou optimisateur à vérifier, ou flux dégradé.",
          },
        ]
      : [];

  const alertLines: AlertLine[] = [];

  for (const id of config.baseAlertOrder) {
    if (id === "pending_past" && metrics.hasPendingOverdue) {
      alertLines.push({
        id: "pending_past",
        severity: "warning",
        text: "Des courses en attente ont un horaire dépassé.",
      });
    } else if (id === "delayed" && typeof metrics.delayedBookings === "number" && metrics.delayedBookings > 0) {
      alertLines.push({
        id: "delayed",
        severity: "warning",
        text: `${metrics.delayedBookings} retard(s) signalé(s) sur le réseau.`,
      });
    } else if (id === "stale" && metrics.isPotentiallyStale) {
      alertLines.push({
        id: "stale",
        severity: "error",
        text: "Données poss. obsolètes — vérifiez le flux et synchronisez.",
      });
    } else if (id === "network" && metrics.isLikelyNetworkError && metrics.errMsg) {
      alertLines.push({ id: "network", severity: "error", text: metrics.errMsg });
    } else if (id === "engine" && engineWarning[0] && !alertLines.some((a) => a.id === "engine")) {
      alertLines.push(...engineWarning);
    }
  }

  if (
    !metrics.isAuthFailure &&
    metrics.isLikelyNetworkError &&
    metrics.errMsg &&
    !alertLines.some((a) => a.id === "network")
  ) {
    alertLines.push({ id: "network", severity: "error", text: metrics.errMsg });
  }

  const optLine = !optimizer.optimizerEnabled
    ? "Inactif"
    : optimizer.optimizerState === "running"
      ? "En cours"
      : optimizer.optimizerState === "failed"
        ? "Inactif"
        : "Actif";

  return {
    operationalQuestion: config.operationalQuestion,
    contextTitle: config.contextTitle,
    contextMessage,
    showAutomationCaution,
    kpi,
    primaryCta: {
      label: primaryCta.label,
      path: primaryCta.path,
      params: primaryCta.params,
    },
    technicalLines: [
      { label: "Mode (configuré)", value: formatDispatchModeFr(config.mode) },
      { label: "État moteur dispatch", value: formatDispatchStateFr(dispatchState) },
      { label: "Optimiseur", value: optLine },
      { label: "Données", value: metrics.dataHealthLabel },
    ],
    alertLines: dedupeAlerts(alertLines),
    dataHealthLabel: metrics.dataHealthLabel,
    optimizerLine: optLine,
  };
}

function dedupeAlerts(a: AlertLine[]): AlertLine[] {
  const seen = new Set<string>();
  return a.filter((l) => {
    if (seen.has(l.id)) return false;
    seen.add(l.id);
    return true;
  });
}

export function formatDispatchStateFr(state: CompanyDispatchState): string {
  switch (state) {
    case "idle":
      return "Inactif";
    case "running":
      return "En cours d’exécution";
    case "degraded":
      return "Dégradé";
    case "failed":
      return "En échec";
    default:
      return "Inconnu";
  }
}

export type DashboardKpiNavigationTarget = { path: string; params?: Record<string, string> };

/**
 * Cible de navigation depuis une tuile KPI (tableau de bord entreprise).
 * Les paramètres sont consommés par `app/(app)/(company)/rides.tsx` (`status`, `filter`).
 */
export function resolveDashboardKpiNavigation(
  key: DashboardKpiKey,
  input: { hasDispatchScreen: boolean }
): DashboardKpiNavigationTarget | null {
  switch (key) {
    case "assign_pending":
      return { path: ROUTE_RIDES, params: { status: "pending" } };
    case "assign_in_progress":
      return { path: ROUTE_RIDES, params: { status: "in_flight" } };
    case "delayed":
      return { path: ROUTE_RIDES, params: { filter: "delayed" } };
    case "exceptions":
      return { path: ROUTE_RIDES, params: { filter: "exceptions" } };
    case "auto_assigned":
      return { path: ROUTE_RIDES, params: { status: "assigned" } };
    case "proposals":
    case "assign_without_proposal":
      if (input.hasDispatchScreen) return { path: ROUTE_DISPATCH };
      return { path: ROUTE_RIDES };
    case "drivers_available":
    case "opportunities":
    case "realtime_health":
      return null;
    default:
      return null;
  }
}

export function formatDispatchModeFr(mode: CompanyDispatchMode): string {
  switch (mode) {
    case "manual":
      return "Manuel";
    case "semi_auto":
      return "Semi-automatique";
    case "fully_auto":
      return "Entièrement auto";
    default:
      return "Inconnu";
  }
}
