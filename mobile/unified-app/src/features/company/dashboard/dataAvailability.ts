/**
 * Métriques “avancées” : tant qu’une entrée vaut `false`, l’UI affiche
 * `—` / “Non disponible” — jamais un 0 implicite.
 * Passer à `true` quand l’API alimente la clé.
 */
export const DISPATCH_DASHBOARD_ADVANCED_METRICS: Record<
  "propositions" | "unassignedWithoutProposition" | "autoAssigned" | "exceptions",
  boolean
> = {
  propositions: false,
  unassignedWithoutProposition: false,
  autoAssigned: false,
  exceptions: false,
};

export type AdvancedMetricKey = keyof typeof DISPATCH_DASHBOARD_ADVANCED_METRICS;

export function isAdvancedMetricAvailable(key: AdvancedMetricKey): boolean {
  return DISPATCH_DASHBOARD_ADVANCED_METRICS[key] === true;
}
