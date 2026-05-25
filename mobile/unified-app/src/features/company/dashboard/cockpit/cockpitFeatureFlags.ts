import { isFeatureEnabled } from "../../../../core/featureFlags/registry";

/**
 * Feature flags cockpit — couches avancées activables indépendamment.
 *
 * ACTIF : safeModeAuto, lowAttentionMode, adaptiveOverlays (via gouvernance),
 *         cockpitMetrics (dev), simplifyClustering (désactive le clustering carte).
 * STUB (flag sans logique dédiée) : predictiveSimplification, attentionRouting
 *         — voir cockpitGovernance.ts ; activer l’env ne change rien pour l’instant.
 * RÉSERVÉ : cockpitPolicyPriority.ts — graphe non branché sur computeCockpitUiState.
 */
export type CockpitAdvancedFlags = {
  predictiveSimplification: boolean;
  attentionRouting: boolean;
  adaptiveOverlays: boolean;
  safeModeAuto: boolean;
  lowAttentionMode: boolean;
  cockpitMetrics: boolean;
  cockpitDebugger: boolean;
};

function envOn(name: string, defaultOn = false): boolean {
  const v =
    name === "EXPO_PUBLIC_COCKPIT_PREDICTIVE_SIMPLIFICATION"
      ? process.env.EXPO_PUBLIC_COCKPIT_PREDICTIVE_SIMPLIFICATION
      : name === "EXPO_PUBLIC_COCKPIT_ATTENTION_ROUTING"
        ? process.env.EXPO_PUBLIC_COCKPIT_ATTENTION_ROUTING
        : name === "EXPO_PUBLIC_COCKPIT_ADAPTIVE_OVERLAYS"
          ? process.env.EXPO_PUBLIC_COCKPIT_ADAPTIVE_OVERLAYS
          : name === "EXPO_PUBLIC_COCKPIT_SAFE_MODE_AUTO"
            ? process.env.EXPO_PUBLIC_COCKPIT_SAFE_MODE_AUTO
            : name === "EXPO_PUBLIC_COCKPIT_LOW_ATTENTION"
              ? process.env.EXPO_PUBLIC_COCKPIT_LOW_ATTENTION
              : name === "EXPO_PUBLIC_COCKPIT_METRICS"
                ? process.env.EXPO_PUBLIC_COCKPIT_METRICS
                : name === "EXPO_PUBLIC_COCKPIT_DEBUGGER"
                  ? process.env.EXPO_PUBLIC_COCKPIT_DEBUGGER
                  : undefined;
  if (v === undefined) return defaultOn;
  return v === "1";
}

/** Flags locaux cockpit (rollout progressif). */
export function getCockpitAdvancedFlags(): CockpitAdvancedFlags {
  return {
    predictiveSimplification: envOn("EXPO_PUBLIC_COCKPIT_PREDICTIVE_SIMPLIFICATION", false),
    attentionRouting: envOn("EXPO_PUBLIC_COCKPIT_ATTENTION_ROUTING", false),
    adaptiveOverlays: envOn("EXPO_PUBLIC_COCKPIT_ADAPTIVE_OVERLAYS", true),
    safeModeAuto: envOn("EXPO_PUBLIC_COCKPIT_SAFE_MODE_AUTO", true),
    lowAttentionMode: envOn("EXPO_PUBLIC_COCKPIT_LOW_ATTENTION", true),
    cockpitMetrics: envOn("EXPO_PUBLIC_COCKPIT_METRICS", __DEV__),
    /** Visible uniquement si EXPO_PUBLIC_COCKPIT_DEBUGGER=1 (jamais par défaut en __DEV__). */
    cockpitDebugger: envOn("EXPO_PUBLIC_COCKPIT_DEBUGGER", false),
  };
}

/** Core toujours actif si dispatch company + map clustering registry. */
export function isCockpitCoreEnabled(): boolean {
  return isFeatureEnabled("company_dispatch_enabled") || isFeatureEnabled("company_realtime_enabled");
}
