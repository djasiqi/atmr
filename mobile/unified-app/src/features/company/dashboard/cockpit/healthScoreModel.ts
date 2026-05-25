import { resolveCockpitLiveStatus } from "./cockpitLiveStatus";

/** Version du modèle — incrémenter si les poids changent. */
export const HEALTH_SCORE_MODEL_VERSION = 1;

/**
 * Plages attendues :
 * - nominal: 70–100
 * - degraded: 35–69
 * - critical: < 35 (safe_mode, reduced complexity)
 */
export type FleetHealthScoreInput = {
  delayedCount: number;
  urgentCount: number;
  unassignedCount: number;
  criticalEtaCount: number;
  realtimeStatus: string;
  realtimeDataFreshness?: string;
  policyFailureCount: number;
  interactionBurstPerMinute: number;
};

/** Seule autorité pour le score santé flotte. */
export function resolveFleetHealthScore(input: FleetHealthScoreInput): number {
  let score = 100;

  // Opérationnel
  score -= input.delayedCount * 8;
  score -= input.urgentCount * 10;
  score -= input.unassignedCount * 5;
  score -= input.criticalEtaCount * 12;

  // Connectivité / fraîcheur
  const live = resolveCockpitLiveStatus(input.realtimeStatus);
  if (live === "reconnecting") score -= 12;
  if (live === "offline") score -= 35;
  if (input.realtimeDataFreshness === "stale") score -= 8;

  // Fiabilité runtime
  score -= input.policyFailureCount * 6;
  if (input.interactionBurstPerMinute > 40) score -= 10;

  return Math.max(0, Math.min(100, score));
}
