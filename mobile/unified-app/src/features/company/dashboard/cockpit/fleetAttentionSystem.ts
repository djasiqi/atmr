/** Niveaux d'attention flotte — tous les événements ne méritent pas l'attention utilisateur. */
export type AttentionLevel = "INFO" | "WARNING" | "CRITICAL";

export type FleetAttentionInput = {
  urgentCount: number;
  delayedCount: number;
  unassignedCount: number;
  criticalEtaCount: number;
  healthScore: number;
};

export type FleetAttentionResult = {
  level: AttentionLevel;
  canInterruptMode: boolean;
  highlightOnly: boolean;
  reason: string;
};

export function resolveFleetAttention(input: FleetAttentionInput): FleetAttentionResult {
  if (input.urgentCount >= 3 || input.healthScore < 25 || input.criticalEtaCount >= 2) {
    return {
      level: "CRITICAL",
      canInterruptMode: true,
      highlightOnly: false,
      reason: "critical_operational_load",
    };
  }
  if (input.delayedCount >= 2 || input.unassignedCount >= 3 || input.healthScore < 45) {
    return {
      level: "WARNING",
      canInterruptMode: false,
      highlightOnly: true,
      reason: "operational_warning",
    };
  }
  return {
    level: "INFO",
    canInterruptMode: false,
    highlightOnly: false,
    reason: "nominal",
  };
}
