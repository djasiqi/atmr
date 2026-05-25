/**
 * Ordre de dominance des policies cockpit (plus haut = prioritaire).
 * RÉSERVÉ — non consommé par computeCockpitUiState() ; priorité codée dans cockpitGovernance.ts.
 */
export const COCKPIT_POLICY_PRIORITY = [
  "safe_mode",
  "no_surprises",
  "human_override",
  "failure_modes",
  "operational_trust",
  "predictive_simplification",
  "attention_routing",
  "fatigue_prevention",
] as const;

export type CockpitPolicyId = (typeof COCKPIT_POLICY_PRIORITY)[number];

export function comparePolicyPriority(a: CockpitPolicyId, b: CockpitPolicyId): number {
  return COCKPIT_POLICY_PRIORITY.indexOf(a) - COCKPIT_POLICY_PRIORITY.indexOf(b);
}
