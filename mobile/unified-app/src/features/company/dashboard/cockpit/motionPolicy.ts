import type { AttentionLevel } from "./fleetAttentionSystem";
import type { DensityLevel } from "./mapDensityGovernor";

export type MotionPolicyState = "ALLOW" | "LIMIT" | "DISABLE";

export type MotionPolicyInput = {
  densityLevel: DensityLevel;
  attentionLevel: AttentionLevel;
  reducedComplexity: boolean;
  safeMode: boolean;
};

export function resolveMotionPolicy(input: MotionPolicyInput): MotionPolicyState {
  if (input.safeMode || input.reducedComplexity) return "DISABLE";
  if (input.densityLevel === "aggregate" || input.densityLevel === "extreme") {
    return "LIMIT";
  }
  if (input.attentionLevel === "CRITICAL") return "ALLOW";
  if (input.densityLevel === "high") return "LIMIT";
  return "ALLOW";
}

export function motionAllowsSelectionPulse(state: MotionPolicyState): boolean {
  return state !== "DISABLE";
}

export function motionAllowsIncidentPulse(state: MotionPolicyState): boolean {
  return state === "ALLOW";
}

export function motionAllowsDecorativeGlow(state: MotionPolicyState): boolean {
  return state === "ALLOW";
}
