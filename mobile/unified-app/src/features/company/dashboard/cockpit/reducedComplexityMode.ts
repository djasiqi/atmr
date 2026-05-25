import type { DensityLevel } from "./mapDensityGovernor";

export type ReducedComplexityInput = {
  densityLevel: DensityLevel;
  healthScore: number;
  lowPowerMode?: boolean;
  fpsBelowThreshold?: boolean;
};

export function shouldEnableReducedComplexity(input: ReducedComplexityInput): boolean {
  if (input.lowPowerMode || input.fpsBelowThreshold) return true;
  if (input.healthScore < 30) return true;
  if (input.densityLevel === "aggregate" || input.densityLevel === "extreme") {
    return true;
  }
  return false;
}

export type ReducedComplexityEffects = {
  disableShadows: boolean;
  disableGlow: boolean;
  disablePulses: boolean;
  disableLiveEtaLabels: boolean;
  disableGradients: boolean;
  simplifyTransitions: boolean;
};

export function resolveReducedComplexityEffects(
  enabled: boolean
): ReducedComplexityEffects {
  if (!enabled) {
    return {
      disableShadows: false,
      disableGlow: false,
      disablePulses: false,
      disableLiveEtaLabels: false,
      disableGradients: false,
      simplifyTransitions: false,
    };
  }
  return {
    disableShadows: true,
    disableGlow: true,
    disablePulses: true,
    disableLiveEtaLabels: true,
    disableGradients: true,
    simplifyTransitions: true,
  };
}
