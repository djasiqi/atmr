import { useMemo, useRef } from "react";
import {
  resolveCockpitOrchestration,
  type CockpitOrchestrationDecision,
  type CockpitOrchestratorInput,
} from "./cockpitOrchestrator";
import type { StabilityBudgetState } from "./cockpitStabilityBudget";
import { createStabilityBudgetState } from "./cockpitStabilityBudget";

export type UseCockpitOrchestrationOptions = CockpitOrchestratorInput;

export function useCockpitOrchestration(
  options: UseCockpitOrchestrationOptions
): CockpitOrchestrationDecision {
  const stabilityRef = useRef<StabilityBudgetState>(createStabilityBudgetState());

  return useMemo(() => {
    const decision = resolveCockpitOrchestration({
      ...options,
      stabilityState: stabilityRef.current,
    });
    stabilityRef.current = decision.stabilityState;
    return decision;
  }, [options]);
}
