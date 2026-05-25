/**
 * Ownership autoritatif des états cockpit — évite double state / dérivations divergentes.
 * Orchestrator = décision finale uniquement (ne possède pas de vérité métier durable).
 */
export const COCKPIT_STATE_OWNERS = {
  fsm: "CockpitFiniteStateMachine",
  attention: "FleetAttentionSystem",
  density: "MapDensityGovernor",
  mapIntent: "MapIntentResolver",
  motion: "MotionPolicy",
  orchestrator: "CockpitOrchestrator",
} as const;

export type CockpitStateOwner = (typeof COCKPIT_STATE_OWNERS)[keyof typeof COCKPIT_STATE_OWNERS];

export type CockpitOwnedDomain =
  | "fsmState"
  | "attentionLevel"
  | "densityLevel"
  | "mapIntent"
  | "motionState";

export const DOMAIN_TO_OWNER: Record<CockpitOwnedDomain, CockpitStateOwner> = {
  fsmState: COCKPIT_STATE_OWNERS.fsm,
  attentionLevel: COCKPIT_STATE_OWNERS.attention,
  densityLevel: COCKPIT_STATE_OWNERS.density,
  mapIntent: COCKPIT_STATE_OWNERS.mapIntent,
  motionState: COCKPIT_STATE_OWNERS.motion,
};
