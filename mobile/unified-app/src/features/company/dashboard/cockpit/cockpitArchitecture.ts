/**
 * Garde-fous architecture cockpit (référence statique).
 * Orchestrator = décision | Managers = calcul pur | UI = rendu
 * Aucun module carte/caméra/motion n'agit sans passer par CockpitOrchestrator.
 */
export const COCKPIT_ARCHITECTURE_LAYERS = {
  decision: "CockpitOrchestrator",
  pureManagers: [
    "CockpitFiniteStateMachine",
    "FleetAttentionSystem",
    "MapDensityGovernor",
    "MapIntentResolver",
    "MotionPolicy",
    "SemanticRouteSystem",
    "CameraPolicyManager",
  ],
  rendering: "React UI components (OperationalFleetMap, DriverBottomSheet, …)",
} as const;
