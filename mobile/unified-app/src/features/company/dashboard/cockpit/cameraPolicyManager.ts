import type { CockpitFsmState } from "./cockpitFiniteStateMachine";
import type { AttentionLevel } from "./fleetAttentionSystem";
import type { MapIntent } from "./mapIntentResolver";

export type CameraPolicy =
  | "global_soft_fit"
  | "driver_focus_fit"
  | "incident_soft_attention"
  | "user_gesture_preserve";

export type CameraPolicyInput = {
  fsmState: CockpitFsmState;
  mapIntent: MapIntent;
  attentionLevel: AttentionLevel;
  manualRecenterOnly: boolean;
  userGestureActive?: boolean;
};

export function resolveCameraPolicy(input: CameraPolicyInput): CameraPolicy {
  if (input.manualRecenterOnly || input.userGestureActive) {
    return "user_gesture_preserve";
  }
  if (input.attentionLevel === "CRITICAL" && input.mapIntent === "IncidentResponseIntent") {
    return "incident_soft_attention";
  }
  if (
    input.fsmState === "DRIVER_FOCUS" ||
    input.mapIntent === "DriverMonitoringIntent"
  ) {
    return "driver_focus_fit";
  }
  return "global_soft_fit";
}
