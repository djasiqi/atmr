import { emitDriverTelemetry } from "../../../core/observability/driverTelemetry";
import { isTrackingActiveStatus } from "../domain/status";
import { DriverMissionStatus } from "../types";
import { evaluateBackgroundTrackingGate } from "./backgroundTrackingGating";
import {
  getTrackingRuntimeState,
  hydrateTrackingRuntimeState,
  updateTrackingRuntimeState,
} from "./trackingRuntime";
import { DriverTrackingMode } from "../runtimeContracts";

function resolveMode(status: DriverMissionStatus | null): DriverTrackingMode {
  if (!status) return "off";
  if (status === "ASSIGNED") return "availability_presence";
  if (isTrackingActiveStatus(status)) return "mission_live";
  return "off";
}

export async function reconcileTrackingRuntime(params: {
  contextId: string;
  missionId: number | null;
  missionStatus: DriverMissionStatus | null;
}): Promise<DriverTrackingMode> {
  await hydrateTrackingRuntimeState();
  const gate = await evaluateBackgroundTrackingGate();
  const desiredMode = resolveMode(params.missionStatus);
  const mode: DriverTrackingMode =
    gate.enabled && gate.permission === "denied" && desiredMode !== "off"
      ? "mission_live_degraded"
      : desiredMode;
  await updateTrackingRuntimeState({
    missionId: params.missionId,
    mode,
  });
  emitDriverTelemetry("driver.runtime.reconcile", {
    source: "driver.tracking.reconcile",
    context_id: params.contextId,
    mission_id: params.missionId,
    tracking_mode: mode,
    background_enabled: gate.enabled,
    background_permission: gate.permission,
    services_enabled: gate.servicesEnabled,
  });
  return getTrackingRuntimeState().mode;
}
