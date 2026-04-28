export const DRIVER_RUNTIME_CONTRACT_VERSION = "decommission-driver-runtime-v1";

export type DriverRuntimeResyncTrigger =
  | "reconnect"
  | "foreground"
  | "mission_detail_open"
  | "push_update";

export type DriverTrackingMode =
  | "off"
  | "availability_presence"
  | "mission_live"
  | "mission_live_degraded"
  | "observability_only";

export type DriverRuntimeContractEvent =
  | "runtime.bootstrap.start"
  | "runtime.bootstrap.ready"
  | "runtime.resync"
  | "runtime.reconcile"
  | "runtime.push.route"
  | "runtime.socket.event";

export type DriverRuntimeContractEnvelope = {
  version: string;
  event: DriverRuntimeContractEvent;
  at: string;
  contextId?: string | null;
  missionId?: number | null;
  trigger?: DriverRuntimeResyncTrigger;
  details?: Record<string, unknown>;
};

export function createRuntimeEnvelope(
  event: DriverRuntimeContractEvent,
  details?: Omit<DriverRuntimeContractEnvelope, "version" | "event" | "at">
): DriverRuntimeContractEnvelope {
  return {
    version: DRIVER_RUNTIME_CONTRACT_VERSION,
    event,
    at: new Date().toISOString(),
    ...details,
  };
}
