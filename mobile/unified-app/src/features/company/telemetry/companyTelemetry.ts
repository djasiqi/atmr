import {
  emitPlatformTelemetry,
  PlatformTelemetryPayload,
} from "../../../core/observability/platformTelemetry";
import { isFeatureEnabled } from "../../../core/featureFlags/registry";

export type CompanyDispatchTelemetryEvent =
  | "company.dispatch.opened"
  | "company.dispatch.driver_selected"
  | "company.dispatch.delay_invalidated"
  | "company.dispatch.optimizer_status_requested"
  | "company.dispatch.socket_state_changed"
  | "company.dispatch.contract_fallback"
  | "company.dispatch.contract_failure"
  | "company.dispatch.auth_failure"
  | "company.dispatch.transfer_conflict";

type EmitCompanyTelemetryOptions = {
  allowWhenDisabled?: boolean;
};

export function emitCompanyDispatchTelemetry(
  event: CompanyDispatchTelemetryEvent,
  payload: PlatformTelemetryPayload,
  options: EmitCompanyTelemetryOptions = {}
): boolean {
  const companyEnabled = isFeatureEnabled("company_dispatch_enabled");
  if (!companyEnabled && !options.allowWhenDisabled) {
    return false;
  }

  const enrichedPayload: PlatformTelemetryPayload = {
    ...payload,
    timestamp_client:
      typeof payload.timestamp_client === "string"
        ? payload.timestamp_client
        : new Date().toISOString(),
  };
  emitPlatformTelemetry("company", event, enrichedPayload);
  return true;
}
