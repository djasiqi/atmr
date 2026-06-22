import { Platform } from "react-native";
import { apiClient } from "../../../core/api/client";

export type CompanyPushTelemetryEvent =
  | "company_push.new_request.opened"
  | "company_push.new_request.tap_without_network"
  | "company_push.new_request.open_to_accept";

type ReportCompanyPushTelemetryInput = {
  event: CompanyPushTelemetryEvent;
  offerId?: number;
  requestId?: number;
  seconds?: number;
  source?: string;
};

export async function reportCompanyPushTelemetry(
  input: ReportCompanyPushTelemetryInput
): Promise<void> {
  try {
    await apiClient.post("/companies/me/telemetry/push", {
      event: input.event,
      offer_id: input.offerId,
      request_id: input.requestId,
      seconds: input.seconds,
      platform: Platform.OS,
      source: input.source ?? "company.notifications.bridge",
    });
  } catch {
    // Télémétrie best-effort — ne bloque pas l'UX
  }
}
