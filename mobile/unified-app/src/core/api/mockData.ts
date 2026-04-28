import { BootstrapResponse, SwitchContextResponse } from "../contracts/auth";

export function buildMockBootstrap(): BootstrapResponse {
  return {
    bootstrap_version: "1.0.0",
    is_authenticated: true,
    user: { id: "u-1", email: "demo@lirie.ch", full_name: "Demo User" },
    account_status: "active",
    onboarding_status: { required: false },
    available_contexts: [
      {
        context_id: "client:self",
        context_type: "client",
        label: "Compte client",
        organization_id: null,
        organization_name: null,
        permissions: ["booking:create", "booking:read:self"],
        is_default: true,
      },
      {
        context_id: "driver:42",
        context_type: "driver",
        label: "Chauffeur",
        organization_id: 9,
        organization_name: "Lirie Transport",
        permissions: ["mission:read", "mission:update_status"],
        is_default: false,
        allow_mobile_context_switch: true,
      },
    ],
    active_context_id: "client:self",
    feature_flags: { driver_live_tracking: true },
    min_supported_app_version: "0.1.0",
    maintenance_mode: false,
    degraded_mode: false,
    server_time: new Date().toISOString(),
    request_id: "mock-bootstrap-001",
  };
}

export function buildMockSwitchContext(targetContextId: string): SwitchContextResponse {
  return {
    success: true,
    active_context_id: targetContextId,
    request_id: "mock-switch-001",
  };
}
