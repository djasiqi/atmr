import { describe, expect, it } from "@jest/globals";
import { resolveInitialRoute } from "./resolveInitialRoute";

function makeBootstrap(overrides: Record<string, unknown> = {}) {
  return {
    bootstrap_version: "1.0.0",
    is_authenticated: true,
    user: { id: "u-1", email: "demo@lirie.ch" },
    account_status: "active",
    onboarding_status: { required: false },
    available_contexts: [
      {
        context_id: "client:self",
        context_type: "client",
        label: "Client",
        permissions: ["booking:read:self"],
        is_default: true,
      },
    ],
    active_context_id: "client:self",
    feature_flags: {},
    min_supported_app_version: "0.1.0",
    maintenance_mode: false,
    degraded_mode: false,
    server_time: new Date().toISOString(),
    request_id: "req-123456",
    ...overrides,
  } as any;
}

describe("resolveInitialRoute", () => {
  it("routes public when not authenticated", () => {
    expect(resolveInitialRoute(makeBootstrap({ is_authenticated: false }))).toBe("/(public)");
  });

  it("routes maintenance when maintenance mode is on", () => {
    expect(resolveInitialRoute(makeBootstrap({ maintenance_mode: true }))).toBe("/(app)/maintenance");
  });

  it("routes onboarding when required", () => {
    expect(resolveInitialRoute(makeBootstrap({ onboarding_status: { required: true } }))).toBe(
      "/(app)/onboarding"
    );
  });

  it("routes by active context", () => {
    expect(resolveInitialRoute(makeBootstrap())).toBe("/(app)/(client)");
  });

  it("routes to context selector when driver gate is off and another context exists", () => {
    const bootstrap = makeBootstrap({
      available_contexts: [
        {
          context_id: "driver:me",
          context_type: "driver",
          label: "Chauffeur",
          permissions: ["driver:missions:read"],
          is_default: true,
        },
        {
          context_id: "company:42",
          context_type: "company",
          label: "Entreprise",
          permissions: ["company:rides:read"],
          is_default: false,
        },
      ],
      active_context_id: "driver:me",
      feature_flags: { driver_unified_enabled: false },
    });
    expect(resolveInitialRoute(bootstrap)).toBe("/(app)/context-selector");
  });

  it("routes to blocked screen when driver gate is off and no alternative context exists", () => {
    const bootstrap = makeBootstrap({
      available_contexts: [
        {
          context_id: "driver:me",
          context_type: "driver",
          label: "Chauffeur",
          permissions: ["driver:missions:read"],
          is_default: true,
        },
      ],
      active_context_id: "driver:me",
      feature_flags: { driver_unified_enabled: false },
    });
    expect(resolveInitialRoute(bootstrap)).toBe("/(app)/blocked?reason=driver_gate");
  });

  it("routes to driver when no driver_unified key (default on, même session qu’un compte chauffeur)", () => {
    const bootstrap = makeBootstrap({
      available_contexts: [
        {
          context_id: "driver:me",
          context_type: "driver",
          label: "Chauffeur",
          permissions: ["driver:missions:read"],
          is_default: true,
        },
      ],
      active_context_id: "driver:me",
      feature_flags: {},
    });
    expect(resolveInitialRoute(bootstrap)).toBe("/(app)/(driver)");
  });

  it("routes company deep link when company context is active", () => {
    const bootstrap = makeBootstrap({
      available_contexts: [
        {
          context_id: "company:42",
          context_type: "company",
          label: "Dispatch",
          permissions: ["company:rides:read"],
          is_default: true,
        },
      ],
      active_context_id: "company:42",
    });
    expect(resolveInitialRoute(bootstrap, "atmr://transfer/99")).toBe(
      "/(app)/(company)/ride-details?rideId=99"
    );
  });
});
