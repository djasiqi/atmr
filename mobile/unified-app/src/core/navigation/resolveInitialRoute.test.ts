import { describe, expect, it } from "@jest/globals";
import { resolveInitialRoute } from "./resolveInitialRoute";
import type { MobileSessionStatus } from "../auth/mobileSessionStatus";

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

const online: MobileSessionStatus = "authenticated_online";

describe("resolveInitialRoute", () => {
  it("routes public when terminal unauthenticated", () => {
    expect(
      resolveInitialRoute(makeBootstrap({ is_authenticated: false }), null, "anonymous")
    ).toBe("/(public)");
  });

  it("P0-4/5 final: recovering / degraded → destination app (pas login, pas null flash)", () => {
    const anon = makeBootstrap({ is_authenticated: false });
    expect(resolveInitialRoute(anon, null, "auth_recovering")).toBe("/(app)/(client)");
    expect(resolveInitialRoute(anon, null, "authenticated_offline")).toBe("/(app)/(client)");
    expect(resolveInitialRoute(makeBootstrap(), null, "initializing")).toBeNull();
  });

  it("P0-4: revoked → /(public)", () => {
    expect(
      resolveInitialRoute(makeBootstrap({ is_authenticated: false }), null, "revoked")
    ).toBe("/(public)");
  });

  it("routes maintenance when maintenance mode is on", () => {
    expect(resolveInitialRoute(makeBootstrap({ maintenance_mode: true }), null, online)).toBe(
      "/(app)/maintenance"
    );
  });

  it("routes onboarding when required", () => {
    expect(
      resolveInitialRoute(makeBootstrap({ onboarding_status: { required: true } }), null, online)
    ).toBe("/(app)/onboarding");
  });

  it("routes by active context", () => {
    expect(resolveInitialRoute(makeBootstrap(), null, online)).toBe("/(app)/(client)");
  });

  it("routes to context selector when driver gate is off and another context exists", () => {
    const bootstrap = makeBootstrap({
      available_contexts: [
        {
          context_id: "driver:1",
          context_type: "driver",
          label: "Chauffeur",
          permissions: ["mission:read"],
          is_default: true,
        },
        {
          context_id: "client:self",
          context_type: "client",
          label: "Client",
          permissions: ["booking:read:self"],
          is_default: false,
        },
      ],
      active_context_id: "driver:1",
      feature_flags: { driver_unified_enabled: false },
    });
    expect(resolveInitialRoute(bootstrap, null, online)).toBe("/(app)/context-selector");
  });

  it("routes blocked when driver gate is off and only driver context", () => {
    const bootstrap = makeBootstrap({
      available_contexts: [
        {
          context_id: "driver:1",
          context_type: "driver",
          label: "Chauffeur",
          permissions: ["mission:read"],
          is_default: true,
        },
      ],
      active_context_id: "driver:1",
      feature_flags: { driver_unified_enabled: false },
    });
    expect(resolveInitialRoute(bootstrap, null, online)).toBe("/(app)/blocked?reason=driver_gate");
  });

  it("routes driver when unified enabled", () => {
    const bootstrap = makeBootstrap({
      available_contexts: [
        {
          context_id: "driver:1",
          context_type: "driver",
          label: "Chauffeur",
          permissions: ["mission:read"],
          is_default: true,
        },
      ],
      active_context_id: "driver:1",
    });
    expect(resolveInitialRoute(bootstrap, null, online)).toBe("/(app)/(driver)");
  });

  it("routes driver deep link ignored when not a driver link → driver home", () => {
    const bootstrap = makeBootstrap({
      available_contexts: [
        {
          context_id: "driver:1",
          context_type: "driver",
          label: "Chauffeur",
          permissions: ["mission:read"],
          is_default: true,
        },
      ],
      active_context_id: "driver:1",
    });
    expect(resolveInitialRoute(bootstrap, "atmr://transfer/99", online)).toBe("/(app)/(driver)");
  });

  it("routes institution gate off with alt context", () => {
    const bootstrap = makeBootstrap({
      available_contexts: [
        {
          context_id: "institution:1",
          context_type: "institution",
          label: "Institution",
          permissions: [],
          is_default: true,
        },
        {
          context_id: "client:self",
          context_type: "client",
          label: "Client",
          permissions: ["booking:read:self"],
          is_default: false,
        },
      ],
      active_context_id: "institution:1",
      feature_flags: { institution_unified_enabled: false },
    });
    expect(resolveInitialRoute(bootstrap, null, online)).toBe("/(app)/context-selector");
  });

  it("routes institution gate off only institution", () => {
    const bootstrap = makeBootstrap({
      available_contexts: [
        {
          context_id: "institution:1",
          context_type: "institution",
          label: "Institution",
          permissions: [],
          is_default: true,
        },
      ],
      active_context_id: "institution:1",
      feature_flags: { institution_unified_enabled: false },
    });
    expect(resolveInitialRoute(bootstrap, null, online)).toBe(
      "/(app)/blocked?reason=institution_gate"
    );
  });

  it("routes institution when enabled", () => {
    const bootstrap = makeBootstrap({
      available_contexts: [
        {
          context_id: "institution:1",
          context_type: "institution",
          label: "Institution",
          permissions: [],
          is_default: true,
        },
      ],
      active_context_id: "institution:1",
      feature_flags: { institution_unified_enabled: true },
    });
    expect(resolveInitialRoute(bootstrap, null, online)).toBe("/(app)/(institution)");
  });
});
