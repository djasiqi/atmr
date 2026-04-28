import { describe, expect, it } from "@jest/globals";
import { bootstrapResponseSchema, hasPermission, resolveDefaultContext } from "./auth";

describe("auth contracts", () => {
  it("validates complete bootstrap payload", () => {
    const parsed = bootstrapResponseSchema.parse({
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
          permissions: ["booking:create"],
          is_default: true,
        },
      ],
      active_context_id: "client:self",
      feature_flags: { flagA: true },
      min_supported_app_version: "0.1.0",
      maintenance_mode: false,
      degraded_mode: false,
      server_time: new Date().toISOString(),
      request_id: "req-123456",
    });
    expect(parsed.available_contexts).toHaveLength(1);
  });

  it("accepts backend bootstrap user shape", () => {
    const parsed = bootstrapResponseSchema.parse({
      bootstrap_version: "1.0.0",
      is_authenticated: true,
      user: {
        id: 1,
        public_id: "u-public",
        username: "demo.user",
        email: "demo@lirie.ch",
        first_name: "Demo",
        last_name: "User",
        role: "CLIENT",
      },
      account_status: "active",
      onboarding_status: { required: false, status: "active" },
      available_contexts: [],
      active_context_id: null,
      feature_flags: {},
      min_supported_app_version: "0.1.0",
      maintenance_mode: false,
      degraded_mode: false,
      server_time: new Date().toISOString(),
      request_id: "req-123456",
    });
    expect(parsed.user?.username).toBe("demo.user");
  });

  it("fails when critical fields are missing", () => {
    expect(() =>
      bootstrapResponseSchema.parse({
        is_authenticated: true,
      })
    ).toThrow();
  });

  it("resolves default context and permissions", () => {
    const contexts = [
      {
        context_id: "client:self",
        context_type: "client" as const,
        label: "Client",
        permissions: ["booking:read:self"],
        is_default: true,
      },
      {
        context_id: "driver:42",
        context_type: "driver" as const,
        label: "Driver",
        permissions: ["mission:read"],
        is_default: false,
      },
    ];
    const resolved = resolveDefaultContext(contexts, null);
    expect(resolved?.context_id).toBe("client:self");
    expect(hasPermission(resolved, "booking:read:self")).toBe(true);
    expect(hasPermission(resolved, "mission:read")).toBe(false);
  });
});
