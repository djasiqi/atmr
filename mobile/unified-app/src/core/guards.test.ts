import { describe, expect, it } from "@jest/globals";
import {
  resolveAuthGuardRedirect,
  resolveCompanyContextGuardRedirect,
  resolveContextGuardRedirect,
  resolveDriverUnifiedGate,
  resolveInstitutionUnifiedGate,
  resolveOnboardingGuardRedirect,
  resolvePermissionGuardRedirect,
  resolveVersionGuardRedirect,
} from "./guardDecisions";

const bootstrapBase = {
  bootstrap_version: "1.0.0",
  is_authenticated: true,
  user: { id: "u-1", email: "demo@lirie.ch" },
  account_status: "active" as const,
  onboarding_status: { required: false },
  available_contexts: [],
  active_context_id: null,
  feature_flags: {},
  min_supported_app_version: "0.1.0",
  maintenance_mode: false,
  degraded_mode: false,
  server_time: new Date().toISOString(),
  request_id: "req-123456",
};

describe("guard decision helpers", () => {
  it("auth guard redirects when unauthenticated", () => {
    expect(resolveAuthGuardRedirect({ ...bootstrapBase, is_authenticated: false })).toBe(
      "/(public)"
    );
    expect(resolveAuthGuardRedirect(bootstrapBase as any)).toBeNull();
  });

  it("context guard redirects when context missing", () => {
    expect(resolveContextGuardRedirect(null)).toBe("/(app)/context-selector");
    expect(
      resolveContextGuardRedirect({
        context_id: "client:self",
        context_type: "client",
        label: "Client",
        permissions: [],
        is_default: true,
      } as any)
    ).toBeNull();
  });

  it("permission guard redirects when denied", () => {
    expect(resolvePermissionGuardRedirect(false)).toBe("/(app)/unauthorized");
    expect(resolvePermissionGuardRedirect(true)).toBeNull();
  });

  it("company context guard rejects non-company contexts", () => {
    expect(resolveCompanyContextGuardRedirect(null)).toBe("/(app)/context-selector");
    expect(
      resolveCompanyContextGuardRedirect({
        context_id: "driver:42",
        context_type: "driver",
        label: "Driver",
        permissions: [],
        is_default: true,
      } as any)
    ).toBe("/(app)/(driver)");
    expect(
      resolveCompanyContextGuardRedirect({
        context_id: "company:42",
        context_type: "company",
        label: "Company",
        permissions: [],
        is_default: true,
      } as any)
    ).toBeNull();
  });

  it("onboarding guard redirects when required", () => {
    expect(
      resolveOnboardingGuardRedirect({ ...bootstrapBase, onboarding_status: { required: true } } as any)
    ).toBe("/(app)/onboarding");
  });

  it("version guard redirects in maintenance mode", () => {
    expect(resolveVersionGuardRedirect({ ...bootstrapBase, maintenance_mode: true } as any)).toBe(
      "/(app)/maintenance"
    );
  });

  // ─── DriverUnifiedGate (Option C) ─────────────────────────────────────────

  it("driver unified gate blocks and redirects with reason when flag is off", () => {
    const result = resolveDriverUnifiedGate(false);
    expect(result.allowed).toBe(false);
    expect(result.option).toBe("C");
    expect(result.redirectTo).toBe("/(app)/blocked?reason=driver_gate");
  });

  it("driver unified gate allows access when flag is on", () => {
    const result = resolveDriverUnifiedGate(true);
    expect(result.allowed).toBe(true);
    expect(result.option).toBe("C");
    expect(result.redirectTo).toBeNull();
  });

  it("driver unified gate redirect includes reason param for UX disambiguation", () => {
    const { redirectTo } = resolveDriverUnifiedGate(false);
    expect(redirectTo).toContain("reason=driver_gate");
  });

  it("institution unified gate blocks until explicitly enabled", () => {
    const blocked = resolveInstitutionUnifiedGate(false);
    expect(blocked.allowed).toBe(false);
    expect(blocked.redirectTo).toBe("/(app)/blocked?reason=institution_gate");

    const allowed = resolveInstitutionUnifiedGate(true);
    expect(allowed.allowed).toBe(true);
    expect(allowed.redirectTo).toBeNull();
  });
});
