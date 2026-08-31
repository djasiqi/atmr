import { describe, expect, it } from "@jest/globals";
import { hasPermission } from "./contracts/auth";
import {
  resolveAuthGuardRedirect,
  resolveCompanyContextGuardRedirect,
  resolveContextGuardRedirect,
  resolveDriverUnifiedGate,
  resolveInstitutionUnifiedGate,
  resolveOnboardingGuardRedirect,
  resolvePermissionGuardRedirect,
  resolveUnauthorizedRecoveryRedirect,
  resolveVersionGuardRedirect,
  shouldUnmountCompanySurface,
  shouldUnmountDriverSurface,
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

  const companyCtx = {
    context_id: "company:1",
    context_type: "company" as const,
    label: "Entreprise",
    permissions: ["company:dashboard:read"],
    is_default: true,
  };
  const driverCtx = {
    context_id: "driver:1",
    context_type: "driver" as const,
    label: "Chauffeur",
    permissions: ["mission:read"],
    is_default: true,
  };

  it("1 DRIVER->COMPANY: ancien écran DRIVER + transition => pas de permission denied", () => {
    expect(hasPermission(companyCtx as any, "mission:read")).toBe(false);
    expect(resolvePermissionGuardRedirect(false, true)).toBeNull();
  });

  it("2 DRIVER->COMPANY: activeContext COMPANY => surface DRIVER démontée", () => {
    expect(shouldUnmountDriverSurface(true, "company")).toBe(true);
    expect(shouldUnmountDriverSurface(true, "driver")).toBe(false);
    expect(shouldUnmountDriverSurface(false, "company")).toBe(false);
  });

  it("3 company context valide => company home", () => {
    expect(resolveUnauthorizedRecoveryRedirect(companyCtx as any)).toBe(
      "/(app)/(company)/dashboard"
    );
    expect(resolveCompanyContextGuardRedirect(companyCtx as any)).toBeNull();
  });

  it("4 nouvel écran COMPANY n'exige pas mission:read", () => {
    expect(hasPermission(companyCtx as any, "mission:read")).toBe(false);
    expect(hasPermission(companyCtx as any, "company:dashboard:read")).toBe(true);
    expect(resolvePermissionGuardRedirect(hasPermission(companyCtx as any, "company:dashboard:read"))).toBeNull();
  });

  it("5 AccessDenied transitoire + COMPANY => redirect company home", () => {
    expect(resolveUnauthorizedRecoveryRedirect(companyCtx as any)).toBe(
      "/(app)/(company)/dashboard"
    );
  });

  it("6 AccessDenied transitoire + DRIVER => redirect driver home", () => {
    expect(resolveUnauthorizedRecoveryRedirect(driverCtx as any)).toBe("/(app)/(driver)");
  });

  it("7 vrai utilisateur sans permission hors transition => AccessDenied", () => {
    expect(resolvePermissionGuardRedirect(false, false)).toBe("/(app)/unauthorized");
    expect(resolvePermissionGuardRedirect(hasPermission(companyCtx as any, "mission:read"), false)).toBe(
      "/(app)/unauthorized"
    );
    const clientSansDroit = {
      context_id: "client:self",
      context_type: "client" as const,
      label: "Client",
      permissions: [],
      is_default: true,
    };
    expect(resolveUnauthorizedRecoveryRedirect(clientSansDroit as any)).toBeNull();
  });

  it("TRANSITION_BYPASS != GLOBAL_PERMISSION_BYPASS", () => {
    expect(resolvePermissionGuardRedirect(false, true)).toBeNull();
    expect(resolvePermissionGuardRedirect(false, false)).toBe("/(app)/unauthorized");
    expect(shouldUnmountCompanySurface(true, "driver")).toBe(true);
    expect(shouldUnmountCompanySurface(false, "driver")).toBe(false);
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
