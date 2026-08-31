import { AuthContext, BootstrapResponse } from "./contracts/auth";

export function resolveAuthGuardRedirect(
  bootstrap: BootstrapResponse | null
): string | null {
  if (!bootstrap?.is_authenticated) return "/(public)";
  return null;
}

export function resolveContextGuardRedirect(activeContext: AuthContext | null): string | null {
  if (!activeContext) return "/(app)/context-selector";
  return null;
}

export function resolvePermissionGuardRedirect(
  canAccess: boolean,
  contextSwitchInFlight = false
): string | null {
  // Bypass limité à la transition : l'ancien écran n'a plus les droits du nouveau contexte.
  // Hors transition, un refus reste un refus (TRANSITION_BYPASS != GLOBAL_PERMISSION_BYPASS).
  if (contextSwitchInFlight) return null;
  if (!canAccess) return "/(app)/unauthorized";
  return null;
}

/** Démonter la surface DRIVER dès que le contexte actif n'est plus driver. */
export function shouldUnmountDriverSurface(
  contextSwitchInFlight: boolean,
  activeContextType: string | null | undefined
): boolean {
  return Boolean(
    contextSwitchInFlight && activeContextType && activeContextType !== "driver"
  );
}

/** Démonter la surface COMPANY dès que le contexte actif n'est plus company. */
export function shouldUnmountCompanySurface(
  contextSwitchInFlight: boolean,
  activeContextType: string | null | undefined
): boolean {
  return Boolean(
    contextSwitchInFlight && activeContextType && activeContextType !== "company"
  );
}

/** Sortie de l'impasse unauthorized après une bascule réussie. */
export function resolveUnauthorizedRecoveryRedirect(
  activeContext: AuthContext | null
): string | null {
  if (!activeContext) return null;
  if (activeContext.context_type === "company") return "/(app)/(company)/dashboard";
  if (activeContext.context_type === "driver") return "/(app)/(driver)";
  return null;
}

export function resolveCompanyContextGuardRedirect(
  activeContext: AuthContext | null
): string | null {
  if (!activeContext) return "/(app)/context-selector";
  if (activeContext.context_type === "driver") return "/(app)/(driver)";
  if (activeContext.context_type !== "company") return "/(app)/unauthorized";
  return null;
}

export function resolveOnboardingGuardRedirect(
  bootstrap: BootstrapResponse | null
): string | null {
  if (bootstrap?.onboarding_status?.required) return "/(app)/onboarding";
  return null;
}

export function resolveVersionGuardRedirect(
  bootstrap: BootstrapResponse | null
): string | null {
  if (bootstrap?.maintenance_mode) return "/(app)/maintenance";
  return null;
}

export type DriverUnifiedGateResult = {
  allowed: boolean;
  redirectTo: string | null;
  option: "C";
};

export function resolveDriverUnifiedGate(enabled: boolean): DriverUnifiedGateResult {
  if (!enabled) {
    return { allowed: false, redirectTo: "/(app)/blocked?reason=driver_gate", option: "C" };
  }
  return { allowed: true, redirectTo: null, option: "C" };
}

export type InstitutionUnifiedGateResult = {
  allowed: boolean;
  redirectTo: string | null;
};

export function resolveInstitutionUnifiedGate(enabled: boolean): InstitutionUnifiedGateResult {
  if (!enabled) {
    return { allowed: false, redirectTo: "/(app)/blocked?reason=institution_gate" };
  }
  return { allowed: true, redirectTo: null };
}
