import { BootstrapResponse, resolveDefaultContext } from "../contracts/auth";
import type { MobileSessionStatus } from "../auth/mobileSessionStatus";
import { resolveAuthGuardDecisionState } from "../auth/authGuardDecision";
import { resolveInstitutionUnifiedEnabledFromBootstrap } from "../featureFlags/registry";
import { resolveCompanyDeepLink, resolveDriverDeepLink } from "./deepLinkHandler";

/**
 * Route initiale post-bootstrap.
 * `mobileSessionStatus` obligatoire (P0-4 final gate) :
 * - TERMINAL → /(public)
 * - BOOTSTRAPPING sans destination → null (cold BootBrand OK)
 * - RECOVERING / DEGRADED / AUTHENTICATED → destination app (pas de flash login ;
 *   warm recovery garde l'arbre authentifié via AuthGuard + route app).
 */
export function resolveInitialRoute(
  bootstrap: BootstrapResponse,
  deepLink: string | null | undefined,
  mobileSessionStatus: MobileSessionStatus
): string | null {
  const decision = resolveAuthGuardDecisionState({ bootstrap, mobileSessionStatus });
  if (decision === "TERMINAL_UNAUTHENTICATED") {
    return "/(public)";
  }
  if (decision === "BOOTSTRAPPING") {
    return null;
  }
  // AUTHENTICATED | RECOVERING | DEGRADED → résoudre la destination app.
  // Même si bootstrap.is_authenticated est momentanément false pendant recovery,
  // on n'envoie jamais vers /(public) ici.

  if (bootstrap.maintenance_mode) return "/(app)/maintenance";
  if (bootstrap.account_status && bootstrap.account_status !== "active") {
    // Pendant recovery, ne pas bloquer sur un snapshot partiel
    if (decision === "AUTHENTICATED") return "/(app)/blocked";
  }
  if (bootstrap.onboarding_status?.required && decision === "AUTHENTICATED") {
    return "/(app)/onboarding";
  }

  const context = resolveDefaultContext(
    bootstrap.available_contexts ?? [],
    bootstrap.active_context_id ?? null
  );
  if (!context) {
    // Warm recovery sans contexte résolvable : rester dans l'arbre app
    if (decision === "RECOVERING" || decision === "DEGRADED_AUTHENTICATED") {
      return "/(app)/context-selector";
    }
    return "/(app)/context-selector";
  }

  switch (context.context_type) {
    case "client":
      return "/(app)/(client)";
    case "driver":
      if (bootstrap.feature_flags?.driver_unified_enabled === false) {
        const hasAlternativeContext = bootstrap.available_contexts.some(
          (candidate) => candidate.context_type !== "driver"
        );
        if (hasAlternativeContext) {
          return "/(app)/context-selector";
        }
        return "/(app)/blocked?reason=driver_gate";
      }
      {
        const deepLinkTarget = resolveDriverDeepLink(deepLink ?? null);
        if (deepLinkTarget?.route) {
          return deepLinkTarget.route;
        }
      }
      return "/(app)/(driver)";
    case "company":
      {
        const companyDeepLink = resolveCompanyDeepLink(deepLink ?? null);
        if (companyDeepLink?.route) {
          return companyDeepLink.route;
        }
      }
      return "/(app)/(company)";
    case "institution": {
      const institutionEnabled = resolveInstitutionUnifiedEnabledFromBootstrap(
        bootstrap.feature_flags
      );
      if (!institutionEnabled) {
        const hasAlternativeContext = bootstrap.available_contexts.some(
          (candidate) => candidate.context_type !== "institution"
        );
        if (hasAlternativeContext) {
          return "/(app)/context-selector";
        }
        return "/(app)/blocked?reason=institution_gate";
      }
      return "/(app)/(institution)";
    }
    default:
      return "/(app)/context-selector";
  }
}
