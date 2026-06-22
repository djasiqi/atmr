import { BootstrapResponse, resolveDefaultContext } from "../contracts/auth";
import { resolveInstitutionUnifiedEnabledFromBootstrap } from "../featureFlags/registry";
import { resolveCompanyDeepLink, resolveDriverDeepLink } from "./deepLinkHandler";

export function resolveInitialRoute(bootstrap: BootstrapResponse, deepLink?: string | null): string {
  if (!bootstrap.is_authenticated) return "/(public)";
  if (bootstrap.maintenance_mode) return "/(app)/maintenance";
  if (bootstrap.account_status !== "active") return "/(app)/blocked";
  if (bootstrap.onboarding_status?.required) return "/(app)/onboarding";

  const context = resolveDefaultContext(
    bootstrap.available_contexts,
    bootstrap.active_context_id ?? null
  );
  if (!context) return "/(app)/context-selector";

  switch (context.context_type) {
    case "client":
      return "/(app)/(client)";
    case "driver":
      /* Opt-out explicite: sans clé (ou `true`) on ouvre l’espace chauffeur comme l’app Chauffeur. */
      if (bootstrap.feature_flags?.driver_unified_enabled === false) {
        const hasAlternativeContext = bootstrap.available_contexts.some(
          (candidate) => candidate.context_type !== "driver"
        );
        if (hasAlternativeContext) {
          return "/(app)/context-selector";
        }
        return "/(app)/blocked?reason=driver_gate";
      }
      const deepLinkTarget = resolveDriverDeepLink(deepLink ?? null);
      if (deepLinkTarget?.route) {
        return deepLinkTarget.route;
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
