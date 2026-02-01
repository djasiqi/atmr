export type ApiKind = "driver" | "enterprise" | "enterpriseStandard";

/**
 * Erreur stable pour rejet de la queue après échec refresh (session invalide).
 * Utilisée par processQueue pour éviter retry spam et erreurs incohérentes.
 */
export class AuthInvalidError extends Error {
  public readonly code = "AUTH_INVALID" as const;
  public readonly route: "driver" | "enterprise";
  public readonly reason: string;

  constructor(params: { route: "driver" | "enterprise"; reason: string }) {
    super(`AUTH_INVALID: ${params.route}:${params.reason}`);
    this.name = "AuthInvalidError";
    this.route = params.route;
    this.reason = params.reason;
  }
}

export function isAuthInvalidError(error: unknown): error is AuthInvalidError {
  return (
    typeof error === "object" &&
    error !== null &&
    (error as any).name === "AuthInvalidError" &&
    (error as any).code === "AUTH_INVALID"
  );
}

export class AuthNotReadyError extends Error {
  public readonly code = "AUTH_NOT_READY" as const;
  public readonly kind: ApiKind;
  public readonly reason: string;
  public readonly url?: string;
  /** Si true, l'UI ne doit pas afficher un second popup (dedupe anti-spam). */
  public readonly silentDedupe?: boolean;

  constructor(params: {
    kind: ApiKind;
    reason: string;
    url?: string;
    silentDedupe?: boolean;
  }) {
    super(`AUTH_NOT_READY: ${params.kind}:${params.reason}${params.url ? ` (${params.url})` : ""}`);
    this.name = "AuthNotReadyError";
    this.kind = params.kind;
    this.reason = params.reason;
    this.url = params.url;
    this.silentDedupe = params.silentDedupe;
  }
}

export function isAuthNotReadyError(error: unknown): error is AuthNotReadyError {
  return (
    typeof error === "object" &&
    error !== null &&
    (error as any).name === "AuthNotReadyError" &&
    (error as any).code === "AUTH_NOT_READY"
  );
}

/** À utiliser avant d'afficher un popup pour AUTH_NOT_READY : si true, ne pas afficher (dedupe). */
export function shouldShowAuthNotReadyAlert(error: unknown): boolean {
  if (!isAuthNotReadyError(error)) return true;
  return !(error as AuthNotReadyError).silentDedupe;
}

/**
 * Option 2 — Métrique légère (placeholder).
 * À appeler quand une AuthNotReadyError est levée (hors silentDedupe).
 * Plus tard : Sentry / Datadog / Firebase — compter l'événement, tag mode: driver | enterprise,
 * pour voir si ça réapparaît après une release.
 */
export function reportAuthNotReadyMetric(params: {
  kind: ApiKind;
  reason: string;
  url?: string;
}): void {
  try {
    if (__DEV__) {
      console.debug(
        "[AUTH] AuthNotReadyError (metric placeholder):",
        params.kind,
        params.reason,
        params.url ?? ""
      );
    }
    // TODO: Sentry/Datadog/Firebase — compter AuthNotReadyError (non silentDedupe), tag mode: driver | enterprise
  } catch {
    // Fire-and-forget : ne jamais faire échouer l'appelant (SDK externe peut throw plus tard).
  }
}

/**
 * Message UX à afficher à l'utilisateur pour AUTH_NOT_READY.
 * Évite d'afficher le message technique "AUTH_NOT_READY: enterprise: missing_refresh_token (/auth/refresh)".
 */
export function getAuthNotReadyDisplayMessage(error: unknown): string | null {
  if (!isAuthNotReadyError(error)) return null;
  const reason = (error as AuthNotReadyError).reason;
  if (reason === "missing_refresh_token") {
    return "Session expirée ou inexistante. Veuillez vous reconnecter.";
  }
  if (reason === "missing_access_token") {
    return "Session non prête. Veuillez patienter ou vous reconnecter.";
  }
  if (reason === "auth_ready_timeout") {
    return "Connexion en cours…";
  }
  if (reason === "missing_token_and_refresh_failed") {
    return "Session expirée. Veuillez vous reconnecter.";
  }
  return "Connexion en cours…";
}

// Centralisation: définition "public endpoint" (pas besoin d'Authorization Bearer)
export function isPublicEndpoint(
  url: string | undefined,
  kind: ApiKind
): boolean {
  const u = (url || "").toLowerCase();
  if (!u) return false;

  // Toujours public
  if (u.includes("/public")) return true;

  if (kind === "driver") {
    // Driver API v1
    if (u.endsWith("/auth/login") || u.includes("/auth/login?")) return true;
    if (u.endsWith("/auth/register") || u.includes("/auth/register?")) return true;
    // CSRF token est requis avant refresh/login sur certains backends
    if (u.endsWith("/auth/csrf-token") || u.includes("/auth/csrf-token?"))
      return true;
    // Version check est public (ne dépend pas de l'auth)
    if (u.endsWith("/app/version-check") || u.includes("/app/version-check?"))
      return true;
    // refresh-token ne nécessite pas Authorization (refresh est en body/header/cookie)
    if (u.endsWith("/auth/refresh-token") || u.includes("/auth/refresh-token?"))
      return true;
    return false;
  }

  // Enterprise (company_mobile)
  if (u.endsWith("/auth/login") || u.includes("/auth/login?")) return true;
  if (u.endsWith("/auth/register") || u.includes("/auth/register?")) return true;
  if (u.endsWith("/auth/csrf-token") || u.includes("/auth/csrf-token?"))
    return true;
  // refresh enterprise ne nécessite pas Authorization (refresh en body)
  if (u.endsWith("/auth/refresh") || u.includes("/auth/refresh?")) return true;
  // MFA endpoints (challenge/verify) sont "publics" côté bearer (token MFA/otp dans body)
  if (u.includes("/auth/mfa")) return true;

  // enterpriseStandard: par défaut, tout est protégé (sauf /public)
  if (kind === "enterpriseStandard") return false;

  return false;
}

