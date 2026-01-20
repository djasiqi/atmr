export type ApiKind = "driver" | "enterprise" | "enterpriseStandard";

export class AuthNotReadyError extends Error {
  public readonly code = "AUTH_NOT_READY" as const;
  public readonly kind: ApiKind;
  public readonly reason: string;
  public readonly url?: string;

  constructor(params: { kind: ApiKind; reason: string; url?: string }) {
    super(`AUTH_NOT_READY: ${params.kind}:${params.reason}${params.url ? ` (${params.url})` : ""}`);
    this.name = "AuthNotReadyError";
    this.kind = params.kind;
    this.reason = params.reason;
    this.url = params.url;
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
    // refresh-token ne nécessite pas Authorization (refresh est en body/header/cookie)
    if (u.endsWith("/auth/refresh-token") || u.includes("/auth/refresh-token?"))
      return true;
    return false;
  }

  // Enterprise (company_mobile)
  if (u.endsWith("/auth/login") || u.includes("/auth/login?")) return true;
  if (u.endsWith("/auth/register") || u.includes("/auth/register?")) return true;
  // refresh enterprise ne nécessite pas Authorization (refresh en body)
  if (u.endsWith("/auth/refresh") || u.includes("/auth/refresh?")) return true;
  // MFA endpoints (challenge/verify) sont "publics" côté bearer (token MFA/otp dans body)
  if (u.includes("/auth/mfa")) return true;

  // enterpriseStandard: par défaut, tout est protégé (sauf /public)
  if (kind === "enterpriseStandard") return false;

  return false;
}

