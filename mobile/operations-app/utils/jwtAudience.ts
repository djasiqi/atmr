/**
 * Utilitaires pour vérifier l'audience des tokens JWT (sans validation de signature).
 * Utilisé pour éviter d'envoyer un token driver à l'endpoint entreprise (et inversement).
 */

const MOBILE_ENTERPRISE_AUDIENCE = "atmr-mobile-enterprise";
const DRIVER_API_AUDIENCE = "atmr-api";

/**
 * Décode le payload JWT (partie centrale) et extrait le claim "aud".
 * Ne valide pas la signature - utilisé uniquement pour vérifier l'audience avant envoi.
 * @returns L'audience du token, ou null si décodage impossible
 */
export function getJwtPayloadAudience(token: string): string | null {
  try {
    const parts = token.split(".");
    if (parts.length !== 3) return null;
    const payloadB64url = parts[1];
    // Base64url → Base64 (atob attend le format standard)
    const base64 = payloadB64url.replace(/-/g, "+").replace(/_/g, "/");
    const pad = base64.length % 4;
    const padded = pad ? base64 + "=".repeat(4 - pad) : base64;
    const decoded = atob(padded);
    const parsed = JSON.parse(decoded) as { aud?: string };
    return parsed.aud ?? null;
  } catch {
    return null;
  }
}

/**
 * Vérifie si le token est un token entreprise (aud = atmr-mobile-enterprise).
 * Retourne false si le token a une autre audience (ex. atmr-api = driver).
 */
export function isEnterpriseRefreshToken(token: string): boolean {
  const aud = getJwtPayloadAudience(token);
  return aud === MOBILE_ENTERPRISE_AUDIENCE;
}

/**
 * Vérifie si le token est un token driver (aud = atmr-api).
 */
export function isDriverRefreshToken(token: string): boolean {
  const aud = getJwtPayloadAudience(token);
  return aud === DRIVER_API_AUDIENCE;
}

export { MOBILE_ENTERPRISE_AUDIENCE, DRIVER_API_AUDIENCE };
