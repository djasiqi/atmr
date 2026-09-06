/**
 * COMPANY-AUTH-GATE-01 — barrière réseau entreprise.
 * Le shell peut se peindre depuis le snapshot ; les GET protégés n’ouvrent
 * qu’après SESSION_READY (même flag que le chauffeur).
 */

import {
  isAuthOnlyRequestUrl,
  isDriverSessionNetworkReady,
  subscribeDriverSessionNetworkReady,
} from "./driverSessionNetworkGate";

export function isCompanySessionNetworkReady(): boolean {
  return isDriverSessionNetworkReady();
}

export function subscribeCompanySessionNetworkReady(listener: () => void): () => void {
  return subscribeDriverSessionNetworkReady(listener);
}

export function isCompanyProtectedRequestUrl(url: string): boolean {
  const normalized = url.startsWith("/") ? url : `/${url}`;
  return (
    normalized.startsWith("/companies") ||
    normalized.startsWith("/company/") ||
    normalized.startsWith("/company_") ||
    normalized.startsWith("/company-") ||
    normalized.startsWith("/dispatch/") ||
    normalized.startsWith("/invoices/companies") ||
    normalized.startsWith("/partnerships") ||
    normalized.startsWith("/pricing/")
  );
}

/**
 * true = ne pas envoyer la requête (pas de 401).
 * /auth/* toujours autorisé. Contexte entreprise : tout le reste attend SESSION_READY.
 */
export function shouldBlockCompanyRequestUntilSessionReady(
  url: string,
  contextId?: string | null
): boolean {
  if (isAuthOnlyRequestUrl(url)) return false;
  if (isCompanySessionNetworkReady()) return false;
  if (isCompanyProtectedRequestUrl(url)) return true;
  return typeof contextId === "string" && contextId.startsWith("company:");
}
