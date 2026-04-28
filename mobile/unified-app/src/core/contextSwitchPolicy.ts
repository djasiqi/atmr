import { Platform } from "react-native";
import type { AuthContext } from "./contracts/auth";

/** Client natif (iOS/Android), exclut le build web. */
export function isExpoNativeMobile(): boolean {
  return Platform.OS === "ios" || Platform.OS === "android";
}

/**
 * Bascule entreprise ↔ chauffeur dans l’app unifiée : mobile **et** web
 * (le build Next / Expo web doit pouvoir renvoyer au bureau comme sur mobile).
 */
export function isContextSwitchClientSupported(): boolean {
  return isExpoNativeMobile() || Platform.OS === "web";
}

export function isCompanyDriverCrossContextSwitch(
  fromCtx: AuthContext | null | undefined,
  toCtx: AuthContext | null | undefined
): boolean {
  if (!fromCtx || !toCtx) return false;
  const a = fromCtx.context_type;
  const b = toCtx.context_type;
  return (a === "company" && b === "driver") || (a === "driver" && b === "company");
}

export function isTransportContextSwitchContext(
  context: AuthContext | null | undefined
): boolean {
  return context?.allow_mobile_context_switch === true;
}

function isCompanyAccountRole(role: string | null | undefined): boolean {
  if (role == null || String(role).trim() === "") return true;
  return String(role).toUpperCase() === "COMPANY";
}

/**
 * Bascule entreprise ↔ chauffeur : uniquement compte entreprise (rôle API `COMPANY`), client web
 * ou mobile, et contextes marqués `allow_mobile_context_switch` (dispatch) par le serveur. Les comptes
 * chauffeur seuls n’ont pas le droit — pas d’accès à la gestion d’entreprise.
 */
export function isCompanyDriverSwitchAllowedForRequest(
  fromCtx: AuthContext | null,
  toCtx: AuthContext | null,
  bootstrapUserRole: string | null | undefined
): boolean {
  if (!isCompanyDriverCrossContextSwitch(fromCtx, toCtx)) return true;
  if (!isCompanyAccountRole(bootstrapUserRole)) return false;
  if (!isContextSwitchClientSupported()) return false;
  return isTransportContextSwitchContext(fromCtx) && isTransportContextSwitchContext(toCtx);
}

export function companyDriverSwitchBlockedReason(
  fromCtx: AuthContext | null,
  toCtx: AuthContext | null,
  bootstrapUserRole: string | null | undefined
): "web" | "not_transport" | "not_company_account" | null {
  if (!isCompanyDriverCrossContextSwitch(fromCtx, toCtx)) return null;
  if (!isCompanyAccountRole(bootstrapUserRole)) return "not_company_account";
  if (!isContextSwitchClientSupported()) return "web";
  if (!isTransportContextSwitchContext(fromCtx) || !isTransportContextSwitchContext(toCtx)) {
    return "not_transport";
  }
  return null;
}

export function companyDriverSwitchBlockedMessage(
  fromCtx: AuthContext | null,
  toCtx: AuthContext | null,
  bootstrapUserRole: string | null | undefined
): string | null {
  const r = companyDriverSwitchBlockedReason(fromCtx, toCtx, bootstrapUserRole);
  if (!r) return null;
  if (r === "web") {
    return "La bascule entreprise / chauffeur n’est pas disponible sur cette plateforme (utilisez l’app unifiée web ou mobile).";
  }
  if (r === "not_company_account") {
    return "Seul le compte entreprise (double casquette) peut basculer entre l’espace entreprise et le chauffeur. Un compte chauffeur seul n’a pas accès à la gestion d’entreprise.";
  }
  return "La bascule n’est pas disponible pour ce compte (compte entreprise requis, dispatch actif).";
}
