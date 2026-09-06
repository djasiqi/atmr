import type { AuthContext, BootstrapResponse } from "../contracts/auth";

/**
 * OPT-07 — entrée app sans attendre le bootstrap réseau.
 * Une session locale est suffisante si le snapshot offline a déjà
 * un bootstrap authentifié et un contexte actif.
 * Révoquée / incohérente / anonyme → ne pas entrer.
 */
export function canEnterFromLocalSession(input: {
  bootstrap?: BootstrapResponse | null;
  activeContext?: AuthContext | null;
}): boolean {
  return Boolean(input.bootstrap?.is_authenticated && input.activeContext?.context_id);
}
