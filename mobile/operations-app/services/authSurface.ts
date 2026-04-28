/**
 * Surface d’auth « chauffeur vs entreprise » pour garde-fous runtime (sync / GPS / mission).
 * Aligné sur le `mode` exposé par useAuth ; mis à jour par sessionModeTransition + effets auth.
 */
import { getLogger } from "@/utils/logger";

const log = getLogger("AuthSurface");

export type AuthSurfaceRole = "driver" | "enterprise";

/** Cohérent avec le défaut initial de `useAuth` (mode entreprise par défaut au cold start). */
let surface: AuthSurfaceRole = "enterprise";

export function setAuthSurfaceRole(role: AuthSurfaceRole): void {
  surface = role;
}

export function getAuthSurfaceRole(): AuthSurfaceRole {
  return surface;
}

export function assertDriverSurface(context: string): boolean {
  if (surface !== "driver") {
    log.warn(`${context}: ignoré hors surface chauffeur`, { surface });
    return false;
  }
  return true;
}
