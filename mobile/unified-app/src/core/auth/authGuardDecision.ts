/**
 * P0-4 — Décision AuthGuard : login uniquement si session terminale / anonyme prouvée.
 * bootstrap === null seul ≠ logout.
 *
 * API runtime : `mobileSessionStatus` est **obligatoire** (pas de fallback
 * `is_authenticated → /(public)` en production).
 */
import type { BootstrapResponse } from "../contracts/auth";
import type { MobileSessionStatus } from "./mobileSessionStatus";

/** Sémantiques produit (noms stables pour tests / docs). */
export type AuthGuardDecisionState =
  | "BOOTSTRAPPING"
  | "AUTHENTICATED"
  | "RECOVERING"
  | "DEGRADED_AUTHENTICATED"
  | "TERMINAL_UNAUTHENTICATED";

export type AuthGuardDecisionInput = {
  bootstrap: BootstrapResponse | null;
  mobileSessionStatus: MobileSessionStatus;
};

export function resolveAuthGuardDecisionState(
  input: AuthGuardDecisionInput
): AuthGuardDecisionState {
  const { bootstrap, mobileSessionStatus } = input;

  if (mobileSessionStatus === "revoked") {
    return "TERMINAL_UNAUTHENTICATED";
  }
  if (mobileSessionStatus === "logging_out") {
    return "TERMINAL_UNAUTHENTICATED";
  }
  if (
    mobileSessionStatus === "auth_recovering" ||
    mobileSessionStatus === "storage_locked" ||
    mobileSessionStatus === "restoring"
  ) {
    return "RECOVERING";
  }
  if (mobileSessionStatus === "initializing") {
    return "BOOTSTRAPPING";
  }
  if (mobileSessionStatus === "authenticated_offline") {
    return "DEGRADED_AUTHENTICATED";
  }
  if (mobileSessionStatus === "authenticated_online") {
    return "AUTHENTICATED";
  }
  // anonymous : bootstrap null = encore en charge ; bootstrap false = terminal
  if (mobileSessionStatus === "anonymous") {
    if (bootstrap == null) {
      return "BOOTSTRAPPING";
    }
    if (bootstrap.is_authenticated) {
      return "AUTHENTICATED";
    }
    return "TERMINAL_UNAUTHENTICATED";
  }

  // Statut inconnu / legacy : bootstrap authentifié gagne, sinon ne pas expulser
  // tant que bootstrap est null (boot).
  if (bootstrap?.is_authenticated) {
    return "AUTHENTICATED";
  }
  if (bootstrap == null) {
    return "BOOTSTRAPPING";
  }
  return "TERMINAL_UNAUTHENTICATED";
}

/**
 * Redirect login uniquement pour TERMINAL_UNAUTHENTICATED.
 * `mobileSessionStatus` obligatoire — aucun fallback legacy `is_authenticated`.
 */
export function resolveAuthGuardRedirect(
  bootstrap: BootstrapResponse | null,
  mobileSessionStatus: MobileSessionStatus
): string | null {
  const state = resolveAuthGuardDecisionState({ bootstrap, mobileSessionStatus });
  if (state === "TERMINAL_UNAUTHENTICATED") return "/(public)";
  return null;
}

/**
 * True si BootBrandSurface plein écran est autorisé (cold start uniquement).
 * Warm RECOVERING / DEGRADED avec arbre déjà prêt → false.
 */
export function shouldShowColdBootBrandSurface(input: {
  status: string;
  mobileSessionStatus: MobileSessionStatus;
}): boolean {
  const { status, mobileSessionStatus } = input;
  if (mobileSessionStatus === "revoked" || mobileSessionStatus === "logging_out") {
    return false;
  }
  if (
    mobileSessionStatus === "auth_recovering" ||
    mobileSessionStatus === "authenticated_offline" ||
    mobileSessionStatus === "authenticated_online" ||
    mobileSessionStatus === "storage_locked"
  ) {
    // Session déjà entrée (ready) ou warm recovery → jamais BootBrand plein écran
    if (status === "ready" || status === "error") {
      return false;
    }
  }
  return (
    status === "idle" ||
    status === "bootstrapping" ||
    mobileSessionStatus === "initializing" ||
    mobileSessionStatus === "restoring"
  );
}
