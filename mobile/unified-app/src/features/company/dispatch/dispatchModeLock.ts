/**
 * DISPATCH MODE — LOCK
 *
 * ACTIVE   : MANUAL = ON
 * INACTIF  : SEMI_AUTO, OPTIMIZER, FULLY_AUTO
 * ENV      : DEV = OFF · PROD = OFF
 *
 * Ce n’est pas un masquage de boutons. Tant qu’un flag est `false`,
 * la branche React correspondante ne doit pas être montée, et toute
 * action/service doit abort avant le premier effet ou HTTP.
 *
 * Le code semi-auto / optimiseur reste dans le dépôt (chantier futur).
 * Pour le réveiller : passer le flag à `true` — rien d’autre.
 */

export const SEMI_AUTO_DISPATCH_ENABLED = false as const;
export const OPTIMIZER_ENABLED = false as const;
export const FULLY_AUTO_DISPATCH_ENABLED = false as const;

export type DispatchLockedFeature = "semi_auto" | "optimizer" | "fully_auto";

export class DispatchFeatureDisabledError extends Error {
  readonly feature: DispatchLockedFeature;

  constructor(feature: DispatchLockedFeature) {
    super(
      feature === "optimizer"
        ? "Optimiseur désactivé (DISPATCH MODE LOCK)."
        : feature === "fully_auto"
          ? "Dispatch automatique désactivé (DISPATCH MODE LOCK)."
          : "Dispatch semi-auto désactivé (DISPATCH MODE LOCK)."
    );
    this.name = "DispatchFeatureDisabledError";
    this.feature = feature;
  }
}

export function shouldMountDispatchEngine(): boolean {
  return SEMI_AUTO_DISPATCH_ENABLED || OPTIMIZER_ENABLED || FULLY_AUTO_DISPATCH_ENABLED;
}

export function assertSemiAutoDispatchEnabled(): void {
  if (!SEMI_AUTO_DISPATCH_ENABLED) {
    throw new DispatchFeatureDisabledError("semi_auto");
  }
}

export function assertOptimizerEnabled(): void {
  if (!OPTIMIZER_ENABLED) {
    throw new DispatchFeatureDisabledError("optimizer");
  }
}

export function assertFullyAutoDispatchEnabled(): void {
  if (!FULLY_AUTO_DISPATCH_ENABLED) {
    throw new DispatchFeatureDisabledError("fully_auto");
  }
}

export function assertDispatchModeSwitchAllowed(
  mode: "manual" | "semi_auto" | "fully_auto"
): void {
  if (mode === "manual") return;
  if (mode === "semi_auto") {
    assertSemiAutoDispatchEnabled();
    return;
  }
  assertFullyAutoDispatchEnabled();
}
