/**
 * Accès recette du panneau chauffeur.
 * Une variable du client ne suffit pas : le serveur doit déclarer le panneau,
 * la variante publiée ne doit pas être la production, et la base doit être atmr_test.
 */

export type SettlementAccessInput = {
  dev: boolean;
  appVariant: string | undefined;
  recipeDatabase: string | null;
  writableRecipe: boolean;
  serverPanelEnabled: boolean;
};

export function settlementRecipeAccessAllowed(input: SettlementAccessInput): boolean {
  if (!input.dev || !input.serverPanelEnabled) return false;
  const variant = (input.appVariant || "").trim().toLowerCase();
  if (variant === "prod" || variant === "production") return false;
  return input.recipeDatabase === "atmr_test" && input.writableRecipe;
}

/**
 * Préparation espèces. Le serveur doit l'avoir ouverte sur une base jetable.
 * Ce n'est pas une activation d'encaissement.
 */
export function settlementCashPrepAllowed(input: {
  dev: boolean;
  appVariant: string | undefined;
  recipeDatabase: string | null;
  serverPrepEnabled: boolean;
}): boolean {
  if (!input.dev || !input.serverPrepEnabled) return false;
  const variant = (input.appVariant || "").trim().toLowerCase();
  if (variant === "prod" || variant === "production") return false;
  const name = (input.recipeDatabase || "").trim();
  if (
    name === "atmr" ||
    name === "atmr_test" ||
    name === "atmr_mig_ci" ||
    name === "atmrstg" ||
    name === "postgres"
  ) {
    return false;
  }
  return (
    name.startsWith("atmr_etape") ||
    name.startsWith("lirie_sim_") ||
    name.startsWith("lirie_e2e_")
  );
}

/** Une coupure réseau ne relance jamais une réception déjà tentée. */
export function cashDeclarationMayRetry(networkFailed: boolean): boolean {
  return networkFailed ? false : true;
}

/** Le téléphone ne décide pas : le serveur doit avoir ouvert le dossier jetable. */
export function settlementLedgerAccessAllowed(input: {
  dev: boolean;
  appVariant: string | undefined;
  recipeDatabase: string | null;
  serverLedgerEnabled: boolean;
}): boolean {
  if (!input.dev || !input.serverLedgerEnabled) return false;
  const variant = (input.appVariant || "").trim().toLowerCase();
  if (variant === "prod" || variant === "production") return false;
  const name = (input.recipeDatabase || "").trim();
  if (
    name === "atmr" ||
    name === "atmr_test" ||
    name === "atmr_mig_ci" ||
    name === "atmrstg"
  ) {
    return false;
  }
  return (
    name.startsWith("atmr_etape") ||
    name.startsWith("lirie_sim_") ||
    name.startsWith("lirie_e2e_")
  );
}

export function settlementRecipeCandidate(input: {
  dev: boolean;
  appVariant: string | undefined;
}): boolean {
  if (!input.dev) return false;
  const variant = (input.appVariant || "").trim().toLowerCase();
  return variant !== "prod" && variant !== "production";
}

/** Le patient est encore là pendant « Patient à bord », puis après la clôture. */
export function settlementMomentOpen(status: string | null | undefined): boolean {
  const value = (status ?? "").trim().toLowerCase();
  return value === "in_progress" || value === "completed" || value === "return_completed";
}

/**
 * « Patient à bord » couvre aussi la conduite. L'arrêt à destination est une
 * confirmation locale du chauffeur, jamais un statut et jamais une position.
 */
export function settlementNeedsHalt(status: string | null | undefined): boolean {
  return (status ?? "").trim().toLowerCase() === "in_progress";
}

/** Mission déjà close : consultation seulement, sans réouverture opérationnelle. */
export function settlementClosed(status: string | null | undefined): boolean {
  const value = (status ?? "").trim().toLowerCase();
  return value === "completed" || value === "return_completed";
}

export function settlementLayout(width: number): {
  contentWidth: number;
  buttonMinHeight: number;
  amountFontSize: number;
  overflows: boolean;
} {
  const padding = 16;
  const contentWidth = width - padding * 2;
  return {
    contentWidth,
    buttonMinHeight: 48,
    amountFontSize: width < 380 ? 24 : 28,
    overflows: contentWidth <= 0,
  };
}

/** Une simulation n'est pas une clôture de transport. */
export function statusAfterSimulatedSettlement(operationalStatus: string): string {
  return operationalStatus;
}
