/**
 * Alias historiques (chauffeur / hors LIRIE).
 * Les queries company / dispatch doivent utiliser `queryCachePolicy.ts` (OPT-05).
 */
export const QUERY_STALE_TIME_MS = {
  /** Aligné sur QueryProvider (défaut global) */
  default: 30_000,
  /** Listes / dispatch / écrans à rafraîchissement fréquent */
  companyList: 20_000,
  /** Détail course, grilles, données « lentes » (ex. factures agrégées) */
  companyDetail: 30_000,
  /** Données peu changeantes (ex. formulaires, catalogues) */
  companySlow: 60_000,
} as const;
