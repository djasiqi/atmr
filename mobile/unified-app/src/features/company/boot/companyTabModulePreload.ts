import { useEffect } from "react";
import { InteractionManager } from "react-native";

/**
 * NAV-01 — aucun preload CODE des onglets hors Cockpit / Courses.
 * `settings` (~1600 modules) et `clients-facturation` (~1600 modules) partaient
 * ensemble dès la lane background et saturaient le heap Metro web (~2 Go).
 * Le chat reste lazy pour la même raison côté Android. Le premier tap charge la route.
 */
export const COMPANY_TAB_CODE_PRELOAD_IDS = [] as const;

export type CompanyTabCodePreloadId = string;

export type CompanyTabCodePreload = {
  id: CompanyTabCodePreloadId;
  load: () => Promise<unknown>;
};

export const COMPANY_TAB_CODE_PRELOADS: readonly CompanyTabCodePreload[] = [];

export async function preloadCompanyTabModules(
  loaders: readonly CompanyTabCodePreload[] = COMPANY_TAB_CODE_PRELOADS,
  isCancelled: () => boolean = () => false
): Promise<void> {
  for (const entry of loaders) {
    if (isCancelled()) return;
    try {
      await entry.load();
    } catch {
      // Le premier tap relancera le lazy ; on ne bloque pas le shell.
    }
  }
}

/**
 * File vide : aucun import() d’écran tant que l’utilisateur n’ouvre pas la route.
 * N’exécute aucun prefetch React Query.
 */
export function usePreloadCompanyTabModules(enabled: boolean): void {
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    const handle = InteractionManager.runAfterInteractions(() => {
      requestAnimationFrame(() => {
        if (cancelled) return;
        void preloadCompanyTabModules(COMPANY_TAB_CODE_PRELOADS, () => cancelled);
      });
    });
    return () => {
      cancelled = true;
      handle.cancel();
    };
  }, [enabled]);
}
