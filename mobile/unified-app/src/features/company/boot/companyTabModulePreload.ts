import { useEffect } from "react";
import { InteractionManager } from "react-native";

/**
 * NAV-01 — preload CODE des onglets barre (pas les écrans, pas leurs GET).
 * RideCreateModal n’est pas import() ici : un second chunk async fait crasher Metro
 * (`unknown module`) au Fast Refresh / idle require.
 */
export const COMPANY_TAB_CODE_PRELOAD_IDS = [
  "chat.module",
  "menu.module",
] as const;

export type CompanyTabCodePreloadId = (typeof COMPANY_TAB_CODE_PRELOAD_IDS)[number];

export type CompanyTabCodePreload = {
  id: CompanyTabCodePreloadId;
  load: () => Promise<unknown>;
};

export const COMPANY_TAB_CODE_PRELOADS: readonly CompanyTabCodePreload[] = [
  {
    id: "chat.module",
    load: () =>
      Promise.all([
        import("../../../../app/(app)/(company)/messages/_layout"),
        import("../../../../app/(app)/(company)/messages/index"),
      ]),
  },
  {
    id: "menu.module",
    load: () =>
      Promise.all([
        import("../../../../app/(app)/(company)/settings"),
        import("../../../../app/(app)/(company)/clients-facturation"),
      ]),
  },
];

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
 * Après le premier écran utile (lane background). N’exécute aucun prefetch React Query.
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
