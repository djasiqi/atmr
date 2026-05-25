import { useMemo } from "react";
import {
  computeClientFloatingBottomPad,
  computeCompanyFloatingBottomPad,
  computeFloatingTabBarClearance,
  computeFloatingTabComposerClearance,
} from "./BaseFloatingBar";
import { useAppViewport } from "../responsive/useAppViewport";

/**
 * Métriques navigation centralisées — Sprint 2 finalisera l'adoption.
 *
 * Sprint 1 : source unique pour LayoutDebugOverlay + base API stable.
 */
export type NavigationMetrics = {
  /** Hauteur indicative en-tête sticky (EnterpriseHeader). Précis Sprint 2. */
  topBarHeight: number;
  /** Hauteur slot bottom bar floatant (pilule + pad) — null si non applicable. */
  bottomBarHeight: number | null;
  /** Hauteur totale réservée par la barre flottante. */
  floatingBarHeight: number | null;
  /** Padding bas (depuis bottomInset). */
  floatingBarBottomPad: number;
  /** Clearance contenu scroll au-dessus de la pilule (preset entreprise). */
  tabBarClearance: number;
  /** Clearance pied de page chat au-dessus tab (preset entreprise). */
  composerClearance: number;
};

const ENTERPRISE_HEADER_HEIGHT = 56;
const FLOATING_TAB_PILL_HEIGHT = 56;

export function useNavigationMetrics(): NavigationMetrics {
  const { bottomInset } = useAppViewport();

  return useMemo(() => {
    const companyPad = computeCompanyFloatingBottomPad(bottomInset);
    const clientPad = computeClientFloatingBottomPad(bottomInset);
    return {
      topBarHeight: ENTERPRISE_HEADER_HEIGHT,
      bottomBarHeight: FLOATING_TAB_PILL_HEIGHT + companyPad,
      floatingBarHeight: FLOATING_TAB_PILL_HEIGHT + companyPad,
      floatingBarBottomPad: Math.max(companyPad, clientPad),
      tabBarClearance: computeFloatingTabBarClearance(bottomInset),
      composerClearance: computeFloatingTabComposerClearance(bottomInset),
    };
  }, [bottomInset]);
}
