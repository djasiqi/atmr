import { useMemo } from "react";
import {
  computeClientFloatingBottomPad,
  computeCompanyFloatingBottomPad,
} from "./BaseFloatingBar";
import {
  computeFloatingBarFallbackClearance,
  FLOATING_BAR_FALLBACK_INNER,
  useFloatingBarClearance,
  useFloatingBarMetrics,
} from "./floatingBarMetrics";
import { useAppViewport } from "../responsive/useAppViewport";

/**
 * Métriques navigation centralisées — source unique LayoutDebugOverlay + API stable.
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

export function useNavigationMetrics(): NavigationMetrics {
  const { bottomInset } = useAppViewport();
  const companyPad = computeCompanyFloatingBottomPad(bottomInset);
  const clientPad = computeClientFloatingBottomPad(bottomInset);
  const metrics = useFloatingBarMetrics("company", companyPad);
  const clearance = useFloatingBarClearance("company", companyPad);

  return useMemo(() => {
    const fallbackClearance = computeFloatingBarFallbackClearance(
      FLOATING_BAR_FALLBACK_INNER.company,
      companyPad
    );
    const height = metrics.clearance || fallbackClearance;
    return {
      topBarHeight: ENTERPRISE_HEADER_HEIGHT,
      bottomBarHeight: height,
      floatingBarHeight: height,
      floatingBarBottomPad: Math.max(companyPad, clientPad),
      tabBarClearance: clearance,
      composerClearance: clearance,
    };
  }, [companyPad, clientPad, metrics.clearance, clearance]);
}
