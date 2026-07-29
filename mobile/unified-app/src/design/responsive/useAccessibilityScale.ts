import { useWindowDimensions } from "react-native";
import { useAppViewport } from "./useAppViewport";
import { CHROME_FONT_CAP, CONTENT_FONT_CAP } from "./fontScaleCaps";

export type AccessibilityScale = {
  fontScale: number;
  isLargeText: boolean;
  isVeryLargeText: boolean;
  /** Empiler les rangées horizontales (très grande police ou écran ≤ 360 px utiles). */
  shouldStackRows: boolean;
  contentMaxFontMultiplier: number;
  chromeMaxFontMultiplier: number;
};

/**
 * Calcule les flags a11y à partir de fontScale + largeur utile (pur, testable).
 *
 * `shouldStackRows` utilise `usableWidth` (pas `contentWidth`) : contentWidth retire
 * déjà les gutters et stackerait trop tôt sur la plupart des téléphones.
 */
export function computeAccessibilityScale(
  fontScale: number,
  usableWidth: number
): AccessibilityScale {
  const safeScale = Number.isFinite(fontScale) && fontScale > 0 ? fontScale : 1;
  const isLargeText = safeScale >= 1.15;
  const isVeryLargeText = safeScale >= 1.3;
  const shouldStackRows = isVeryLargeText || usableWidth <= 360;

  return {
    fontScale: safeScale,
    isLargeText,
    isVeryLargeText,
    shouldStackRows,
    contentMaxFontMultiplier: CONTENT_FONT_CAP,
    chromeMaxFontMultiplier: CHROME_FONT_CAP,
  };
}

/**
 * Zoom police système / accessibilité.
 *
 * - Source : `useWindowDimensions().fontScale` (réactif aux changements système).
 * - Ne pas utiliser `allowFontScaling={false}` globalement.
 * - Contenu : `CONTENT_FONT_CAP` (2.0) ; chrome : `CHROME_FONT_CAP` (1.3).
 */
export function useAccessibilityScale(): AccessibilityScale {
  const { fontScale } = useWindowDimensions();
  const { usableWidth } = useAppViewport();
  return computeAccessibilityScale(fontScale, usableWidth);
}
