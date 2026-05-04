import { PixelRatio } from "react-native";

export type AccessibilityScale = {
  fontScale: number;
  isLargeText: boolean;
  isVeryLargeText: boolean;
};

/**
 * Zoom police système / accessibilité (V1).
 *
 * - Source : `PixelRatio.getFontScale()` à **chaque rendu** (pas de polling).
 * - Ne pas utiliser `allowFontScaling={false}` globalement.
 * - UI courte (tabs, CTA, badges) : `maxFontSizeMultiplier` ~1.2–1.3 + truncate si besoin.
 * - Contenu long : multiplicateur plus élevé (~1.5) ou défaut RN.
 *
 * V2 (hors périmètre V1) : si « Larger Text » à chaud reste obsolète sur device,
 * brancher `AccessibilityInfo` / `AppState` ou équivalent.
 */
export function useAccessibilityScale(): AccessibilityScale {
  const fontScale = PixelRatio.getFontScale();
  return {
    fontScale,
    isLargeText: fontScale >= 1.15,
    isVeryLargeText: fontScale >= 1.3,
  };
}
