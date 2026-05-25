/** Marge visuelle composeur ↔ limite bas visible (objectif 8 px). */
export const COMPOSER_EDGE_GAP = 8;

/**
 * Marge de sécurité Android (base) quand le clavier est visible.
 *
 * Les claviers OEM (Samsung, certains Pixel) exposent une toolbar
 * (suggestions, raccourcis, Bixby/Gboard) dont la hauteur n'est pas
 * totalement reportée par `keyboardDidShow.endCoordinates`. La marge
 * couvre l'écart résiduel pour éviter que le composeur touche la
 * toolbar.
 *
 * Utiliser {@link computeOemSafetyMargin} qui applique un scale par
 * `fontScale` au-dessus de 1.0 (composer plus grand → marge plus large).
 */
export const OEM_TOOLBAR_SAFETY_MARGIN_PX = 8;

/** Plafond de la marge de sécurité (évite des espaces excessifs en très large text). */
export const OEM_TOOLBAR_SAFETY_MARGIN_MAX_PX = 20;

/**
 * Marge de sécurité Android, scalée par `fontScale` au-dessus de 1.0.
 *
 * - `fontScale ≤ 1` (S23 0.8, défaut) → 8 px (rendu "propre" confirmé QA terrain).
 * - `fontScale = 1.15` (Pixel 9 Pro accessibilité légère) → 12 px.
 * - `fontScale = 1.3` (large text) → 15 px.
 * - Plafonné à {@link OEM_TOOLBAR_SAFETY_MARGIN_MAX_PX}.
 *
 * Le coefficient 24 a été calibré sur le delta `fontScale 0.8 → 1.15` pour passer
 * de "tout juste" à "aéré" sur Pixel 9 Pro (cf. snapshot 2026-05-21 14:32).
 */
export function computeOemSafetyMargin(fontScale: number): number {
  if (!Number.isFinite(fontScale) || fontScale <= 1) {
    return OEM_TOOLBAR_SAFETY_MARGIN_PX;
  }
  const extra = Math.round((fontScale - 1) * 24);
  return Math.min(
    OEM_TOOLBAR_SAFETY_MARGIN_MAX_PX,
    OEM_TOOLBAR_SAFETY_MARGIN_PX + extra
  );
}

/** Seuil minimal de resize fenêtre pour le considérer fiable (adjustResize). */
const WINDOW_RESIZE_TRUST_THRESHOLD = 12;

export type KeyboardLayoutInput = {
  baselineWindowHeight: number;
  windowHeight: number;
  keyboardHeight: number;
  screenY: number;
};

export type VisibleBottomMetrics = {
  resizeDelta: number;
  fromScreenY: number;
  fromHeight: number;
  measuredSlack: number;
  effectiveKeyboardTopY: number;
  visibleBottomInset: number;
};

/**
 * Slack > 0 = écart mesuré entre screenY et height.
 * La toolbar/accessoires se situent typiquement dans cette zone au-dessus de screenY.
 */
export function computeEffectiveKeyboardTopY(
  screenY: number,
  measuredSlack: number
): number {
  if (measuredSlack > 0) {
    return screenY - measuredSlack;
  }
  return screenY;
}

/**
 * Inset bas visible — mesure où le contenu s'arrête, sans hauteur codée en dur.
 */
export function computeVisibleBottomInsets(
  input: KeyboardLayoutInput
): VisibleBottomMetrics {
  const { baselineWindowHeight, windowHeight, keyboardHeight, screenY } = input;
  const resizeDelta = Math.max(0, baselineWindowHeight - windowHeight);
  const hasReliableResize = resizeDelta >= WINDOW_RESIZE_TRUST_THRESHOLD;
  const isScreenYCorrupted =
    screenY <= 1 || (!hasReliableResize && screenY >= windowHeight - 4);
  const safeScreenY = isScreenYCorrupted ? Math.max(0, windowHeight - keyboardHeight) : screenY;

  if (__DEV__ && isScreenYCorrupted) {
    console.warn("[keyboard-layout] corrupted screenY", {
      screenY,
      keyboardHeight,
      windowHeight,
      safeScreenY,
    });
  }

  const fromScreenY = Math.max(0, windowHeight - safeScreenY);
  const fromHeight = Math.max(0, keyboardHeight);
  const measuredSlack = Math.max(0, fromScreenY - fromHeight);
  const effectiveKeyboardTopY = computeEffectiveKeyboardTopY(safeScreenY, measuredSlack);
  const fromEffectiveTop = Math.max(0, windowHeight - effectiveKeyboardTopY);

  let visibleBottomInset: number;

  if (resizeDelta >= WINDOW_RESIZE_TRUST_THRESHOLD) {
    visibleBottomInset = Math.max(resizeDelta, fromScreenY, fromEffectiveTop);
  } else {
    visibleBottomInset = Math.max(fromHeight, fromScreenY, fromEffectiveTop);
  }

  return {
    resizeDelta,
    fromScreenY,
    fromHeight,
    measuredSlack,
    effectiveKeyboardTopY,
    visibleBottomInset,
  };
}

/** Convertit un inset fenêtre en offset `bottom` relatif au shell conversation. */
export function shellFooterOffset(
  visibleBottomInset: number,
  shellBottomGap: number
): number {
  return Math.max(0, visibleBottomInset - shellBottomGap);
}

/**
 * Correction mesurée après layout.
 *
 * On ne corrige que les chevauchements positifs (footer **sous** le haut clavier).
 * Les écarts négatifs sont attendus : ils correspondent à la marge de sécurité
 * `OEM_TOOLBAR_SAFETY_MARGIN_PX` injectée par {@link useChatFooterLayout} pour
 * compenser les toolbars OEM non mesurées. Les annuler créait un cycle
 * "remonte 8 px / corrige -8 px" visible sur S23.
 */
export function computeFooterLiftCorrection(
  footerBottomY: number,
  keyboardTopY: number
): number {
  const correction = footerBottomY - keyboardTopY;
  if (correction <= 2) return 0;
  return Math.round(correction);
}
