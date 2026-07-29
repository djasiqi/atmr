/**
 * Caps Larger Text / Dynamic Type — séparés pour ne pas gonfler
 * la densité horizontale au même rythme que le contenu.
 */

/** Contenu essentiel (adresses, titres, erreurs, labels CTA). */
export const CONTENT_FONT_CAP = 2.0;

/** UI courte chrome (tabs, badges, meta dense). */
export const CHROME_FONT_CAP = 1.3;

/** Boost minHeight / paddingVertical utiles. */
export const VERTICAL_LAYOUT_SCALE_CAP = 1.5;

/** Gaps et padding horizontal. */
export const DENSITY_SCALE_CAP = 1.2;

/** Rayons de bordure. */
export const RADIUS_SCALE_CAP = 1.1;

export type AppTextScaleRole = "content" | "chrome";

export function fontCapForScaleRole(role: AppTextScaleRole): number {
  return role === "chrome" ? CHROME_FONT_CAP : CONTENT_FONT_CAP;
}

/** Borne une échelle brute par un plafond dédié. */
export function clampScale(fontScale: number, cap: number): number {
  if (!Number.isFinite(fontScale) || fontScale <= 0) return 1;
  return Math.min(fontScale, cap);
}
